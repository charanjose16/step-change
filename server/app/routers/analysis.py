import asyncio
import os
import re
import shutil
import tempfile
from typing import List
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, validator
from app.utils.file_utils import cleanup_temp_dir, get_hierarchical_files
from app.utils.s3_utils import download_s3_folder
from app.services.analysis_service import (
    generate_requirements_for_files_whole,
    generate_graph_from_requirement,
    GraphResponse,
    FilesRequirements
)
from app.utils import logger
from threading import Lock

log_lock = Lock()

router = APIRouter(prefix="/analysis", tags=["Analysis"])

class AnalyzeRequest(BaseModel):
    folder_name: str

    @validator("folder_name")
    def validate_folder_name(cls, value):
        # Basic validation for S3 folder name (alphanumeric, slashes, hyphens, underscores)
        if not re.match(r'^[\w\-/]+/?$', value):
            raise ValueError("Invalid folder name. Use alphanumeric characters, slashes, hyphens, or underscores.")
        return value

class GraphAllRequest(BaseModel):
    requirement: str

def clean_folder_name(name: str) -> str:
    """
    Clean folder name for use as root name in hierarchy.
    
    Args:
        name (str): Raw folder name
    
    Returns:
        str: Cleaned folder name
    """
    if not name:
        return "project-root"
    
    # Remove trailing/leading slashes, hyphens, and whitespace
    cleaned = name.strip('/\\- ')
    
    # Remove 'main' or 'main/' at the end
    cleaned = re.sub(r'main/?$', '', cleaned)
    
    # Remove any remaining trailing hyphens
    cleaned = cleaned.rstrip('-')
    
    # Extract basename
    cleaned = os.path.basename(cleaned)
    
    # Replace invalid characters with underscores
    cleaned = re.sub(r'[^\w\-]', '_', cleaned)
    
    # Ensure non-empty name
    return cleaned or "project-root"

@router.post("", summary="Download S3 folder and generate requirements")
async def analyze_s3_folder(data: AnalyzeRequest, background_tasks: BackgroundTasks):
    folder_name = data.folder_name
    is_temp = True

    # Clean folder name for display
    cleaned_folder_name = clean_folder_name(folder_name)
    with log_lock:
        logger.debug(f"Original folder_name: {folder_name}, Cleaned folder_name: {cleaned_folder_name}")

    # Download the S3 folder to a temporary directory
    try:
        directory = await download_s3_folder(folder_name)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download S3 folder: {str(e)}"
        )

    # Schedule cleanup of temporary directory
    background_tasks.add_task(cleanup_temp_dir, directory)

    # Process the directory to generate requirements for each file
    try:
        files_requirements: FilesRequirements = await generate_requirements_for_files_whole(directory)
    except Exception as e:
        if is_temp:
            await asyncio.get_running_loop().run_in_executor(None, shutil.rmtree, directory, True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating requirements: {str(e)}"
        )

    # Get hierarchical file structure with cleaned folder name
    file_hierarchy = get_hierarchical_files(directory, root_name=cleaned_folder_name)

    # Return the base directory path, requirements, and file hierarchy
    return {
        "success": True,
        "directory": directory,
        "requirements": [
            {"file_name": fr.file_name, "relative_path": fr.relative_path, "requirements": fr.requirements}
            for fr in files_requirements.files
        ],
        "file_hierarchy": file_hierarchy,
        "summary":files_requirements.project_summary
    }

@router.post("/graphs", response_model=List[GraphResponse],

             summary="Generate graphs from file requirement")

async def get_all_graphs(payload: GraphAllRequest):

    """

    Generate graphs for the provided requirement.

    Predefined target graphs are 'Entity Relationship Diagram' and

    'Requirement Diagram'.

    Returns an array of GraphResponse objects.

    Implements caching based on a hash of the file content to avoid

    repeated LLM calls.

    """

    import hashlib

    import os

    import json

    # Retrieve file content

    try:

        file_content = await retrieve_file_content(payload.requirement)

    except Exception as e:

        logger.warning(f"Could not retrieve file content: {e}")

        file_content = payload.requirement

    # Ensure we have a string to hash

    if not isinstance(file_content, str):

        file_content = str(file_content)

    # Compute unique cache ID

    content_hash = hashlib.sha256(

        file_content.encode("utf-8")

    ).hexdigest()

    base_cache_dir = "graphs"

    cache_dir = os.path.join(base_cache_dir, content_hash)

    # Define the set of target graphs

    target_graphs = [

        "Entity Relationship Diagram",

        "Requirement Diagram"

    ]

    # Attempt to load from cache

    if os.path.isdir(cache_dir):

        cached_results = []

        for tg in target_graphs:

            # sanitize filename by replacing spaces with underscores

            fname = f"{tg.replace(' ', '_')}.json"

            fpath = os.path.join(cache_dir, fname)

            if not os.path.isfile(fpath):

                # Incomplete cache, skip to generation

                cached_results = None

                break

            with open(fpath, "r", encoding="utf-8") as f:

                data = json.load(f)

            # Reconstruct GraphResponse

            cached_results.append(GraphResponse(**data))

        if cached_results is not None:

            return cached_results

    # Cache miss: generate the graphs

    tasks = [

        generate_graph_from_requirement(payload.requirement, tg, file_content)

        for tg in target_graphs

    ]

    try:

        results = await asyncio.gather(*tasks)

    except Exception as e:

        logger.error(f"Error generating graphs: {str(e)}")

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=f"Error generating graphs: {str(e)}"

        )

    # Persist to cache

    os.makedirs(cache_dir, exist_ok=True)

    for res in results:

        fname = f"{res.target_graph.replace(' ', '_')}.json"

        fpath = os.path.join(cache_dir, fname)

        with open(fpath, "w", encoding="utf-8") as f:

            json.dump({

                "target_graph": res.target_graph,

                "generated_code": res.generated_code

            }, f, ensure_ascii=False, indent=2)

    return results
 
async def retrieve_file_content(requirement):
    return requirement

from fastapi import Body
from app.utils.file_utils import analyze_project_dependencies

class ProjectGraphRequest(BaseModel):
    folder_name: str

    @validator("folder_name")
    def validate_folder_name(cls, value):
        if not re.match(r'^[\w\-/]+/?$', value):
            raise ValueError("Invalid folder name. Use alphanumeric characters, slashes, hyphens, or underscores.")
        return value

@router.post("/project-graph", summary="Generate a unified D3 hierarchical graph of the full project")
async def project_graph(data: ProjectGraphRequest = Body(...)):
    folder_name = data.folder_name
    # Download and extract S3 folder
    try:
        directory = await download_s3_folder(folder_name)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error downloading S3 folder: {str(e)}"
        )
    # Build unified hierarchical project graph
    from app.utils.file_utils import build_project_graph
    from .analysis import clean_folder_name
    try:
        root_name = clean_folder_name(folder_name)
        graph = build_project_graph(directory, root_name=root_name)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error building project graph: {str(e)}"
        )
    # Optionally clean up the temp dir in background
    # background_tasks.add_task(cleanup_temp_dir, directory)
    return {"success": True, "graph": graph}
