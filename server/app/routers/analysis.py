
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
        "file_hierarchy": file_hierarchy
    }

@router.post("/graphs", response_model=List[GraphResponse], summary="Generate graphs from file requirement")
async def get_all_graphs(payload: GraphAllRequest):
    """
    Generate graphs for the provided requirement.
    Predefined target graphs are 'Entity Relationship Diagram' and 'Requirement Diagram'.
    Returns an array of GraphResponse objects.
    """
    file_content = None
    try:
        file_content = await retrieve_file_content(payload.requirement)
    except Exception as e:
        logger.warning(f"Could not retrieve file content: {e}")

    target_graphs = [
        "Entity Relationship Diagram",
        "Requirement Diagram"
    ]
    
    tasks = [
        generate_graph_from_requirement(payload.requirement, tg, file_content)
        for tg in target_graphs
    ]
    
    try:
        results = await asyncio.gather(*tasks)
        return results
    except Exception as e:
        logger.error(f"Error generating graphs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating graphs: {str(e)}"
        )

async def retrieve_file_content(requirement):
    return requirement
