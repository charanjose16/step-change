import asyncio
import os
import re
import shutil
import tempfile
from typing import List
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, validator
from app.utils.file_utils import cleanup_temp_dir
from app.utils.git_utils import clone_repository
from app.services.analysis_service import (
    generate_requirements_for_files_whole,
    generate_graph_from_requirement,
    GraphResponse,
    FilesRequirements
)

router = APIRouter(prefix="/analysis", tags=["Analysis"])

class AnalyzeRequest(BaseModel):
    link: str

    @validator("link")
    def validate_link(cls, value):
        # Accept either an HTTP URL or an absolute file path.
        if value.startswith("http"):
            if not re.match(r"^https?://\S+$", value):
                raise ValueError("Invalid URL format")
            return value
        # Check for Windows absolute path (e.g., C:\...) or Unix-like absolute path (e.g., /...)
        if not (re.match(r"^[A-Za-z]:\\", value) or value.startswith("/")):
             # Basic check for relative paths or invalid formats
            if not os.path.isabs(value):
                 raise ValueError("Link must be either an HTTP URL or an absolute file path")
        return value

class GraphAllRequest(BaseModel):
    requirement: str

@router.post("", summary="Clone repository/local folder and generate requirements")
async def analyze_repo(data: AnalyzeRequest, background_tasks: BackgroundTasks):
    link = data.link
    is_temp = False
    if link.startswith("http"):
        # Clone the repository into a temporary directory (via git_utils)
        try:
            directory = await clone_repository(link)
            is_temp = True
        except Exception as e:
             raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to clone repository: {str(e)}"
            )
    else:
        # Use the provided absolute path
        if not os.path.isdir(link):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provided path does not exist or is not a directory"
            )
        directory = link

    # Process the directory to generate requirements for each file.
    try:
        files_requirements: FilesRequirements = await generate_requirements_for_files_whole(directory)
    except Exception as e:
        if is_temp:
            # If an error occurred and this is a temporary directory, clean it up immediately.
            await asyncio.get_running_loop().run_in_executor(None, shutil.rmtree, directory, True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating requirements: {str(e)}"
        )

    requirements_output = [
        {"file_name": fr.file_name, "relative_path": fr.relative_path, "requirements": fr.requirements}
        for fr in files_requirements.files
    ]

    # If it's a temporary directory, schedule cleanup after sending the response.
    if is_temp:
        background_tasks.add_task(cleanup_temp_dir, directory)

    # Return the base directory path along with requirements
    return {"success": True, "directory": directory, "requirements": requirements_output}


@router.post("/graphs", response_model=List[GraphResponse], summary="Generate graphs from file requirement")
async def get_all_graphs(payload: GraphAllRequest):
    """
    Generate graphs for the provided requirement.
    Predefined target graphs are 'Entity Relationship Diagram' and 'Requirement Diagram'.
    Returns an array of GraphResponse objects.
    """
    # Attempt to retrieve the file content based on the requirement
    file_content = None
    try:
        # You might need to implement a method to retrieve file content
        # This is a placeholder - adjust according to your actual file retrieval mechanism
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

# You'll need to implement this function based on your file retrieval logic
async def retrieve_file_content(requirement):
    # This is a placeholder. Implement actual file content retrieval
    # You might need to pass additional context or use a database/file system lookup
    return requirement  # Temporary fallback