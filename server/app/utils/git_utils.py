import os
import asyncio
import subprocess
import shutil
import tempfile
from fastapi import HTTPException, status

async def clone_repository(repo_url: str) -> str:
    """
    Clone a git repository from the given URL into a temporary directory.
    Returns the path where the repository was cloned.
    Raises an HTTPException on failure.
    """
    # Create a temporary directory for cloning
    temp_dir = tempfile.mkdtemp()
    
    # Extract the repository name from the URL, strip trailing '.git' if present.
    repo_name = repo_url.rstrip('/').split('/')[-1]
    if repo_name.endswith('.git'):
        repo_name = repo_name[:-4]
    
    clone_path = os.path.join(temp_dir, repo_name)
    
    def run_clone():
        return subprocess.run(
            ['git', 'clone', '--depth', '1', repo_url, clone_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    process = await asyncio.to_thread(run_clone)
    
    if process.returncode != 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error cloning repository: " + process.stderr.decode()
        )
    
    # Return the path where the repository was cloned
    return clone_path