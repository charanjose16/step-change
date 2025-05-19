import asyncio
import shutil
import os
from typing import List, Tuple

async def cleanup_temp_dir(temp_dir: str):
    # Wait 5 minutes before cleaning up the temporary directory (non-blocking)
    await asyncio.sleep(300)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, shutil.rmtree, temp_dir, True)

def get_code_files(directory: str) -> List[Tuple[str, str]]:
    """
    Recursively gather all file paths from the given directory,
    skipping hidden directories (e.g. .git) to avoid binary files.
    
    Returns a list of tuples where each tuple contains:
        (full_file_path, "")
    The second value is kept empty for now.
    """
    code_files: List[Tuple[str, str]] = []
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories (those starting with a dot)
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            full_path = os.path.join(root, file)
            code_files.append((full_path, ""))
    return code_files