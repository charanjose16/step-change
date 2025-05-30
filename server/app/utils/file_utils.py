
import asyncio
import shutil
import os
import re
from typing import List, Tuple, Dict, Any

# Define IGNORE_PATTERNS at module scope
IGNORE_PATTERNS = [
    # Version Control
    r'\.git$',
    r'\.gitignore$',
    r'\.gitmodules$',
    
    # Environment Files
    r'\.env$',
    r'\.env\.*',
    
    # Dependency Directories
    r'node_modules$',
    r'vendor$',
    r'__pycache__$',
    r'\.venv$',
    r'venv$',
    
    # Build and Compilation Outputs
    r'build$',
    r'dist$',
    r'target$',
    r'out$',
    r'\.next$',
    r'\.nuxt$',
    
    # Package Management
    r'package-lock\.json$',
    r'yarn\.lock$',
    r'poetry\.lock$',
    r'Pipfile\.lock$',
    r'requirements\.txt$',
    r'pyproject\.toml$',
    r'package\.json$',
    
    # Configuration Files
    r'\.eslintrc$',
    r'\.prettierrc$',
    r'tsconfig\.json$',
    r'babel\.config\.js$',
    r'webpack\.config\.js$',
    
    # IDE and Editor Files
    r'\.vscode$',
    r'\.idea$',
    r'\.eclipse$',
    r'\.vs$',
    
    # Logs and Temporary Files
    r'\.log$',
    r'logs$',
    r'tmp$',
    
    # Documentation and Media
    r'README\.md$',
    r'LICENSE$',
    r'\.(png|jpg|jpeg|gif|svg|ico)$',
    r'\.(mp3|mp4|wav)$',
    
    # Compiled and Binary Files
    r'\.(class|jar|pyc|pyo|pyd)$',
    r'\.(exe|dll|so|dylib)$',
    
    # Backup and System Files
    r'\.DS_Store$',
    r'Thumbs\.db$',
]

# Compile ignore patterns once at module level
IGNORE_PATTERN_COMPILED = [re.compile(pattern) for pattern in IGNORE_PATTERNS]

def should_ignore(path: str) -> bool:
    """
    Determine if a file or directory should be ignored.
    
    Args:
        path (str): Full path to the file or directory
    
    Returns:
        bool: True if the path should be ignored, False otherwise
    """
    for pattern in IGNORE_PATTERN_COMPILED:
        if pattern.search(path):
            return True
    
    filename = os.path.basename(path)
    if filename.startswith('.'):
        return True
    
    return False

async def cleanup_temp_dir(temp_dir: str):
    """
    Clean up temporary directory after a delay.
    
    Args:
        temp_dir (str): Path to the temporary directory
    """
    await asyncio.sleep(300)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, shutil.rmtree, temp_dir, True)

def get_code_files(directory: str) -> List[Tuple[str, str]]:
    """
    Recursively gather code files while excluding unnecessary files.
    
    Args:
        directory (str): Directory to scan
    
    Returns:
        List of tuples containing (file_path, language)
    """
    code_files = []
    
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not should_ignore(os.path.join(root, d))]
        
        for file in files:
            full_path = os.path.join(root, file)
            
            if should_ignore(full_path):
                continue
            
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    f.read(1024)
                
                language = get_file_language(full_path)
                code_files.append((full_path, language))
            
            except (UnicodeDecodeError, IOError):
                continue
    
    return code_files

def get_file_language(file_path: str) -> str:
    """
    Detect programming language based on file extension.
    
    Args:
        file_path (str): Path to the file
    
    Returns:
        str: Detected language or empty string
    """
    LANGUAGE_MAP = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.jsx': 'JavaScript',
        '.ts': 'TypeScript',
        '.tsx': 'TypeScript',
        '.java': 'Java',
        '.scala': 'Scala',
        '.rb': 'Ruby',
        '.go': 'Go',
        '.cpp': 'C++',
        '.c': 'C',
        '.rs': 'Rust',
        '.kt': 'Kotlin',
        '.swift': 'Swift',
        '.php': 'PHP',
        '.cs': 'C#'
    }
    
    _, ext = os.path.splitext(file_path)
    
    from app.utils import logger
    logger.debug(f"File: {file_path}, Extension: {ext}, Detected Language: {LANGUAGE_MAP.get(ext.lower(), '')}")
    
    return LANGUAGE_MAP.get(ext.lower(), '')

def clean_folder_name(name: str) -> str:
    """
    Clean folder name by removing trailing hyphens, slashes, and invalid characters.
    
    Args:
        name (str): Raw folder name
    
    Returns:
        str: Cleaned folder name
    """
    if not name:
        return "project-root"
    
    # Remove trailing/leading slashes, hyphens, and whitespace
    cleaned = name.strip('/\\- ')
    
    # Remove any remaining trailing hyphens
    cleaned = cleaned.rstrip('-')
    
    # Replace invalid characters with underscores
    cleaned = re.sub(r'[^\w\-]', '_', cleaned)
    
    # Ensure non-empty name
    return cleaned or "project-root"

def get_hierarchical_files(directory: str, root_name: str = None) -> Dict[str, Any]:
    """
    Recursively build a hierarchical file structure.
    
    Args:
        directory (str): Base directory path
        root_name (str, optional): Custom name for the root folder, defaults to directory basename
    
    Returns:
        Dict representing the hierarchical file structure
    """
    def build_hierarchy(path):
        hierarchy = {
            'name': os.path.basename(path),
            'path': path,
            'type': 'directory' if os.path.isdir(path) else 'file',
            'children': []
        }
        
        if os.path.isdir(path):
            try:
                for item in sorted(os.listdir(path)):
                    full_path = os.path.join(path, item)
                    
                    if should_ignore(full_path):
                        continue
                    
                    child_item = build_hierarchy(full_path)
                    hierarchy['children'].append(child_item)
            except PermissionError:
                pass
        
        return hierarchy

    hierarchy = build_hierarchy(directory)
    if root_name:
        hierarchy['name'] = clean_folder_name(root_name)
    else:
        hierarchy['name'] = clean_folder_name(os.path.basename(directory))
    return hierarchy
