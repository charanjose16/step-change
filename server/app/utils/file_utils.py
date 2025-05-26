import asyncio
import shutil
import os
from typing import List, Tuple, Dict, Any
import re

async def cleanup_temp_dir(temp_dir: str):
    # Wait 5 minutes before cleaning up the temporary directory (non-blocking)
    await asyncio.sleep(300)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, shutil.rmtree, temp_dir, True)

def get_code_files(directory: str) -> List[Tuple[str, str]]:
    """
    Recursively gather code files while excluding unnecessary files.
    
    Exclusion Criteria:
    1. Hidden directories and files
    2. Configuration and metadata files
    3. Dependency and build-related files
    4. Binary and compiled files
    5. Documentation and media files
    """
    # Comprehensive list of files and directories to ignore
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

    # Compile regex patterns for efficiency
    ignore_patterns = [re.compile(pattern) for pattern in IGNORE_PATTERNS]

    def should_ignore(path):
        """
        Determine if a file or directory should be ignored.
        
        Args:
            path (str): Full path to the file or directory
        
        Returns:
            bool: True if the path should be ignored, False otherwise
        """
        # Check against ignore patterns
        for pattern in ignore_patterns:
            if pattern.search(path):
                return True
        
        # Additional custom checks
        filename = os.path.basename(path)
        
        # Ignore files starting with dot
        if filename.startswith('.'):
            return True
        
        return False

    code_files = []
    
    # Walk through directory
    for root, dirs, files in os.walk(directory):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if not should_ignore(os.path.join(root, d))]
        
        for file in files:
            full_path = os.path.join(root, file)
            
            # Skip ignored files
            if should_ignore(full_path):
                continue
            
            # Optional: Add more sophisticated file type detection
            try:
                # Basic file type detection
                with open(full_path, 'r', encoding='utf-8') as f:
                    # Read first few lines to validate text file
                    f.read(1024)
                
                # If successful, add to code files with detected language
                language = get_file_language(full_path)
                code_files.append((full_path, language))
            
            except (UnicodeDecodeError, IOError):
                # Skip binary or non-readable files
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
    # Language extension mapping
    LANGUAGE_MAP = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.jsx': 'JavaScript',  # Treat JSX as JavaScript for requirements generation
        '.ts': 'TypeScript',
        '.tsx': 'TypeScript',  # Treat TSX as TypeScript
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
    
    # Get file extension
    _, ext = os.path.splitext(file_path)
    
    # Log the extension and detected language for debugging
    detected_language = LANGUAGE_MAP.get(ext.lower(), '')
    from app.utils import logger
    logger.debug(f"File: {file_path}, Extension: {ext}, Detected Language: {detected_language}")
    
    # Return language or empty string
    return detected_language

def get_hierarchical_files(directory: str) -> Dict[str, Any]:
    """
    Recursively build a hierarchical file structure.
    
    Args:
        directory (str): Base directory path
    
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
        
        # If it's a directory, recursively add its contents
        if os.path.isdir(path):
            try:
                for item in sorted(os.listdir(path)):
                    full_path = os.path.join(path, item)
                    
                    # Skip ignored files and directories
                    if should_ignore(full_path):
                        continue
                    
                    child_item = build_hierarchy(full_path)
                    hierarchy['children'].append(child_item)
            except PermissionError:
                # Handle permission issues gracefully
                pass
        
        return hierarchy

    # Reuse the existing should_ignore logic from get_code_files
    def should_ignore(path):
        IGNORE_PATTERNS = [
            # ... (keep the existing ignore patterns from get_code_files)
        ]
        
        # Compile regex patterns for efficiency
        ignore_patterns = [re.compile(pattern) for pattern in IGNORE_PATTERNS]
        
        # Check against ignore patterns
        for pattern in ignore_patterns:
            if pattern.search(path):
                return True
        
        # Additional custom checks
        filename = os.path.basename(path)
        
        # Ignore files starting with dot
        if filename.startswith('.'):
            return True
        
        return False

    # Build and return the hierarchical structure
    return build_hierarchy(directory)