import asyncio
import shutil
import os
from typing import List, Tuple
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
                
                # If successful, add to code files
                code_files.append((full_path, ""))
            
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
        '.ts': 'TypeScript',
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
    
    # Return language or empty string
    return LANGUAGE_MAP.get(ext.lower(), '')