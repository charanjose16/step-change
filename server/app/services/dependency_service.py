
import os
import re
from typing import List
from pydantic import BaseModel, ValidationError
from app.utils import logger
from app.utils.file_utils import get_code_files
from threading import Lock

log_lock = Lock()

class FileDependency(BaseModel):
    file_name: str
    relative_path: str
    dependency_reason: str

# Language-specific configuration for dependency detection
LANGUAGE_CONFIG = {
    'javascript': {
        'extensions': ['.js', '.jsx'],
        'patterns': [
            r'^\s*import\s+[\w\s{},*]*\s*from\s*[\'"]((\.\.?/[^\'"]*))[\'"](?:\s*;)?$',
            r'^\s*require\s*\(\s*[\'"]((\.\.?/[^\'"]*))[\'"]\s*\)',
        ],
        'reason': "Imports {import_name}, providing components or utilities for {file_name}.",
    },
    'typescript': {
        'extensions': ['.ts', '.tsx'],
        'patterns': [
            r'^\s*import\s+[\w\s{},*]*\s*from\s*[\'"]((\.\.?/[^\'"]*))[\'"](?:\s*;)?$',
        ],
        'reason': "Imports {import_name}, providing components, services, or modules for {file_name} (Angular/TypeScript).",
    },
    'python': {
        'extensions': ['.py'],
        'patterns': [
            r'^(?:from|import)\s+([\w\.]+)(?:\s+import|\s*,\s*)',
        ],
        'reason': "Imports {import_name} module, providing functionality for {file_name}.",
    },
    'csharp': {
        'extensions': ['.cs'],
        'patterns': [
            r'^\s*using\s+([\w\.]+)\s*;',  # Matches 'using System.Text;'
        ],
        'reason': "References {import_name} namespace, providing classes or utilities for {file_name} (.NET/C#).",
    },
    'java': {
        'extensions': ['.java'],
        'patterns': [
            r'^\s*import\s+([\w\.]+)\s*;',  # Matches 'import java.util.List;'
        ],
        'reason': "Imports {import_name} package or class, providing functionality for {file_name} (Java).",
    },
    'ruby': {
        'extensions': ['.rb'],
        'patterns': [
            r'^\s*require\s+[\'"]([^\'"]+)[\'"]',  # Matches 'require "module"'
        ],
        'reason': "Requires {import_name} module, providing functionality for {file_name} (Ruby).",
    },
}

async def detect_dependencies(file_path: str, content: str, language: str, base_dir: str) -> List[FileDependency]:
    """
    Detect dependencies in the file content based on language-specific patterns.
    Returns a list of FileDependency objects with file_name, relative_path, and dependency_reason.
    
    Args:
        file_path (str): Path to the file being analyzed
        content (str): Content of the file
        language (str): Programming language of the file
        base_dir (str): Base directory for resolving relative paths
    
    Returns:
        List[FileDependency]: List of detected dependencies
    """
    dependencies = []
    try:
        # Normalize language to lowercase
        language = language.lower()

        # Precompute a map of file names (with and without extensions) to their relative paths
        code_files_map = {}
        for file_full_path, _ in get_code_files(base_dir):
            file_name = os.path.basename(file_full_path)
            rel_path = os.path.relpath(file_full_path, base_dir)
            code_files_map[file_name] = rel_path
            base_name, _ = os.path.splitext(file_name)
            code_files_map[base_name] = rel_path

        with log_lock:
            logger.debug(f"Processing dependencies for {file_path}. Available files: {list(code_files_map.keys())}")

        # Find matching language configuration
        lang_config = None
        for lang, config in LANGUAGE_CONFIG.items():
            if language in lang or any(file_path.endswith(ext) for ext in config['extensions']):
                lang_config = config
                language = lang  # Standardize language name
                break

        if not lang_config:
            with log_lock:
                logger.warning(f"No dependency detection configured for language: {language}")
            return []

        # Process each pattern for the language
        for pattern in lang_config['patterns']:
            matches = re.findall(pattern, content, re.MULTILINE)
            with log_lock:
                logger.debug(f"Found {len(matches)} import matches in {file_path} for {language}: {matches}")

            for match in matches:
                import_name = match if isinstance(match, str) else match[1]  # Handle tuple or string
                import_dir = os.path.dirname(import_name) if language in ['javascript', 'typescript'] else ''
                import_base = os.path.basename(import_name)

                # Skip standard library or external modules
                if language in ['python', 'csharp', 'java'] and '.' in import_name:
                    continue  # e.g., 'System.Text' or 'java.util'

                # Normalize import path for JavaScript/TypeScript
                if language in ['javascript', 'typescript']:
                    import_path = os.path.normpath(os.path.join(import_dir, import_base).lstrip('./'))
                else:
                    import_path = import_base

                # Check for matching file with possible extensions
                possible_extensions = lang_config['extensions'] + ['']  # Include extension-less
                for ext in possible_extensions:
                    candidate_name = import_path + ext if ext else import_path
                    if language in ['python', 'ruby', 'csharp', 'java']:
                        candidate_name = import_base + ext if ext else import_base

                    if candidate_name in code_files_map:
                        rel_path = code_files_map[candidate_name]
                        full_path = os.path.join(base_dir, rel_path)
                        if os.path.exists(full_path):
                            try:
                                dep = FileDependency(
                                    file_name=os.path.basename(full_path),
                                    relative_path=rel_path,
                                    dependency_reason=lang_config['reason'].format(
                                        import_name=import_base,
                                        file_name=os.path.basename(file_path)
                                    )
                                )
                                dependencies.append(dep)
                                with log_lock:
                                    logger.debug(f"Detected {language} dependency: {candidate_name} -> {rel_path}")
                                break
                            except ValidationError as ve:
                                with log_lock:
                                    logger.error(f"Validation error for FileDependency {candidate_name}: {ve}")
                    elif import_base in code_files_map:
                        rel_path = code_files_map[import_base]
                        full_path = os.path.join(base_dir, rel_path)
                        if os.path.exists(full_path):
                            try:
                                dep = FileDependency(
                                    file_name=os.path.basename(full_path),
                                    relative_path=rel_path,
                                    dependency_reason=lang_config['reason'].format(
                                        import_name=import_base,
                                        file_name=os.path.basename(file_path)
                                    )
                                )
                                dependencies.append(dep)
                                with log_lock:
                                    logger.debug(f"Detected {language} dependency (extension-less): {import_base} -> {rel_path}")
                                break
                            except ValidationError as ve:
                                with log_lock:
                                    logger.error(f"Validation error for FileDependency {import_base}: {ve}")
                    else:
                        with log_lock:
                            logger.debug(f"No matching file found for {language} import: {import_path}")

        # Remove duplicates
        seen = set()
        unique_dependencies = []
        for dep in dependencies:
            dep_key = f"{dep.file_name}:{dep.relative_path}"
            if dep_key not in seen:
                seen.add(dep_key)
                unique_dependencies.append(dep)

        with log_lock:
            logger.info(f"Detected {len(unique_dependencies)} unique dependencies for {file_path}: {[dep.file_name for dep in unique_dependencies]}")
            logger.debug(f"Dependencies: {[dep.dict() for dep in unique_dependencies]}")
        return unique_dependencies

    except Exception as e:
        with log_lock:
            logger.error(f"Error detecting dependencies for {file_path}: {str(e)}")
        return []
