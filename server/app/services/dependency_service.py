import os
import re
from typing import List
from pydantic import BaseModel, ValidationError
from app.utils import logger
from app.utils.file_utils import get_code_files
from threading import Lock
import aiofiles

log_lock = Lock()

class FileDependency(BaseModel):
    file_name: str
    relative_path: str
    dependency_reason: str

async def detect_dependencies(file_path: str, content: str, language: str, base_dir: str) -> List[FileDependency]:
    """
    Detect dependencies in the file content based on language-specific patterns.
    Returns a list of FileDependency objects with file_name, relative_path, and dependency_reason.
    """
    dependencies = []
    try:
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

        if language.lower() in ['javascript', 'jsx', 'typescript', 'tsx']:
            # Updated regex to match relative imports
            import_pattern = r'^\s*import\s+[\w\s{},*]*\s*from\s*[\'"]((\.\.?/[^\'"]*))[\'"](?:\s*;)?$'
            matches = re.findall(import_pattern, content, re.MULTILINE)
            with log_lock:
                logger.debug(f"Found {len(matches)} import matches in {file_path}: {[m[1] for m in matches]}")

            for _, import_path in matches:
                # Normalize import path
                import_name = os.path.basename(import_path)
                import_dir = os.path.dirname(import_path)
                with log_lock:
                    logger.debug(f"Checking import: {import_path} (base: {import_name}, dir: {import_dir})")

                # Check for matching file with possible extensions
                possible_extensions = ['.js', '.jsx', '.ts', '.tsx', '']
                for ext in possible_extensions:
                    candidate_name = import_name + ext if ext else import_name
                    candidate_path = os.path.normpath(os.path.join(import_dir, candidate_name).lstrip('./'))
                    if candidate_name in code_files_map:
                        rel_path = code_files_map[candidate_name]
                        full_path = os.path.join(base_dir, rel_path)
                        if os.path.exists(full_path):
                            try:
                                dep = FileDependency(
                                    file_name=os.path.basename(full_path),
                                    relative_path=rel_path,
                                    dependency_reason=f"Imports {import_name}, providing components or utilities for {os.path.basename(file_path)}."
                                )
                                dependencies.append(dep)
                                with log_lock:
                                    logger.debug(f"Detected dependency: {candidate_name} -> {rel_path}")
                                break
                            except ValidationError as ve:
                                with log_lock:
                                    logger.error(f"Validation error for FileDependency {candidate_name}: {ve}")
                    elif import_name in code_files_map:
                        rel_path = code_files_map[import_name]
                        full_path = os.path.join(base_dir, rel_path)
                        if os.path.exists(full_path):
                            try:
                                dep = FileDependency(
                                    file_name=os.path.basename(full_path),
                                    relative_path=rel_path,
                                    dependency_reason=f"Imports {import_name}, providing components or utilities for {os.path.basename(file_path)}."
                                )
                                dependencies.append(dep)
                                with log_lock:
                                    logger.debug(f"Detected dependency (extension-less): {import_name} -> {rel_path}")
                                break
                            except ValidationError as ve:
                                with log_lock:
                                    logger.error(f"Validation error for FileDependency {import_name}: {ve}")
                    else:
                        with log_lock:
                            logger.debug(f"No matching file found for import: {import_path}")

        elif language.lower() in ['python']:
            import_pattern = r'^(?:from|import)\s+([\w\.]+)(?:\s+import|\s*,\s*)'
            matches = re.findall(import_pattern, content, re.MULTILINE)
            for match in matches:
                module = match.split('.')[-1]
                for ext in ['.py']:
                    possible_file = f"{module}{ext}"
                    if possible_file in code_files_map:
                        rel_path = code_files_map[possible_file]
                        try:
                            dep = FileDependency(
                                file_name=possible_file,
                                relative_path=rel_path,
                                dependency_reason=f"Imports {module} module, providing functionality for {os.path.basename(file_path)}."
                            )
                            dependencies.append(dep)
                            with log_lock:
                                logger.debug(f"Detected Python dependency: {possible_file} -> {rel_path}")
                        except ValidationError as ve:
                            with log_lock:
                                logger.error(f"Validation error for FileDependency {possible_file}: {ve}")

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
            logger.debug(f"Dependencies before return: {[dep.dict() for dep in unique_dependencies]}")
        return unique_dependencies

    except Exception as e:
        with log_lock:
            logger.error(f"Error detecting dependencies for {file_path}: {str(e)}")
        return []