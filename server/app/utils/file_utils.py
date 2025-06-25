
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
            # Exclude CSS files
            if file.lower().endswith('.css'):
                continue
            # Only include code and specified non-code files
            allowed_exts = [
                '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.scala', '.rb', '.go', '.cpp', '.c', '.rs', '.kt', '.swift', '.php',
                '.cs', '.vb', '.fs', '.csproj', '.vbproj', '.fsproj', '.sln',
                '.json', '.txt', '.xlsx', '.xlsm', '.doc', '.md'
            ]
            if not any(file.lower().endswith(ext) for ext in allowed_exts):
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

import ast
import json
from collections import defaultdict


def build_project_graph(directory: str, root_name: str = None) -> dict:
    """
    Build a unified project graph combining folder/file hierarchy and code dependencies.
    Returns a dict with nodes, edges, and hierarchy (tree structure).
    Highlights important/mandatory functions by name.
    Adds explicit parent-child hierarchy edges.
    """
    # 1. Get hierarchy tree
    hierarchy = get_hierarchical_files(directory, root_name)

    # 2. Get flat dependency graph
    dep_graph = analyze_project_dependencies(directory)
    nodes = dep_graph['nodes']
    dep_edges = dep_graph['edges']

    # 3. Build folder/file/function nodes from hierarchy
    def flatten_hierarchy(node, parent_id=None, flat_nodes=None, hierarchy_edges=None):
        if flat_nodes is None:
            flat_nodes = []
        if hierarchy_edges is None:
            hierarchy_edges = []
        node_type = node.get('type')
        node_id = node.get('path')
        flat_node = {
            'id': node_id,
            'name': node.get('name'),
            'type': node_type,
            'parent': parent_id
        }
        flat_nodes.append(flat_node)
        if parent_id is not None:
            hierarchy_edges.append({'source': parent_id, 'target': node_id, 'type': 'hierarchy'})
        for child in node.get('children', []):
            flatten_hierarchy(child, node_id, flat_nodes, hierarchy_edges)
        return flat_nodes, hierarchy_edges

    flat_hierarchy_nodes, hierarchy_edges = flatten_hierarchy(hierarchy)
    flat_hierarchy_ids = set(n['id'] for n in flat_hierarchy_nodes)

    # 4. Merge function/file nodes into hierarchy nodes
    merged_nodes = {n['id']: n for n in flat_hierarchy_nodes}
    for n in nodes:
        if n['type'] == 'file' and n['id'] not in merged_nodes:
            merged_nodes[n['id']] = {
                'id': n['id'],
                'name': n['id'],
                'type': 'file',
                'parent': None
            }
        if n['type'] == 'function':
            # Parent is the file (should exist in merged_nodes)
            merged_nodes[n['id']] = {
                'id': n['id'],
                'name': n['id'].split(':')[-1],
                'type': 'function',
                'parent': n.get('parent')
            }
            # Add hierarchy edge from file to function
            parent_file = n.get('parent')
            if parent_file:
                hierarchy_edges.append({'source': parent_file, 'target': n['id'], 'type': 'hierarchy'})

    # 5. Highlight important/mandatory functions
    important_keywords = ['main', 'run', 'start', 'init', 'entry', 'execute']
    for node in merged_nodes.values():
        if node['type'] == 'function':
            lname = node['name'].lower()
            if any(kw in lname for kw in important_keywords):
                node['important'] = True
                node['type'] = 'important_function'

    # 6. Output unified structure
    all_edges = hierarchy_edges + dep_edges
    return {
        'nodes': list(merged_nodes.values()),
        'edges': all_edges,
        'hierarchy': hierarchy
    }

def analyze_project_dependencies(directory: str) -> dict:
    """
    Analyze code dependencies and function relationships for Python and JS files.
    Returns a hierarchical structure for D3 visualization.
    """
    code_files = get_code_files(directory)
    file_nodes = {}
    edges = []

    def analyze_python(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            try:
                tree = ast.parse(source, filename=file_path)
                parse_chunks = False
            except (SyntaxError, IndentationError) as e:
                from app.utils import logger
                logger.warning(f"Parse error in file: {file_path} ({e.__class__.__name__}: {e}). Attempting to extract valid blocks.")
                parse_chunks = True
            file_name = os.path.relpath(file_path, directory)
            functions = []
            function_calls = defaultdict(list)
            imports = []

            def analyze_ast(tree):
                class Analyzer(ast.NodeVisitor):
                    def visit_Import(self, node):
                        for alias in node.names:
                            imports.append(alias.name)
                        self.generic_visit(node)
                    def visit_ImportFrom(self, node):
                        if node.module:
                            imports.append(node.module)
                        self.generic_visit(node)
                    def visit_FunctionDef(self, node):
                        functions.append(node.name)
                        for n in ast.walk(node):
                            if isinstance(n, ast.Call):
                                if isinstance(n.func, ast.Name):
                                    function_calls[node.name].append(n.func.id)
                                elif isinstance(n.func, ast.Attribute):
                                    function_calls[node.name].append(n.func.attr)
                        self.generic_visit(node)
                Analyzer().visit(tree)

            if not parse_chunks:
                analyze_ast(tree)
            else:
                # Try to parse each top-level function/class block individually
                import re
                block_starts = [m.start() for m in re.finditer(r'^(def |class )', source, re.MULTILINE)]
                block_starts.append(len(source))
                for i in range(len(block_starts) - 1):
                    chunk = source[block_starts[i]:block_starts[i+1]]
                    try:
                        chunk_tree = ast.parse(chunk, filename=file_path)
                        analyze_ast(chunk_tree)
                    except (SyntaxError, IndentationError) as e:
                        from app.utils import logger
                        logger.warning(f"Skipping block in {file_path} due to parse error: {e.__class__.__name__}: {e}")
                        continue
            return {
                'file': file_name,
                'language': 'Python',
                'imports': imports,
                'functions': functions,
                'function_calls': dict(function_calls)
            }
        except Exception as e:
            from app.utils import logger
            logger.warning(f"Failed to analyze file: {file_path} ({e.__class__.__name__}: {e})")
            return None  # Skip this file


    def analyze_js(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
        file_name = os.path.relpath(file_path, directory)
        imports = re.findall(r'import\s+(?:[\w*{}, ]+\s+from\s+)?[\'\"]([\w\-/\.]+)[\'\"]', source)
        functions = re.findall(r'function\s+(\w+)\s*\(', source)
        function_calls = defaultdict(list)
        for func in functions:
            body_match = re.search(r'function\s+' + re.escape(func) + r'\s*\([^)]*\)\s*{([^}]*)}', source, re.DOTALL)
            if body_match:
                body = body_match.group(1)
                calls = re.findall(r'(\w+)\s*\(', body)
                function_calls[func].extend(calls)
        return {
            'file': file_name,
            'language': 'JavaScript',
            'imports': imports,
            'functions': functions,
            'function_calls': dict(function_calls)
        }

    # Build nodes and edges
    for file_path, language in code_files:
        if language == 'Python':
            result = analyze_python(file_path)
        elif language == 'JavaScript':
            result = analyze_js(file_path)
        else:
            continue
        file_nodes[result['file']] = result

    # Build hierarchical structure for D3
    nodes = []
    node_indices = {}
    idx = 0
    for file, data in file_nodes.items():
        nodes.append({'id': file, 'type': 'file', 'language': data['language']})
        node_indices[file] = idx
        idx += 1
        for func in data['functions']:
            func_id = f"{file}:{func}"
            nodes.append({'id': func_id, 'type': 'function', 'parent': file})
            node_indices[func_id] = idx
            idx += 1
    # File import edges
    for file, data in file_nodes.items():
        for imp in data['imports']:
            # Try to resolve import to another file node
            for target_file in file_nodes:
                if imp in target_file or imp.split('.')[-1] in target_file:
                    edges.append({'source': file, 'target': target_file, 'type': 'import'})
    # Function call edges
    for file, data in file_nodes.items():
        for func, calls in data['function_calls'].items():
            for call in calls:
                for target_func in data['functions']:
                    if call == target_func:
                        edges.append({'source': f"{file}:{func}", 'target': f"{file}:{target_func}", 'type': 'call'})
    return {'nodes': nodes, 'edges': edges}


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
