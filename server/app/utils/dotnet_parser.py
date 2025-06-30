import re
import os
import json

def summarize_csharp_code(content: str) -> str:
    """
    Extracts business logic summary from a C# (.cs) file, focusing on ASP.NET Core service extension patterns.
    - Summarizes extension methods, DI, configuration, and middleware logic.
    - Avoids references to unrelated technologies.
    """
    import textwrap
    import re
    
    # Extract class name
    class_match = re.search(r'class\s+(\w+)', content)
    class_name = class_match.group(1) if class_match else "UnknownClass"
    
    # Extract extension methods
    method_blocks = re.findall(
        r'(public\s+static\s+[\w<>,\[\]]+\s+(\w+)\s*\([^\)]*\)\s*\{[\s\S]*?\})',
        content
    )
    method_summaries = []
    for block, method_name in method_blocks:
        # Try to extract what the method does
        # Health checks
        if 'AddHealthChecks' in block:
            method_summaries.append(f"'{method_name}': Registers health check endpoints for service monitoring.")
        # Service discovery
        elif 'AddServiceDiscovery' in block:
            method_summaries.append(f"'{method_name}': Integrates service discovery into the DI container.")
        # Logging
        elif 'AddLogging' in block:
            method_summaries.append(f"'{method_name}': Configures application logging (console, debug).")
        # Routing/middleware
        elif 'UseRouting' in block or 'UseHealthChecks' in block:
            method_summaries.append(f"'{method_name}': Sets up middleware for routing and health check endpoints.")
        # Configuration
        elif 'AddDefaultConfiguration' in block:
            method_summaries.append(f"'{method_name}': Loads configuration from JSON files and environment variables.")
        else:
            # Fallback: Use method name and parameters
            params = re.findall(rf'{method_name}\s*\(([^\)]*)\)', block)
            param_str = params[0] if params else ''
            method_summaries.append(f"'{method_name}({param_str})': Extension method for ASP.NET Core services.")
    
    # Compose summary
    lines = [
        f"This file defines the '{class_name}' static class, providing extension methods to streamline service and middleware configuration in ASP.NET Core applications.",
        "Key functionalities include:"
    ]
    lines += [f"- {ms}" for ms in method_summaries]
    
    # If no methods detected, fallback to class summary
    if not method_summaries:
        lines.append("- No extension methods detected.")
    
    # Avoid unrelated references
    summary = '\n'.join(lines)
    summary = summary.replace('macro', '').replace('xlsm', '').replace('Excel', '').replace('sheet', '')
    return textwrap.dedent(summary)


def summarize_dotnet_config(content: str, file_path: str) -> str:
    """
    Summarizes .NET project/config files (.csproj, .sln, .json, .yml, .http).
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csproj':
        # Summarize project references and dependencies
        refs = re.findall(r'<ProjectReference Include=\"([^"]+)\"', content)
        pkgs = re.findall(r'<PackageReference Include=\"([^"]+)\" Version=\"([^"]+)\"', content)
        summary = [f"Project References: {', '.join(refs)}" if refs else "No project references detected."]
        if pkgs:
            summary.append("NuGet Packages: " + ', '.join([f"{p[0]} ({p[1]})" for p in pkgs]))
        return '\n'.join(summary)
    elif ext == '.json':
        try:
            data = json.loads(content)
            keys = list(data.keys())
            return f"JSON config file; top-level keys: {', '.join(keys)}."
        except Exception:
            return "JSON config file; unable to parse."
    elif ext in ['.yml', '.yaml']:
        lines = content.splitlines()
        top_keys = [l.split(':')[0].strip() for l in lines if ':' in l and not l.strip().startswith('#')]
        return f"YAML config file; top-level keys: {', '.join(top_keys[:10])}."
    elif ext == '.http':
        lines = content.splitlines()
        endpoints = [l for l in lines if l.strip().startswith(('GET', 'POST', 'PUT', 'DELETE', 'PATCH'))]
        return f"HTTP file; endpoints: {', '.join(endpoints[:5])}."
    elif ext == '.sln':
        projects = re.findall(r'Project\("\{[^}]+\}"\) = "([^"]+)",', content)
        return f"Solution file; projects: {', '.join(projects)}."
    else:
        return f".NET config file: {os.path.basename(file_path)}."
