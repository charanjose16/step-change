import re
import os

def summarize_angular_file(content: str, file_path: str) -> str:
    """
    Extracts business logic summary from Angular files (.ts, .html).
    - For .ts: Summarizes components, services, and modules.
    - For .html: Summarizes main UI elements.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.ts':
        # Detect Angular decorators for components/services/modules
        decorators = re.findall(r'@(Component|Injectable|NgModule)\s*\(', content)
        classes = re.findall(r'class\s+(\w+)', content)
        summary = []
        if decorators:
            summary.append(f"Decorators: {', '.join(set(decorators))}.")
        if classes:
            summary.append(f"Classes: {', '.join(classes)}.")
        # Extract selector and template info
        selectors = re.findall(r'selector:\s*[\'\"]([\w-]+)[\'\"]', content)
        if selectors:
            summary.append(f"Selectors: {', '.join(selectors)}.")
        return ' '.join(summary) if summary else "Angular TypeScript file; no decorators or classes detected."
    elif ext == '.html':
        # Summarize main UI tags
        tags = re.findall(r'<([a-zA-Z0-9-_]+)', content)
        tag_counts = {}
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        main_tags = sorted(tag_counts, key=tag_counts.get, reverse=True)[:5]
        return f"HTML template file; main tags: {', '.join(main_tags)}."
    elif ext == '.json':
        # For Angular .json (e.g. angular.json)
        import json
        try:
            data = json.loads(content)
            keys = list(data.keys())
            return f"Angular JSON config; top-level keys: {', '.join(keys)}."
        except Exception:
            return "Angular JSON config file; unable to parse."
    else:
        return f"Angular file: {os.path.basename(file_path)}."
