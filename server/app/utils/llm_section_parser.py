import re
from typing import Dict, List, Union

SECTION_HEADERS = [
    "Overview",
    "Objective",
    "Use Case",
    "Key Functionalities",
    "Workflow Summary",
    "Dependent Files"
]

SECTION_REGEX = re.compile(rf"^({'|'.join(re.escape(h) for h in SECTION_HEADERS)})\\s*$", re.MULTILINE)

def parse_llm_business_logic_sections(
    llm_output: str
) -> Dict[str, Union[str, List[str]]]:
    """
    Parse LLM output into a structured dict by section headers.
    Returns a dict with keys for each section. Key Functionalities is a list. Others are strings.
    """
    # Split by double newlines and find sections
    sections = {}
    chunks = re.split(r"\n\n", llm_output.strip())
    current_header = None
    for chunk in chunks:
        lines = [l.strip() for l in chunk.splitlines() if l.strip()]
        if not lines:
            continue
        header = None
        for h in SECTION_HEADERS:
            if lines[0].startswith(h):
                header = h
                break
        if header:
            content_lines = lines[1:] if len(lines) > 1 else []
            if header == "Key Functionalities":
                # Parse as numbered list
                numbered = [l for l in content_lines if re.match(r"^\d+\. ", l)]
                if not numbered:
                    # fallback: treat each line as a functionality
                    numbered = content_lines
                sections[header] = [re.sub(r"^\d+\. ", "", l).strip() for l in numbered if l.strip()]
            elif header == "Dependent Files":
                # Each line is a dependency (may be empty)
                sections[header] = [l for l in content_lines if l.strip()]
            else:
                sections[header] = " ".join(content_lines).strip() if content_lines else ""
            current_header = header
        elif current_header:
            # Continuation of previous section
            if current_header == "Key Functionalities":
                if current_header not in sections:
                    sections[current_header] = []
                sections[current_header].extend([re.sub(r"^\d+\. ", "", l).strip() for l in lines if l.strip()])
            elif current_header == "Dependent Files":
                if current_header not in sections:
                    sections[current_header] = []
                sections[current_header].extend([l for l in lines if l.strip()])
            else:
                prev = sections.get(current_header, "")
                add = " ".join(lines).strip()
                sections[current_header] = (prev + " " + add).strip()
    # Ensure all sections exist
    for h in SECTION_HEADERS:
        if h not in sections:
            sections[h] = [] if h in ["Key Functionalities", "Dependent Files"] else ""
    # Output as snake_case keys
    return {
        "overview": sections["Overview"],
        "objective": sections["Objective"],
        "use_case": sections["Use Case"],
        "key_functionalities": sections["Key Functionalities"],
        "workflow_summary": sections["Workflow Summary"],
        "dependent_files": sections["Dependent Files"]
    }
