import re

def is_generic_llm_output(text: str) -> bool:
    """
    Returns True if the LLM output is likely generic, empty, or not business-meaningful.
    Heuristics:
      - Output is empty or very short
      - Contains fallback phrases (e.g., 'No business logic detected', 'This file', 'utility file', etc.)
      - Lacks numbered Key Functionalities or required section headers
    """
    if not text or len(text.strip()) < 50:
        return True
    generic_patterns = [
        r"No business logic detected",
        r"utility file",
        r"This file (does|is|contains)",
        r"No explicit business logic",
        r"Error: LLM failed",
        r"Overview\s*\n\s*Objective\s*\n\s*Use Case",  # All sections empty
    ]
    for pat in generic_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    # Check if Key Functionalities section is missing or too short
    if "Key Functionalities" not in text or len(re.findall(r"\d+\. ", text)) < 1:
        return True
    return False
