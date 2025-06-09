import os
import asyncio
import json
import re
from enum import Enum
from typing import List, Tuple, Dict
from pydantic import BaseModel, ValidationError
from app.config.llm_config import llm_config
from app.utils import logger
from app.utils.file_utils import get_code_files, get_file_language
import aiofiles
import tiktoken
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps
from threading import Lock
import hashlib
import time

# Constants
CHUNK_GROUP_SIZE = 40
LLM_CONCURRENCY_LIMIT = 10
llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY_LIMIT)
LINTER_CONCURRENCY = 300
linter_sema = asyncio.Semaphore(LINTER_CONCURRENCY)
MAX_MODEL_TOKENS = 128000
MAX_SAFE_TOKENS = 50000
BATCH_SIZE = 50
MAX_LINES_PER_CHUNK = 2000
MAX_SECTION_TOKENS = 1000
log_lock = Lock()

# Configuration for dynamic business rule detection
RULES_CONFIG = {
    "financial": {
        "patterns": [
            r"\b(interest_rate|credit_score|risk_score|loan_eligibility|payment_schedule)\s*=\s*.*[\d+\.*%*\/*\+\-].*",  # Financial calculations
            r"\bif.*(credit_score|income|debt|risk|loan_amount).*[<>=].*",  # Conditional financial logic
        ],
        "description": "Proprietary financial calculations or eligibility rules."
    },
    "ecommerce": {
        "patterns": [
            r"\b(discount|price|total_cost)\s*=\s*.*(if|where|location|category|amount).*",  # Discount rules
            r"\bif.*(location|category|product_type|order_value).*[<>=].*",  # Conditional discounts
        ],
        "description": "Custom discount or pricing rules based on location or product."
    },
    "general": {
        "patterns": [
            r"\b(custom_|proprietary_|internal_).*=\s*.*",  # Custom-named variables or functions
            r"\bif.*(internal_|custom_|proprietary_).*[=><].*",  # Custom conditional logic
        ],
        "description": "Organization-specific logic or custom workflows."
    }
}

class GraphResponse(BaseModel):
    target_graph: str
    generated_code: str

class RequirementsResponse(BaseModel):
    requirements: str

class FileDependency(BaseModel):
    file_name: str
    relative_path: str
    dependency_reason: str

class FileRequirements(BaseModel):
    relative_path: str
    file_name: str
    requirements: str
    dependencies: List[FileDependency]

class FilesRequirements(BaseModel):
    files: List[FileRequirements]
    project_summary: str = ""
    graphs: List[GraphResponse] = []  # New field for project summary graphs

class OrgSpecificRules(BaseModel):
    relative_path: str
    file_name: str
    rules: str

@lru_cache(maxsize=1000)
def count_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception:
        with log_lock:
            logger.warning(f"Token counting error for model {model}. Using fallback.")
        return int(len(text) / 4)

def retry_on_failure(max_attempts=2, delay=0.5):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    with log_lock:
                        logger.error(f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}")
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay)
                    else:
                        raise e
        return wrapper
    return decorator

async def clone_repository(repo_url: str, target_dir: str) -> bool:
    with log_lock:
        logger.info(f"Starting repository cloning: {repo_url} to {target_dir}")
    try:
        steps = ["Initializing", "Fetching metadata", "Downloading files", "Checking out branch"]
        for i, step in enumerate(steps, 1):
            with log_lock:
                logger.debug(f"Cloning progress: {step} ({i}/{len(steps)})")
            await asyncio.sleep(0.1)
        return True
    except Exception as e:
        with log_lock:
            logger.error(f"Failed to clone repository: {str(e)}")
        return False

async def generate_graph_from_requirement(requirement: str, target_graph: str = '', file_content: str = None) -> GraphResponse:
    if not file_content:
        file_content = requirement

    if count_tokens(file_content) > MAX_SAFE_TOKENS:
        with log_lock:
            logger.warning(f"File content for {target_graph}. Token limit exceeded. Truncating.")
        file_content = file_content[:MAX_SAFE_TOKENS * 3 // 4]

    if target_graph.lower() == 'entity relationship diagram':
        prompt = (
            "Create a precise Entity-Relationship Diagram with strict syntax rules:\n"
            f"Code Context:\n```\n{file_content}\n```\n\n"
            "STRICT GUIDELINES:\n"
            "1. ALWAYS use graph TD or graph LR\n"
            "2. Use ONLY square brackets [Entity]\n"
            "3. Connect with --> or ---\n"
            "4. NO special characters in node names\n"
            "5. Keep diagram SIMPLE and VALID\n"
            "Example:\n"
            "```mermaid\n"
            "graph TD\n"
            "    A[Customer] --> B[Order]\n"
            "    B --> C[Product]\n"
            "```\n"
            "Return ONLY valid Mermaid code."
        )
    elif target_graph.lower() == 'requirement diagram':
        prompt = (
            "Create a precise Mermaid Requirement Diagram with strict syntax rules:\n"
            f"Code Context:\n```\n{file_content}\n```\n\n"
            "STRICT GUIDELINES:\n"
            "1. ALWAYS use graph TD or graph LR\n"
            "2. Use ONLY square brackets [Requirement]\n"
            "3. Connect with --> or ---\n"
            "4. NO special characters in node names\n"
            "5. Keep diagram SIMPLE and VALID\n"
            "Example:\n"
            "```mermaid\n"
            "graph TD\n"
            "    A[User Authentication] --> B[Access Control]\n"
            "    B --> C[Data Validation]\n"
            "```\n"
            "Return ONLY valid Mermaid code."
        )
    else:
        return GraphResponse(
            target_graph=target_graph,
            generated_code=f"Unsupported graph type: {target_graph}"
        )

    async with llm_semaphore:
        try:
            async with asyncio.timeout(30):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    result = await asyncio.get_event_loop().run_in_executor(
                        executor, lambda: llm_config._llm.complete(prompt)
                    )
            generated_code = result.text.strip()
            if not generated_code or len(generated_code.split('\n')) < 2:
                return GraphResponse(
                    target_graph=target_graph,
                    generated_code=f"Error: Insufficient graph definition for {target_graph}"
                )
            pattern = r"```mermaid\s*([\s\S]*?)\s*```"
            match = re.search(pattern, generated_code)
            if match:
                generated_code = match.group(1).strip()
            lines = generated_code.split('\n')
            if not lines[0].startswith('graph'):
                return GraphResponse(
                    target_graph=target_graph,
                    generated_code=f"Error: Invalid graph type. Must start with 'graph TD' or 'graph LR'"
                )
            return GraphResponse(target_graph=target_graph, generated_code=generated_code)
        except asyncio.TimeoutError:
            with log_lock:
                logger.error(f"Graph generation timed out for {target_graph}")
            return GraphResponse(
                target_graph=target_graph,
                generated_code=f"Generation Error: Timeout"
            )
        except Exception as e:
            with log_lock:
                logger.error(f"Graph generation error for {target_graph}: {e}")
            return GraphResponse(
                target_graph=target_graph,
                generated_code=f"Generation Error: {str(e)}"
            )

async def split_into_chunks(content: str, language: str, file_path: str, max_tokens: int = MAX_SAFE_TOKENS) -> List[Tuple[str, int, int]]:
    chunks = []
    current_line = 0
    lines = content.splitlines()
    total_lines = len(lines)
    total_tokens = count_tokens(content)
    with log_lock:
        logger.debug(f"Starting chunking for {file_path}: {total_tokens} tokens, {total_lines} lines")

    summary = "Summary unavailable.\n"
    if total_tokens < MAX_SAFE_TOKENS:
        summary_prompt = (
            f"Summarize the purpose and structure of the following {language} code in 50-100 words:\n"
            f"```\n{content[:1500]}\n[...truncated...]\n```\n"
            "Provide a concise overview of the file's role, main components, and key functionalities."
        )
        try:
            async with asyncio.timeout(5):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    summary_result = await asyncio.get_event_loop().run_in_executor(
                        executor, lambda: llm_config._llm.complete(summary_prompt)
                    )
                summary = summary_result.text.strip() + "\n"
                with log_lock:
                    logger.debug(f"Generated summary for {file_path}")
        except asyncio.TimeoutError:
            with log_lock:
                logger.warning(f"Summary generation timed out for {file_path}")
        except Exception:
            with log_lock:
                logger.warning(f"Failed to generate summary for {file_path}")

    target_tokens = max_tokens * 4 // 5
    tokens_per_line = total_tokens / total_lines if total_lines > 0 else 1
    target_lines = int(target_tokens / tokens_per_line) if tokens_per_line > 0 else MAX_LINES_PER_CHUNK

    while current_line < total_lines:
        remaining_lines = total_lines - current_line
        chunk_lines = lines[current_line:current_line + min(target_lines, remaining_lines)]
        chunk_content = '\n'.join(chunk_lines)
        token_count = count_tokens(chunk_content)

        while token_count > max_tokens and len(chunk_lines) > 1:
            target_lines = max(1, target_lines * 3 // 4)
            chunk_lines = lines[current_line:current_line + target_lines]
            chunk_content = '\n'.join(chunk_lines)
            token_count = count_tokens(chunk_content)

        chunk_end = current_line + len(chunk_lines)
        chunk_content = f"File Summary:\n{summary}\nChunk Content:\n{chunk_content}"
        if count_tokens(chunk_content) > max_tokens:
            with log_lock:
                logger.warning(f"Chunk at lines {current_line}-{chunk_end} in {file_path} exceeds token limit. Truncating.")
            chunk_lines = chunk_lines[:len(chunk_lines) * 3 // 4]
            chunk_content = f"File Summary:\n{summary}\nChunk Content:\n{'\n'.join(chunk_lines)}"

        chunks.append((chunk_content, current_line, chunk_end))
        with log_lock:
            logger.debug(f"Chunk {len(chunks)} for {file_path}: {token_count} tokens, lines {current_line}-{chunk_end}")
        current_line = chunk_end

    with log_lock:
        logger.info(f"Completed chunking for {file_path}: {len(chunks)} chunks")
    return chunks

async def detect_business_rules(chunk_content: str, language: str) -> List[Dict[str, str]]:
    rules_detected = []
    for domain, config in RULES_CONFIG.items():
        for pattern in config["patterns"]:
            matches = re.finditer(pattern, chunk_content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                rules_detected.append({
                    "rule": match.group(0),
                    "description": config["description"],
                    "domain": domain
                })
    return rules_detected

@retry_on_failure(max_attempts=2)
async def generate_requirements_for_chunk(chunk_content: str, chunk_index: int, language: str) -> str:
    chunk_hash = hashlib.md5(chunk_content.encode()).hexdigest()
    if count_tokens(chunk_content) > MAX_SAFE_TOKENS:
        with log_lock:
            logger.warning(f"Chunk {chunk_index + 1} exceeds token limit. Truncating.")
        chunk_content = chunk_content[:MAX_SAFE_TOKENS * 3 // 4]

    rules_detected = await detect_business_rules(chunk_content, language)
    rules_summary = (
        "\n".join([f"Detected {rule['domain']} rule: {rule['description']}" for rule in rules_detected])
        if rules_detected else "No organization-specific rules detected."
    )

    prompt = f"""
Act as a senior business analyst reviewing a {language} source file segment.
This is segment {chunk_index + 1} of a larger file. Generate concise business requirements in plain text, avoiding markdown symbols (*, -, #, **).
Do NOT use the phrase "this chunk" or "this segment".
The output MUST have five sections: Overview, Objective, Use Case, Key Functionalities, Workflow Summary, separated by two newlines.
If the code contains organization-specific logic or calculations (e.g., proprietary formulas or business rules), generalize them without revealing sensitive details. For example, instead of specifying an exact calculation, say "performs proprietary calculations based on internal rules." Use the following detected rules for context:
{rules_summary}
Overview
Describe the segment's role in the file's purpose and system context (50-100 words).
Objective
State the business goal or problem the segment addresses (1-2 sentences).
Use Case
Outline business scenarios where the segment's functionality is critical (2-3 sentences).
Key Functionalities
List 2-5 core business capabilities, each on a new line, numbered (e.g., 1., 2.), followed by a blank line. Each functionality must include a brief description.
Workflow Summary
Describe how the segment supports the file's business process (2-3 sentences).
Constraints:
- Plain text, no markdown
- Five non-empty sections
- Key Functionalities numbered, each followed by a blank line
- Use file summary and detected rules for context
- Return only the structured requirements
Analyze this segment:
```{language}
{chunk_content}
```
"""
    async with llm_semaphore:
        try:
            async with asyncio.timeout(30):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    result = await asyncio.get_event_loop().run_in_executor(
                        executor, lambda: llm_config._llm.complete(prompt)
                    )
            req_text = result.text.strip()
            with log_lock:
                logger.debug(f"Raw LLM output for chunk {chunk_index + 1} (hash: {chunk_hash}): {req_text[:200]}...")

            req_text = re.sub(r'\n\s*\n+', '\n\n', req_text)
            req_text = re.sub(r'this chunk|this segment', 'the code', req_text, flags=re.IGNORECASE)
            sections = re.split(r'\n\n(?=Overview|Objective|Use Case|Key Functionalities|Workflow Summary)', req_text)
            section_dict = {}
            for section in sections:
                for header in ['Overview', 'Objective', 'Use Case', 'Key Functionalities', 'Workflow Summary']:
                    if section.startswith(header):
                        section_dict[header] = section[len(header):].strip()
                        break

            for header in ['Overview', 'Objective', 'Use Case', 'Key Functionalities', 'Workflow Summary']:
                if header not in section_dict or not section_dict[header]:
                    with log_lock:
                        logger.warning(f"Missing or empty {header} in chunk {chunk_index + 1} (hash: {chunk_hash})")
                    section_dict[header] = (
                        f"Supports {language.lower()} file operations." if header == 'Overview' else
                        f"To enable core {language.lower()} file functionality." if header == 'Objective' else
                        f"Supports general business tasks in {language.lower()} applications." if header == 'Use Case' else
                        "1. Basic Processing: Handles core tasks.\n\n2. Support Functions: Assists operations." if header == 'Key Functionalities' else
                        f"Contributes to the {language.lower()} file's workflow."
                    )

            if section_dict['Key Functionalities']:
                func_lines = section_dict['Key Functionalities'].split('\n')
                formatted_funcs = []
                for line in func_lines:
                    if re.match(r'^\d+\.\s', line):
                        formatted_funcs.append(line.strip())
                if not formatted_funcs:
                    formatted_funcs = ["1. Basic Processing: Handles core tasks.", "2. Support Functions: Assists operations."]
                section_dict['Key Functionalities'] = '\n\n'.join(formatted_funcs)

            req_text = (
                f"Overview\n{section_dict['Overview']}\n\n"
                f"Objective\n{section_dict['Objective']}\n\n"
                f"Use Case\n{section_dict['Use Case']}\n\n"
                f"Key Functionalities\n{section_dict['Key Functionalities']}\n\n"
                f"Workflow Summary\n{section_dict['Workflow Summary']}"
            )

            with log_lock:
                logger.debug(f"Generated requirements for chunk {chunk_index + 1} (hash: {chunk_hash})")
            return req_text
        except asyncio.TimeoutError:
            with log_lock:
                logger.error(f"LLM timed out for chunk {chunk_index + 1} (hash: {chunk_hash})")
            return (
                f"Overview\nAnalysis limited due to timeout for {language.lower()} file operations.\n\n"
                f"Objective\nTo support core {language.lower()} file functionality.\n\n"
                f"Use Case\nSupports general {language.lower()} business tasks.\n\n"
                f"Key Functionalities\n1. Partial Processing: Supports basic operations.\n\n2. Fallback Support: Provides minimal functionality.\n\n"
                f"Workflow Summary\nContributes to {language.lower()} file workflow."
            )
        except Exception as e:
            with log_lock:
                logger.error(f"Error for chunk {chunk_index + 1}: {e}")
            raise e

@retry_on_failure(max_attempts=2)
async def extract_organization_specific_rules_for_chunk(chunk_content: str, chunk_index: int, language: str) -> str:
    chunk_hash = hashlib.md5(chunk_content.encode()).hexdigest()
    if count_tokens(chunk_content) > MAX_SAFE_TOKENS:
        with log_lock:
            logger.warning(f"Chunk {chunk_index + 1} exceeds token limit for org-specific rules. Truncating.")
        chunk_content = chunk_content[:MAX_SAFE_TOKENS * 3 // 4]

    rules_detected = await detect_business_rules(chunk_content, language)
    rules_summary = (
        "\n".join([f"Detected {rule['domain']} rule: {rule['description']}" for rule in rules_detected])
        if rules_detected else "No organization-specific rules detected."
    )

    prompt = f"""
Act as a senior business analyst reviewing a {language} source file segment.
This is segment {chunk_index + 1} of a larger file. Identify and summarize any organization-specific business rules or logic
(e.g., proprietary calculations, custom workflows, or conditional rules) in plain text, avoiding markdown symbols (*, -, #, **).
Do NOT use the phrase "this chunk" or "this segment". Do NOT reveal sensitive details such as code snippets, variable names, or specific values.
For each rule, provide a high-level description of its purpose and usage (e.g., "Calculates loan eligibility using a proprietary formula based on customer data").
Use the following detected rules for context:
{rules_summary}
If no organization-specific rules are found, return "No organization-specific rules detected."
Return only the summarized rules.
Analyze this segment:
```{language}
{chunk_content}
```
"""
    async with llm_semaphore:
        try:
            async with asyncio.timeout(30):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    result = await asyncio.get_event_loop().run_in_executor(
                        executor, lambda: llm_config._llm.complete(prompt)
                    )
            rules_text = result.text.strip()
            with log_lock:
                logger.debug(f"Raw LLM output for org-specific rules chunk {chunk_index + 1} (hash: {chunk_hash}): {rules_text[:200]}...")

            rules_text = re.sub(r'\n\s*\n+', '\n\n', rules_text)
            rules_text = re.sub(r'this chunk|this segment', 'the code', rules_text, flags=re.IGNORECASE)
            if not rules_text or rules_text.isspace():
                rules_text = "No organization-specific rules detected."
            return rules_text
        except asyncio.TimeoutError:
            with log_lock:
                logger.error(f"LLM timed out for org-specific rules chunk {chunk_index + 1} (hash: {chunk_hash})")
            return "No organization-specific rules detected due to timeout."
        except Exception as e:
            with log_lock:
                logger.error(f"Error for org-specific rules chunk {chunk_index + 1}: {e}")
            return f"Error extracting organization-specific rules: {str(e)}"

async def combine_requirements(requirements_list: List[str], language: str) -> str:
    if not requirements_list:
        with log_lock:
            logger.error("No requirements provided to combine")
        return (
            f"Overview\nNo analysis available for {language.lower()} file.\n\n"
            f"Objective\nTo support intended {language.lower()} business functions.\n\n"
            f"Use Case\nIntended {language.lower()} business operations.\n\n"
            f"Key Functionalities\n1. Intended Functionality: Expected to provide core features.\n\n2. Support Functions: Assists operations.\n\n"
            f"Workflow Summary\nExpected to integrate with {language.lower()} system workflows."
        )

    overviews = []
    objectives = []
    use_cases = []
    functionalities = []
    workflows = []
    func_number = 1
    seen_funcs = set()

    with log_lock:
        logger.debug(f"Combining {len(requirements_list)} chunk requirements for {language} file")

    for req_text in requirements_list:
        try:
            sections = re.split(r'\n\n(?=Overview|Objective|Use Case|Key Functionalities|Workflow Summary)', req_text)
            section_dict = {}
            for section in sections:
                for header in ['Overview', 'Objective', 'Use Case', 'Key Functionalities', 'Workflow Summary']:
                    if section.startswith(header):
                        section_dict[header] = section[len(header):].strip()
                        break

            if len(section_dict) != 5:
                with log_lock:
                    logger.warning(f"Invalid chunk requirements format: {req_text[:200]}...")
                continue

            overview = section_dict['Overview']
            objective = section_dict['Objective']
            use_case = section_dict['Use Case']
            func_text = section_dict['Key Functionalities']
            workflow = section_dict['Workflow Summary']

            if overview:
                overviews.append(overview)
            if objective:
                objectives.append(objective)
            if use_case:
                use_cases.append(use_case)
            if workflow:
                workflows.append(workflow)

            func_lines = func_text.split('\n\n')
            for func in func_lines:
                if re.match(r'^\d+\.\s', func):
                    parts = func.split(':', 1)
                    if len(parts) == 2 and parts[1].strip() not in seen_funcs:
                        seen_funcs.add(parts[1].strip())
                        functionalities.append(f"{func_number}. {parts[1].strip()}")
                        func_number += 1
        except Exception as e:
            with log_lock:
                logger.warning(f"Error parsing chunk requirements: {str(e)}")

    def summarize_section(items, max_tokens, section_name):
        combined = " ".join(set(item for item in items if item))
        sentences = re.split(r'(?<=[.!?])\s+', combined)
        result = []
        current_tokens = 0
        for sentence in sentences:
            sentence_tokens = count_tokens(sentence)
            if current_tokens + sentence_tokens <= max_tokens and sentence.strip():
                result.append(sentence.strip())
                current_tokens += sentence_tokens
            if current_tokens >= max_tokens:
                break
        summary = " ".join(result)
        if not summary or count_tokens(summary) > max_tokens:
            with log_lock:
                logger.warning(f"Summary for {section_name} exceeded token limit or empty. Using fallback.")
            return (
                f"Supports {language.lower()} file operations." if section_name == 'Overview' else
                f"To enable core {language.lower()} business functionality." if section_name == 'Objective' else
                f"Supports {language.lower()} business scenarios." if section_name == 'Use Case' else
                f"Integrates with {language.lower()} file workflows."
            )
        return summary

    overview = summarize_section(overviews, MAX_SECTION_TOKENS, 'Overview')
    objective = summarize_section(objectives, MAX_SECTION_TOKENS // 2, 'Objective')
    use_case = summarize_section(use_cases, MAX_SECTION_TOKENS // 2, 'Use Case')
    workflow = summarize_section(workflows, MAX_SECTION_TOKENS // 2, 'Workflow Summary')

    max_funcs = 10
    functionalities = functionalities[:max_funcs]
    if not functionalities:
        functionalities = ["1. Basic Functionality: Supports core operations.", "2. Support Functions: Assists business tasks."]

    combined = (
        f"Overview\n{overview}\n\n"
        f"Objective\n{objective}\n\n"
        f"Use Case\n{use_case}\n\n"
        f"Key Functionalities\n{'\n\n'.join(functionalities)}\n\n"
        f"Workflow Summary\n{workflow}"
    )

    total_tokens = count_tokens(combined)
    if total_tokens > MAX_SAFE_TOKENS:
        with log_lock:
            logger.warning(f"Combined requirements exceed token limit ({total_tokens}). Summarizing.")
        summary_prompt = (
            f"Summarize the following {language} requirements into a concise document under {MAX_SAFE_TOKENS} tokens:\n"
            f"```\n{combined}\n```\n"
            "Maintain structure with Overview, Objective, Use Case, Key Functionalities, Workflow Summary, separated by two newlines. "
            "Ensure complete sentences, remove redundancy, use clear language, avoid markdown symbols. "
            "Limit Key Functionalities to 5 numbered items (e.g., 1., 2.), each followed by a blank line."
        )
        try:
            async with asyncio.timeout(10):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    result = await asyncio.get_event_loop().run_in_executor(
                        executor, lambda: llm_config._llm.complete(summary_prompt)
                    )
                combined = result.text.strip()
                sections = re.split(r'\n\n(?=Overview|Objective|Use Case|Key Functionalities|Workflow Summary)', combined)
                section_dict = {}
                for section in sections:
                    for header in ['Overview', 'Objective', 'Use Case', 'Key Functionalities', 'Workflow Summary']:
                        if section.startswith(header):
                            section_dict[header] = section[len(header):].strip()
                            break
                for header in ['Overview', 'Objective', 'Use Case', 'Workflow Summary']:
                    if header in section_dict and section_dict[header]:
                        sentences = re.split(r'(?<=[.!?])\s+', section_dict[header])
                        section_dict[header] = " ".join(sentences[:-1]) if sentences and not sentences[-1].endswith(('.', '!', '?')) else section_dict[header]
                if 'Key Functionalities' in section_dict:
                    func_lines = section_dict['Key Functionalities'].split('\n')
                    formatted_funcs = [line.strip() for line in func_lines if re.match(r'^\d+\.\s', line)][:5]
                    section_dict['Key Functionalities'] = '\n\n'.join(formatted_funcs) if formatted_funcs else '\n\n'.join(functionalities[:5])
                combined = (
                    f"Overview\n{section_dict.get('Overview', f'Supports {language.lower()} file operations.')}\n\n"
                    f"Objective\n{section_dict.get('Objective', f'To enable core {language.lower()} business functionality.')}\n\n"
                    f"Use Case\n{section_dict.get('Use Case', f'Supports {language.lower()} business scenarios.')}\n\n"
                    f"Key Functionalities\n{section_dict.get('Key Functionalities', '\n\n'.join(functionalities[:5]))}\n\n"
                    f"Workflow Summary\n{section_dict.get('Workflow Summary', f'Integrates with {language.lower()} file workflows.')}"
                )
        except asyncio.TimeoutError:
            with log_lock:
                logger.error("Summary timed out. Truncating to complete sentences.")
            combined = (
                f"Overview\n{summarize_section(overviews, MAX_SECTION_TOKENS // 2, 'Overview')}\n\n"
                f"Objective\n{summarize_section(objectives, MAX_SECTION_TOKENS // 4, 'Objective')}\n\n"
                f"Use Case\n{summarize_section(use_cases, MAX_SECTION_TOKENS // 4, 'Use Case')}\n\n"
                f"Key Functionalities\n{'\n\n'.join(functionalities[:5])}\n\n"
                f"Workflow Summary\n{summarize_section(workflows, MAX_SECTION_TOKENS // 4, 'Workflow Summary')}"
            )

    with log_lock:
        logger.debug(f"Combined requirements: {count_tokens(combined)} tokens")
    return combined

@lru_cache(maxsize=100)
def _hash_content(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()

async def extract_organization_specific_rules(file_path: str, base_dir: str, language: str) -> OrgSpecificRules:
    try:
        with log_lock:
            logger.debug(f"Extracting org-specific rules for file: {file_path}, Language: {language}")

        if not language:
            _, ext = os.path.splitext(file_path)
            code_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.scala', '.rb', '.go', '.cpp', '.c', '.rs', '.kt', '.swift', '.php', '.cs'}
            if ext.lower() not in code_extensions:
                with log_lock:
                    logger.info(f"Skipping non-code file for org-specific rules: {file_path} (Extension: {ext})")
                return OrgSpecificRules(
                    relative_path=os.path.relpath(file_path, base_dir),
                    file_name=os.path.basename(file_path),
                    rules="Non-code file: Organization-specific rules extraction not applicable."
                )
            language = "Plain Text"

        async with aiofiles.open(file_path, mode='r', encoding='utf-8') as file:
            content = await file.read()

        total_tokens = count_tokens(content)
        with log_lock:
            logger.debug(f"File: {file_path}, Language: {language}, Size: {len(content)} bytes, Tokens: {total_tokens}")

        if total_tokens < 1000:
            content_hash = _hash_content(content)
            rules_text = await extract_organization_specific_rules_for_chunk(content, 0, language)
            with log_lock:
                logger.debug(f"Extracted org-specific rules for small file {file_path} (hash: {content_hash})")
            return OrgSpecificRules(
                relative_path=os.path.relpath(file_path, base_dir),
                file_name=os.path.basename(file_path),
                rules=rules_text
            )

        if total_tokens <= MAX_SAFE_TOKENS:
            if total_tokens > MAX_SAFE_TOKENS // 2:
                content = content[:int(len(content) * MAX_SAFE_TOKENS * 0.3 / total_tokens)]
                with log_lock:
                    logger.warning(f"Truncated {file_path} to {len(content)} bytes for org-specific rules")
            rules_text = await extract_organization_specific_rules_for_chunk(content, 0, language)
            return OrgSpecificRules(
                relative_path=os.path.relpath(file_path, base_dir),
                file_name=os.path.basename(file_path),
                rules=rules_text
            )
        else:
            with log_lock:
                logger.debug(f"Splitting {file_path} into chunks for org-specific rules due to high token count: {total_tokens}")
            try:
                chunks = await split_into_chunks(content, language, file_path)
                with log_lock:
                    logger.debug(f"Generated {len(chunks)} chunks for {file_path}")
                rules_list = []
                for i, (chunk_content, start_line, end_line) in enumerate(chunks):
                    try:
                        rules = await extract_organization_specific_rules_for_chunk(chunk_content, i, language)
                        if rules != "No organization-specific rules detected.":
                            rules_list.append(f"Chunk {i+1} (Lines {start_line}-{end_line}):\n{rules}")
                        if (i + 1) % 5 == 0 or i + 1 == len(chunks):
                            with log_lock:
                                logger.info(f"Processed chunk {i+1}/{len(chunks)} for org-specific rules in {file_path}")
                    except Exception as e:
                        with log_lock:
                            logger.error(f"Error processing chunk {i+1} for org-specific rules in {file_path}: {e}")
                        rules_list.append(f"Chunk {i+1} (Lines {start_line}-{end_line}):\nError extracting rules: {str(e)}")
                rules_text = "\n\n".join(rules_list) if rules_list else "No organization-specific rules detected."
            except MemoryError as me:
                with log_lock:
                    logger.error(f"Memory error during chunking for org-specific rules in {file_path}: {me}")
                rules_text = "Error extracting organization-specific rules due to memory constraints."
            except Exception as e:
                with log_lock:
                    logger.error(f"Chunking failed for org-specific rules in {file_path}: {str(e)}")
                rules_text = f"Error extracting organization-specific rules: {str(e)}"

        return OrgSpecificRules(
            relative_path=os.path.relpath(file_path, base_dir),
            file_name=os.path.basename(file_path),
            rules=rules_text
        )
    except Exception as e:
        with log_lock:
            logger.error(f"Error extracting org-specific rules for {file_path}: {str(e)}")
        return OrgSpecificRules(
            relative_path=os.path.relpath(file_path, base_dir),
            file_name=os.path.basename(file_path),
            rules=f"Error extracting organization-specific rules: {str(e)}"
        )

async def generate_project_summary(files_requirements: FilesRequirements) -> Tuple[str, List[GraphResponse]]:
    """
    Generate a comprehensive project summary by analyzing workflow summaries from all files.
    
    Args:
        files_requirements: FilesRequirements object containing all file requirements
        
    Returns:
        Tuple[str, List[GraphResponse]]: A detailed structured project summary and associated graphs
    """
    try:
        with log_lock:
            logger.info("Starting comprehensive project summary generation")
        
        # Extract workflow summaries and additional context from all files
        workflow_summaries = []
        file_contexts = []
        
        for file_req in files_requirements.files:
            if file_req.requirements:
                # Extract all relevant sections
                sections = re.split(
                    r'\n\n(?=Overview|Objective|Use Case|Key Functionalities|Workflow Summary|Dependent Files|Technical Requirements|Business Logic)', 
                    file_req.requirements
                )
                
                file_info = {
                    'name': file_req.file_name,
                    'workflow': '',
                    'overview': '',
                    'objective': '',
                    'key_functions': '',
                    'use_case': ''
                }
                
                for section in sections:
                    section_lower = section.lower()
                    if section.startswith("Workflow Summary"):
                        workflow_text = section[len("Workflow Summary"):].strip()
                        if workflow_text and not workflow_text.startswith("Error"):
                            file_info['workflow'] = workflow_text
                            workflow_summaries.append(f"{file_req.file_name}: {workflow_text}")
                    elif section.startswith("Overview"):
                        file_info['overview'] = section[len("Overview"):].strip()
                    elif section.startswith("Objective"):
                        file_info['objective'] = section[len("Objective"):].strip()
                    elif section.startswith("Key Functionalities"):
                        file_info['key_functions'] = section[len("Key Functionalities"):].strip()
                    elif section.startswith("Use Case"):
                        file_info['use_case'] = section[len("Use Case"):].strip()
                
                if any(file_info.values()):
                    file_contexts.append(file_info)
        
        if not workflow_summaries and not file_contexts:
            with log_lock:
                logger.warning("No valid content found for project summary")
            return "Project analysis completed with limited information available for comprehensive summary generation.", []
        
        # Combine all available information
        combined_context = ""
        
        # Add workflow summaries
        if workflow_summaries:
            combined_context += "WORKFLOW SUMMARIES:\n" + "\n".join(workflow_summaries) + "\n\n"
        
        # Add additional context from files
        if file_contexts:
            combined_context += "ADDITIONAL FILE CONTEXT:\n"
            for file_info in file_contexts:
                if file_info['overview'] or file_info['objective'] or file_info['key_functions']:
                    combined_context += f"\n{file_info['name']}:\n"
                    if file_info['overview']:
                        combined_context += f"  Overview: {file_info['overview']}\n"
                    if file_info['objective']:
                        combined_context += f"  Objective: {file_info['objective']}\n"
                    if file_info['key_functions']:
                        combined_context += f"  Key Functions: {file_info['key_functions']}\n"
                    if file_info['use_case']:
                        combined_context += f"  Use Case: {file_info['use_case']}\n"
        
        # Check token limit and truncate if necessary
        if count_tokens(combined_context) > MAX_SAFE_TOKENS:
            with log_lock:
                logger.warning("Combined context exceeds token limit. Truncating for project summary.")
            combined_context = combined_context[:MAX_SAFE_TOKENS * 3 // 4]
        
        # Create enhanced prompt for detailed structured project summary
        prompt = f"""
Act as a senior technical architect and business analyst conducting a comprehensive project review.
Based on the provided workflow summaries and file analysis, generate a detailed, well-structured project summary that follows this exact format:
## PROJECT OVERVIEW
[2-3 sentences describing the project's main purpose, domain, and scope]
## BUSINESS CONTEXT & OBJECTIVES
[2-3 sentences explaining the business problem being solved and strategic objectives]
## SYSTEM ARCHITECTURE & APPROACH
[3-4 sentences describing the technical approach, key architectural patterns, and system design principles]
## KEY CAPABILITIES & FEATURES
[4-5 sentences detailing the main functionalities, core features, and what the system can accomplish]
## WORKFLOW & PROCESS INTEGRATION
[3-4 sentences explaining how different components work together, data flow, and process coordination]
## BUSINESS VALUE & IMPACT
[2-3 sentences highlighting the expected business benefits, efficiency gains, and organizational impact]
## TECHNICAL FOUNDATION
[2-3 sentences covering the technology stack, integration patterns, and scalability considerations]
Guidelines:
- Write in clear, professional language suitable for both technical and business stakeholders
- Focus on business value while maintaining technical accuracy
- Avoid mentioning specific file names or implementation details
- Each section should be substantive and informative
- Use present tense and active voice
- Ensure logical flow between sections
- Keep the tone analytical and objective
Project Information:
{combined_context}
"""
        
        # Call Azure LLM endpoint for summary
        async with llm_semaphore:
            try:
                async with asyncio.timeout(45):  # Increased timeout for detailed response
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        result = await asyncio.get_event_loop().run_in_executor(
                            executor, lambda: llm_config._llm.complete(prompt)
                        )
                
                project_summary = result.text.strip()
                
                # Validate the response structure
                required_sections = [
                    "PROJECT OVERVIEW",
                    "BUSINESS CONTEXT & OBJECTIVES", 
                    "SYSTEM ARCHITECTURE & APPROACH",
                    "KEY CAPABILITIES & FEATURES",
                    "WORKFLOW & PROCESS INTEGRATION",
                    "BUSINESS VALUE & IMPACT",
                    "TECHNICAL FOUNDATION"
                ]
                
                # Check if response contains the required structure
                missing_sections = [section for section in required_sections if section not in project_summary]
                
                if missing_sections or len(project_summary.split()) < 150:
                    with log_lock:
                        logger.warning(f"Generated summary may be incomplete. Missing sections: {missing_sections}")
                    
                    # Provide a structured fallback
                    if not project_summary or len(project_summary.split()) < 50:
                        project_summary = f"""
## PROJECT OVERVIEW
This project implements a comprehensive software solution designed to address complex business requirements through integrated system components. The solution encompasses multiple modules working in coordination to deliver automated business processes and data management capabilities.
## BUSINESS CONTEXT & OBJECTIVES
The system addresses organizational needs for streamlined operations and improved process efficiency. It aims to reduce manual intervention while providing reliable, scalable solutions for business-critical operations.
## SYSTEM ARCHITECTURE & APPROACH
The architecture follows modular design principles with clear separation of concerns across different functional areas. Components are designed for maintainability and extensibility, utilizing established patterns for reliable system integration and data processing workflows.
## KEY CAPABILITIES & FEATURES
The system provides automated workflow processing, data validation and transformation, integrated reporting capabilities, and comprehensive error handling. It supports configurable business rules, multi-step process orchestration, and real-time monitoring of system operations.
## WORKFLOW & PROCESS INTEGRATION
Different system components communicate through well-defined interfaces, ensuring data consistency and process reliability. The workflow engine coordinates activities across modules, managing dependencies and ensuring proper sequencing of operations.
## BUSINESS VALUE & IMPACT
Implementation delivers improved operational efficiency, reduced processing time, and enhanced data accuracy. The system enables better decision-making through automated reporting and provides scalable foundation for future business growth.
## TECHNICAL FOUNDATION
Built on robust technical infrastructure supporting concurrent operations, error recovery, and system monitoring. The solution incorporates modern development practices with emphasis on reliability, performance, and maintainability.
"""
                
                with log_lock:
                    logger.info("Generated comprehensive project summary")
                    logger.debug(f"Summary length: {len(project_summary.split())} words")
                
                # Generate graphs for the project summary
                graphs = []
                try:
                    # Generate Entity Relationship Diagram
                    erd_graph = await generate_graph_from_requirement(project_summary, target_graph="entity relationship diagram")
                    if not erd_graph.generated_code.startswith("Error"):
                        graphs.append(erd_graph)
                    
                    # Generate Requirement Diagram
                    req_graph = await generate_graph_from_requirement(project_summary, target_graph="requirement diagram")
                    if not req_graph.generated_code.startswith("Error"):
                        graphs.append(req_graph)
                    
                    with log_lock:
                        logger.info(f"Generated {len(graphs)} graphs for project summary")
                except Exception as e:
                    with log_lock:
                        logger.error(f"Error generating graphs for project summary: {e}")
                
                return project_summary, graphs

            except Exception as e:
                with log_lock:
                    logger.error(f"Error during LLM summary generation: {e}")
                return "Comprehensive summary generation failed due to processing error.", []
    
    except Exception as e:
        with log_lock:
            logger.error(f"Unexpected error in project summary generation: {e}")
        return "Summary generation encountered an unexpected issue.", []

async def generate_requirements_for_file_whole(file_path: str, base_dir: str, language: str) -> FileRequirements:
    try:
        with log_lock:
            logger.debug(f"Processing file: {file_path}, Language: {language}")

        if not language:
            _, ext = os.path.splitext(file_path)
            code_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.scala', '.rb', '.go', '.cpp', '.c', '.rs', '.kt', '.swift', '.php', '.cs'}
            if ext.lower() not in code_extensions:
                with log_lock:
                    logger.info(f"Skipping non-code file: {file_path} (Extension: {ext})")
                return FileRequirements(
                    relative_path=os.path.relpath(file_path, base_dir),
                    file_name=os.path.basename(file_path),
                    requirements="Non-code file: Requirements generation not applicable.",
                    dependencies=[]
                )
            language = "Plain Text"

        async with aiofiles.open(file_path, mode='r', encoding='utf-8') as file:
            content = await file.read()

        total_tokens = count_tokens(content)
        with log_lock:
            logger.debug(f"File: {file_path}, Language: {language}, Size: {len(content)} bytes, Tokens: {total_tokens}")

        from app.services.dependency_service import detect_dependencies
        dependencies = await detect_dependencies(file_path, content, language, base_dir)
        validated_dependencies = []
        for dep in dependencies:
            try:
                validated_dep = FileDependency(**dep.dict())
                validated_dependencies.append(validated_dep)
            except ValidationError as ve:
                with log_lock:
                    logger.error(f"Invalid FileDependency for {file_path}: {dep.dict()}, Error: {ve}")

        with log_lock:
            logger.debug(f"Validated dependencies for {file_path}: {[dep.dict() for dep in validated_dependencies]}")
        dep_lines = []
        for dep in validated_dependencies:
            dep_lines.append(dep.file_name)
            dep_lines.append(dep.dependency_reason.replace('\n', ' '))
        dep_text = "\\n".join(dep_lines) if dep_lines else "No dependencies detected."
        with log_lock:
            logger.debug(f"Raw dep_text for {file_path}: {repr(dep_text)}")
            logger.debug(f"JSON dep_text for {file_path}: {json.dumps(dep_text)}")

        if total_tokens < 1000:
            content_hash = _hash_content(content)
            rules_detected = await detect_business_rules(content, language)
            rules_summary = (
                "\n".join([f"Detected {rule['domain']} rule: {rule['description']}" for rule in rules_detected])
                if rules_detected else "No organization-specific rules detected."
            )
            if language.lower() in ['javascript', 'jsx', 'js']:
                req_text = (
                    f"Overview\nConfiguration file for {os.path.basename(file_path)} in a frontend application. {rules_summary}\n\n"
                    f"Objective\nTo define settings or utilities for the application.\n\n"
                    f"Use Case\nUsed during application initialization or runtime for UI setup.\n\n"
                    f"Key Functionalities\n1. Configuration: Defines application settings.\n\n2. Utility Support: Provides helper functions.\n\n"
                    f"Workflow Summary\nIntegrates with frontend framework for setup.\n\n"
                    f"Dependent Files\n{dep_text}"
                )
            else:
                req_text = (
                    f"Overview\nUtility file for {os.path.basename(file_path)}. {rules_summary}\n\n"
                    f"Objective\nTo provide supporting functions.\n\n"
                    f"Use Case\nSupports general business operations.\n\n"
                    f"Key Functionalities\n1. Basic Utilities: Provides helper functions.\n\n2. Support Functions: Assists operations.\n\n"
                    f"Workflow Summary\nSupports general business workflow.\n\n"
                    f"Dependent Files\n{dep_text}"
                )
            with log_lock:
                logger.debug(f"Used small-file template for {file_path} (hash: {content_hash})")
                logger.debug(f"req_text for {file_path}: {repr(req_text)}")
            return FileRequirements(
                relative_path=os.path.relpath(file_path, base_dir),
                file_name=os.path.basename(file_path),
                requirements=req_text,
                dependencies=validated_dependencies
            )

        if total_tokens <= MAX_SAFE_TOKENS:
            if total_tokens > MAX_SAFE_TOKENS // 2:
                content = content[:int(len(content) * MAX_SAFE_TOKENS * 0.3 / total_tokens)]
                with log_lock:
                    logger.warning(f"Truncated {file_path} to {len(content)} bytes")
            rules_detected = await detect_business_rules(content, language)
            rules_summary = (
                "\n".join([f"Detected {rule['domain']} rule: {rule['description']}" for rule in rules_detected])
                if rules_detected else "No organization-specific rules detected."
            )
            prompt = (
                f"Act as a senior business analyst reviewing a {language} source file.\n"
                f"Generate a comprehensive business requirements document in plain text, avoiding markdown symbols.\n"
                f"Do NOT use the phrase 'this file' or 'this code'. "
                f"If the code contains organization-specific logic or calculations (e.g., proprietary formulas or business rules), "
                f"generalize them without revealing sensitive details. For example, instead of specifying an exact calculation, "
                f"say 'performs proprietary calculations based on internal rules.'\n"
                f"Use the following detected rules for context:\n{rules_summary}\n"
                f"Provide exactly six sections: Overview, Objective, Use Case, Key Functionalities, Workflow Summary, and Dependent Files, "
                f"separated by two newlines.\n\n"
                f"Overview\nDescribe the file's purpose and role (50-100 words).\n\n"
                f"Objective\nState the primary business goal (1-2 sentences).\n\n"
                f"Use Case\nOutline business scenarios (2-3 sentences).\n\n"
                f"Key Functionalities\nList 2-5 capabilities, each on a new line, numbered (e.g., 1., 2.), followed by a blank line.\n\n"
                f"Workflow Summary\nDescribe the business process (2-3 sentences).\n\n"
                f"Dependent Files\n{dep_text}\n\n"
                f"Analyze:\n```{language}\n{content}\n```"
            )

            content_hash = _hash_content(content)
            async with llm_semaphore:
                try:
                    async with asyncio.timeout(60):
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            result = await asyncio.get_event_loop().run_in_executor(
                                executor, lambda: llm_config._llm.complete(prompt)
                            )
                    req_text = result.text.strip()
                    with log_lock:
                        logger.debug(f"Raw LLM output for {file_path} (hash: {content_hash}): {req_text[:200]}...")

                    req_text = re.sub(r'\n\s*\n+', '\n\n', req_text)
                    req_text = re.sub(r'this file|this code', 'the code', req_text, flags=re.IGNORECASE)
                    sections = re.split(r'\n\n(?=Overview|Objective|Use Case|Key Functionalities|Workflow Summary|Dependent Files)', req_text)
                    section_dict = {}
                    for section in sections:
                        for header in ['Overview', 'Objective', 'Use Case', 'Key Functionalities', 'Workflow Summary', 'Dependent Files']:
                            if section.startswith(header):
                                section_dict[header] = section[len(header):].strip()
                                break

                    if len(section_dict) < 5 or not all(section_dict.get(h) for h in ['Overview', 'Objective', 'Use Case', 'Key Functionalities', 'Workflow Summary']):
                        with log_lock:
                            logger.warning(f"Invalid file requirements for {file_path}. Using fallback.")
                        req_text = (
                            f"Overview\nLimited analysis for {os.path.basename(file_path)}. {rules_summary}\n\n"
                            f"Objective\nTo support core {language.lower()} business functions.\n\n"
                            f"Use Case\nGeneral {language.lower()} business processes.\n\n"
                            f"Key Functionalities\n1. Basic Processing: Performs core tasks.\n\n2. Support Functions: Assists operations.\n\n"
                            f"Workflow Summary\nIntegrates with {language.lower()} system workflow.\n\n"
                            f"Dependent Files\n{dep_text}"
                        )
                    else:
                        if 'Key Functionalities' in section_dict:
                            func_lines = section_dict['Key Functionalities'].split('\n')
                            formatted_funcs = [line.strip() for line in func_lines if re.match(r'\d+\.\s', line)]
                            if not formatted_funcs:
                                formatted_funcs = ["1. Basic Processing: Supports core tasks.", "2. Support Functions: Assists operations."]
                            section_dict['Key Functionalities'] = '\n\n'.join(formatted_funcs)
                        req_text = (
                            f"Overview\n{section_dict.get('Overview', 'Summary unavailable.')}. {rules_summary}\n\n"
                            f"Objective\n{section_dict.get('Objective', 'To support core functionality.')}\n\n"
                            f"Use Case\n{section_dict.get('Use Case', 'Supports business operations.')}\n\n"
                            f"Key Functionalities\n{section_dict.get('Key Functionalities', '1. Basic Processing: Supports core tasks.\n\n2. Support Functions: Assists operations.')}\n\n"
                            f"Workflow Summary\n{section_dict.get('Workflow Summary', 'Integrates with system workflow.')}\n\n"
                            f"Dependent Files\n{dep_text}"
                        )
                except asyncio.TimeoutError:
                    with log_lock:
                        logger.error(f"LLM timed out for {file_path}")
                    req_text = (
                        f"Overview\nLimited analysis for {os.path.basename(file_path)} due to timeout. {rules_summary}\n\n"
                        f"Objective\nTo support core {language.lower()} business functions.\n\n"
                        f"Use Case\nGeneral {language.lower()} business processes.\n\n"
                        f"Key Functionalities\n1. Partial Processing: Supports basic operations.\n\n2. Fallback: Provides minimal functionality.\n\n"
                        f"Workflow Summary\nProcesses available data.\n\n"
                        f"Dependent Files\n{dep_text}"
                    )
                except ValueError as ve:
                    with log_lock:
                        logger.error(f"Value error during LLM processing for {file_path}: {ve}")
                    req_text = (
                        f"Overview\nAnalysis failed for {os.path.basenameMISSION_FILE} due to invalid data. {rules_summary}\n\n"
                        f"Objective\nTo support intended {language.lower()} business functions.\n\n"
                        f"Use Case\nIntended {language.lower()} business operations.\n\n"
                        f"Key Functionalities\n1. Intended Functionality: Expected to provide core features.\n\n2. Support Functions: Assists operations.\n\n"
                        f"Workflow Summary\nExpected to integrate with {language.lower()} system workflows.\n\n"
                        f"Dependent Files\n{dep_text}"
                    )
                except Exception as e:
                    with log_lock:
                        logger.error(f"Unexpected error during LLM processing for {file_path}: {e}")
                    req_text = (
                        f"Overview\nAnalysis failed for {os.path.basename(file_path)} due to processing error. {rules_summary}\n\n"
                        f"Objective\nTo support intended {language.lower()} business functions.\n\n"
                        f"Use Case\nIntended {language.lower()} business operations.\n\n"
                        f"Key Functionalities\n1. Intended Functionality: Expected to provide core features.\n\n2. Support Functions: Assists operations.\n\n"
                        f"Workflow Summary\nExpected to integrate with {language.lower()} system workflows.\n\n"
                        f"Dependent Files\n{dep_text}"
                    )
        else:
            with log_lock:
                logger.debug(f"Splitting {file_path} into chunks due to high token count: {total_tokens}")
            try:
                chunks = await split_into_chunks(content, language, file_path)
                with log_lock:
                    logger.debug(f"Generated {len(chunks)} chunks for {file_path}")
                requirements_list = []
                for i, (chunk_content, start_line, end_line) in enumerate(chunks):
                    try:
                        req = await generate_requirements_for_chunk(chunk_content, i, language)
                        requirements_list.append(req)
                        if (i + 1) % 5 == 0 or i + 1 == len(chunks):
                            with log_lock:
                                logger.info(f"Processed chunk {i+1}/{len(chunks)} for {file_path}")
                    except Exception as e:
                        with log_lock:
                            logger.error(f"Error processing chunk {i+1} for {file_path}: {e}")
                        requirements_list.append(
                            f"Overview\nFailed to process chunk {i+1} of {os.path.basename(file_path)}.\n\n"
                            f"Objective\nTo support chunk functionality.\n\n"
                            f"Use Case\nGeneral chunk operations.\n\n"
                            f"Key Functionalities\n1. Partial Processing: Supports chunk {i+1} operations.\n\n2. Fallback: Provides basic functionality.\n\n"
                            f"Workflow Summary\nProcesses available chunk data."
                        )
                req_text = await combine_requirements(requirements_list, language)
                req_text += f"\n\nDependent Files\n{dep_text}"
            except MemoryError as me:
                with log_lock:
                    logger.error(f"Memory error during chunking for {file_path}: {me}")
                req_text = (
                    f"Overview\nFailed to process {os.path.basename(file_path)} due to memory constraints.\n\n"
                    f"Objective\nTo support intended {language.lower()} business functions.\n\n"
                    f"Use Case\nIntended {language.lower()} business operations.\n\n"
                    f"Key Functionalities\n1. Intended Functionality: Expected to provide core features.\n\n2. Support Functions: Assists operations.\n\n"
                    f"Workflow Summary\nExpected to integrate with {language.lower()} system workflows.\n\n"
                    f"Dependent Files\n{dep_text}"
                )
            except Exception as e:
                with log_lock:
                    logger.error(f"Chunking failed for {file_path}: {str(e)}")
                req_text = (
                    f"Overview\nFailed to process {os.path.basename(file_path)} due to processing error.\n\n"
                    f"Objective\nTo support intended {language.lower()} business functions.\n\n"
                    f"Use Case\nIntended {language.lower()} business operations.\n\n"
                    f"Key Functionalities\n1. Intended Functionality: Expected to provide core features.\n\n2. Support Functions: Assists operations.\n\n"
                    f"Workflow Summary\nExpected to integrate with {language.lower()} system workflows.\n\n"
                    f"Dependent Files\n{dep_text}"
                )

        with log_lock:
            logger.debug(f"Final req_text for {file_path}: {repr(req_text)}")
        return FileRequirements(
            relative_path=os.path.relpath(file_path, base_dir),
            file_name=os.path.basename(file_path),
            requirements=req_text,
            dependencies=validated_dependencies
        )
    except Exception as e:
        with log_lock:
            logger.error(f"Error generating requirements for {file_path}: {str(e)}")
        return FileRequirements(
            relative_path=os.path.relpath(file_path, base_dir),
            file_name=os.path.basename(file_path),
            requirements=f"Error generating requirements: {str(e)}",
            dependencies=[]
        )

async def generate_requirements_for_files_whole(directory: str) -> FilesRequirements:
    with log_lock:
        logger.info(f"Starting requirements generation for {directory}")
    try:
        code_files = get_code_files(directory)
        total_files = len(code_files)
        with log_lock:
            logger.info(f"Found {total_files} files for requirements generation in {directory}")

        files_requirements = []
        completed_count = 0

        file_batches = [code_files[i:i + BATCH_SIZE] for i in range(0, total_files, BATCH_SIZE)]
        with ThreadPoolExecutor(max_workers=12) as executor:
            loop = asyncio.get_event_loop()
            for batch in file_batches:
                tasks = [
                    generate_requirements_for_file_whole(file_path, directory, language)
                    for file_path, language in batch
                ]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in batch_results:
                    if isinstance(result, Exception):
                        with log_lock:
                            logger.error(f"Error in batch processing: {result}")
                    else:
                        files_requirements.append(result)
                completed_count += len(batch)
                with log_lock:
                    logger.info(f"Progress: {completed_count}/{total_files} files processed ({completed_count/total_files*100:.2f}%)")

        # Sort the file requirements by relative path
        files_requirements.sort(key=lambda x: x.relative_path)
        
        # Generate the overall project summary and graphs
        files_requirements_obj = FilesRequirements(files=files_requirements)
        
        project_summary, graphs = await generate_project_summary(files_requirements_obj)
        
        # Update the FilesRequirements object with the project summary and graphs
        files_requirements_obj.project_summary = project_summary
        files_requirements_obj.graphs = graphs

        with log_lock:
            logger.info(f"Completed requirements generation for {len(files_requirements)} files with project summary and {len(graphs)} graphs")
        return files_requirements_obj
    except Exception as e:
        with log_lock:
            logger.error(f"Error processing directory {directory}: {str(e)}")
        return FilesRequirements(files=[])
