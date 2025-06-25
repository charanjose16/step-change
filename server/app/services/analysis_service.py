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
import openpyxl
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter
from .vector_store_service import VectorStoreService

# Constants
CHUNK_GROUP_SIZE = 40
LLM_CONCURRENCY_LIMIT = 10
llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY_LIMIT)
LINTER_CONCURRENCY = 300
linter_sema = asyncio.Semaphore(LINTER_CONCURRENCY)
MAX_MODEL_TOKENS = 128000
MAX_SAFE_TOKENS = 50000
MAX_EMBEDDING_TOKENS = 8192
BATCH_SIZE = 50
MAX_LINES_PER_CHUNK = 2000
MAX_SECTION_TOKENS = 1000
log_lock = Lock()
vector_store = VectorStoreService()
VECTOR_STORE_DIR = "vector_store"

# Tokenizer and Chunking Utilities
@lru_cache(maxsize=10)
def get_tokenizer(model: str = "text-embedding-ada-002"):
    return tiktoken.encoding_for_model(model)

def count_tokens(text: str, model: str = "text-embedding-ada-002") -> int:
    tokenizer = get_tokenizer(model)
    return len(tokenizer.encode(text))

def chunk_text_for_embedding(text: str, model: str = "text-embedding-ada-002", max_tokens: int = MAX_EMBEDDING_TOKENS) -> List[str]:
    tokenizer = get_tokenizer(model)
    tokens = tokenizer.encode(text)

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk))

    return chunks

# Enhanced configuration for dynamic business rule detection (updated for VBA)
RULES_CONFIG = {
    "financial": {
        "patterns": [
            r"\b(interest_rate|credit_score|risk_score|loan_eligibility|payment_schedule)\s*[+\-*/=]\s*[\d+\.*%*].*",  # Specific financial calculations
            r"\bif.*\b(credit_score|income|debt|risk|loan_amount)\b.*[<>=].*\b(approve|reject|calculate|assess)\b.*",  # Financial decision logic
            r"\bdef\s+(calculate_|compute_|assess_)(interest|loan|risk|score)\b.*",  # Financial calculation functions
            r"\bSub\s+(calculate_|compute_|assess_)(cost|total|adjustment|summary)\b.*",  # VBA financial subs
        ],
        "description": "Proprietary financial calculations, risk assessments, or loan eligibility logic."
    },
    "ecommerce": {
        "patterns": [
            r"\b(discount|price|total_cost|tax|shipping)\s*[+\-*/=]\s*.*(if|where|category|location|amount).*",  # Pricing and discount rules
            r"\bif.*\b(location|category|product_type|order_value|customer_type)\b.*[<>=].*\b(discount|price|promotion)\b.*",  # Conditional pricing logic
            r"\bdef\s+(apply_|calculate_)(discount|promotion|price)\b.*",  # Ecommerce-specific functions
            r"\bSub\s+(apply_|calculate_)(discount|promotion|cost|total)\b.*",  # VBA ecommerce subs
        ],
        "description": "Custom pricing, discount, or promotion rules based on business criteria."
    },
    "general": {
        "patterns": [
            r"\b(custom_|proprietary_|internal_)(rule|logic|calc|workflow)\b.*=\s*.*",  # Custom-named business logic
            r"\bif.*\b(custom_|proprietary_|internal_)\b.*[<>=].*\b(process|validate|compute)\b.*",  # Custom conditional logic
            r"\bdef\s+(custom_|proprietary_|internal_)(process|validate|compute)\b.*",  # Custom processing functions
            r"\bSub\s+(custom_|proprietary_|internal_)(process|validate|compute)\b.*",  # VBA custom subs
        ],
        "description": "Organization-specific business logic, workflows, or proprietary computations."
    },
    "excel": {
        "patterns": [
            r"FORMULA:.*\b(SUM|AVERAGE|IF|VLOOKUP|HLOOKUP|INDEX|MATCH|COUNTIF|SUMIF|ROUND|PMT|FV|PV)\b\s*\(.*\)",  # Excel formulas
            r"FORMULA:.*\b(AND|OR|NOT|IFERROR|IFNA)\b\s*\(.*\)",  # Logical Excel formulas
            r"DATA_VALIDATION:.*\b(list|whole|decimal|textLength|date|time|custom)\b.*",  # Data validation rules
            r"CONDITIONAL_FORMATTING:.*\b(cellIs|expression|colorScale|dataBar|iconSet)\b.*",  # Conditional formatting
            r"FORMULA:.*\[.*\].*",  # Formulas with named ranges or custom references
            r"\bSub\s+.*\b(totalcost|finalcost|tax|discount|overhead)\b.*",  # VBA macros with business logic
        ],
        "description": "Excel-specific business rules including complex formulas, data validation, conditional formatting, and VBA macros."
    }
}

# Model definitions (unchanged)
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
    graphs: List[GraphResponse] = []

class OrgSpecificRules(BaseModel):
    relative_path: str
    file_name: str
    rules: str

def parse_vba_macros_from_content(content: str) -> Dict[str, Dict[str, str]]:
    """
    Extract VBA macro names, parameters, and comments from the provided content.
    """
    macros = {}
    lines = content.splitlines()
    current_macro = None
    macro_lines = []
    for line in lines:
        if line.strip().startswith('--- VBA Module:'):
            if current_macro and macro_lines:
                macros[current_macro] = {
                    'parameters': extract_macro_parameters(macro_lines),
                    'comments': extract_macro_comments(macro_lines),
                    'logic': '\n'.join(macro_lines)
                }
            current_macro = line.strip().replace('--- VBA Module:', '').strip()
            macro_lines = []
        elif current_macro is not None:
            macro_lines.append(line)
    if current_macro and macro_lines:
        macros[current_macro] = {
            'parameters': extract_macro_parameters(macro_lines),
            'comments': extract_macro_comments(macro_lines),
            'logic': '\n'.join(macro_lines)
        }
    return macros

def extract_macro_parameters(macro_lines: List[str]) -> str:
    """
    Extract macro parameters from the provided macro lines.
    """
    parameters = []
    for line in macro_lines:
        match = re.search(r'\bSub\s+(\w+)\s*\((.*?)\)', line, re.IGNORECASE)
        if match:
            parameters.append(f"{match.group(1)}({match.group(2)})")
    return ', '.join(parameters)

def extract_macro_comments(macro_lines: List[str]) -> str:
    """
    Extract macro comments from the provided macro lines.
    """
    comments = []
    for line in macro_lines:
        match = re.search(r"'(.*)", line, re.IGNORECASE)
        if match:
            comments.append(match.group(1).strip())
    return '\n'.join(comments)

def summarize_excel_vba_logic(content: str) -> str:
    """
    Parse the extracted Excel/VBA content and produce a detailed, step-by-step summary of all business logic, rules, calculations, and workflow automation as found in the file.
    - For VBA: List each Sub/Function, describe its purpose, logic, and steps in plain business language.
    - For formulas: List each formula, its cell, and what it calculates in business terms.
    - For validations: List all data validation and conditional formatting rules.
    - For named ranges and pivots: List and describe their business meaning.
    """
    lines = content.splitlines()
    summary = []
    macros = parse_vba_macros_from_content(content)
    for macro, details in macros.items():
        summary.append(f"Macro: {macro}")
        summary.append(f"Parameters: {details['parameters']}")
        summary.append(f"Comments: {details['comments']}")
        summary.append(f"Logic: {details['logic']}")
    # Formulas
    for line in lines:
        m = re.match(r'FORMULA: ([A-Z]+\d+) = (.+)', line)
        if m:
            cell, formula = m.groups()
            summary.append(f"Formula in {cell}: {formula}")
    # Data Validations
    for line in lines:
        if line.strip().startswith('DATA_VALIDATION:'):
            summary.append(f"Data Validation Rule: {line.strip().replace('DATA_VALIDATION:', '').strip()}")
    # Conditional Formatting
    for line in lines:
        if line.strip().startswith('CONDITIONAL_FORMATTING:'):
            summary.append(f"Conditional Formatting: {line.strip().replace('CONDITIONAL_FORMATTING:', '').strip()}")
    # Named Ranges
    for line in lines:
        if line.strip().startswith('- ') and ':' in line:
            summary.append(f"Named Range: {line.strip()}")
    # Pivot Tables
    for line in lines:
        if line.strip().startswith('Pivot Tables in') or line.strip().startswith('- Name:'):
            summary.append(line.strip())
    # External Links
    for line in lines:
        if line.strip().startswith('External Link:'):
            summary.append(line.strip())
    # If nothing found, fallback
    if not summary:
        return "No business logic, rules, or automation found in the file."
    return '\n'.join(summary)

async def narrative_business_logic_summary(technical_logic: str, file_path: str) -> dict:
    from app.utils.logger import logger
    logger.info(f"[LLM ENTRY] narrative_business_logic_summary called for file: {file_path}")
    
    try:
        prompt = f"""You are a business analyst. Given the following technical summary of an Excel/VBA file, write a detailed business logic summary for stakeholders.\n\nInstructions:\n- Identify and describe each macro/subroutine, its business purpose, parameters, and how it processes data.\n- Explain input validation, calculations, adjustments, and reporting in business terms.\n- Connect technical elements (like macro names, parameters, and key formulas) to their real-world business function.\n\nExample Output (JSON):\n{{\n  \"overview\": \"The ComplexProjectCalculation macro in rule.xlsm automates project cost calculations across multiple worksheets. It processes input data, applies financial adjustments, and generates summarized reports for project budgeting and financial oversight.\",\n  \"objective\": \"To calculate and summarize project costs with adjustments for taxes, discounts, and overheads, ensuring accurate financial reporting.\",\n  \"use_case\": \"Project managers use the macro to input project data, validate entries, compute adjusted costs, and review budget status for internal and external projects during financial planning and audits.\",\n  \"key_functionalities\": [\n    \"Input Validation: Verifies that project type, hours, and rate are provided to prevent calculation errors.\",\n    \"Total Cost Calculation: Computes base project costs by multiplying hours, rate, and a project-type-specific multiplier, setting the multiplier to 1 for internal projects.\",\n    \"Cost Adjustments: Applies tax, discount, and overhead adjustments to base costs to determine final project costs.\",\n    \"Summary Generation: Populates a summary table with project names, final costs, budget status, and error messages for invalid inputs.\",\n    \"Grand Total Reporting: Calculates and displays the sum of all adjusted project costs in the summary worksheet.\"\n  ],\n  \"workflow\": \"The macro processes project data from input worksheets, performs validations and calculations, applies financial adjustments, and generates a summary report with budget alerts and grand totals.\"\n}}\n\nTECHNICAL SUMMARY (includes macro names, parameters, comments, and business logic steps):\n{technical_logic}\n\nOutput ONLY a valid JSON object with the following keys: overview, objective, use_case, key_functionalities, workflow. Do not include any text before or after the JSON. Each section should be a paragraph or bulleted list (where appropriate), written for a business audience."""
        logger.debug(f"[LLM DEBUG] Prompt sent to LLM for file {file_path}: {prompt[:1000]}...")
        
        try:
            if not hasattr(llm_config, '_llm') or llm_config._llm is None:
                logger.error(f"[LLM ERROR] LLM client (_llm) is not initialized on llm_config for file {file_path}. Fallback will be used.")
                raise RuntimeError("LLM client (_llm) is not initialized.")
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: llm_config._llm.complete(
                    prompt=prompt,
                    max_tokens=700,
                    temperature=0.2,
                    stop=None,
                    model=getattr(llm_config, 'azure_openai_model', None)
                )
            )
            # Extract response text
            if hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            logger.debug(f"[LLM DEBUG] Raw LLM response for file {file_path}: {response_text[:200]}...")
        except Exception as e:
            logger.error(f"[LLM ERROR] Exception during LLM call for {file_path}: {e}")
            raise
        
        # Extract JSON from Markdown code block, if present
        pattern = r"```json\s*([\s\S]*?)\s*```"
        match = re.search(pattern, response_text)
        if match:
            response_text = match.group(1).strip()
            logger.debug(f"[LLM DEBUG] Extracted JSON from Markdown block for {file_path}")
        else:
            logger.debug(f"[LLM DEBUG] No Markdown JSON block found; using raw response for {file_path}")
        
        # Try to parse JSON from response
        try:
            parsed = json.loads(response_text)
            # Handle double-encoded JSON (string inside string)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            logger.info(f"[LLM SUCCESS] Parsed LLM response for file {file_path}: {parsed}")
            # Ensure all keys exist
            for k in ["overview", "objective", "use_case", "key_functionalities", "workflow"]:
                if k not in parsed:
                    parsed[k] = "[Not provided]"
            return parsed
        except json.JSONDecodeError as e:
            logger.error(f"[LLM ERROR] Failed to parse LLM response as JSON for {file_path}: {e}\nRaw response: {response_text}")
            # Return response as overview if not valid JSON
            return {
                "overview": response_text,
                "objective": "[Not provided]",
                "use_case": "[Not provided]",
                "key_functionalities": "[Not provided]",
                "workflow": "[Not provided]"
            }
    except Exception as e:
        logger.error(f"[LLM FALLBACK] Exception in narrative_business_logic_summary for {file_path}: {e}", exc_info=True)
        logger.info(f"[LLM FALLBACK] Using fallback narrative generation for file {file_path}")
        # Fallback: template-based summary with keyword breakdown
        lines = [l.strip() for l in technical_logic.splitlines() if l.strip()]
        validations = [l for l in lines if 'validat' in l.lower() or 'check' in l.lower() or 'error' in l.lower()]
        calculations = [l for l in lines if 'calculat' in l.lower() or 'compute' in l.lower() or 'cost' in l.lower() or 'total' in l.lower() or 'sum' in l.lower()]
        adjustments = [l for l in lines if 'adjust' in l.lower() or 'tax' in l.lower() or 'discount' in l.lower() or 'overhead' in l.lower()]
        summaries = [l for l in lines if 'summary' in l.lower() or 'report' in l.lower() or 'grand total' in l.lower()]
        overview = f"This file automates business calculations and reporting for key processes, including validation, calculation, adjustment, and summary reporting." if lines else "[No business logic detected]"
        objective = f"To automate business calculations and ensure accurate reporting and validation for business processes."
        use_case = f"Used by business users to input data, trigger calculations, apply adjustments, and generate summary reports for operational and financial decision-making."
        key_func_parts = []
        if validations:
            key_func_parts.append("1. Input Validation: " + "; ".join(validations[:2]))
        if calculations:
            key_func_parts.append(f"{len(key_func_parts)+1}. Calculation: " + "; ".join(calculations[:2]))
        if adjustments:
            key_func_parts.append(f"{len(key_func_parts)+1}. Adjustments: " + "; ".join(adjustments[:2]))
        if summaries:
            key_func_parts.append(f"{len(key_func_parts)+1}. Reporting: " + "; ".join(summaries[:2]))
        if not key_func_parts:
            key_func_parts = ["- [No detailed logic detected]"]
        workflow = "The workflow includes input validation, business calculations, adjustment of results, and summary reporting as described above." if lines else "[No workflow detected]"
        return {
            "overview": overview,
            "objective": objective,
            "use_case": use_case,
            "key_functionalities": "\n".join(key_func_parts),
            "workflow": workflow
        }
    
# Utility functions (unchanged)
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
            f"Summarize the purpose and structure of the following {language} content in 50-100 words:\n"
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

async def detect_business_rules(content: str, language: str) -> List[Dict[str, str]]:
    rules_detected = []
    domains = ["financial", "ecommerce", "general"]
    if language.lower() in ["excel", "vba"]:
        domains.append("excel")

    # Regex-based detection
    for domain in domains:
        config = RULES_CONFIG[domain]
        for pattern in config["patterns"]:
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                rule_text = match.group(0)
                rules_detected.append({
                    "rule": rule_text,
                    "description": config["description"],
                    "domain": domain,
                    "source": "regex"
                })
                with log_lock:
                    logger.debug(f"Detected {domain} rule via regex: {rule_text[:100]}...")

    # Semantic-based detection
    if len(rules_detected) < 3:
        semantic_prompt = (
            f"Analyze the following {language} content for organization-specific business rules or logic not captured by regex patterns:\n"
            f"```\n{content[:2000]}\n[...truncated...]\n```\n"
            "Identify up to 3 organization-specific rules (e.g., proprietary calculations, custom workflows, or conditional logic). "
            "For each rule, provide a high-level description without revealing sensitive details. "
            "Return a JSON list of objects with 'rule', 'description', and 'domain' fields. If no rules are found, return an empty list."
        )

        try:
            async with llm_semaphore:
                async with asyncio.timeout(15):
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        result = await asyncio.get_event_loop().run_in_executor(
                            executor, lambda: llm_config._llm.complete(semantic_prompt)
                        )
            output_text = result.text.strip()

            with log_lock:
                logger.debug(f"Raw LLM output: {output_text[:500]}...")

            if not output_text:
                raise ValueError("Empty response from LLM")

            semantic_rules = json.loads(output_text)
            if not isinstance(semantic_rules, list):
                raise ValueError("LLM response is not a list")

            for rule in semantic_rules:
                rule["source"] = "semantic"
                rules_detected.append(rule)
                with log_lock:
                    logger.debug(f"Detected {rule.get('domain', 'unknown')} rule via semantic: {rule.get('description', '')[:100]}...")

        except (json.JSONDecodeError, ValueError, asyncio.TimeoutError, Exception) as e:
            with log_lock:
                logger.warning(f"Semantic rule detection failed: {str(e)}")

    # Final log summary
    if not rules_detected:
        with log_lock:
            logger.debug(f"No business rules detected in content for language {language}")
    else:
        with log_lock:
            logger.info(f"Detected {len(rules_detected)} business rules for language {language}")

    return rules_detected

@retry_on_failure(max_attempts=2)
async def generate_requirements_for_chunk(chunk_content: str, chunk_index: int, language: str) -> str:
    """Generate requirements for a code chunk with enhanced RAG"""
    try:
        # Calculate tokens and handle limits
        chunk_tokens = count_tokens(chunk_content)
        chunk_hash = hashlib.md5(chunk_content.encode()).hexdigest()
        
        # Get relevant context from vector store with filtering
        context_data = await vector_store.get_related_context(
            query=chunk_content,
            k=3,  # Reduced from 5 to save tokens
            filters={"language": language}
        )
        
        # Calculate total tokens including context
        context_tokens = count_tokens(context_data.get('context', ''))
        total_tokens = chunk_tokens + context_tokens
        
        # If total tokens exceed limit, truncate context
        if total_tokens > MAX_SAFE_TOKENS:
            with log_lock:
                logger.warning(f"Total tokens ({total_tokens}) exceed limit. Truncating context.")
            # Keep chunk content but reduce context
            context_data['context'] = context_data.get('context', '')[:MAX_SAFE_TOKENS // 4]
        
        # If chunk itself is too large, truncate it
        if chunk_tokens > MAX_SAFE_TOKENS * 3 // 4:
            with log_lock:
                logger.warning(f"Chunk {chunk_index + 1} exceeds token limit. Truncating.")
            chunk_content = chunk_content[:MAX_SAFE_TOKENS * 3 // 4]
        
        # Detect business rules
        rules_detected = await detect_business_rules(chunk_content, language)
        rules_summary = (
            "\n".join([f"Detected {rule['domain']} rule ({rule['source']}): {rule['description']}" for rule in rules_detected])
            if rules_detected else ""
        )
        
        # Build enhanced prompt with metadata
        prompt = f"""Act as a senior business analyst reviewing a {language} source file segment.
This is segment {chunk_index + 1} of a larger file. Generate concise business requirements in plain text, avoiding markdown symbols (*, -, #, **).
Do NOT use the phrase "this chunk" or "this segment".

The output MUST have five sections: Overview, Objective, Use Case, Key Functionalities, Workflow Summary, separated by two newlines.

If the code contains organization-specific logic or calculations (e.g., proprietary formulas or business rules), generalize them without revealing sensitive details. For example, instead of specifying an exact calculation, say "performs proprietary calculations based on internal rules."

Use the following detected rules for context:
{rules_summary}"""

        # Add context if available
        if context_data.get('context'):
            metadata = context_data.get('metadata', {})
            related_files = context_data.get('related_files', [])
            prompt += f"""

Relevant Context from Similar Code:
{context_data['context']}

Context Metadata:
- Total related files: {metadata.get('total_files', 0)}
- Most relevant file: {related_files[0]['file_path'] if related_files else 'None'}"""

        prompt += f"""

Analyze this segment:
```{language}
{chunk_content}
```

Constraints:
- Plain text, no markdown
- Five non-empty sections
- Key Functionalities numbered, each followed by a blank line
- Use file summary and detected rules for context
- Return only the structured requirements

Generate the following sections:
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
"""
        
        async with llm_semaphore:
            try:
                async with asyncio.timeout(30):
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        result = await asyncio.get_event_loop().run_in_executor(
                            executor, lambda: llm_config._llm.complete(
                                prompt=prompt,
                                max_tokens=500,
                                temperature=0.2
                            )
                        )
                
                req_text = result.text.strip()
                with log_lock:
                    logger.debug(f"Raw LLM output for chunk {chunk_index + 1} (hash: {chunk_hash}): {req_text[:200]}...")
                
                # Process and format the response
                req_text = re.sub(r'\n\s*\n+', '\n\n', req_text)
                req_text = re.sub(r'this chunk|this segment', 'the code', req_text, flags=re.IGNORECASE)
                
                # Split into sections
                sections = re.split(r'\n\n(?=Overview|Objective|Use Case|Key Functionalities|Workflow Summary)', req_text)
                section_dict = {}
                
                for section in sections:
                    for header in ['Overview', 'Objective', 'Use Case', 'Key Functionalities', 'Workflow Summary']:
                        if section.startswith(header):
                            section_dict[header] = section[len(header):].strip()
                            break
                
                # Ensure all sections exist
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
                
                # Format Key Functionalities
                if section_dict['Key Functionalities']:
                    func_lines = section_dict['Key Functionalities'].split('\n')
                    formatted_funcs = []
                    for line in func_lines:
                        if re.match(r'^\d+\.\s', line):
                            formatted_funcs.append(line.strip())
                    if not formatted_funcs:
                        formatted_funcs = ["1. Basic Processing: Handles core tasks.", "2. Support Functions: Assists operations."]
                    section_dict['Key Functionalities'] = '\n\n'.join(formatted_funcs)
                
                # Combine sections
                output_sections = [
                    f"Overview\n{section_dict['Overview']}",
                    f"Objective\n{section_dict['Objective']}",
                    f"Use Case\n{section_dict['Use Case']}",
                    f"Key Functionalities\n{section_dict['Key Functionalities']}",
                    f"Workflow Summary\n{section_dict['Workflow Summary']}"
                ]
                
                return '\n\n'.join(output_sections)
                
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
    except Exception as e:
        with log_lock:
            logger.error(f"Error in generate_requirements_for_chunk: {str(e)}")
        return (
            f"Overview\nError analyzing {language.lower()} code segment.\n\n"
            f"Objective\nUnable to determine specific objectives.\n\n"
            f"Use Case\nGeneral {language.lower()} file operations.\n\n"
            f"Key Functionalities\n1. Basic Processing: Handles core tasks.\n\n2. Error Recovery: Provides fallback functionality.\n\n"
            f"Workflow Summary\nSupports basic {language.lower()} file workflow."
        )

@retry_on_failure(max_attempts=2)
async def extract_organization_specific_rules_for_chunk(chunk_content: str, chunk_index: int, language: str) -> str:
    chunk_hash = hashlib.md5(chunk_content.encode()).hexdigest()
    if count_tokens(chunk_content) > MAX_SAFE_TOKENS:
        with log_lock:
            logger.warning(f"Chunk {chunk_index + 1} exceeds token limit for org-specific rules. Truncating.")
        chunk_content = chunk_content[:MAX_SAFE_TOKENS * 3 // 4]

    rules_detected = await detect_business_rules(chunk_content, language)
    rules_summary = (
        "\n".join([f"Detected {rule['domain']} rule ({rule['source']}): {rule['description']}" for rule in rules_detected])
        if rules_detected else ""
    )

    min_lines = 3
    if len(chunk_content) > 1000:
        min_lines = 5
    elif len(chunk_content) > 5000:
        min_lines = 7
    elif len(chunk_content) > 10000:
        min_lines = 10

    prompt = (
        f"You are a senior business analyst. Carefully review the following {language} source file segment and produce a highly detailed, elaborate, and comprehensive business logic summary.\n"
        f"For each section below (Overview, Objective, Use Case, Key Functionalities, Workflow Summary), provide an exhaustive, content-rich explanation based strictly on the actual file content.\n"
        f"- Expand on every available detail: describe all functions, classes, workflows, and business scenarios in depth.\n"
        f"- Avoid vague, generic, or filler statements. Do NOT repeat phrases like 'utility file' or 'supports operations' unless those are explicitly and uniquely justified by the file content.\n"
        f"- For small files, elaborate as much as possible on every element, inferring business purpose, logic, and workflow from names, comments, and code structure.\n"
        f"- For each rule, provide a high-level description of its purpose and usage (e.g., 'Calculates loan eligibility using a proprietary formula based on customer data').\n"
        f"- Use the following detected rules for context:\n{rules_summary}\n"
        f"- Your output MUST be at least {min_lines} lines, with every line relevant and derived from the file. If the file is small, expand each section as much as possible based on available information.\n"
        f"- Do NOT add generic, filler, or placeholder lines. If there is insufficient business logic, only output what is truly present in the file, but do your best to elaborate on every detail.\n"
        f"If no organization-specific rules are found, return \"\"\n"
        f"Return only the summarized rules.\n"
        f"Analyze this segment:\n"
        f"```{language}\n{chunk_content}\n```\n"
    )
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
            rules_text = re.sub(r'this chunk|this segment', 'the content', rules_text, flags=re.IGNORECASE)
            if not rules_text or rules_text.isspace():
                rules_text = ""
            return rules_text
        except asyncio.TimeoutError:
            with log_lock:
                logger.error(f"LLM timed out for org-specific rules chunk {chunk_index + 1} (hash: {chunk_hash})")
            return ""
        except Exception as e:
            with log_lock:
                logger.error(f"Error for org-specific rules chunk {chunk_index + 1}: {e}")
            return ""

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
    functionalities_section = ''
    if functionalities:
        functionalities_section = f"Key Functionalities\n{'\n\n'.join(functionalities)}\n\n"
    # else: leave functionalities_section empty (or optionally: functionalities_section = 'Key Functionalities\nNo explicit functionalities detected.\n\n')

    combined = (
        f"Overview\n{overview}\n\n"
        f"Objective\n{objective}\n\n"
        f"Use Case\n{use_case}\n\n"
        f"{functionalities_section}"
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

async def extract_vba_macros(file_path: str) -> str:
    """Extract VBA macros from .xlsm, .xlsb, or .xls files using oletools or fallback methods. Also parse macro names, parameters, and comments for richer summaries."""
    try:
        import subprocess
        import sys
        import os
        import re
        
        # Check if oletools is installed
        try:
            import oletools.olevba as olevba
            vba_parser = olevba.VBA_Parser(file_path)
            vba_modules = []
            
            if vba_parser.detect_vba_macros():
                for (filename, stream_path, vba_filename, vba_code) in vba_parser.extract_macros():
                    if vba_code.strip():
                        # Parse macro names, parameters, and comments
                        macro_summaries = []
                        for match in re.finditer(r'((Sub|Function)\s+(\w+)\s*\((.*?)\))', vba_code, re.IGNORECASE):
                            header, kind, name, params = match.groups()
                            # Extract comments before macro
                            pre_lines = vba_code[:match.start()].splitlines()[-3:]
                            comments = [l.strip() for l in pre_lines if l.strip().startswith("'")]
                            macro_summaries.append(f"Macro: {name}\nType: {kind}\nParameters: {params}\nComments: {' '.join(comments) if comments else '[No comments]'}\n")
                        vba_modules.append(f"--- VBA Module: {vba_filename} ---\n{vba_code}\n\nSummary:\n" + "\n".join(macro_summaries))
                vba_parser.close()
                
                if vba_modules:
                    return "\n\n".join(vba_modules)
                return "No VBA macros found or could not be extracted."
            else:
                return "No VBA macros detected in the file."
        except ImportError:
            # Fallback to basic detection without oletools
            if file_path.lower().endswith(('.xlsm', '.xlsb', '.xls')):
                try:
                    # Attempt to read VBA project data directly (basic approach)
                    with open(file_path, 'rb') as f:
                        content = f.read().decode('latin1', errors='ignore')
                        if 'Sub ' in content or 'Function ' in content:
                            return "VBA macros detected (detailed analysis requires oletools package: pip install oletools)"
                        return "No VBA macros detected in the file."
                except Exception as e:
                    with log_lock:
                        logger.warning(f"Fallback VBA detection failed for {file_path}: {str(e)}")
                    return "Error extracting VBA macros: oletools not installed and fallback detection failed."
        except Exception as e:
            with log_lock:
                logger.warning(f"Error extracting VBA macros from {file_path}: {str(e)}")
            return f"Error extracting VBA macros: {str(e)}"
            
    except Exception as e:
        with log_lock:
            logger.error(f"Unexpected error in VBA macro extraction for {file_path}: {str(e)}")
        return f"Error extracting VBA macros: {str(e)}"

async def extract_excel_content(file_path: str) -> str:
    """Extract content from Excel files (.xlsx, .xlsm, .xls) including VBA macros, formulas, validations, and formatting."""
    try:
        content_lines = []
        _, ext = os.path.splitext(file_path)
        is_macro_enabled = ext.lower() in {'.xlsm', '.xlsb', '.xls'}

        # Always try to extract VBA macros for macro-enabled files (even if workbook fails)
        vba_content = None
        if is_macro_enabled:
            try:
                vba_content = await extract_vba_macros(file_path)
                if vba_content and not vba_content.startswith("Error"):
                    content_lines.append("=== VBA MACROS ===")
                    content_lines.append(vba_content)
                    content_lines.append("")
                elif vba_content and vba_content.startswith("Error"):
                    content_lines.append(f"Error extracting VBA macros: {vba_content}")
                    content_lines.append("")
            except Exception as e:
                content_lines.append(f"Error extracting VBA macros: {str(e)}")
                content_lines.append("")

        # Try to open the workbook for reading
        try:
            workbook = openpyxl.load_workbook(file_path, data_only=False, read_only=True)
        except Exception as e:
            content_lines.append(f"Error reading workbook content: {str(e)}")
            # If we got any VBA, at least return that
            if content_lines:
                return "\n".join(content_lines)
            else:
                return f"Error reading workbook content: {str(e)}"

        
        try:
            content_lines.append("=== WORKSHEET DATA ===")
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                content_lines.append(f"\n--- Sheet: {sheet_name} ---")
                
                # Extract cell values and formulas (limit to 100 rows to avoid performance issues)
                max_rows = min(100, sheet.max_row) if sheet.max_row else 100
                for row in sheet.iter_rows(max_row=max_rows):
                    for cell in row:
                        if cell.value is not None:
                            cell_ref = f"{get_column_letter(cell.column)}{cell.row}"
                            if cell.data_type == 'f':  # Formula
                                content_lines.append(f"FORMULA: {cell_ref} = {cell.value}")
                            else:
                                content_lines.append(f"CELL: {cell_ref} = {cell.value}")
                
                # Extract data validation rules
                if hasattr(sheet, 'data_validations') and sheet.data_validations.dataValidation:
                    content_lines.append("\n  Data Validations:")
                    for dv in sheet.data_validations.dataValidation:
                        try:
                            range_str = dv.sqref.ranges[0].coord if hasattr(dv, 'sqref') and dv.sqref else 'Unknown range'
                            validation_type = getattr(dv, 'type', 'custom')
                            formula1 = getattr(dv, 'formula1', None)
                            formula2 = getattr(dv, 'formula2', None)
                            
                            validation_info = [f"Range: {range_str}", f"Type: {validation_type}"]
                            if formula1:
                                validation_info.append(f"Formula1: {formula1}")
                            if formula2:
                                validation_info.append(f"Formula2: {formula2}")
                                
                            content_lines.append(f"DATA_VALIDATION: {' | '.join(validation_info)}")
                        except Exception as e:
                            with log_lock:
                                logger.warning(f"Error processing data validation in {sheet_name}: {str(e)}")
                
                # Extract conditional formatting
                if hasattr(sheet, 'conditional_formatting') and sheet.conditional_formatting:
                    content_lines.append("\n  Conditional Formatting:")
                    for range_str, rules in sheet.conditional_formatting._cf_rules.items():
                        for rule in rules:
                            try:
                                rule_type = rule.type
                                formula = getattr(rule, 'formula', None)
                                if formula:
                                    if isinstance(formula, (list, tuple)):
                                        formula = "; ".join([str(f) for f in formula if f])
                                    content_lines.append(
                                        f"CONDITIONAL_FORMATTING: Range: {range_str} | Type: {rule_type} | Formula: {formula}"
                                    )
                            except Exception as e:
                                with log_lock:
                                    logger.warning(f"Error processing conditional formatting in {sheet_name}: {str(e)}")
                
                # Extract named ranges
                if hasattr(workbook, 'defined_names') and workbook.defined_names:
                    content_lines.append("\n  Named Ranges:")
                    if hasattr(workbook.defined_names, 'definedName'):  # Old versions
                        for name in workbook.defined_names.definedName:
                            content_lines.append(f"  - {name.name}: {name.value}")
                    else:  # New versions (4.0+)
                        for name, named_range in workbook.defined_names.items():
                            content_lines.append(f"  - {name}: {named_range.value}")
            
            # Extract pivot tables
            for sheet in workbook.worksheets:
                if hasattr(sheet, '_pivots') and sheet._pivots:
                    content_lines.append(f"\nPivot Tables in {sheet.title}:")
                    for pivot in sheet._pivots:
                        content_lines.append(f"  - Name: {pivot.name}")
                        content_lines.append(f"    Data Source: {pivot.cache.cacheSource.worksheetSource.ref}")
            
            # Extract external links
            if hasattr(workbook, 'external_links') and workbook.external_links:
                content_lines.append("\nExternal Links:")
                for link in workbook.external_links:
                    content_lines.append(f"  - {link.target}")
            
            # Extract data connections
            if hasattr(workbook, 'connections') and workbook.connections:
                content_lines.append("\nData Connections:")
                for conn in workbook.connections:
                    content_lines.append(f"  - {conn.name}: {conn.connection_string}")
            
        finally:
            workbook.close()
            
        return "\n".join(content_lines) if content_lines else "No content extracted from Excel file."
        
    except Exception as e:
        with log_lock:
            logger.error(f"Error extracting Excel content from {file_path}: {str(e)}")
        return f"Error extracting Excel content: {str(e)}"

async def extract_organization_specific_rules(file_path: str, base_dir: str, language: str) -> OrgSpecificRules:
    try:
        with log_lock:
            logger.debug(f"Extracting org-specific rules for file: {file_path}, Language: {language}")

        _, ext = os.path.splitext(file_path)
        is_excel = ext.lower() in {'.xlsx', '.xlsm', '.xls', '.xlsb'}
        
        if is_excel:
            language = "Excel" if ext.lower() == '.xlsx' else "VBA"
            content = await extract_excel_content(file_path)
        else:
            code_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.scala', '.rb', '.go', '.cpp', '.c', '.rs', '.kt', '.swift', '.php', '.cs'}
            if ext.lower() not in code_extensions:
                with log_lock:
                    logger.info(f"Skipping non-code file for org-specific rules: {file_path} (Extension: {ext})")
                return OrgSpecificRules(
                    relative_path=os.path.relpath(file_path, base_dir),
                    file_name=os.path.basename(file_path),
                    rules="Non-code file: Organization-specific rules extraction not applicable."
                )
            language = language or "Plain Text"
            async with aiofiles.open(file_path, mode='r', encoding='latin-1') as file:
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
                        if rules.strip():
                            rules_list.append(f"Chunk {i+1} (Lines {start_line}-{end_line}):\n{rules}")
                        if (i + 1) % 5 == 0 or i + 1 == len(chunks):
                            with log_lock:
                                logger.info(f"Processed chunk {i+1}/{len(chunks)} for org-specific rules in {file_path}")
                    except Exception as e:
                        with log_lock:
                            logger.error(f"Error processing chunk {i+1} for org-specific rules in {file_path}: {e}")
                        rules_list.append(f"Chunk {i+1} (Lines {start_line}-{end_line}):\nError extracting rules: {str(e)}")
                rules_text = "\n\n".join(rules_list) if rules_list else ""
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
    try:
        with log_lock:
            logger.info("Starting comprehensive project summary generation")
        
        workflow_summaries = []
        file_contexts = []
        
        for file_req in files_requirements.files:
            if file_req.requirements:
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
        
        combined_context = ""
        
        if workflow_summaries:
            combined_context += "WORKFLOW SUMMARIES:\n" + "\n".join(workflow_summaries) + "\n\n"
        
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
        
        if count_tokens(combined_context) > MAX_SAFE_TOKENS:
            with log_lock:
                logger.warning("Combined context exceeds token limit. Truncating for project summary.")
            combined_context = combined_context[:MAX_SAFE_TOKENS * 3 // 4]
        
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
        
        async with llm_semaphore:
            try:
                async with asyncio.timeout(45):
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        result = await asyncio.get_event_loop().run_in_executor(
                            executor, lambda: llm_config._llm.complete(prompt)
                        )
                
                project_summary = result.text.strip()
                
                required_sections = [
                    "PROJECT OVERVIEW",
                    "BUSINESS CONTEXT & OBJECTIVES", 
                    "SYSTEM ARCHITECTURE & APPROACH",
                    "KEY CAPABILITIES & FEATURES",
                    "WORKFLOW & PROCESS INTEGRATION",
                    "BUSINESS VALUE & IMPACT",
                    "TECHNICAL FOUNDATION"
                ]
                
                missing_sections = [section for section in required_sections if section not in project_summary]
                
                if missing_sections or len(project_summary.split()) < 150:
                    with log_lock:
                        logger.warning(f"Generated summary may be incomplete. Missing sections: {missing_sections}")
                    
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
                
                graphs = []
                try:
                    erd_graph = await generate_graph_from_requirement(project_summary, target_graph="entity relationship diagram")
                    if not erd_graph.generated_code.startswith("Error"):
                        graphs.append(erd_graph)
                    
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
    """Generate requirements for a file with optimized RAG integration"""
    try:
        with log_lock:
            logger.debug(f"Processing file: {file_path}, Language: {language}")
        
        _, ext = os.path.splitext(file_path)
        is_excel = ext.lower() in {'.xlsx', '.xlsm', '.xls', '.xlsb'}
        
        if is_excel:
            language = "Excel" if ext.lower() == '.xlsx' else "VBA"
            content = await extract_excel_content(file_path)
        else:
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
            
            # Read file content with improved error handling
            content = None
            try:
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
            except UnicodeDecodeError:
                # Try alternative encodings efficiently
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                            content = await f.read()
                        logger.warning(f"Successfully read {file_path} with {encoding} encoding")
                        break
                    except Exception:
                        continue
                
                # Final fallback to binary mode
                if content is None:
                    try:
                        async with aiofiles.open(file_path, 'rb') as f:
                            raw_content = await f.read()
                            content = raw_content.decode('latin-1', errors='replace')
                        logger.warning(f"Read {file_path} in binary mode with error replacement")
                    except Exception as e:
                        logger.error(f"Failed to read {file_path}: {str(e)}")
                        return _create_fallback_requirements(file_path, base_dir, language, "encoding error")
            except Exception as e:
                logger.error(f"Unexpected error reading {file_path}: {str(e)}")
                return _create_fallback_requirements(file_path, base_dir, language, "file read error")
        
        if content is None:
            return _create_fallback_requirements(file_path, base_dir, language, "could not decode file content")

        # Count tokens and determine processing strategy
        total_tokens = count_tokens(content)
        with log_lock:
            logger.debug(f"File: {file_path}, Language: {language}, Size: {len(content)} bytes, Tokens: {total_tokens}")
        
        # Process RAG storage in background (non-blocking)
        asyncio.create_task(_store_in_vector_db_async(content, file_path, language, base_dir))
        
        # Extract dependencies in parallel with requirements generation
        dependencies_task = asyncio.create_task(_extract_dependencies_async(file_path, content, language, base_dir))
        
        # Generate requirements based on file size
        if total_tokens < 1000:
            requirements_text = await _generate_small_file_requirements(content, file_path, language)
        elif total_tokens <= MAX_SAFE_TOKENS:
            requirements_text = await _generate_medium_file_requirements(content, file_path, language, total_tokens)
        else:
            requirements_text = await _generate_large_file_requirements(content, file_path, language, total_tokens)
        
        # Wait for dependencies to complete
        dependencies = await dependencies_task
        
        # Add dependencies to requirements text
        dep_text = _format_dependencies(dependencies)
        requirements_text += f"\n\nDependent Files\n{dep_text}"
        
        return FileRequirements(
            relative_path=os.path.relpath(file_path, base_dir),
            file_name=os.path.basename(file_path),
            requirements=requirements_text,
            dependencies=dependencies
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


async def _store_in_vector_db_async(content: str, file_path: str, language: str, base_dir: str):
    """Store content in vector database asynchronously without blocking main processing"""
    try:
        # Only chunk if content is large enough to warrant it
        if count_tokens(content) > 2000:
            chunks = await split_into_chunks(content, language, file_path)
        else:
            # For smaller files, store as single chunk
            chunks = [(content, 1, content.count('\n') + 1)]
        
        documents = []
        for i, (chunk, start_line, end_line) in enumerate(chunks):
            documents.append({
                "content": chunk,
                "metadata": {
                    "file_path": file_path,
                    "language": language,
                    "start_line": start_line,
                    "end_line": end_line,
                    "base_dir": base_dir,
                    "chunk_index": i
                }
            })
        
        # Store in vector database
        await vector_store.add_documents(documents)
        # Save only periodically or at the end to reduce I/O
        if len(documents) > 10:  # Only save for larger document sets
            vector_store.save(VECTOR_STORE_DIR)
            
    except Exception as e:
        logger.error(f"Failed to store chunks in vector store for {file_path}: {str(e)}")


async def _extract_dependencies_async(file_path: str, content: str, language: str, base_dir: str):
    """Extract dependencies asynchronously"""
    try:
        from app.services.dependency_service import detect_dependencies
        dependencies = await detect_dependencies(file_path, content, language, base_dir)
        
        validated_dependencies = []
        for dep in dependencies:
            try:
                validated_dep = FileDependency(**dep.dict())
                validated_dependencies.append(validated_dep)
            except ValidationError as ve:
                with log_lock:
                    logger.error(f"Invalid FileDependency for {file_path}: {dep.dict()}, Error: {str(ve)}")
        
        with log_lock:
            logger.debug(f"Validated dependencies for {file_path}: {[dep.dict() for dep in validated_dependencies]}")
        
        return validated_dependencies
    except Exception as e:
        logger.error(f"Error extracting dependencies for {file_path}: {str(e)}")
        return []


async def _generate_small_file_requirements(content: str, file_path: str, language: str) -> str:
    """Generate requirements for small files using templates"""
    content_hash = _hash_content(content)
    rules_detected = await detect_business_rules(content, language)
    rules_summary = (
        "\n".join([f"Detected {rule['domain']} rule ({rule['source']}): {rule['description']}" 
                  for rule in rules_detected])
        if rules_detected else ""
    )
    
    with log_lock:
        logger.debug(f"Used small-file template for {file_path} (hash: {content_hash})")
    
    if language.lower() in ['javascript', 'jsx', 'js']:
        return (
            f"Overview\nConfiguration file for {os.path.basename(file_path)} in a frontend application. {rules_summary}\n\n"
            f"Objective\nTo define settings or utilities for the application.\n\n"
            f"Use Case\nUsed during application initialization or runtime for UI setup.\n\n"
            f"Key Functionalities\n1. Configuration: Defines application settings.\n\n2. Utility Support: Provides helper functions.\n\n"
            f"Workflow Summary\nIntegrates with frontend framework for setup."
        )
    elif language.lower() in ["excel", "vba"]:
        # Enhanced VBA/Excel processing similar to first file
        logic_summary = summarize_excel_vba_logic(content)
        # Generate a business narrative summary from the extracted logic
        business_narrative = await narrative_business_logic_summary(logic_summary, file_path)
        
        # Format key_functionalities as numbered points with blank lines
        key_functionalities = business_narrative['key_functionalities']
        if isinstance(key_functionalities, list):
            formatted_key_funcs = [f"{i+1}. {func}" for i, func in enumerate(key_functionalities)]
            key_functionalities = "\n\n".join(formatted_key_funcs)
        elif isinstance(key_functionalities, str) and key_functionalities.startswith("- "):
            # Handle fallback case where key_functionalities is a string with bullet points
            funcs = [f.strip("- ") for f in key_functionalities.split("\n") if f.strip().startswith("- ")]
            formatted_key_funcs = [f"{i+1}. {func}" for i, func in enumerate(funcs)]
            key_functionalities = "\n\n".join(formatted_key_funcs)
        
        return (
            f"Overview\n{business_narrative['overview']}\n\n"
            f"Objective\n{business_narrative['objective']}\n\n"
            f"Use Case\n{business_narrative['use_case']}\n\n"
            f"Key Functionalities\n{key_functionalities}\n\n"
            f"Workflow Summary\n{business_narrative['workflow']}"
        )
    else:
        return (
            f"Overview\nUtility file for {os.path.basename(file_path)}. {rules_summary}\n\n"
            f"Objective\nTo provide supporting functions.\n\n"
            f"Use Case\nSupports general business operations.\n\n"
            f"Key Functionalities\n1. Basic Utilities: Provides helper functions.\n\n2. Support Functions: Assists operations.\n\n"
            f"Workflow Summary\nSupports general business workflow."
        )


async def _generate_medium_file_requirements(content: str, file_path: str, language: str, total_tokens: int) -> str:
    """Generate requirements for medium-sized files using LLM"""
    # Truncate if necessary
    if total_tokens > MAX_SAFE_TOKENS // 2:
        content = content[:int(len(content) * MAX_SAFE_TOKENS * 0.3 / total_tokens)]
        with log_lock:
            logger.warning(f"Truncated {file_path} to {len(content)} bytes")
    
    rules_detected = await detect_business_rules(content, language)
    rules_summary = (
        "\n".join([f"Detected {rule['domain']} rule ({rule['source']}): {rule['description']}" 
                  for rule in rules_detected])
        if rules_detected else ""
    )
    
    prompt = (
        f"Act as a senior business analyst reviewing a {language} source file.\n"
        f"Generate a comprehensive business requirements document in plain text, avoiding markdown symbols.\n"
        f"Do NOT use the phrase 'this file' or 'this code'. "
        f"If the content contains organization-specific logic or calculations (e.g., proprietary formulas or business rules), "
        f"generalize them without revealing sensitive details. For example, instead of specifying an exact calculation, "
        f"say 'performs proprietary calculations based on internal rules.'\n"
        f"Use the following detected rules for context:\n{rules_summary}\n"
        f"Provide exactly five sections: Overview, Objective, Use Case, Key Functionalities, and Workflow Summary, "
        f"separated by two newlines.\n\n"
        f"Overview\nDescribe the file's purpose and role (50-100 words).\n\n"
        f"Objective\nState the primary business goal (1-2 sentences).\n\n"
        f"Use Case\nOutline business scenarios (2-3 sentences).\n\n"
        f"Key Functionalities\nList 2-5 capabilities, each on a new line, numbered (e.g., 1., 2.), followed by a blank line.\n\n"
        f"Workflow Summary\nDescribe the business process (2-3 sentences).\n\n"
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
            req_text = re.sub(r'this file|this code', 'the content', req_text, flags=re.IGNORECASE)
            
            # Validate and format the response
            return _validate_and_format_requirements(req_text, file_path, language, rules_summary)
            
        except asyncio.TimeoutError:
            with log_lock:
                logger.error(f"LLM timed out for {file_path}")
            return _create_fallback_requirements_text(file_path, language, "timeout", rules_summary)
        except ValueError as ve:
            with log_lock:
                logger.error(f"Value error during LLM processing for {file_path}: {ve}")
            return _create_fallback_requirements_text(file_path, language, "invalid data", rules_summary)
        except Exception as e:
            with log_lock:
                logger.error(f"Error during LLM processing for {file_path}: {e}")
            return _create_fallback_requirements_text(file_path, language, "processing error", rules_summary)


async def _generate_large_file_requirements(content: str, file_path: str, language: str, total_tokens: int) -> str:
    """Generate requirements for large files using chunking strategy"""
    try:
        with log_lock:
            logger.debug(f"Splitting {file_path} into chunks due to high token count: {total_tokens}")
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
                requirements_list.append(_create_chunk_fallback(i + 1, file_path))
        
        # Combine all requirements
        combined_req = await combine_requirements(requirements_list, language)
        return combined_req
        
    except MemoryError as me:
        with log_lock:
            logger.error(f"Memory error during chunking for {file_path}: {me}")
        return _create_fallback_requirements_text(file_path, language, "memory constraints", "")
    except Exception as e:
        with log_lock:
            logger.error(f"Chunking failed for {file_path}: {str(e)}")
        return _create_fallback_requirements_text(file_path, language, "processing error", "")


# Helper functions
def _create_fallback_requirements(file_path: str, base_dir: str, language: str, error_type: str) -> FileRequirements:
    """Create fallback requirements when processing fails"""
    req_text = (
        f"Overview\nFailed to process {os.path.basename(file_path)} due to {error_type}.\n\n"
        f"Objective\nTo support intended {language.lower()} business functions.\n\n"
        f"Use Case\nIntended {language.lower()} business operations.\n\n"
        f"Key Functionalities\n1. Intended Functionality: Expected to provide core features.\n\n2. Support Functions: Assists operations.\n\n"
        f"Workflow Summary\nExpected to integrate with {language.lower()} system workflows.\n\n"
        f"Dependent Files\nNo dependencies detected."
    )
    
    return FileRequirements(
        relative_path=os.path.relpath(file_path, base_dir),
        file_name=os.path.basename(file_path),
        requirements=req_text,
        dependencies=[]
    )


def _create_fallback_requirements_text(file_path: str, language: str, error_type: str, rules_summary: str) -> str:
    """Create fallback requirements text"""
    return (
        f"Overview\nLimited analysis for {os.path.basename(file_path)} due to {error_type}. {rules_summary}\n\n"
        f"Objective\nTo support core {language.lower()} business functions.\n\n"
        f"Use Case\nGeneral {language.lower()} business processes.\n\n"
        f"Key Functionalities\n1. Basic Processing: Performs core tasks.\n\n2. Support Functions: Assists operations.\n\n"
        f"Workflow Summary\nIntegrates with {language.lower()} system workflow."
    )


def _validate_and_format_requirements(req_text: str, file_path: str, language: str, rules_summary: str) -> str:
    """Validate and format LLM-generated requirements"""
    sections = re.split(r'\n\n(?=Overview|Objective|Use Case|Key Functionalities|Workflow Summary|Dependent Files)', req_text)
    section_dict = {}
    
    for section in sections:
        for header in ['Overview', 'Objective', 'Use Case', 'Key Functionalities', 'Workflow Summary', 'Dependent Files']:
            if section.startswith(header):
                section_dict[header] = section[len(header):].strip()
                break

    # Validate required sections
    if len(section_dict) < 5 or not all(section_dict.get(h) for h in ['Overview', 'Objective', 'Use Case', 'Key Functionalities', 'Workflow Summary']):
        with log_lock:
            logger.warning(f"Invalid file requirements for {file_path}. Using fallback.")
        return _create_fallback_requirements_text(file_path, language, "invalid format", rules_summary)
    
    # Format Key Functionalities properly
    if 'Key Functionalities' in section_dict:
        func_lines = section_dict['Key Functionalities'].split('\n')
        formatted_funcs = [line.strip() for line in func_lines if re.match(r'\d+\.\s', line)]
        if not formatted_funcs:
            formatted_funcs = ["1. Basic Processing: Supports core tasks.", "2. Support Functions: Assists operations."]
        section_dict['Key Functionalities'] = '\n\n'.join(formatted_funcs)
    
    return (
        f"Overview\n{section_dict.get('Overview', 'Summary unavailable.')}. {rules_summary}\n\n"
        f"Objective\n{section_dict.get('Objective', 'To support core functionality.')}\n\n"
        f"Use Case\n{section_dict.get('Use Case', 'Supports business operations.')}\n\n"
        f"Key Functionalities\n{section_dict.get('Key Functionalities', '1. Basic Processing: Supports core tasks.\n\n2. Support Functions: Assists operations.')}\n\n"
        f"Workflow Summary\n{section_dict.get('Workflow Summary', 'Integrates with system workflow.')}"
    )


def _format_dependencies(dependencies: list) -> str:
    """Format dependencies for display"""
    if not dependencies:
        return "No dependencies detected."
    
    dep_lines = []
    for dep in dependencies:
        dep_lines.append(dep.file_name)
        dep_lines.append(dep.dependency_reason.replace('\n', ' '))
    
    return "\\n".join(dep_lines)


def _create_chunk_fallback(chunk_num: int, file_path: str) -> str:
    """Create fallback text for failed chunk processing"""
    return (
        f"Overview\nFailed to process chunk {chunk_num} of {os.path.basename(file_path)}.\n\n"
        f"Objective\nTo support chunk functionality.\n\n"
        f"Use Case\nGeneral chunk operations.\n\n"
        f"Key Functionalities\n1. Partial Processing: Supports chunk {chunk_num} operations.\n\n2. Fallback: Provides basic functionality.\n\n"
        f"Workflow Summary\nProcesses available chunk data."
    )


async def generate_requirements_for_files_whole(directory: str) -> FilesRequirements:
    """Optimized batch processing of files"""
    logger.info(f"Starting requirements generation for {directory}")
    
    try:
        # Get all files to process
        code_files = get_code_files(directory)
        all_files = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file)
                
                if ext.lower() in {'.xlsx', '.xlsm', '.xls', '.xlsb'}:
                    language = "Excel" if ext.lower() == '.xlsx' else "VBA"
                    all_files.append((file_path, language))
                elif (file_path, get_file_language(file_path)) in code_files:
                    all_files.append((file_path, get_file_language(file_path)))

        total_files = len(all_files)
        logger.info(f"Found {total_files} files for requirements generation in {directory}")

        files_requirements = []
        completed_count = 0

        # Process files in smaller batches with controlled concurrency
        batch_size = min(BATCH_SIZE, 6)  # Limit batch size for better control
        file_batches = [all_files[i:i + batch_size] for i in range(0, total_files, batch_size)]
        
        for batch_num, batch in enumerate(file_batches):
            logger.info(f"Processing batch {batch_num + 1}/{len(file_batches)}")
            
            tasks = [
                generate_requirements_for_file_whole(file_path, directory, language)
                for file_path, language in batch
            ]
            
            # Use asyncio.gather with return_exceptions to handle failures gracefully
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing {batch[i][0]}: {result}")
                    # Create fallback requirement
                    file_path, language = batch[i]
                    fallback = _create_fallback_requirements(file_path, directory, language, "processing error")
                    files_requirements.append(fallback)
                else:
                    files_requirements.append(result)
            
            completed_count += len(batch)
            progress = completed_count / total_files * 100
            logger.info(f"Progress: {completed_count}/{total_files} files processed ({progress:.1f}%)")
            
            # Brief pause between batches to prevent overwhelming the system
            if batch_num < len(file_batches) - 1:
                await asyncio.sleep(0.1)

        # Sort results by relative path
        files_requirements.sort(key=lambda x: x.relative_path)
        
        # Create final object
        files_requirements_obj = FilesRequirements(files=files_requirements)
        
        # Generate project summary and graphs
        project_summary, graphs = await generate_project_summary(files_requirements_obj)
        files_requirements_obj.project_summary = project_summary
        files_requirements_obj.graphs = graphs

        # Save vector store at the end
        try:
            vector_store.save(VECTOR_STORE_DIR)
            logger.info("Vector store saved successfully")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")

        logger.info(f"Completed requirements generation for {len(files_requirements)} files with project summary and {len(graphs)} graphs")
        return files_requirements_obj
        
    except Exception as e:
        logger.error(f"Error processing directory {directory}: {str(e)}")
        return FilesRequirements(files=[])