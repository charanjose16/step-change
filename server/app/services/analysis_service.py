import os
import asyncio
import json
from enum import Enum
from typing import List, Tuple, Dict
from pydantic import BaseModel
from app.config.llm_config import llm_config
from app.utils import logger
from app.utils.file_utils import get_code_files
import aiofiles
import re
import math
import tiktoken

CHUNK_GROUP_SIZE = 40
LLM_CONCURRENCY_LIMIT = 150
llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY_LIMIT)
LINTER_CONCURRENCY = 300
linter_sema = asyncio.Semaphore(LINTER_CONCURRENCY)
MAX_MODEL_TOKENS = 128000
MAX_SAFE_TOKENS = 80000

class GraphResponse(BaseModel):
    target_graph: str
    generated_code: str

class RequirementsResponse(BaseModel):
    requirements: str

class FileRequirements(BaseModel):
    relative_path: str
    file_name: str
    requirements: str

class FilesRequirements(BaseModel):
    files: List[FileRequirements]

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

async def generate_graph_from_requirement(requirement: str, target_graph: str, file_content: str = None) -> GraphResponse:
    if not file_content:
        file_content = requirement

    if target_graph.lower() == "entity relationship diagram":
        prompt = (
            "Create a precise Mermaid Entity-Relationship Diagram with strict syntax rules:\n"
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
    elif target_graph.lower() == "requirement diagram":
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

    try:
        result = await asyncio.to_thread(llm_config._llm.complete, prompt)
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
    except Exception as e:
        logger.error(f"Graph generation error for {target_graph}: {e}")
        return GraphResponse(
            target_graph=target_graph, 
            generated_code=f"Generation Error: {str(e)}"
        )

async def generate_requirements_for_file_whole(file_path, base_dir, language):
    try:
        async with aiofiles.open(file_path, mode='r', encoding='utf-8') as file:
            content = await file.read()

        prompt = f"""
        Act as a senior business analyst reviewing a {language} source file. 
        Generate a comprehensive business requirements document with clear, concise language.
        Avoid using markdown symbols like *, -, #, or **. 
        Focus on presenting the requirements in a clean, professional plain text format.

        Provide the following structured sections, with each section separated by two newlines:

        Overview
        Describe the file's purpose, its role in the broader system, and its strategic importance.

        Objective
        Clearly state the primary business goal or problem this component addresses.
        Explain the key business value it delivers.

        Use Case
        Outline the specific business scenarios where this component is critical.
        Describe how it supports business processes or user interactions.

        Key Functionalities
        List the core business capabilities implemented in this component.
        Each functionality must be numbered (e.g., 1., 2.) and placed on a new line.
        Each point must be followed by a blank line to separate it from the next point.
        Focus on what the functionality achieves from a business perspective, not technical details.
        Example format:
        1. Product Display: The component fetches and displays a list of products...

        2. Wishlist Management: Users can add products to their wishlist...

        Workflow Summary
        Describe the end-to-end business process or workflow this component supports.
        Explain how it integrates with other system components.

        Constraints:
        - Use plain text only, no markdown symbols (*, -, #, **)
        - Ensure each section is separated by two newlines
        - For Key Functionalities, use numbered points with a blank line between each point
        - Avoid technical jargon and focus on business value

        Analyze the following file content and generate requirements accordingly:
        ```{language}
        {content}
        ```
        """

        async with llm_semaphore:
            result = await asyncio.to_thread(llm_config._llm.complete, prompt)
        
        req_text = result.text.strip()
        return FileRequirements(
            relative_path=os.path.relpath(file_path, base_dir),
            file_name=os.path.basename(file_path),
            requirements=req_text
        )
    except Exception as e:
        logger.error(f"Error generating requirements for {file_path}: {e}", exc_info=True)
        return FileRequirements(
            relative_path=os.path.relpath(file_path, base_dir),
            file_name=os.path.basename(file_path),
            requirements=f"Error generating requirements: {str(e)}"
        )

async def generate_requirements_for_files_whole(directory: str) -> FilesRequirements:
    logger.info("fast requirements: starting pass over files")
    code_files = get_code_files(directory)
    total_files = len(code_files)
    logger.info(f"Found {total_files} files for requirements generation.")
    tasks = [
        generate_requirements_for_file_whole(fp, directory, lang)
        for fp, lang in code_files
    ]

    files_requirements: List[FileRequirements] = []
    completed_count = 0
    for coro in asyncio.as_completed(tasks):
        try:
            file_req = await coro
            files_requirements.append(file_req)
        except Exception as e:
            logger.error(f"Unexpected error retrieving result from requirements task: {e}", exc_info=True)
        finally:
            completed_count += 1
            if completed_count % 20 == 0 or completed_count == total_files:
                logger.info(f"Requirements generation progress: {completed_count}/{total_files} files processed.")

    logger.info(f"Finished requirements generation for all {total_files} files.")
    files_requirements.sort(key=lambda fr: fr.relative_path)
    return FilesRequirements(files=files_requirements)