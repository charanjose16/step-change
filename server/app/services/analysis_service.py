import os
import asyncio
import json
from enum import Enum
from typing import List, Tuple,Dict
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

# Use a safety threshold below the maximum context length (e.g. 100000 tokens)
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
    # If no file content is provided, use a generic approach
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

        # Validate and clean the generated code
        if not generated_code or len(generated_code.split('\n')) < 2:
            return GraphResponse(
                target_graph=target_graph, 
                generated_code=f"Error: Insufficient graph definition for {target_graph}"
            )

        # Remove markdown code block markers if present
        pattern = r"```mermaid\s*([\s\S]*?)\s*```"
        match = re.search(pattern, generated_code)
        if match:
            generated_code = match.group(1).strip()

        # Additional syntax validation
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




async def generate_requirements_for_file_whole(
    file_path: str,
    base_dir: str,
    language: str
) -> FileRequirements:
    logger.info(f"fast requirements: processing file {file_path}")
    req_text = ""
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()

        token_count = count_tokens(content)

        # Check token count BEFORE making the API call
        if token_count > MAX_SAFE_TOKENS:
            num_chunks = math.ceil(token_count / MAX_SAFE_TOKENS)
            logger.info(
                f"File {file_path} exceeds safe token limit ({MAX_SAFE_TOKENS}): {token_count} tokens. Splitting into {num_chunks} chunks."
            )
            # Ensure chunk size is at least 1
            chunk_size = max(1, len(content) // num_chunks)
            req_text_parts = []
            tasks = []

            for i in range(num_chunks):
                chunk_start = i * chunk_size
                # Ensure the last chunk takes the remainder
                chunk_end = (i + 1) * chunk_size if i < num_chunks - 1 else len(content)
                chunk = content[chunk_start:chunk_end]

                prompt_chunk = (
                    f"Act as a meticulous product analyst and extract *all* functional requirements, rules, and logic from part {i+1}/{num_chunks} of this {language} source file. "
                    "Pay close attention to even small details, conditions, constraints, and specific behaviors implemented in the code. "
                    "Describe the key functionalities, user benefits, system interactions, data flows, and any implemented business rules comprehensively. "
                    "Present the findings as clear, cohesive paragraphs. Do not omit any details, no matter how minor they seem. "
                    "Return only the requirements text.\n\n"
                    "```" + chunk + "```"
                )
                # Define the task for the LLM call within the semaphore context
                async def chunk_task(p):
                     async with llm_semaphore:
                         return await asyncio.to_thread(llm_config._llm.complete, p)

                tasks.append(chunk_task(prompt_chunk))

            # Gather results from all chunk tasks
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result_chunk in enumerate(chunk_results):
                 if isinstance(result_chunk, Exception):
                      logger.error(f"Error processing chunk {i+1}/{num_chunks} for {file_path}: {result_chunk}")
                      req_text_parts.append(f"[Error processing chunk {i+1}]") # Add error placeholder
                 elif hasattr(result_chunk, "text"):
                      req_text_parts.append(result_chunk.text.strip())
                 else:
                      logger.warning(f"Unexpected result type for chunk {i+1}/{num_chunks} of {file_path}: {type(result_chunk)}")
                      req_text_parts.append("[Unexpected result format]")


            req_text = "\n\n---\n\n".join(req_text_parts) # Join parts with a separator

        else:
            # Token count is within limits, process the whole file
            logger.info(f"File {file_path} is within safe token limit ({token_count} tokens). Processing as a whole.")
            prompt = (
                f"Act as a meticulous product analyst and extract *all* functional requirements, rules, and logic from this {language} source file. "
                "Pay close attention to even small details, conditions, constraints, and specific behaviors implemented in the code. "
                "For each identified requirement or rule, specify:\n"
                "  1. The feature, capability, or rule name/description.\n"
                "  2. Its business purpose, user benefit, or system constraint it enforces.\n"
                "  3. The key user/system interactions, data flows, or specific logic involved.\n"
                "Write the analysis as clear, comprehensive, and cohesive paragraphs. Ensure no functionality, rule, or logic, however minor, is missed. "
                "Do not include code references or implementation details unless essential for describing the requirement. "
                "Return only the requirements text.\n\n"
                "```" + content + "```"
            )
            try:
                async with llm_semaphore:
                    result = await asyncio.to_thread(llm_config._llm.complete, prompt)
                req_text = result.text.strip()
            except Exception as e:
                 logger.error(f"LLM error processing whole file {file_path} (within token limit): {e}", exc_info=True)
                 req_text = f"[Error processing file: {e}]" # Indicate error in output

    except FileNotFoundError:
        logger.error(f"File not found during requirements generation: {file_path}")
        req_text = "[Error: File not found]"
    except Exception as e:
        logger.error(f"General error processing requirements for {file_path}: {e}", exc_info=True)
        req_text = f"[General Error: {e}]" # Indicate general error

    return FileRequirements(
        relative_path=os.path.relpath(file_path, base_dir),
        file_name=os.path.basename(file_path),
        requirements=req_text
    )


async def generate_requirements_for_files_whole(
    directory: str
) -> FilesRequirements:
    """
    Parallel requirement generation by sending each file as a whole or in chunks if too large.
    """
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
    # Use asyncio.as_completed to process results as they finish and log progress
    for coro in asyncio.as_completed(tasks):
        try:
            file_req = await coro
            files_requirements.append(file_req)
        except Exception as e:
            # This shouldn't happen often if generate_requirements_for_file_whole handles its errors,
            # but log just in case. The specific file causing the error isn't easily known here.
            logger.error(f"Unexpected error retrieving result from requirements task: {e}", exc_info=True)
        finally:
            completed_count += 1
            if completed_count % 20 == 0 or completed_count == total_files:
                 logger.info(f"Requirements generation progress: {completed_count}/{total_files} files processed.")

    logger.info(f"Finished requirements generation for all {total_files} files.")
    # Sort results by relative path for consistency, if desired
    files_requirements.sort(key=lambda fr: fr.relative_path)
    return FilesRequirements(files=files_requirements)