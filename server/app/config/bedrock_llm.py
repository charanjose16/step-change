import logging
import boto3
import json
import asyncio
from typing import Optional, Any, Callable

logger = logging.getLogger("bedrock_llm_config")

# Bedrock client and model config
_bedrock_client = boto3.client("bedrock-runtime", region_name="eu-west-2")
_SONNET_MODEL_ID = "anthropic.claude-3-7-sonnet-20250219-v1:0"
_TITAN_EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"  # Set this to the correct embedding model for your region

# Helper to get embedding dimension
async def get_embedding_dimension():
    # Use a short string to get the embedding and return its length
    body = {"inputText": "test"}
    response = _bedrock_client.invoke_model(
        modelId=_TITAN_EMBED_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )
    output = response["body"].read()
    data = json.loads(output)
    embedding = data.get("embedding", [])
    return len(embedding)

def _truncate_text(text, max_tokens=8192):
    # Truncate text to max_tokens using a simple word split (for safety)
    # In production, use a tokenizer for exact token count
    words = text.split()
    if len(words) > max_tokens:
        return ' '.join(words[:max_tokens])
    return text

async def call_sonnet(prompt_or_messages, max_tokens=1000, temperature=0.2, stop=None):
    """
    Async function to call Claude Sonnet via Bedrock.
    Accepts either a prompt string or a list of messages (Anthropic format).
    """
    import logging
    logger = logging.getLogger("bedrock_llm_config")
    loop = asyncio.get_event_loop()
    def _invoke():
        if isinstance(prompt_or_messages, str):
            messages = [{"role": "user", "content": prompt_or_messages}]
        else:
            messages = prompt_or_messages
        body = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "anthropic_version": "bedrock-2023-05-31"
        }
        if stop:
            body["stop_sequences"] = stop
        logger.debug(f"[Bedrock LLM] Prompt: {messages[0]['content'][:1000]}... | max_tokens={max_tokens}")
        response = _bedrock_client.invoke_model(
            modelId=_SONNET_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        output = response["body"].read()
        logger.debug(f"[Bedrock LLM] Token usage: max_tokens={max_tokens}")
        try:
            data = json.loads(output)
            if "content" in data and isinstance(data["content"], list) and data["content"]:
                return data["content"][0]["text"]
            elif "content" in data and isinstance(data["content"], str):
                return data["content"]
            else:
                return str(data)
        except Exception as e:
            logger.error(f"[Bedrock LLM] Error parsing response: {e}")
            return output.decode("utf-8")
    return await loop.run_in_executor(None, _invoke)

async def bedrock_embed(text):
    """
    Calls AWS Bedrock Titan Embeddings model and returns the embedding vector.
    Truncates input to 8192 tokens for safety.
    """
    loop = asyncio.get_event_loop()
    def _invoke():
        try:
            safe_text = _truncate_text(text, 8192)
            body = {"inputText": safe_text}
            response = _bedrock_client.invoke_model(
                modelId=_TITAN_EMBED_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body)
            )
            output = response["body"].read()
            data = json.loads(output)
            embedding = data.get("embedding", [0.0] * 1024)
            if not embedding or all(v == 0.0 for v in embedding):
                logger.error(f"[Bedrock Embed] All-zero or missing embedding for input: {text[:100]}")
            return embedding
        except Exception as e:
            logger.error(f"[Bedrock Embed] Exception: {e}")
            return [0.0] * 1024
    return await loop.run_in_executor(None, _invoke)

def bedrock_embed_sync(text):
    """
    Synchronous version of the Bedrock embedding call.
    Truncates input to 8192 tokens for safety.
    """
    try:
        safe_text = _truncate_text(text, 8192)
        body = {"inputText": safe_text}
        response = _bedrock_client.invoke_model(
            modelId=_TITAN_EMBED_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        output = response["body"].read()
        data = json.loads(output)
        embedding = data.get("embedding", [0.0] * 1024)
        if not embedding or all(v == 0.0 for v in embedding):
            logger.error(f"[Bedrock Embed] All-zero or missing embedding for input: {text[:100]}")
        return embedding
    except Exception as e:
        logger.error(f"[Bedrock Embed] Exception: {e}")
        return [0.0] * 1024

class BedrockLLMConfig:
    """
    Configuration and interface for Bedrock LLM and Embeddings.
    Provides .llm (Claude/Sonnet) and .embed_model (Titan Embeddings) compatible with previous usage.
    """
    def __init__(self):
        self._llm = self._get_llm()
        self._embed_model = self._get_embed_model()

    def _get_llm(self) -> Callable:
        # Returns a function compatible with previous .complete() usage
        async def complete(prompt_or_messages, max_tokens=4096, temperature=0.1, stop=None):
            # Accepts either a prompt string or a list of messages
            return await call_sonnet(prompt_or_messages, max_tokens=max_tokens, temperature=temperature, stop=stop)
        return complete

    def _get_embed_model(self):
        # Provide both async and sync for compatibility
        class EmbedModel:
            @staticmethod
            async def aget_text_embedding(text):
                return await bedrock_embed(text)

            @staticmethod
            def get_text_embedding(text):
                return bedrock_embed_sync(text)
        return EmbedModel

    @property
    def llm(self):
        return self._llm

    @property
    def embed_model(self):
        return self._embed_model

    @staticmethod
    async def get_embedding_dimension():
        return await get_embedding_dimension()

    @staticmethod
    def count_tokens(text: str) -> int:
        # Claude token heuristic: 1 token ≈ 4 chars
        return int(len(text) / 4)


# Singleton instance for use across the app
llm_config = BedrockLLMConfig()