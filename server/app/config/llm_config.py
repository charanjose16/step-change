from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr, PrivateAttr
from functools import lru_cache
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import Settings
from typing import Optional



class LLMConfig(BaseSettings):
    # Azure OpenAI Configuration
    azure_openai_api_key: SecretStr = Field(..., validation_alias="AZURE_OPENAI_API_KEY")
    azure_openai_api_version: str = Field(..., validation_alias="AZURE_OPENAI_API_VERSION")
    azure_openai_endpoint: str = Field(..., validation_alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment_name: str = Field(..., validation_alias="AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_openai_model: str = Field(..., validation_alias="AZURE_OPENAI_MODEL")

    # Azure OpenAI Embedding Configuration
    azure_openai_embed_api_endpoint: str = Field(..., validation_alias="AZURE_OPENAI_EMBED_API_ENDPOINT")
    azure_openai_embed_api_key: SecretStr = Field(..., validation_alias="AZURE_OPENAI_EMBED_API_KEY")
    azure_openai_embed_model: str = Field(..., validation_alias="AZURE_OPENAI_EMBED_MODEL")
    azure_openai_embed_version: str = Field(..., validation_alias="AZURE_OPENAI_EMBED_VERSION")

    # Private attributes to hold the instantiated models
    _llm: Optional[AzureOpenAI] = PrivateAttr(default=None)
    _embed_model: Optional[AzureOpenAIEmbedding] = PrivateAttr(default=None)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    def init_llamaindex(self) -> None:
        """Initialize LlamaIndex with Azure OpenAI components."""
        self._llm = AzureOpenAI(
            model=self.azure_openai_model,
            deployment_name=self.azure_openai_deployment_name,
            api_key=self.azure_openai_api_key.get_secret_value(),
            azure_endpoint=self.azure_openai_endpoint,
            api_version=self.azure_openai_api_version,
            max_tokens=4096,
            temperature=0.1
        )

        self._embed_model = AzureOpenAIEmbedding(
            model=self.azure_openai_embed_model,
            api_key=self.azure_openai_embed_api_key.get_secret_value(),
            azure_endpoint=self.azure_openai_embed_api_endpoint,
            api_version=self.azure_openai_embed_version,
        )

        # Configure global LlamaIndex settings
        Settings.llm = self._llm
        Settings.embed_model = self._embed_model


# Create a singleton instance and initialize models once at startup
llm_config = LLMConfig()
llm_config.init_llamaindex()