from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    # LLM configuration
    azure_openai_api_key: str = Field(..., env="AZURE_OPENAI_API_KEY")
    azure_openai_api_version: str = Field(..., env="AZURE_OPENAI_API_VERSION")
    azure_openai_endpoint: str = Field(..., env="AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment_name: str = Field(..., env="AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_openai_model: str = Field(..., env="AZURE_OPENAI_MODEL")

    azure_openai_embed_api_endpoint: str = Field(..., env="AZURE_OPENAI_EMBED_API_ENDPOINT")
    azure_openai_embed_api_key: str = Field(..., env="AZURE_OPENAI_EMBED_API_KEY")
    azure_openai_embed_model: str = Field(..., env="AZURE_OPENAI_EMBED_MODEL")
    azure_openai_embed_version: str = Field(..., env="AZURE_OPENAI_EMBED_VERSION")
    azure_openai_embed_deployment_name: str = Field(..., env="AZURE_OPENAI_EMBED_DEPLOYMENT_NAME")

    # Logging configuration    
    env: str = Field("dev", env="ENV")  
    log_level: str = Field("DEBUG", env="LOG_LEVEL")
    log_format: str = Field( "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s", env="LOG_FORMAT")
    logs_dir: Path = Field(Path("logs"), env="LOGS_DIR")

    # Database configuration
    DATABASE_URL: str = Field(..., env="DATABASE_URL")

    # JWT configuration
    JWT_SECRET: str = Field(..., env="JWT_SECRET")
    JWT_ALGORITHM: str = Field(..., env="JWT_ALGORITHM")

    #user credentials
    AUTH_USERNAME: str = Field(..., env="AUTH_USERNAME")
    AUTH_PASSWORD: str = Field(..., env="AUTH_PASSWORD")



    class Config:
        env_file = ".env"

settings = Settings()