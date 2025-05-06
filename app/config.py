"""
Configuration management for the RAGify core service.
Uses Pydantic's BaseSettings for robust environment variable handling and validation.
"""

from typing import Optional
from pydantic import BaseSettings, validator
from loguru import logger
import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables with sensible defaults.
    """
    # Vector Database Configuration
    QDRANT_URL: str
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "documents"
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "openai"
    EMBEDDING_DIMENSION: int = 1536  # OpenAI ada-002 default
    
    # OpenAI Configuration
    USE_OPENAI: bool = True
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4-turbo"
    
    # Service Configuration
    SERVICE_PORT: int = 50051
    SERVICE_HOST: str = "0.0.0.0"
    WORKERS: int = 4
    
    # Redis Configuration (for task queue)
    REDIS_URL: Optional[str] = None
    REDIS_PASSWORD: Optional[str] = None
    
    # Chunking Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    
    @validator("OPENAI_API_KEY")
    def validate_openai_key(cls, v: str, values: dict) -> str:
        """Validate that OpenAI API key is present if USE_OPENAI is True."""
        if values.get("USE_OPENAI", True) and not v:
            raise ValueError("OPENAI_API_KEY is required when USE_OPENAI is True")
        return v
    
    @validator("QDRANT_URL")
    def validate_qdrant_url(cls, v: str) -> str:
        """Validate that Qdrant URL is properly formatted."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("QDRANT_URL must start with http:// or https://")
        return v
    
    class Config:
        """Pydantic config class."""
        case_sensitive = True
        env_file = ".env"

# Create global settings instance
settings = Settings()

# Configure logging
logger.remove()  # Remove default handler
logger.add(
    "logs/ragify.log",
    rotation="500 MB",
    retention="10 days",
    level=settings.LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)
logger.add(lambda msg: print(msg), level=settings.LOG_LEVEL)

logger.info("Configuration loaded successfully")