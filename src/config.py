"""
Configuration module for MD&A Generator
Loads environment variables and provides application settings
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Gemini API Configuration
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_api_key_2: str = Field(default="", alias="GEMINI_API_KEY_2") # Secondary key for rotation
    gemini_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_MODEL")
    
    # Groq API Configuration (Fallback)
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    
    # Embedding Configuration
    embedding_provider: str = Field(default="gemini", alias="EMBEDDING_PROVIDER") # Options: "gemini", "huggingface"
    embedding_model: str = Field(default="models/embedding-001", alias="EMBEDDING_MODEL") # For Gemini
    huggingface_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", alias="HUGGINGFACE_MODEL") # For Local/Colab
    
    # ChromaDB Settings
    chroma_persist_dir: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIR")
    collection_name: str = Field(default="financial_statements", alias="COLLECTION_NAME")
    
    # Processing Settings
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    top_k_results: int = Field(default=5, alias="TOP_K_RESULTS")
    
    # Paths
    data_dir: Path = Field(default=Path("./data"))
    output_dir: Path = Field(default=Path("./output/generated_mda"))
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True  # Allow reading env vars by alias
    )
    
    def validate_api_key(self) -> bool:
        """Check if API key is configured"""
        return bool(self.gemini_api_key and self.gemini_api_key != "your_gemini_api_key_here")
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
