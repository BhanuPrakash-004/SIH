# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
import json

class Settings(BaseSettings):
    # Load settings from .env file
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    # LLM & Embedding Models
    EMBEDDING_MODEL: str = "mxbai-embed-large:335m"
    LLM_MODEL: str = "llama3"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"

    # Application Paths
    FAISS_PATH: str = "vector_store.faiss"
    UPLOAD_DIRECTORY: str = "uploaded_files"

    # n8n Integration
    N8N_WEBHOOK_URLS_JSON: Optional[str] = '[]'

    @property
    def N8N_WEBHOOK_URLS(self) -> List[str]:
        return json.loads(self.N8N_WEBHOOK_URLS_JSON or '[]')

settings = Settings()