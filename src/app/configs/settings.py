# app/configs/settings.py
from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # provider switch
    llm_provider: Literal["ollama", "openai", "vllm"]
    embedding_provider: Literal["ollama", "openai", "hf"]

    # ollama
    base_url: str | None = None
    llm_model: str | None = None
    embed_model: str | None = None
    temperature: float = 0.2
    num_ctx: int = 4096

    milvus_host: str = "localhost"
    milvus_port: int = 19530

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
