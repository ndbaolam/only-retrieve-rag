# app/configs/settings.py
from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # provider switch
    llm_provider: Literal["ollama", "openai", "vllm"]
    embedding_provider: Literal["ollama", "openai", "hf"]

    # ollama
    ollama_url: str | None = "http://localhost:11434"
    llm_model: str | None = None
    embed_model: str | None = None
    temperature: float = 0.2
    num_ctx: int = 4096

    milvus_url: str = "http://localhost:19530"
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
