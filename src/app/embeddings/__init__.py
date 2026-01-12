# app/embeddings/__init__.py
from langchain.embeddings import init_embeddings
from app.configs.settings import settings

def create_embedding():
    return init_embeddings(
            model=settings.embed_model,
            provider=settings.embedding_provider,
        )