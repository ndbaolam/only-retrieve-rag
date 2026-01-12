# app/vectorstores/__init__.py
from pathlib import Path

from langchain_milvus import Milvus

from app.embeddings import create_embedding
from app.configs.settings import settings

def get_vectordb(
    collection_name: str = "docs",
):
    """
    Create / load Milvus Lite vector store
    """
    embedding = create_embedding()

    vectordb = Milvus(
        embedding_function=embedding,
        collection_name=collection_name,
        connection_args={
            "host": settings.milvus_host,
            "port": settings.milvus_port,
        },
    )

    return vectordb
