# app/vectorstores/__init__.py
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import init_embeddings

from app.configs.settings import settings

_vectordb = None

def get_vectordb(collection_name: str):
    global _vectordb
    if _vectordb is not None:
        return _vectordb

    # ------------------
    # Dense embedding
    # ------------------
    dense_embedding = init_embeddings(
        model=settings.embed_model,
        provider=settings.embedding_provider,
        base_url=settings.ollama_url,
    )

    # ------------------
    # Sparse embedding
    # ------------------
    sparse_embedding = HuggingFaceEmbeddings(
        model_name="ibm-granite/granite-embedding-30m-sparse",
        encode_kwargs={"normalize_embeddings": False},
    )

    _vectordb = Milvus(
        embedding_function=[dense_embedding, sparse_embedding],
        vector_field=["dense", "sparse"],
        collection_name=collection_name,
        connection_args={
            "uri": settings.milvus_url
        },
        consistency_level="Bounded",
        drop_old=False,
    )

    return _vectordb
