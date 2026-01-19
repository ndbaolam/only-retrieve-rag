# app/vectorstores/__init__.py
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from langchain.embeddings import init_embeddings

from app.configs.settings import settings

_vectordb = None

def get_vectordb(collection_name: str):
    global _vectordb
    if _vectordb is not None:
        return _vectordb

    dense = init_embeddings("ollama:bge-m3:567m")
    # sparse = BM25SparseEmbedding()

    _vectordb = Milvus(
        embedding_function=[
            dense
        ],
        builtin_function=BM25BuiltInFunction(),
        vector_field=["dense", "sparse"],
        index_params={},
        collection_name=collection_name,
        connection_args={
            "uri": settings.milvus_url
        },
        consistency_level="Bounded",
        drop_old=True,
    )

    return _vectordb