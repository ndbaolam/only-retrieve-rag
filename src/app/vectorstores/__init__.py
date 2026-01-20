from langchain_milvus import Milvus
from langchain_milvus.utils.sparse import BaseSparseEmbedding
from langchain_core.embeddings import Embeddings
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from typing import List, Dict, Any
import scipy.sparse as sp
import logging

from app.configs.settings import settings

logger = logging.getLogger(__name__)

class DenseEmbeddings(Embeddings):
    def __init__(self, model):
        self.ef = model
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.ef.encode_documents(texts)["dense"]
    def embed_query(self, text: str) -> List[float]:
        return self.ef.encode_queries([text])["dense"][0]

class SparseEmbeddings(BaseSparseEmbedding):
    def __init__(self, model):
        self.ef = model
        self.min_sparse_values = 1

    def embed_query(self, text: str) -> Dict[int, float]:
        sparse = self.ef.encode_queries([text])["sparse"]
        result = self._sparse_to_dict(sparse)
        if not result or len(result) < self.min_sparse_values:
            logger.debug("Empty/minimal sparse vector for query, adding dummy")
            result = {0: 0.1}
        return result

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        sparse_arrays = self.ef.encode_documents(texts)["sparse"]

        if sparse_arrays.ndim == 1:
            result = self._sparse_to_dict(sparse_arrays)
            if not result or len(result) < self.min_sparse_values:
                logger.debug("Empty/minimal sparse vector for document 0, adding dummy")
                result = {0: 0.1}
            return [result]

        if not sp.isspmatrix_csr(sparse_arrays):
            sparse_arrays = sp.csr_matrix(sparse_arrays)
        
        results = []
        for i in range(sparse_arrays.shape[0]):
            sparse_dict = self._sparse_to_dict(self._get_row(sparse_arrays, i))
            if not sparse_dict or len(sparse_dict) < self.min_sparse_values:
                logger.debug(f"Empty/minimal sparse vector for document {i}, adding dummy")
                sparse_dict = {0: 0.1}
            results.append(sparse_dict)
        return results

    def _get_row(self, sparse_matrix, row_idx):
        """Safely extract a single row from a sparse matrix/array."""
        if hasattr(sparse_matrix, 'getrow'):
            return sparse_matrix.getrow(row_idx)
        else:
            return sparse_matrix[row_idx:row_idx+1]

    def _sparse_to_dict(self, sparse_array: Any) -> Dict[int, float]:
        """Convert sparse array to dictionary format."""
        if hasattr(sparse_array, 'tocoo'):
            coo = sparse_array.tocoo()
        else:
            coo = sparse_array
            
        if coo.ndim == 1:  
            indices = coo.nonzero()[0]
            return {
                int(i): float(coo[i])
                for i in indices
            }
        else:  
            if not sp.isspmatrix_coo(coo):
                coo = coo.tocoo()
            return {
                int(col): float(val)
                for col, val in zip(coo.col, coo.data)
            }

_vectordb_cache = {}

def get_vectordb(collection_name: str):
    global _vectordb_cache
    
    # Return cached vectordb for this collection
    if collection_name in _vectordb_cache:
        return _vectordb_cache[collection_name]

    # Use correct model name format (without colon for HuggingFace)
    model = BGEM3EmbeddingFunction(
        model_name="BAAI/bge-m3"
    )

    vectordb = Milvus(
        embedding_function=[
            DenseEmbeddings(model),
            SparseEmbeddings(model)
        ],
        vector_field=["dense", "sparse"],
        collection_name=collection_name,
        connection_args={
            "uri": settings.milvus_url
        },
        consistency_level="Bounded",
        drop_old=False,
    )

    _vectordb_cache[collection_name] = vectordb
    return vectordb