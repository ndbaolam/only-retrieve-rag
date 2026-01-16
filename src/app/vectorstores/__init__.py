# app/vectorstores/__init__.py
from langchain_milvus import Milvus
from langchain_milvus.utils.sparse import BaseSparseEmbedding
from langchain_core.embeddings import Embeddings
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from typing import List, Dict, Any
import scipy.sparse as sp

from app.configs.settings import settings

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

    def embed_query(self, text: str) -> Dict[int, float]:
        sparse = self.ef.encode_queries([text])["sparse"]
        return self._sparse_to_dict(sparse)

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        sparse_arrays = self.ef.encode_documents(texts)["sparse"]

        # Handle 1D sparse array (single document)
        if sparse_arrays.ndim == 1:
            return [self._sparse_to_dict(sparse_arrays)]

        # Handle 2D sparse array (multiple documents)
        # Convert to CSR for efficient row access
        if not sp.isspmatrix_csr(sparse_arrays):
            sparse_arrays = sp.csr_matrix(sparse_arrays)
        
        return [
            self._sparse_to_dict(self._get_row(sparse_arrays, i))
            for i in range(sparse_arrays.shape[0])
        ]

    def _get_row(self, sparse_matrix, row_idx):
        """Safely extract a single row from a sparse matrix/array."""
        if hasattr(sparse_matrix, 'getrow'):
            # csr_matrix has getrow
            return sparse_matrix.getrow(row_idx)
        else:
            # csr_array: use slicing
            return sparse_matrix[row_idx:row_idx+1]

    def _sparse_to_dict(self, sparse_array: Any) -> Dict[int, float]:
        """Convert sparse array to dictionary format."""
        # Ensure we're working with COO format
        if hasattr(sparse_array, 'tocoo'):
            coo = sparse_array.tocoo()
        else:
            coo = sparse_array
            
        if coo.ndim == 1:  
            # 1D sparse array
            indices = coo.nonzero()[0]
            return {
                int(i): float(coo[i])
                for i in indices
            }
        else:  
            # 2D sparse array (convert to COO for iteration)
            if not sp.isspmatrix_coo(coo):
                coo = coo.tocoo()
            return {
                int(col): float(val)
                for col, val in zip(coo.col, coo.data)
            }

_vectordb = None

def get_vectordb(collection_name: str):
    global _vectordb
    if _vectordb is not None:
        return _vectordb

    model = BGEM3EmbeddingFunction(
        model_name="BAAI/bge-large-en-v1.5"
    )

    _vectordb = Milvus(
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

    return _vectordb
