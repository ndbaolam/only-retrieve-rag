# app/retrievers/vector_retriever.py
from typing import List

from langchain_core.documents import Document

def get_vector_retriever(
    vectorstore,
    k: int = 5,
    search_type: str = "similarity",
):
    """
    Return a LangChain retriever from a vector store.

    vectorstore: Chroma, FAISS, etc.
    """
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )

def retrieve(
    query: str,
    vectorstore,
    k: int = 5,
) -> List[Document]:
    """
    One-shot retrieval helper (for scripts / debug).
    """
    retriever = get_vector_retriever(vectorstore, k=k)
    return retriever.get_relevant_documents(query)
