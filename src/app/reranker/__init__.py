# app/retrievers/vector_retriever.py
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document

from app.schemas.rerank_schema import RerankInput

_ce_model = None

def get_ce_model():
    global _ce_model
    if _ce_model is None:
        _ce_model = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            model_kwargs={"device": "cpu"},
        )
    return _ce_model

def rerank_docs(inputs: RerankInput, top_k: int = 5) -> list[Document]:
    question = inputs["question"]
    docs = inputs["documents"]

    ce_rf = get_ce_model()

    pairs = [(question, doc.page_content) for doc in docs]
    scores = ce_rf.score(pairs)

    for doc, score in zip(docs, scores):
        doc.metadata["ce_score"] = float(score)

    return sorted(
        docs,
        key=lambda d: d.metadata["ce_score"],
        reverse=True,
    )[:top_k]