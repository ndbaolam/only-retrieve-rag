from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from app.vectorstores import get_vectordb
from app.schemas.retrieve_schema import RetrieveInput
from app.reranker import rerank_docs

def retrieve_router(input: RetrieveInput):
    collection = input["collection"]
    query = input["query"]

    vectordb  = get_vectordb(collection_name=collection)
    if vectordb is None:
        raise ValueError(f"Unknown collection: {collection}")

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10, "rerank_type": "rrf"},
    )

    docs =  retriever.invoke(query)

    results = rerank_docs(
        inputs={
            "documents": docs,
            "question": query
        }
    )

    return results


def build_retrieve_chain():
    return RunnableLambda(retrieve_router)