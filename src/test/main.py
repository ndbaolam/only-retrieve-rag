from app.vectorstores import get_vectordb
from app.ingestion import dataframe_to_documents, chunking
from uuid import uuid4
import pandas as pd

import asyncio

async def main():
    vector_store = get_vectordb(
        collection_name="test_chunk"
    )

    df = pd.read_csv("data/kb_info.csv").iloc[:3]

    docs = dataframe_to_documents(
        df,
        content_cols=["summary"],
        metadata_cols=["documentid", "last_updated", "reference", "applies_to", "cause", "product_versions", "service", "title", "solution"]
    )

    chunks = chunking(docs)

    ids = [str(uuid4()) for _ in range(len(chunks))]

    await vector_store.aadd_documents(chunks, ids=ids)

if __name__ == "__main__":
    asyncio.run(main())
    