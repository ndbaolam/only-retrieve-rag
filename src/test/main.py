from app.vectorstores import get_vectordb
from app.ingestion import upsert_documents
import pandas as pd

import asyncio

async def main():
    vector_store = get_vectordb(
        collection_name="test_chunk"
    )

    for i, df_batch in enumerate(
        pd.read_csv("data/kb_info.csv", chunksize=1)
    ):
        print(f"Processing CSV batch {i}")

        await upsert_documents(
            vector_store=vector_store,
            df=df_batch,
            content_cols=["summary"],
            metadata_cols=[
                "documentid",
                "last_updated",
                "reference",
                "applies_to",
                "cause",
                "product_versions",
                "service",
                "title",
                "solution",
            ],
            collection_name="test_chunk",
            is_chunked=False
        )

        break


if __name__ == "__main__":
    asyncio.run(main())
    