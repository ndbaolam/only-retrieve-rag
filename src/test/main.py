from app.vectorstores import get_vectordb
from app.ingestion import dataframe_to_documents
from uuid import uuid4
import pandas as pd

if __name__ == "__main__":
    vector_store = get_vectordb(
        collection_name="test_3"
    )

    df = pd.read_csv("data/kb_info.csv").sample(n=3)

    docs = dataframe_to_documents(
        df,
        content_cols=["summary", "solution"],
        metadata_cols=["documentid", "last_updated", "reference", "applies_to", "cause", "product_versions", "service", "title"]
    )

    # client = vector_store.client
    # schema = client.create_schema(
    #     auto_id=False,
    #     enable_dynamic_field=True,
    # )

    # schema.add_field(
    #     field_name="pk",
    #     datatype=DataType.VARCHAR,
    #     max_length=16,
    #     is_primary=True,
    # )

    ids = [str(uuid4()) for _ in range(len(docs))]

    vector_store.add_documents(docs, ids=ids)
    