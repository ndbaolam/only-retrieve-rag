import pandas as pd
from langchain_core.documents import Document
from langchain_milvus import Milvus
from pymilvus import DataType

def dataframe_to_documents(
    df: pd.DataFrame,
    content_cols: list[str],
    metadata_cols: list[str],
) -> list[Document]:
    documents = []

    for _, row in df.iterrows():
        content = "\n".join(
            f"{col}: {row[col] if pd.notna(row[col]) else 'N/A'}"
            for col in content_cols
        )

        metadata = {
            col: (row[col] if pd.notna(row[col]) else "N/A")
            for col in metadata_cols
        }

        documents.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )

    return documents
