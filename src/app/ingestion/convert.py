import pandas as pd
from langchain_core.documents import Document

def dataframe_to_documents(
    df: pd.DataFrame,
    content_cols: list[str],
    metadata_cols: list[str],
) -> list[Document]:

    # 1. Fill N/A chỉ cho content
    df = df.copy()
    df = df.fillna("N/A")

    documents = []

    for row in df.itertuples(index=False):
        row_dict = row._asdict()

        # 2. Build page_content
        content = "\n".join(
            f"{row_dict[col]}"
            for col in content_cols
        )

        # 3. Build metadata (skip null / empty)
        metadata = {
            col: row_dict[col]
            for col in metadata_cols
            if pd.notna(row_dict[col]) and row_dict[col] != ""
        }

        documents.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )

    return documents