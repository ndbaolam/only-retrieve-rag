import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

def dataframe_to_documents(
    df: pd.DataFrame,
    content_cols: list[str],
    metadata_cols: list[str],
) -> list[Document]:

    # 1. Fill N/A chỉ cho content
    df = df.copy()
    df[content_cols] = df[content_cols].fillna("N/A")

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

def normalize_text(text: str) -> str:
    import re

    if not isinstance(text, str):
        return ""

    text = (
        text
        .replace("\xa0", " ")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace(" \n", "\n")
        .replace("\n ", "\n")
        .strip()
    )

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text

def chunking(
    docs: list[Document],
    chunk_size: int = 120,
    chunk_overlap: int = 20,
    batch_size: int = 100,
) -> list[Document]:
    """
    Efficient chunking for large number of documents
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[Document] = []

    batch: list[Document] = []

    for doc in docs:
        # normalize without mutating original
        normalized_doc = Document(
            page_content=normalize_text(doc.page_content),
            metadata=doc.metadata,
        )

        batch.append(normalized_doc)

        if len(batch) >= batch_size:
            chunks.extend(splitter.split_documents(batch))
            batch.clear()

    # process remaining docs
    if batch:
        chunks.extend(splitter.split_documents(batch))

    return chunks