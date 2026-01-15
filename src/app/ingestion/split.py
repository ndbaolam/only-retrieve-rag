from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

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
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # separators=["\n\n", "\n", " ", ""],
    )

    for doc in docs:
        normalized = normalize_text(doc.page_content)

        chunks = splitter.split_text(normalized)

        for idx, chunk in enumerate(chunks):
            yield Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "chunk_index": idx,
                },
            )
