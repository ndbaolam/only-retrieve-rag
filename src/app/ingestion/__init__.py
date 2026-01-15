from .split import chunking
from .convert import dataframe_to_documents

import pandas as pd
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

async def upsert_documents(
    vector_store,
    df: pd.DataFrame,
    content_cols: list[str],
    metadata_cols: list[str],
    collection_name: str,
    batch_size: int = 100,
    is_chunked: bool = True
):
    """
    Upsert KB documents into vector store (delete old chunks, insert new ones)
    """

    # ---------- 1. Convert dataframe -> documents ----------
    try:
        docs = dataframe_to_documents(
            df,
            content_cols=content_cols,
            metadata_cols=metadata_cols,
        )
    except Exception as e:
        logger.exception("Failed to convert dataframe to documents")
        raise

    if not docs:
        logger.warning("No documents to ingest (empty docs)")
        return

    # ---------- 2. Collect valid documentids ----------
    document_ids = set()
    for doc in docs:
        doc_id = doc.metadata.get("documentid")
        if isinstance(doc_id, str) and doc_id.strip():
            document_ids.add(doc_id)
        else:
            logger.warning(
                "Skip document without valid documentid: %s",
                repr(doc.metadata)
            )

    # ---------- 3. Delete old chunks ----------
    for doc_id in document_ids:
        try:
            existed = vector_store.search_by_metadata(
                expr=f'documentid == "{doc_id}"'
            )
        except Exception as e:
            logger.warning(
                "Metadata search failed for documentid=%s: %s",
                doc_id, e
            )
            existed = False

        if existed:
            try:
                expr = " or ".join(
                    f'documentid == "{doc_id}"' for doc_id in document_ids
                )
                vector_store.client.delete(
                    collection_name=collection_name,
                    filter=expr
                )
                logger.info("Deleted old chunks for documentid=%s", doc_id)
            except Exception as e:
                # delete fail → KHÔNG nên insert mới (tránh duplicate)
                logger.exception(
                    "Failed to delete old chunks for documentid=%s",
                    doc_id
                )
                continue

    # ---------- 4. Chunk + batch insert ----------
    batch_docs: list[Document] = []
    batch_ids: list[str] = []

    if(is_chunked):
        for chunk in chunking(docs):
            doc_id = chunk.metadata.get("documentid")
            chunk_index = chunk.metadata.get("chunk_index")

            if not isinstance(doc_id, str) or not doc_id.strip():
                logger.warning("Skip chunk without documentid")
                continue

            if not isinstance(chunk_index, int):
                logger.warning(
                    "Skip chunk without valid chunk_index: %s",
                    chunk.metadata
                )
                continue

            chunk_id = f"{doc_id}_{chunk_index}"

            batch_docs.append(chunk)
            batch_ids.append(chunk_id)

            if len(batch_docs) >= batch_size:
                try:
                    await vector_store.aadd_documents(
                        batch_docs,
                        ids=batch_ids
                    )
                    logger.info("Inserted %d chunks", len(batch_docs))
                except Exception as e:
                    logger.exception(
                        "Failed to insert batch (size=%d). First chunk_id=%s",
                        len(batch_docs),
                        batch_ids[0] if batch_ids else None
                    )
                    # Chiến lược: bỏ batch lỗi, không retry ở đây
                finally:
                    batch_docs.clear()
                    batch_ids.clear()
    else: 
        from uuid import uuid4
        batch_docs = docs
        batch_ids = [str(uuid4()) for _ in docs]

    # ---------- 5. Flush remaining ----------
    if batch_docs:
        try:
            await vector_store.aadd_documents(
                batch_docs,
                ids=batch_ids
            )
            logger.info("Inserted %d chunks (final batch)", len(batch_docs))
        except Exception:
            logger.exception(
                "Failed to insert final batch (size=%d)",
                len(batch_docs)
            )