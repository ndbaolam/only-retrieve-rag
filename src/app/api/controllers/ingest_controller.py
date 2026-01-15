import os
import tempfile
import logging
import pandas as pd
from fastapi import UploadFile, HTTPException

from app.ingestion import upsert_documents
from app.vectorstores import get_vectordb

logger = logging.getLogger(__name__)

async def ingest_csv_controller(
    file: UploadFile,
    collection_name: str = "docs",
    chunksize: int = 100,
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported",
        )

    try:
        vectordb = get_vectordb(collection_name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        batches = 0

        for i, df_batch in enumerate(
            pd.read_csv(tmp_path, chunksize=chunksize)
        ):
            logger.info(
                "Ingesting CSV batch %s (rows=%s)",
                i,
                len(df_batch),
            )

            await upsert_documents(
                vector_store=vectordb,
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
                collection_name=collection_name,
                is_chunked=False,
            )

            batches += 1

        return {
            "status": "success",
            "filename": file.filename,
            "batches_processed": batches,
            "chunksize": chunksize,
        }

    except Exception as e:
        logger.exception("CSV ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
