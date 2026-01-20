import os
import json
import tempfile
import logging
import pandas as pd
from fastapi import UploadFile, HTTPException

from app.ingestion import upsert_documents
from app.vectorstores import get_vectordb

logger = logging.getLogger(__name__)

# Shared metadata columns
METADATA_COLS = [
    "documentid",
    "last_updated",
    "reference",
    "applies_to",
    "cause",
    "product_versions",
    "service",
    "title",
    "solution",
]

async def ingest_controller(
    file: UploadFile,
    collection_name: str = "docs",
    chunksize: int = 100,
):
    """Unified ingestion controller for CSV and JSON files"""
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in [".csv", ".json"]:
        raise HTTPException(
            status_code=400,
            detail="Only CSV and JSON files are supported",
        )

    try:
        vectordb = get_vectordb(collection_name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Parse file based on extension
        if file_ext == ".csv":
            df_iterator = pd.read_csv(tmp_path, chunksize=chunksize)
        else:  # .json
            df_iterator = _parse_json_file(tmp_path, chunksize)

        batches = 0

        for i, df_batch in enumerate(df_iterator):
            logger.info(
                "Ingesting %s batch %s (rows=%s)",
                file_ext.upper(),
                i,
                len(df_batch),
            )

            await upsert_documents(
                vector_store=vectordb,
                df=df_batch,
                content_cols=["summary"],
                metadata_cols=METADATA_COLS,
                collection_name=collection_name,
                is_chunked=False,
            )

            batches += 1

        return {
            "status": "success",
            "filename": file.filename,
            "file_type": file_ext[1:].upper(),
            "batches_processed": batches,
            "chunksize": chunksize,
        }

    except json.JSONDecodeError as e:
        logger.exception("JSON parsing failed")
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        logger.exception(f"{file_ext.upper()} ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _parse_json_file(file_path: str, chunksize: int):
    """Parse JSON file and yield DataFrames in chunks"""
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Handle both array and nested structure
    if isinstance(json_data, dict) and 'kb_info' in json_data:
        records = json_data['kb_info']
    elif isinstance(json_data, list):
        records = json_data
    else:
        raise ValueError("JSON must be array or contain 'kb_info' key with array value")

    df = pd.DataFrame(records)
    
    # Yield chunks
    for i in range(0, len(df), chunksize):
        yield df.iloc[i:i+chunksize]