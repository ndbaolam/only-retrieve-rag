from fastapi import APIRouter, UploadFile, File, Query

from app.api.controllers.ingest_controller import ingest_controller

router = APIRouter(
    prefix="/ingest",
    tags=["Ingestion"],
)

@router.post("/upload")
async def ingest_file(
    file: UploadFile = File(...),
    collection_name: str = "docs",
    chunksize: int = Query(100, ge=1, le=5000),
):
    """Upload CSV or JSON file for ingestion"""
    return await ingest_controller(
        file=file,
        collection_name=collection_name,
        chunksize=chunksize,
    )