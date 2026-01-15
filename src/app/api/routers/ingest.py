from fastapi import APIRouter, UploadFile, File, Query

from app.api.controllers.ingest_controller import ingest_csv_controller

router = APIRouter(
    prefix="/ingest",
    tags=["Ingestion"],
)

@router.post("/csv")
async def ingest_csv(
    file: UploadFile = File(...),
    collection_name: str = "docs",
    chunksize: int = Query(100, ge=1, le=5000),
):
    return await ingest_csv_controller(
        file=file,
        collection_name=collection_name,
        chunksize=chunksize,
    )
