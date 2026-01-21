from fastapi import FastAPI

from app.api.routers.ingest import router as ingest_router
from app.api.routers.retrieve import register_retrieve_routes

app = FastAPI(
    title="RAG Server",
    version="1.0",
)

# REST routers
app.include_router(ingest_router)

# LangServe routes
register_retrieve_routes(app)