from fastapi import APIRouter
from langserve import add_routes

from app.api.controllers.retrieve_controller import build_retrieve_chain

router = APIRouter(
    tags=["Retrieval"],
)

chain = build_retrieve_chain()

def register_retrieve_routes(app):
    add_routes(
        app,
        chain,
        path="/retrieve",
    )
