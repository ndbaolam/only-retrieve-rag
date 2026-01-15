from pydantic import BaseModel, Field
from langchain_core.documents import Document

class RerankInput(BaseModel):
    documents: list[Document] = Field(default_factory=list)
    question: str