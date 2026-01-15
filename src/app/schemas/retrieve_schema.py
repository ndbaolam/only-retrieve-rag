from pydantic import BaseModel

class RetrieveInput(BaseModel):
    query: str
    collection: str