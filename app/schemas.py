from pydantic import BaseModel

class SummarizeRequest(BaseModel):
    content: str
    url: str | None = None
    metadata: dict | None = None

class SummarizeResponse(BaseModel):
    summary: str
    document_type: str
    truncated: bool