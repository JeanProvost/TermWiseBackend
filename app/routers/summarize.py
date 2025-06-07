from fastapi import APIRouter, HTTPException
from app.schemas import SummarizeRequest, SummarizeResponse
from app.core.ai_summarize import summarize_content
from pydantic import BaseModel

router = APIRouter(
    prefix="/summarize",
    tags=["Summarization"],
)

class SummarizationRequest(BaseModel):
    text: str

class SummarizationResponse(BaseModel):
    summary: str

@router.post("/", response_model=SummarizationResponse)
def summarize_text(request: SummarizationRequest):
    """
    Receives text and returns a placeholder summary.
    This will be replaced with actual model inference.
    """
    # Placeholder logic
    summary = f"This is a summary of the text: {request.text[:30]}..."
    return {"summary": summary}

@router.post("/summarize", response_model=SummarizeResponse)
async def create_summary(request: SummarizeRequest):
    try:
        summary = await summarize_content(request.content)
        return {
            "summary": summary,
            "document_type": "DetectedType",
            "truncated": False,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))