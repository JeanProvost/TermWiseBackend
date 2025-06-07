from fastapi import APIRouter, HTTPException
from app.schemas import SummarizeRequest, SummarizeResponse
from app.core.ai_summarize import summarize_content

router = APIRouter()

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