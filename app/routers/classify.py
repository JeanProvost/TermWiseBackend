from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(
    prefix="/classify",
    tags=["Classification"],
)

class ClassificationRequest(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    document_type: str
    confidence: float

@router.post("/", response_model=ClassificationResponse)
def classify_document(request: ClassificationRequest):
    """
    Receives text and returns a placeholder document classification.
    This will be replaced with actual model inference later.
    """
    # Placeholder logic
    return {
        "document_type": "Privacy Policy",
        "confidence": 0.95
    }
