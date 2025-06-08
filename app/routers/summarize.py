from fastapi import APIRouter, HTTPException
from ..schemas import SummarizationRequest, SummarizationResponse
from ..core.ai import summarizer

router = APIRouter(
    prefix="/summarize",
    tags=["Summarization"],
)

@router.post("/", response_model=SummarizationResponse)
def summarize_text(request: SummarizationRequest):
    """
    Receives text, sends it for structured analysis,
    and returns the result.
    """
    try:
        # Call the new function to get the structured dictionary
        structured_summary = summarizer.get_structured_summary(request.text)
        
        # Pydantic will automatically validate the dictionary and format the response
        return structured_summary
    except Exception as e:
        # Log the exception for debugging
        print(f"An error occurred during summarization: {e}")
        raise HTTPException(
            status_code=500, 
            detail="There was an error processing your request."
        )