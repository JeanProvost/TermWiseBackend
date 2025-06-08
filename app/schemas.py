from pydantic import BaseModel

class SummarizationRequest(BaseModel):
    """The request model for the summarization endpoint."""
    text: str

class SectionSummary(BaseModel):
    """A model to hold the summary of a document section."""
    section_title: str
    summary: str

class SummarizationResponse(BaseModel):
    """The structured response model for the summarization endpoint."""
    document_type: str
    key_terms: list[str]
    sectional_summaries: list[SectionSummary]