from pydantic import BaseModel
from typing import List

class SummarizationRequest(BaseModel):
    """The request model for the summarization endpoint."""
    text: str

class KeyTerm(BaseModel):
    """A model for a key term and its definition."""
    term: str
    definition: str

class SectionSummary(BaseModel):
    """A model to hold the detailed summary of a single document section."""
    section_title: str
    detailed_summary: str
    simple_summary: str

class SummarizationResponse(BaseModel):
    """The structured response model for the summarization endpoint."""
    document_type: str
    overall_summary: str
    key_terms: List[KeyTerm]
    sectional_summaries: List[SectionSummary]
