import json
import torch
from transformers import pipeline
from typing import Dict, Any
from ...config import settings
# from transformers import BitsAndBytesConfig  # Commented out for GPT model

# Initialize the pipeline globally to avoid reloading
_pipeline = None

def _get_pipeline():
    """
    Lazy-load the pipeline to avoid loading it at import time.
    This ensures it's only loaded when first needed.
    """
    global _pipeline
    
    if _pipeline is None:
        print(f"Loading {settings.MODEL_NAME} model...")
        
        # Create pipeline using the GPT model configuration
        _pipeline = pipeline(
            "text-generation",
            model=settings.MODEL_NAME,
            torch_dtype="auto",
            device_map="auto" if settings.USE_GPU and torch.cuda.is_available() else None,
        )
        
        device = "GPU" if settings.USE_GPU and torch.cuda.is_available() else "CPU"
        print(f"Model loaded successfully on {device}!")
    
    return _pipeline

def get_layman_summary(text: str) -> Dict[str, Any]:
    """
    Uses the openai/gpt-oss-120b model to generate a structured, 
    easy-to-understand analysis of a document.

    Args:
        text: The input text to be analyzed.

    Returns:
        A dictionary containing the structured analysis.
    
    Raises:
        Exception: If the model generation fails or the response is not valid JSON.
    """
    pipe = _get_pipeline()
    
    # Truncate text if it's too long to avoid exceeding context window
    if len(text) > settings.MAX_INPUT_LENGTH:
        text = text[:settings.MAX_INPUT_LENGTH] + "..."
    
    # Format the prompt for GPT model
    user_prompt = f"""You are a specialized API that converts legal documents into structured JSON.
Analyze the following document and respond with ONLY a single, valid JSON object.
Do not include any introductory text, explanations, or markdown formatting like ```json. Your entire response must be the raw JSON object.

The JSON object must strictly follow this structure:
{{
  "document_type": "string",
  "overall_summary": "string",
  "key_terms": [
    {{
      "term": "string",
      "definition": "string"
    }}
  ],
  "sectional_summaries": [
    {{
      "section_title": "string",
      "detailed_summary": "string"
    }}
  ]
}}

Document to analyze:
{text}

JSON Response:"""

    # Use the messages format for the pipeline
    messages = [
        {"role": "user", "content": user_prompt}
    ]

    try:
        # Generate response using pipeline
        outputs = pipe(
            messages,
            max_new_tokens=settings.MAX_NEW_TOKENS,
        )
        
        # Extract the generated text from pipeline output
        generated_text = outputs[0]["generated_text"][-1]["content"]
        
        # Try to extract JSON from the response
        # Sometimes models add extra text, so we'll try to find the JSON part
        generated_text = generated_text.strip()
        
        # If the response starts with ```json, extract the content
        if generated_text.startswith("```json"):
            generated_text = generated_text[7:]  # Remove ```json
            if generated_text.endswith("```"):
                generated_text = generated_text[:-3]  # Remove closing ```
        
        # Parse the JSON
        try:
            structured_data = json.loads(generated_text)
        except json.JSONDecodeError:
            # If direct parsing fails, try to find JSON object in the text
            import re
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                structured_data = json.loads(json_match.group())
            else:
                raise ValueError("Could not extract valid JSON from model response")
        
        # Validate the structure
        required_keys = ["document_type", "overall_summary", "key_terms", "sectional_summaries"]
        for key in required_keys:
            if key not in structured_data:
                # Provide a default structure if parsing failed
                structured_data = {
                    "document_type": "Unknown Document",
                    "overall_summary": "Failed to generate summary. The document appears to be: " + text[:200] + "...",
                    "key_terms": [],
                    "sectional_summaries": []
                }
                break
        
        return structured_data

    except Exception as e:
        print(f"Error during model generation: {e}")
        print(f"Generated text: {generated_text if 'generated_text' in locals() else 'No output generated'}")
        
        # Return a fallback response
        return {
            "document_type": "Unknown Document",
            "overall_summary": f"An error occurred while processing this document: {str(e)}",
            "key_terms": [],
            "sectional_summaries": []
        }
