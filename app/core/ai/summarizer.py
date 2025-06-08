import requests
import json
from ...config import settings

# We are upgrading to a powerful instruction-following model
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"

def get_structured_summary(text: str) -> dict:
    """
    Invokes a powerful instruction-following model on Hugging Face to generate a
    structured analysis of the provided text.

    Args:
        text: The input text to be analyzed.

    Returns:
        A dictionary containing the structured analysis.
    
    Raises:
        Exception: If the API call fails or the response is not valid JSON.
    """
    if not settings.HUGGINGFACE_API_TOKEN:
        raise ValueError("HUGGINGFACE_API_TOKEN is not set in the configuration.")

    # This prompt instructs the model to act as an analyst and return a specific JSON structure.
    prompt = f"""
    As a legal document analyst, your task is to analyze the following document.
    Please provide a response in a valid JSON format. The JSON object should have three keys:
    1. "document_type": A string identifying the type of document (e.g., "Privacy Policy", "Terms of Service", "Legal Contract").
    2. "key_terms": An array of strings, where each string is a significant keyword or phrase from the document.
    3. "sectional_summaries": An array of objects, where each object has two keys:
        - "section_title": A string containing a descriptive title for a document section.
        - "summary": A string containing a concise summary of that section's content.

    Analyze the following document and provide your response in the specified JSON format:

    ---
    DOCUMENT:
    {text}
    ---
    """

    headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_TOKEN}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 2048, # Increase token limit for complex JSON
            "return_full_text": False
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        # The model's response is a string inside a JSON object, so we need to extract it.
        generated_text = result[0].get("generated_text")
        
        if not generated_text:
            raise ValueError("The model did not return any generated text.")

        # The generated text itself should be a JSON string, so we parse it.
        structured_data = json.loads(generated_text)
        return structured_data

    except requests.exceptions.RequestException as e:
        print(f"Error calling Hugging Face API: {e}")
        print(f"Response body: {e.response.text if e.response else 'No response'}")
        raise
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from model response: {e}")
        print(f"Model output that failed parsing: {generated_text}")
        raise
    except (KeyError, IndexError) as e:
        print(f"Could not parse Hugging Face response structure: {e}")
        print(f"Full response: {result}")
        raise
