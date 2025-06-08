import requests
import json
from ...config import settings

# Removed simple_summary for now 
# - "simple_summary": A string that starts with "In simple terms:" and re-explains the section's meaning in plain, everyday language.


# We are upgrading to a powerful instruction-following model
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"

def get_layman_summary(text: str) -> dict:
    """
    Invokes an instruction-following model to generate a structured, 
    easy-to-understand analysis of a document.

    Args:
        text: The input text to be analyzed.

    Returns:
        A dictionary containing the structured analysis.
    
    Raises:
        Exception: If the API call fails or the response is not valid JSON.
    """
    if not settings.HUGGINGFACE_API_TOKEN:
        raise ValueError("HUGGINGFACE_API_TOKEN is not set in the configuration.")

    prompt = f"""
    You are an expert legal analyst who specializes in explaining complex documents to a layperson.
    Your task is to analyze the following document and provide a structured, easy-to-understand breakdown in a valid JSON format.

    The JSON object must have the following four keys:
    1. "document_type": A string identifying the type of document (e.g., "Privacy Policy", "Terms of Service").
    2. "overall_summary": A string providing a brief, one-paragraph overview of the document's main purpose.
    3. "key_terms": An array of objects. Each object must have two keys:
        - "term": A string containing a significant keyword or phrase (e.g., "Personal Information").
        - "definition": A string explaining what that term means in simple language.
    4. "sectional_summaries": An array of objects. Each object must represent a distinct section of the document and have three keys:
        - "section_title": A string with a descriptive title for that section.
        - "detailed_summary": A string providing a comprehensive summary of the section's content.


    Please analyze the following document and generate the complete JSON object as described.

    ---
    DOCUMENT:
    {text}
    ---
    """

    headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_TOKEN}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 2048,
            "return_full_text": False
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        generated_text = result[0].get("generated_text")
        
        if not generated_text:
            raise ValueError("The model did not return any generated text.")

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
