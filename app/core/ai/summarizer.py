import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any
from ...config import settings
from transformers import BitsAndBytesConfig

# Initialize the model and tokenizer globally to avoid reloading
_model = None
_tokenizer = None

def _get_model_and_tokenizer():
    """
    Lazy-load the model and tokenizer to avoid loading them at import time.
    This ensures they are only loaded when first needed.
    """
    global _model, _tokenizer
    
    if _model is None or _tokenizer is None:
        print(f"Loading {settings.MODEL_NAME} model...")
        # Try to use fast tokenizer to avoid sentencepiece dependency
        try:
            _tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME, use_fast=True)
        except Exception as e:
            print(f"Fast tokenizer failed: {e}")
            # Fallback to slow tokenizer
            _tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME, use_fast=False)
        
        # Set device_map based on GPU availability
        device_map = "auto" if settings.USE_GPU and torch.cuda.is_available() else "cpu"
        
        quantization_config = None
        if settings.QUANTIZE_MODEL:
            # This configures the 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        # Determine which attention implementation to use
        # NOTE: Flash Attention 2 can be difficult to install on Windows.
        # It's disabled by default here to ensure compatibility.
        attn_implementation = "eager"
        if settings.USE_FLASH_ATTENTION_2:
            try:
                # Check if flash-attn is available
                import flash_attn
                attn_implementation = "flash_attention_2"
            except ImportError:
                print("Flash Attention 2 not available. Falling back to eager attention.")

        _model = AutoModelForCausalLM.from_pretrained(
            settings.MODEL_NAME,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if settings.USE_GPU else torch.float32,
            quantization_config=quantization_config,
            attn_implementation=attn_implementation,
        )
        
        device = "GPU" if settings.USE_GPU and torch.cuda.is_available() else "CPU"
        print(f"Model loaded successfully on {device}!")
    
    return _model, _tokenizer

def get_layman_summary(text: str) -> Dict[str, Any]:
    """
    Uses the local Gemma-2-9b-it model to generate a structured, 
    easy-to-understand analysis of a document.

    Args:
        text: The input text to be analyzed.

    Returns:
        A dictionary containing the structured analysis.
    
    Raises:
        Exception: If the model generation fails or the response is not valid JSON.
    """
    model, tokenizer = _get_model_and_tokenizer()
    
    # Truncate text if it's too long to avoid exceeding context window
    if len(text) > settings.MAX_INPUT_LENGTH:
        text = text[:settings.MAX_INPUT_LENGTH] + "..."
    
    # Format the prompt for Phi-3 using chat template
    system_prompt = """You are an expert legal analyst who specializes in explaining complex documents to a layperson.
Your task is to analyze documents and provide a structured, easy-to-understand breakdown in a valid JSON format.

The JSON object must have the following four keys:
1. "document_type": A string identifying the type of document (e.g., "Privacy Policy", "Terms of Service").
2. "overall_summary": A string providing a brief, one-paragraph overview of the document's main purpose.
3. "key_terms": An array of objects. Each object must have two keys:
    - "term": A string containing a significant keyword or phrase (e.g., "Personal Information").
    - "definition": A string explaining what that term means in simple language.
4. "sectional_summaries": An array of objects. Each object must represent a distinct section of the document and have three keys:
    - "section_title": A string with a descriptive title for that section.
    - "detailed_summary": A string providing a comprehensive summary of the section's content.

Generate ONLY the JSON object as described, with no additional text before or after."""

    user_prompt = f"""Please analyze the following document:

{text}

JSON Response:"""

    # Use Mistral's chat template format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        # Tokenize using the chat template
        input_ids = tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt", 
            add_generation_prompt=True,
            truncation=True,
            max_length=4096
        )
        
        # Move to GPU if available
        device = next(model.parameters()).device
        if isinstance(input_ids, dict):
            input_ids = {k: v.to(device) for k, v in input_ids.items()}
        else:
            # For chat template, input_ids is a tensor
            input_ids = {"input_ids": input_ids.to(device)}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **input_ids,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the output, skipping the input prompt
        generated_text = tokenizer.decode(outputs[0][input_ids['input_ids'].shape[1]:], skip_special_tokens=True)
        
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
