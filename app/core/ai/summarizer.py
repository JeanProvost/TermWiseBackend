import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any
from ...config import settings
# from transformers import BitsAndBytesConfig  # Commented out for GPT model

# Initialize the pipeline globally to avoid reloading
_pipeline = None
_raw_model = None
_tokenizer = None

def _get_pipeline():
    """
    Lazy-load the pipeline to avoid loading it at import time.
    This ensures it's only loaded when first needed.
    """
    global _pipeline
    
    if _pipeline is None:
        model_name = settings.MODEL_NAME
        print(f"Loading primary model: {model_name}")
        load_kwargs = {
            "trust_remote_code": settings.TRUST_REMOTE_CODE,
            "low_cpu_mem_usage": getattr(settings, "LOW_CPU_MEM_USAGE", True),
        }
        dtype = settings.TORCH_DTYPE
        if dtype != "auto":
            import torch as _torch
            load_kwargs["torch_dtype"] = getattr(_torch, dtype)
        if settings.USE_4BIT:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                print("4-bit quantization enabled (simplified config).")
            except Exception as qe:
                print(f"Could not enable 4-bit quantization: {qe}")
                # Continue without quantization
                pass
        try:
            global _raw_model, _tokenizer
            print("Loading tokenizer...")
            _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=settings.TRUST_REMOTE_CODE)
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
            print("Tokenizer loaded successfully.")
            
            print("Loading model with quantization...")
            print(f"Load kwargs: {load_kwargs}")
            _raw_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if settings.USE_GPU and torch.cuda.is_available() else None,
                **load_kwargs,
            )
            # Don't create pipeline - use direct generation only
            _pipeline = "direct"  # Flag to indicate direct mode
            print("Model loaded in direct generation mode.")
        except Exception as e:
            print(f"Model load failed with error: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            
            # Try fallback model if available
            if hasattr(settings, 'FALLBACK_MODEL') and settings.FALLBACK_MODEL:
                print(f"Attempting fallback model: {settings.FALLBACK_MODEL}")
                try:
                    _tokenizer = AutoTokenizer.from_pretrained(settings.FALLBACK_MODEL)
                    if _tokenizer.pad_token is None:
                        _tokenizer.pad_token = _tokenizer.eos_token
                    _raw_model = AutoModelForCausalLM.from_pretrained(
                        settings.FALLBACK_MODEL,
                        device_map="auto" if settings.USE_GPU and torch.cuda.is_available() else None,
                        torch_dtype=torch.float16 if settings.USE_GPU else torch.float32,
                        low_cpu_mem_usage=True,
                    )
                    _pipeline = "fallback"  # Flag to indicate fallback mode
                    print("Fallback model loaded successfully.")
                except Exception as fe:
                    print(f"Fallback model also failed: {fe}")
                    _pipeline = None
            else:
                _pipeline = None
    
    return _pipeline

def get_layman_summary(text: str) -> Dict[str, Any]:
    """
    Uses the openai/gpt-oss-20b model to generate a structured, 
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

    try:
        # Generate response using plain string input (text-generation pipeline)
        # Safety: ensure we don't request more tokens than remaining context budget.
        # Very rough token estimate: assume ~4 chars per token (English average).
        est_prompt_tokens = max(1, len(user_prompt) // 4)
        remaining_budget = max(settings.CONTEXT_WINDOW - est_prompt_tokens, 32)
        max_new = min(settings.MAX_NEW_TOKENS, remaining_budget)

        if pipe is None:
            raise RuntimeError("Model pipeline not available; model failed to load.")

        generated_text = None
        # Use direct generation only (no pipeline to avoid MoE issues)
        if _raw_model is not None and _tokenizer is not None:
            # Adjust prompt for fallback model if needed
            if _pipeline == "fallback":
                # Enhanced prompt for legal document analysis with fallback model
                simplified_prompt = f"""Analyze this legal document and provide a summary:

Document:
{text}

Please provide:
1. Document type (Terms of Service, Privacy Policy, or EULA)
2. Main purpose and scope
3. Key user obligations
4. Important terms and definitions
5. Notable restrictions or limitations

Summary:"""
                prompt_to_use = simplified_prompt
            else:
                prompt_to_use = user_prompt
                
            inputs = _tokenizer(prompt_to_use, return_tensors="pt", padding=True, truncation=True)
            if settings.USE_GPU and torch.cuda.is_available():
                inputs = {k: v.to(_raw_model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                output_ids = _raw_model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=0.7 if _pipeline == "fallback" else 0.2,
                    pad_token_id=_tokenizer.pad_token_id,
                    eos_token_id=_tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )
            gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            generated_text = _tokenizer.decode(gen_ids, skip_special_tokens=True)
        else:
            raise RuntimeError("Raw model/tokenizer not available")

        # If the original prompt is echoed back, strip everything up to the JSON Response: marker
        if "JSON Response:" in generated_text:
            generated_text = generated_text.split("JSON Response:", 1)[1].strip()
        
        # Try to extract JSON from the response
        # Sometimes models add extra text, so we'll try to find the JSON part
        generated_text = generated_text.strip()
        
        # Handle fallback model responses that may not follow JSON format perfectly
        if _pipeline == "fallback":
            # Parse the generated text to extract legal document insights
            lines = generated_text.strip().split('\n')
            doc_type = "Legal Document"
            summary_text = generated_text
            key_terms = []
            sections = []
            
            # Try to extract document type
            for line in lines:
                if any(term in line.lower() for term in ['terms of service', 'privacy policy', 'eula', 'end user license']):
                    if 'terms of service' in line.lower():
                        doc_type = "Terms of Service"
                    elif 'privacy policy' in line.lower():
                        doc_type = "Privacy Policy"
                    elif 'eula' in line.lower() or 'end user license' in line.lower():
                        doc_type = "End User License Agreement"
                    break
            
            # Extract key terms from common legal phrases in the original text
            legal_keywords = {
                'data collection': 'Information gathered about users',
                'third parties': 'External companies or services',
                'cookies': 'Small files stored on your device for tracking',
                'termination': 'Ending of service or agreement',
                'liability': 'Legal responsibility for damages',
                'intellectual property': 'Copyrights, trademarks, and patents',
                'user content': 'Content created or uploaded by users',
                'prohibited conduct': 'Actions that are not allowed',
                'privacy settings': 'Controls for personal information sharing',
                'data retention': 'How long personal information is kept'
            }
            
            text_lower = text.lower()
            for term, definition in legal_keywords.items():
                if term in text_lower:
                    key_terms.append({"term": term.title(), "definition": definition})
            
            # Create sections based on content analysis
            if len(generated_text) > 200:
                mid_point = len(generated_text) // 2
                sections = [
                    {
                        "section_title": "Primary Obligations and Rights", 
                        "detailed_summary": generated_text[:mid_point].strip()
                    },
                    {
                        "section_title": "Additional Terms and Conditions", 
                        "detailed_summary": generated_text[mid_point:].strip()
                    }
                ]
            else:
                sections = [
                    {
                        "section_title": "Document Summary", 
                        "detailed_summary": generated_text
                    }
                ]
            
            return {
                "document_type": doc_type,
                "overall_summary": f"This {doc_type.lower()} outlines the key terms and conditions. {summary_text[:300]}{'...' if len(summary_text) > 300 else ''}",
                "key_terms": key_terms if key_terms else [
                    {"term": "Legal Agreement", "definition": "A binding contract between the service provider and user"}
                ],
                "sectional_summaries": sections
            }
        
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
        raise
