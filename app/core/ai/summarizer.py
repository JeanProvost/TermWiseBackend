import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any
from ...config import settings
# from transformers import BitsAndBytesConfig  # Commented out for GPT model

# Initialize global model/tokenizer (no fallback variants)
_pipeline = None  # flag string 'direct' once loaded
_raw_model = None
_tokenizer = None
_last_load_error: str | None = None

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
            "low_cpu_mem_usage": getattr(settings, "LOW_CPU_MEM_USAGE", False),
        }
        dtype = settings.TORCH_DTYPE
        if dtype != "auto":
            import torch as _torch
            load_kwargs["torch_dtype"] = getattr(_torch, dtype)
        if settings.USE_4BIT:
            import platform as _platform
            if not (settings.USE_GPU and torch.cuda.is_available()) or _platform.system() == 'Windows':
                # Skip 4-bit on unsupported environments (e.g., Windows w/out proper bitsandbytes build)
                print("Skipping 4-bit quantization (unsupported environment: Windows or no CUDA GPU).")
            else:
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16 if settings.TORCH_DTYPE == "bfloat16" else torch.float16,
                    )
                    print("4-bit quantization enabled.")
                except Exception as qe:
                    print(f"Could not enable 4-bit quantization: {qe}")
                    # Continue without quantization
                    pass
        try:
            global _raw_model, _tokenizer
            print("Loading tokenizer (fast)...")
            try:
                # Pre-check sentencepiece presence (helps Windows confusion)
                try:
                    import sentencepiece  # noqa: F401
                except Exception as sp_e:
                    print("sentencepiece import pre-check failed (will still attempt load):", sp_e)
                _tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=settings.TRUST_REMOTE_CODE,
                    use_fast=True,
                )
            except Exception as te:
                print(f"Fast tokenizer failed: {te}\nRetrying with slow tokenizer (requires sentencepiece)...")
                try:
                    _tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        trust_remote_code=settings.TRUST_REMOTE_CODE,
                        use_fast=False,
                    )
                except Exception as ste:
                    print("Slow tokenizer also failed.")
                    raise ste
            # Ensure pad token
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
            print("Tokenizer loaded.")

            print("Loading model (this can take a while)...")
            _raw_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if settings.USE_GPU and torch.cuda.is_available() else None,
                **load_kwargs,
            )
            _pipeline = "direct"
            device = str(getattr(_raw_model, 'device', 'cpu'))
            print(f"Model loaded on {device}.")
        except Exception as e:
            import traceback
            print("FAILED to load mistral model:", e)
            if 'SentencePiece' in str(e) or 'sentencepiece' in str(e):
                print("Hint: Ensure sentencepiece is installed in the SAME virtual environment used to run the app. If just installed, restart the server.")
            print(traceback.format_exc())
            global _last_load_error
            _last_load_error = f"Tokenizer/model load failure: {e}"
            _pipeline = None
    
    return _pipeline

def get_layman_summary(text: str) -> Dict[str, Any]:
    """Generate a structured legal-document summary using the single Mistral instruct model.

    Returns JSON with keys: document_type, overall_summary, key_terms, sectional_summaries.
    """
    pipe = _get_pipeline()
    
    # Truncate text if it's too long to avoid exceeding context window
    if len(text) > settings.MAX_INPUT_LENGTH:
        text = text[:settings.MAX_INPUT_LENGTH] + "..."
    
    # Chat-style messages to leverage instruct fine-tuning
    system_msg = (
        "You are a legal document summarization engine. Return ONLY valid JSON matching the schema: "
        "{ 'document_type': str, 'overall_summary': str, 'key_terms': [ {'term': str, 'definition': str} ], "
        "'sectional_summaries': [ {'section_title': str, 'detailed_summary': str} ] }." \
        " Classify document_type as one of: Terms of Service, Privacy Policy, EULA, or Other."
    )
    user_msg = (
        "Summarize the following document into the JSON schema. Keep key_terms concise and 5-12 entries if possible.\n\n" + text
    )

    # Use tokenizer chat template if available; else manual format
    try:
        user_prompt = _tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        user_prompt = system_msg + "\n\nUSER:\n" + user_msg + "\n\nJSON Response:"  # fallback string

    try:
        # Generate response using plain string input (text-generation pipeline)
        # Safety: ensure we don't request more tokens than remaining context budget.
        # Very rough token estimate: assume ~4 chars per token (English average).
        est_prompt_tokens = max(1, len(user_prompt) // 4)
        remaining_budget = max(settings.CONTEXT_WINDOW - est_prompt_tokens, 32)
        max_new = min(settings.MAX_NEW_TOKENS, remaining_budget)

        if pipe is None:
            raise RuntimeError(f"Model pipeline not available; model failed to load. Details: {_last_load_error}")

        generated_text = None
        # Use direct generation only (no pipeline to avoid MoE issues)
        if _raw_model is not None and _tokenizer is not None:
            inputs = _tokenizer(user_prompt, return_tensors="pt", padding=True, truncation=True, max_length=settings.MAX_INPUT_LENGTH)
            if settings.USE_GPU and torch.cuda.is_available():
                inputs = {k: v.to(_raw_model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                output_ids = _raw_model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=0.2,
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
        
    # (No fallback path; single model only)
        
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
