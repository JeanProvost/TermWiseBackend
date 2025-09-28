from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any
from ...config import settings
import os
import json
import torch
import traceback
import platform as _platform
import boto3
from botocore.exceptions import BotoCoreError, ClientError

# Initialize global model/tokenizer (no fallback variants)
_pipeline = None  # 'direct' for local transformers, 'bedrock' when using AWS
_raw_model = None
_tokenizer = None
_last_load_error: str | None = None
_bedrock_client = None


def _get_pipeline():
    """
    Lazy-load the pipeline to avoid loading it at import time.
    This ensures it's only loaded when first needed.
    """
    global _pipeline

    if _pipeline is None:
        if settings.MODEL_PROVIDER == "bedrock":
            # No local model to load; defer to Bedrock invocation path
            _pipeline = "bedrock"
            return _pipeline

        model_name = settings.MODEL_NAME
        hf_token = getattr(settings, "HF_TOKEN", None)
        print(f"Loading primary model: {model_name}")
        load_kwargs = {
            "trust_remote_code": settings.TRUST_REMOTE_CODE,
            "low_cpu_mem_usage": getattr(settings, "LOW_CPU_MEM_USAGE", False),
            # gpt-oss needs eager attention (no SDPA support)
            "attn_implementation": "sdpa",
        }
        attn_impl = "sdpa"
        if "gpt-oss" in model_name.lower() or "gpt_oss" in model_name.lower():
            attn_impl = "eager"
        load_kwargs = {
            "trust_remote_code": settings.TRUST_REMOTE_CODE,
            "low_cpu_mem_usage": getattr(settings, "LOW_CPU_MEM_USAGE", False),
            "attn_implementation": attn_impl,
        }
        # dtype handling
        dtype = (settings.TORCH_DTYPE or "auto").lower()
        if dtype != "auto":
            if dtype in ("bfloat16", "bf16"):
                load_kwargs["dtype"] = torch.bfloat16
            elif dtype in ("float16", "fp16", "half"):
                load_kwargs["dtype"] = torch.float16
            elif dtype in ("float32", "fp32"):
                load_kwargs["dtype"] = torch.float32

        # device and offload
        offload_dir = os.path.join(os.getcwd(), ".hf_offload")
        os.makedirs(offload_dir, exist_ok=True)
        use_gpu = settings.USE_GPU and torch.cuda.is_available()
        # MoE models are more stable with 'balanced_low_0' on consumer GPUs
        device_map = "auto" if use_gpu else "cpu"
        load_kwargs["device_map"] = device_map
        load_kwargs["offload_folder"] = offload_dir
        if hf_token:
            load_kwargs["token"] = hf_token
        try:
            global _raw_model, _tokenizer
            print("Loading tokenizer (fast)...")
            try:
                try:
                    import sentencepiece  # noqa: F401
                except Exception as sp_e:
                    print(
                        "sentencepiece import pre-check failed (will still attempt load):",
                        sp_e,
                    )
                tok_kwargs = {
                    "trust_remote_code": settings.TRUST_REMOTE_CODE,
                    "use_fast": True,
                }
                if hf_token:
                    tok_kwargs["token"] = hf_token
                _tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
            except Exception as te:
                print(
                    f"Fast tokenizer failed: {te}\nRetrying with slow tokenizer (requires sentencepiece)..."
                )
                tok_kwargs = {
                    "trust_remote_code": settings.TRUST_REMOTE_CODE,
                    "use_fast": False,
                }
                if hf_token:
                    tok_kwargs["token"] = hf_token
                _tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
            # Ensure pad token
            if _tokenizer.pad_token is None:
                # Avoid adding new tokens (no resize); reuse eos
                if _tokenizer.eos_token is not None:
                    _tokenizer.pad_token = _tokenizer.eos_token
                    _tokenizer.pad_token_id = _tokenizer.eos_token_id
            _tokenizer.padding_side = "left"
            print("Tokenizer loaded.")

            print("Loading model (this can take a while)...")
            _raw_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            _raw_model.eval()
            # Ensure model has pad/eos ids
            if getattr(_raw_model.config, "pad_token_id", None) is None:
                _raw_model.config.pad_token_id = _tokenizer.pad_token_id
            if getattr(_raw_model.generation_config, "pad_token_id", None) is None:
                _raw_model.generation_config.pad_token_id = _tokenizer.pad_token_id
            if getattr(_raw_model.generation_config, "eos_token_id", None) is None and _tokenizer.eos_token_id is not None:
                _raw_model.generation_config.eos_token_id = _tokenizer.eos_token_id
            _pipeline = "direct"
            device = str(getattr(_raw_model, "device", "cpu"))
            print(f"Model loaded on {device}.")
        except Exception as e:
            print("FAILED to load model:", e)
            print(traceback.format_exc())
            global _last_load_error
            _last_load_error = f"Tokenizer/model load failure: {e}"
            _pipeline = None

    return _pipeline


def get_layman_summary(text: str) -> Dict[str, Any]:
    """Generate a structured legal-document summary using the configured model.

    Returns JSON with keys: document_type, overall_summary, key_terms, sectional_summaries.
    """
    pipe = _get_pipeline()

    # Truncate text if it's too long to avoid exceeding context window
    if len(text) > settings.MAX_INPUT_LENGTH:
        text = text[: settings.MAX_INPUT_LENGTH] + "..."

    # Chat-style messages
    system_msg = (
        "You are a legal document summarization engine. Return ONLY valid JSON matching the schema: "
        "{ 'document_type': str, 'overall_summary': str, 'key_terms': [ {'term': str, 'definition': str} ], "
        "'sectional_summaries': [ {'section_title': str, 'detailed_summary': str} ] }."
        " Classify document_type as one of: Terms of Service, Privacy Policy, EULA, or Other."
    )
    user_msg = (
        "Summarize the following document into the JSON schema. Keep key_terms concise and 5-12 entries if possible.\n\n"
        + text
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
        user_prompt = system_msg + "\n\nUSER:\n" + user_msg + "\n\nJSON Response:"

    try:
        est_prompt_tokens = max(1, len(user_prompt) // 4)
        remaining_budget = max(settings.CONTEXT_WINDOW - est_prompt_tokens, 32)
        max_new = min(settings.MAX_NEW_TOKENS, remaining_budget)

        if pipe is None:
            raise RuntimeError(
                f"Model pipeline not available; model failed to load. Details: {_last_load_error}"
            )

        generated_text = None
        if pipe == "bedrock":
            generated_text = _invoke_bedrock(system_msg, user_msg, max_new)
        elif _raw_model is not None and _tokenizer is not None:
            inputs = _tokenizer(
                user_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=settings.MAX_INPUT_LENGTH,
            )
            target_device = "cuda" if settings.USE_GPU and torch.cuda.is_available() else "cpu"
            try:
                inputs = inputs.to(target_device)
            except AttributeError:
                inputs = {k: v.to(target_device) for k, v in inputs.items()}

            with torch.inference_mode():
                output_ids = _raw_model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,  # deterministic
                    eos_token_id=_tokenizer.eos_token_id,
                    pad_token_id=_tokenizer.pad_token_id,
                )

            gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
            generated_text = _tokenizer.decode(gen_ids, skip_special_tokens=True)
        else:
            raise RuntimeError("Raw model/tokenizer not available")

        if "JSON Response:" in generated_text:
            generated_text = generated_text.split("JSON Response:", 1)[1].strip()

        generated_text = generated_text.strip()

        if generated_text.startswith("```json"):
            generated_text = generated_text[7:]
        if generated_text.endswith("```"):
            generated_text = generated_text[:-3]

        # Parse the JSON
        try:
            structured_data = json.loads(generated_text)
        except json.JSONDecodeError:
            start = generated_text.find("{")
            end = generated_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                structured_data = json.loads(generated_text[start : end + 1])
            else:
                raise

        # Validate the structure
        required_keys = [
            "document_type",
            "overall_summary",
            "key_terms",
            "sectional_summaries",
        ]
        for key in required_keys:
            if key not in structured_data:
                raise ValueError(f"Missing key in model output: {key}")

        return structured_data

    except Exception as e:
        print(f"Error during model generation: {e}")
        print(
            f"Generated text: {generated_text if 'generated_text' in locals() else 'No output generated'}"
        )
        raise


def _get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is not None:
        return _bedrock_client

    region = settings.BEDROCK_REGION or settings.AWS_REGION
    if not region:
        raise ValueError("BEDROCK_REGION or AWS_REGION must be set for Bedrock usage")

    session_kwargs = {}
    if settings.BEDROCK_PROFILE:
        session_kwargs["profile_name"] = settings.BEDROCK_PROFILE

    session = boto3.Session(**session_kwargs)
    client = session.client(
        "bedrock-runtime",
        region_name=region,
        endpoint_url=settings.BEDROCK_ENDPOINT_URL,
    )

    _bedrock_client = client
    return _bedrock_client


def _invoke_bedrock(system_msg: str, user_msg: str, max_new_tokens: int) -> str:
    if not settings.BEDROCK_MODEL_ID:
        raise ValueError("BEDROCK_MODEL_ID must be provided when using the Bedrock provider")

    client = _get_bedrock_client()

    payload = {
        "system": system_msg,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_msg,
                    }
                ],
            }
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
    }

    try:
        response = client.invoke_model(
            modelId=settings.BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload).encode("utf-8"),
        )
    except (BotoCoreError, ClientError) as err:
        raise RuntimeError(f"Bedrock invocation failed: {err}") from err

    body = response.get("body")
    if hasattr(body, "read"):
        raw = body.read()
    else:
        raw = body

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Unable to parse Bedrock response: {exc}. Raw: {raw!r}") from exc

    # Anthropic Claude responses (Bedrock)
    if isinstance(parsed, dict):
        if "content" in parsed and isinstance(parsed["content"], list):
            text_chunks = [
                block.get("text", "")
                for block in parsed["content"]
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if text_chunks:
                return "".join(text_chunks)

        if "outputText" in parsed:
            return parsed["outputText"]

        if "generated_text" in parsed:
            return parsed["generated_text"]

    raise RuntimeError(f"Unexpected Bedrock response format: {parsed}")
