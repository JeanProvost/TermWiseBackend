from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Manages application settings using Pydantic, loading from environment variables.
    """
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding='utf-8', 
        extra='ignore'
    )

    AWS_REGION: str = "us-east-1"
    
    # Model configuration
    MODEL_NAME: str = "microsoft/DialoGPT-large"  # More reliable alternative to gpt-oss-20b
    PREMIUM_MODEL: str = "openai/gpt-oss-20b"  # Original model to try if you have enough resources
    FALLBACK_MODEL: str = "gpt2-medium"  # Lightweight fallback
    USE_GPU: bool = True
    MAX_INPUT_LENGTH: int = 3000
    MAX_NEW_TOKENS: int = 512  # Increased for better legal summaries
    CONTEXT_WINDOW: int = 8192  # Approximate context window for safety budgeting
    # Model loading options
    TRUST_REMOTE_CODE: bool = True
    TORCH_DTYPE: str = "float16"  # Explicit dtype to avoid auto issues
    USE_4BIT: bool = False  # Disable 4-bit quantization to avoid compatibility issues
    LOW_CPU_MEM_USAGE: bool = True  # Reduce memory usage during loading

    #Performance configuration
    # QUANTIZE_MODEL: bool = True  # Commented out for GPT model
    # USE_FLASH_ATTENTION_2: bool = True  # Commented out for GPT model
    # USE_MISTRAL_NATIVE: bool = False  # Commented out, not needed for GPT

settings = Settings()
