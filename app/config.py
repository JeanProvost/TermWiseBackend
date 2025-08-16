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
    MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.3"  # Sole model used
    USE_GPU: bool = True
    MAX_INPUT_LENGTH: int = 4096  # Increased for Mistral
    MAX_NEW_TOKENS: int = 1024  # Increased for better summaries
    CONTEXT_WINDOW: int = 8192  # Approximate context window for safety budgeting
    # Model loading options
    TRUST_REMOTE_CODE: bool = True
    TORCH_DTYPE: str = "auto"  # Let transformers handle dtype
    USE_4BIT: bool = False  # Disabled on Windows / CPU for stability; re-enable later on Linux+GPU
    LOW_CPU_MEM_USAGE: bool = True  # Reduce memory usage during loading

    #Performance configuration
    # QUANTIZE_MODEL: bool = True  # Commented out for GPT model
    # USE_FLASH_ATTENTION_2: bool = True  # Commented out for GPT model
    # USE_MISTRAL_NATIVE: bool = False  # Comment out, not needed for GPT

settings = Settings()
