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
    MODEL_NAME: str = "microsoft/Phi-3-mini-4k-instruct"
    USE_GPU: bool = True
    MAX_INPUT_LENGTH: int = 3000
    MAX_NEW_TOKENS: int = 2048

    #Performance configuration
    QUANTIZE_MODEL: bool = True  # 4-bit quantization for RTX 4060 8GB
    USE_FLASH_ATTENTION_2: bool = True
    USE_MISTRAL_NATIVE: bool = False  # Use HF transformers instead of mistral_inference

settings = Settings()
