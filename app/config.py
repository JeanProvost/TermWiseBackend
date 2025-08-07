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
    MODEL_NAME: str = "openai/gpt-oss-120b"
    USE_GPU: bool = True
    MAX_INPUT_LENGTH: int = 3000
    MAX_NEW_TOKENS: int = 256  # Updated to match example

    #Performance configuration
    # QUANTIZE_MODEL: bool = True  # Commented out for GPT model
    # USE_FLASH_ATTENTION_2: bool = True  # Commented out for GPT model
    # USE_MISTRAL_NATIVE: bool = False  # Commented out, not needed for GPT

settings = Settings()
