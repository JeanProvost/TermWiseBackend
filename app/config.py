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
    MODEL_NAME: str = "google/gemma-2-9b-it"
    USE_GPU: bool = True
    MAX_INPUT_LENGTH: int = 3000
    MAX_NEW_TOKENS: int = 2048

settings = Settings()
