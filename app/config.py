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
    BEDROCK_MODEL_ID: str = "deepseek-llm-r1-distill-llama-70b"
    HUGGINGFACE_API_TOKEN: str = ""

settings = Settings()
