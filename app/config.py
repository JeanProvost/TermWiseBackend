import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Manages application settings using Pydantic, loading from environment variables.
    """
    model_config = SettingsConfigDict(
        env_file_encoding="utf-8", 
        extra="ignore"
    )

    # General environment
    APP_ENV: Literal["local", "staging", "production"] = "local"
    MODEL_PROVIDER: Literal["huggingface", "bedrock"] = "huggingface"

    # Shared runtime settings
    AWS_REGION: str = "us-east-1"
    USE_GPU: bool = True
    MAX_INPUT_LENGTH: int = 2048
    MAX_NEW_TOKENS: int = 512
    CONTEXT_WINDOW: int = 4096

    # Hugging Face / local transformers configuration (default path)
    MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
    TRUST_REMOTE_CODE: bool = True
    TORCH_DTYPE: str = "bfloat16"
    USE_4BIT: bool = True
    LOW_CPU_MEM_USAGE: bool = True
    HF_TOKEN: str | None = None

    # AWS Bedrock configuration (used when MODEL_PROVIDER == "bedrock")
    BEDROCK_REGION: str | None = None
    BEDROCK_MODEL_ID: str | None = None
    BEDROCK_ENDPOINT_URL: str | None = None
    BEDROCK_PROFILE: str | None = None
    BEDROCK_ASSUME_ROLE_ARN: str | None = None


def _resolve_env_files() -> list[Path]:
    base_files: list[Path] = []
    default_env = Path(".env")
    if default_env.exists():
        base_files.append(default_env)

    current_env = os.getenv("APP_ENV", os.getenv("TERMWS_ENV", "local"))
    env_override = Path(f".env.{current_env}")
    if env_override.exists():
        base_files.append(env_override)

    return base_files


settings = Settings(_env_file=_resolve_env_files())
