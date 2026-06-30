"""Runtime configuration for the agentx library (pydantic-settings).

Reads from environment / a local ``.env``. Only generic, cross-provider knobs
live here; provider credentials are read by each provider's SDK from their own
standard env vars (see ``agentx.providers.registry``).
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Default provider/model used when none is passed explicitly.
    default_provider: str = Field(default="openai", alias="AGENTX_PROVIDER")
    default_model: str = Field(default="", alias="AGENTX_MODEL")
    temperature: float = Field(default=0.3, alias="AGENTX_TEMPERATURE")
    max_tokens: int | None = Field(default=None, alias="AGENTX_MAX_TOKENS")
    request_timeout: int = Field(default=120, alias="AGENTX_REQUEST_TIMEOUT")

    # Local backends.
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
