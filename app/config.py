from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "rag_career_agent"

    daily_embedding_budget_usd: float = 0.50

    admin_token_ttl_minutes: int = 10
    admin_session_ttl_minutes: int = 120
