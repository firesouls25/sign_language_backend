from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
import os


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Allow extra fields in .env
    )

    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/lsc_db"
    REDIS_URL: str = "redis://localhost:6379"

    SECRET_KEY: str = "change-this-to-a-random-secret-key-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    GOOGLE_APPLICATION_CREDENTIALS: str = ""
    GCS_BUCKET_NAME: str = "lsc-videos-bucket"
    UPLOAD_DIR: str = "uploads"

    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    APPLE_CLIENT_ID: str = ""
    APPLE_CLIENT_SECRET: str = ""
    APPLE_TEAM_ID: str = ""
    APPLE_KEY_ID: str = ""
    APPLE_PRIVATE_KEY: str = ""

    # Groq / LiteLLM
    GROQ_API_KEY: str = ""
    LITELLM_MODEL: str = "groq/llama-3.1-8b-instant"

    # Kaggle
    KAGGLE_USERNAME: str = ""
    KAGGLE_API_TOKEN: str = ""

    FRONTEND_URL: str = "http://localhost:5173"
    BACKEND_URL: str = "http://localhost:8000"

    APP_ENV: str = "development"
    DEBUG: bool = True
    ENABLE_DEV_ROUTES: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
