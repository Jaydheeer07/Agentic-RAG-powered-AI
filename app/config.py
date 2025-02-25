from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # PostgreSQL settings
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int

    # Database URL
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # pgAdmin settings
    PGADMIN_EMAIL: str
    PGADMIN_PASSWORD: str

    # Application settings
    MAX_CONCURRENT_CRAWLS: int
    OUTPUT_DIR: str

    # OpenAI settings
    OPENAI_API_KEY: str
    OPENAI_ORGANIZATION: Optional[str] = None

    # Embedding settings
    EMBEDDING_MODEL: str
    EMBEDDING_DIMENSION: int
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int

    # Chat model
    CHAT_MODEL: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


# Create a global settings instance
settings = Settings()
