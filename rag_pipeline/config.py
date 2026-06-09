from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    cohere_api_key: str = Field(..., alias="COHERE_API_KEY")
    cohere_embed_model: str = Field(
        default="embed-english-v3.0",
        alias="COHERE_EMBED_MODEL",
    )
    database_url: str = Field(
        default="postgresql://rag:rag@localhost:5432/rag_db",
        alias="DATABASE_URL",
    )
    semantic_chunk_breakpoint_percentile: float = Field(
        default=90.0,
        alias="SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE",
        ge=0.0,
        le=100.0,
    )
    semantic_chunk_max_chars: int = Field(
        default=2000,
        alias="SEMANTIC_CHUNK_MAX_CHARS",
        gt=0,
    )
    embed_batch_size: int = Field(default=96, alias="EMBED_BATCH_SIZE", gt=0)

    @property
    def embedding_dimensions(self) -> int:
        """Return vector dimensions for the configured Cohere embed model."""
        model_dims = {
            "embed-english-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-v3.0": 1024,
            "embed-multilingual-light-v3.0": 384,
        }
        return model_dims.get(self.cohere_embed_model, 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
