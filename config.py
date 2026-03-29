"""
SIGMA Configuration.
All environment variables and constants.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys (set via environment variables)
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""  # for embeddings
    GROQ_API_KEY: str = ""

    # LLM Provider: "groq" or "anthropic"
    LLM_PROVIDER: str = "groq"

    # Models
    REASONING_MODEL: str = "claude-sonnet-4-6"
    FAST_MODEL: str = "claude-haiku-4-5-20251001"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Groq Models
    GROQ_REASONING_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_FAST_MODEL: str = "llama-3.1-8b-instant"

    # RAG - ChromaDB Cloud (set via environment variables)
    CHROMA_API_KEY: str = ""
    CHROMA_TENANT: str = ""
    CHROMA_DATABASE: str = "SIGMA"
    CHROMA_PERSIST_DIR: str = "./chroma_db"  # fallback for local
    USE_CHROMA_CLOUD: bool = True
    RAG_TOP_K: int = 5

    # Pipeline
    BULK_DEAL_POLL_INTERVAL_SEC: int = 300
    NEWS_POLL_INTERVAL_SEC: int = 60

    # Thresholds
    HIGH_CONFIDENCE_THRESHOLD: float = 0.65
    LOW_CONFIDENCE_THRESHOLD: float = 0.40
    BREAKOUT_VOLUME_RATIO_MIN: float = 1.5
    RSI_OVERBOUGHT: float = 70.0
    RSI_OVERSOLD: float = 30.0
    MAX_SINGLE_STOCK_WEIGHT_PCT: float = 10.0
    MAX_SECTOR_WEIGHT_PCT: float = 30.0

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
