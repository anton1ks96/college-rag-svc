from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # service
    rag_service_token: str = "dev-secret-token"
    port: int = 8001

    # qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "dataset_chunks"
    qdrant_distance: str = "COSINE"  # TODO: поддержка COSINE / DOT

    # rag params
    chunk_size: int = 1200
    chunk_overlap: int = 100
    k_top: int = 6
    min_score: float = 0.0
    max_ctx_chars: int = 8000

    # embeddings
    embedding_provider: str = "sbert"   # openai | sbert
    embedding_model: str = "BAAI/bge-m3"

    # llm
    llm_provider: str = "ollama"
    ollama_model: str = "gemma3:1b"
    openai_api_key: str | None = None
    openai_model: str = "gemma3:1b"
    openai_temperature: float = 0.2
    openai_max_tokens: int = 800

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()
