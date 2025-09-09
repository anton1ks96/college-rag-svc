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
    k_top: int = 6  # Финальное количество чанков после reranking
    k_retrieval: int = 20  # Количество чанков для initial retrieval (больше чем k_top)
    min_score: float = 0.0
    max_ctx_chars: int = 8000

    # reranking параметры
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_use_fp16: bool = True  # Использовать FP16 для экономии памяти
    reranker_device: str | None = None  # None = auto-detect, "cuda", "cpu"
    reranker_batch_size: int = 32  # Размер батча для обработки
    reranker_top_k: int = 6  # Сколько чанков оставить после reranking
    reranker_min_score: float = 0.01  # Минимальный score для фильтрации
    reranker_normalize: bool = True  # Нормализовать scores в [0, 1]

    # Экспериментальные параметры для fine-tuning
    reranker_score_weight: float = 0.7  # Вес reranker score vs original score
    reranker_use_hybrid: bool = False  # Использовать гибридный scoring

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
