"""
Модуль для переранжирования результатов поиска с использованием BGE-reranker.
Улучшает релевантность найденных чанков перед передачей в LLM.
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from config import settings

logger = logging.getLogger(__name__)

try:
    from FlagEmbedding import FlagReranker
except ImportError as e:
    FlagReranker = None
    _import_error = e
else:
    _import_error = None

_reranker_model = None


@dataclass
class RerankResult:
    """Результат переранжирования"""
    chunk_id: int
    text: str
    original_score: float
    rerank_score: float
    metadata: Dict[str, Any]


def _load_reranker_model():
    """
    Загружает и кеширует модель reranker.
    Использует ленивую загрузку для экономии памяти.
    """
    global _reranker_model

    if _reranker_model is not None:
        return _reranker_model

    if FlagReranker is None:
        raise ImportError(
            "FlagEmbedding is required for reranking. "
            "Install with: pip install -U FlagEmbedding\n"
            f"Original error: {_import_error}"
        )

    model_name = getattr(settings, "reranker_model", "BAAI/bge-reranker-v2-m3")
    use_fp16 = getattr(settings, "reranker_use_fp16", True)
    device = getattr(settings, "reranker_device", None)

    logger.info(f"Loading reranker model: {model_name}")

    try:
        _reranker_model = FlagReranker(
            model_name,
            use_fp16=use_fp16,
            device=device
        )
        logger.info("Reranker model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load reranker model: {e}")
        raise

    return _reranker_model


def rerank_chunks(
        question: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        batch_size: Optional[int] = None,
        normalize_scores: bool = True
) -> List[RerankResult]:
    """
    Переранжирует чанки на основе их релевантности к вопросу.

    Args:
        question: Вопрос пользователя
        chunks: Список чанков из векторного поиска
        top_k: Количество лучших результатов для возврата
        min_score: Минимальный score для фильтрации
        batch_size: Размер батча для обработки
        normalize_scores: Нормализовать scores в диапазон [0, 1]

    Returns:
        Список переранжированных результатов
    """
    if not chunks:
        return []

    top_k = top_k or getattr(settings, "reranker_top_k", 10)
    min_score = min_score if min_score is not None else getattr(settings, "reranker_min_score", 0.01)
    batch_size = batch_size or getattr(settings, "reranker_batch_size", 32)

    reranker = _load_reranker_model()

    pairs = []
    chunk_data = []

    for chunk in chunks:
        if isinstance(chunk, tuple):
            score, payload = chunk
            text = payload.get("text", "")
            chunk_id = payload.get("chunk_id", -1)
            metadata = payload
        else:
            text = chunk.get("text", "")
            score = chunk.get("score", 0.0)
            chunk_id = chunk.get("chunk_id", -1)
            metadata = chunk.get("metadata", {})

        pairs.append([question, text])
        chunk_data.append({
            "chunk_id": chunk_id,
            "text": text,
            "original_score": score,
            "metadata": metadata
        })

    if not pairs:
        logger.warning("No valid pairs for reranking")
        return []

    try:
        if len(pairs) <= batch_size:
            scores = reranker.compute_score(
                pairs,
                batch_size=batch_size,
                normalize=normalize_scores
            )
        else:
            scores = []
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_scores = reranker.compute_score(
                    batch_pairs,
                    batch_size=batch_size,
                    normalize=normalize_scores
                )
                if isinstance(batch_scores, float):
                    scores.append(batch_scores)
                else:
                    scores.extend(batch_scores)
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        scores = [chunk["original_score"] for chunk in chunk_data]

    if isinstance(scores, float):
        scores = [scores]

    results = []
    for score, chunk_info in zip(scores, chunk_data):
        if score >= min_score:
            results.append(RerankResult(
                chunk_id=chunk_info["chunk_id"],
                text=chunk_info["text"],
                original_score=chunk_info["original_score"],
                rerank_score=float(score),
                metadata=chunk_info["metadata"]
            ))

    results.sort(key=lambda x: x.rerank_score, reverse=True)

    return results[:top_k]


def rerank_with_fallback(
        question: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        enable_reranking: Optional[bool] = None
) -> List[Dict[str, Any]]:
    """
    Обертка для reranking с fallback на оригинальные результаты.
    Удобна для постепенной миграции.

    Args:
        question: Вопрос пользователя
        chunks: Чанки из векторного поиска
        top_k: Количество результатов
        enable_reranking: Включить/выключить reranking

    Returns:
        Список чанков (переранжированных или оригинальных)
    """
    if enable_reranking is None:
        enable_reranking = getattr(settings, "reranker_enabled", False)

    if not enable_reranking:
        return chunks[:top_k] if top_k else chunks

    try:
        reranked = rerank_chunks(question, chunks, top_k=top_k)

        result_chunks = []
        for item in reranked:
            result_chunks.append({
                "chunk_id": item.chunk_id,
                "text": item.text,
                "score": item.rerank_score,
                "original_score": item.original_score,
                "metadata": item.metadata
            })

        logger.info(f"Reranking successful: {len(chunks)} -> {len(result_chunks)} chunks")
        return result_chunks

    except Exception as e:
        logger.error(f"Reranking failed, using fallback: {e}")
        return chunks[:top_k] if top_k else chunks


def get_reranker_stats() -> Dict[str, Any]:
    """
    Возвращает статистику использования reranker.
    """
    return {
        "model_loaded": _reranker_model is not None,
        "model_name": getattr(settings, "reranker_model", "BAAI/bge-reranker-v2-m3"),
        "enabled": getattr(settings, "reranker_enabled", False),
        "top_k": getattr(settings, "reranker_top_k", 10),
        "min_score": getattr(settings, "reranker_min_score", 0.01)
    }


def unload_reranker_model():
    """
    Выгружает модель из памяти для освобождения ресурсов.
    """
    global _reranker_model
    if _reranker_model is not None:
        del _reranker_model
        _reranker_model = None
        logger.info("Reranker model unloaded")


# Дополнительные утилиты для анализа

def analyze_reranking_impact(
        question: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 10
) -> Dict[str, Any]:
    """
    Анализирует влияние reranking на результаты.
    Полезно для отладки и оптимизации.
    """
    if not chunks:
        return {"error": "No chunks provided"}

    # Оригинальный порядок
    original_order = [c.get("chunk_id", i) for i, c in enumerate(chunks[:top_k])]

    # Reranked порядок
    try:
        reranked = rerank_chunks(question, chunks, top_k=top_k)
        reranked_order = [r.chunk_id for r in reranked]

        # Считаем метрики
        position_changes = []
        for new_pos, chunk_id in enumerate(reranked_order):
            if chunk_id in original_order:
                old_pos = original_order.index(chunk_id)
                position_changes.append(old_pos - new_pos)

        # Score изменения
        score_improvements = []
        for r in reranked:
            improvement = r.rerank_score - r.original_score
            score_improvements.append(improvement)

        return {
            "original_order": original_order,
            "reranked_order": reranked_order,
            "avg_position_change": sum(position_changes) / len(position_changes) if position_changes else 0,
            "max_position_jump": max(position_changes) if position_changes else 0,
            "avg_score_improvement": sum(score_improvements) / len(score_improvements) if score_improvements else 0,
            "new_chunks_in_top": len(set(reranked_order) - set(original_order))
        }
    except Exception as e:
        return {"error": str(e)}