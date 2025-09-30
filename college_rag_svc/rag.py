from typing import Dict, List
import logging
from config import settings
from normalize import normalize_markdown, NormalizationConfig
from chunking import chunk_student_markdown, ChunkerConfig
from embeddings import embed_texts
from qdrant_store import ensure_collection, upsert_chunks, search
from reranking import rerank_with_fallback, analyze_reranking_impact, RerankResult
from llm import generate_answer

logger = logging.getLogger(__name__)


def index_dataset(dataset_id: str, version: int, title: str | None, text: str, overwrite: bool = True) -> int:
    """
    Индексация текста: normalize -> chunk -> embed -> upsert.
    """
    norm_config = NormalizationConfig()
    normalized_text = normalize_markdown(text, norm_config)

    chunk_config = ChunkerConfig(
        chunk_tokens=settings.chunk_size // 4,
        overlap_tokens=settings.chunk_overlap // 4,
        max_code_chunk_tokens=(settings.chunk_size // 4) + 120
    )

    documents = chunk_student_markdown(
        text_md=normalized_text,
        student_id="default",  # TODO: добавить реальный student_id
        assignment_id=dataset_id,
        version=version,
        cfg=chunk_config,
        source_name=title or "document"
    )

    chunks = []
    for idx, doc in enumerate(documents):
        chunks.append({
            "idx": idx,
            "text": doc.page_content,
            "metadata": doc.metadata
        })

    vecs = embed_texts([c["text"] for c in chunks])

    if not vecs:
        return 0

    vector_size = len(vecs[0])
    ensure_collection(vector_size)

    count = upsert_chunks(dataset_id, version, title, chunks, vecs)
    return count


def ask_dataset(
        dataset_id: str,
        version: int,
        question: str,
        k: int | None = None,
        min_score: float | None = None,
        max_ctx_chars: int | None = None,
        use_reranking: bool | None = None,
        debug_reranking: bool = False
) -> Dict:
    """
    Поиск по датасету с опциональным reranking и генерация ответа с цитатами.

    Args:
        dataset_id: ID датасета
        version: Версия датасета
        question: Вопрос пользователя
        k: Количество финальных чанков для LLM
        min_score: Минимальный score для фильтрации
        max_ctx_chars: Максимальное количество символов контекста
        use_reranking: Принудительно включить/выключить reranking
        debug_reranking: Добавить debug информацию о reranking в metrics
    """
    k_final = k or settings.k_top
    min_score = settings.min_score if min_score is None else min_score
    max_ctx_chars = max_ctx_chars or settings.max_ctx_chars

    if use_reranking is None:
        use_reranking = settings.reranker_enabled

    if use_reranking:
        k_retrieval = settings.k_retrieval or k_final * 3
    else:
        k_retrieval = k_final

    q_vec = embed_texts([question])[0] if question else []
    if not q_vec:
        return {
            "answer": "Ошибка: не удалось обработать вопрос",
            "citations": [],
            "metrics": {"error": "embedding_failed"}
        }

    hits = search(dataset_id, version, q_vec, k_retrieval)

    if not hits:
        return {
            "answer": "Не найдено релевантной информации для ответа на вопрос.",
            "citations": [],
            "metrics": {"chunks_found": 0, "reranking_used": False}
        }

    chunks_for_reranking = []
    for score, payload in hits:
        if score >= min_score:
            chunks_for_reranking.append({
                "chunk_id": payload.get("chunk_id", 0),
                "text": payload.get("text", ""),
                "score": score,
                "metadata": payload
            })

    if use_reranking and chunks_for_reranking:
        logger.info(f"Applying reranking: {len(chunks_for_reranking)} candidates -> top {k_final}")

        reranking_impact = None
        if debug_reranking:
            reranking_impact = analyze_reranking_impact(
                question,
                chunks_for_reranking,
                top_k=k_final
            )

        reranked_chunks = rerank_with_fallback(
            question=question,
            chunks=chunks_for_reranking,
            top_k=k_final,
            enable_reranking=True
        )

        contexts = [
            {
                "chunk_id": chunk.get("chunk_id", idx),
                "text": chunk["text"],
                "score": chunk.get("score", 0.0),
                "original_score": chunk.get("original_score", 0.0)
            }
            for idx, chunk in enumerate(reranked_chunks)
        ]
    else:
        contexts = [
            {
                "chunk_id": payload.get("chunk_id", idx),
                "text": payload["text"],
                "score": score,
                "original_score": score
            }
            for idx, (score, payload) in enumerate(hits[:k_final])
            if score >= min_score
        ]
        reranking_impact = None

    system_prompt = (
        "Ты ассистент для ответов на вопросы на основе предоставленного контекста.\n\n"
        "КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:\n"
        "1. Отвечай СТРОГО на основе информации из тегов <context>. "
        "ЗАПРЕЩЕНО использовать твои внутренние знания или предположения.\n"
        "2. Если в контексте НЕТ информации для ответа на вопрос - честно скажи: "
        "\"В предоставленных материалах недостаточно информации для ответа на этот вопрос.\"\n"
        "3. Если контекст содержит противоречивую, неполную или неясную информацию - "
        "укажи на это явно, не придумывай недостающие детали.\n"
        "4. Цитируй факты из контекста напрямую, не перефразируй если не уверен.\n"
        "5. Если вопрос частично покрыт контекстом - ответь только на ту часть, которая есть, "
        "и укажи что остальное отсутствует.\n\n"
        "НИКОГДА не додумывай, не предполагай, не обобщай за пределами данного контекста."
    )

    if contexts:
        answer = generate_answer(question, contexts, system_prompt=system_prompt)
    else:
        answer = "Не найдено релевантной информации для ответа на вопрос."

    metrics = {
        "chunks_found": len(hits),
        "chunks_after_filtering": len(chunks_for_reranking) if use_reranking else len(contexts),
        "chunks_after_reranking": len(contexts),
        "chunks_used": len(contexts),
        "reranking_used": use_reranking,
        "top_scores": [round(ctx["score"], 6) for ctx in contexts[:5]],
        "insufficient_data": "недостаточно данных" in answer.lower()
    }

    if debug_reranking and reranking_impact:
        metrics["reranking_impact"] = reranking_impact

    citations = []
    for c in contexts:
        citation = {
            "chunk_id": c["chunk_id"],
            "score": round(c["score"], 6)
        }
        if "original_score" in c and c["original_score"] != c["score"]:
            citation["original_score"] = round(c["original_score"], 6)
            citation["score_improvement"] = round(c["score"] - c["original_score"], 6)
        citations.append(citation)

    return {
        "answer": answer,
        "citations": citations,
        "metrics": metrics
    }


def hybrid_ask_dataset(
        dataset_id: str,
        version: int,
        question: str,
        **kwargs
) -> Dict:
    """
    Экспериментальный метод с гибридным scoring (vector + reranker).
    Комбинирует оба score для лучшего результата.
    """
    weight = settings.reranker_score_weight
    k_final = kwargs.get("k", settings.k_top)

    result = ask_dataset(
        dataset_id=dataset_id,
        version=version,
        question=question,
        use_reranking=True,
        debug_reranking=True,
        **kwargs
    )

    if "reranking_impact" in result.get("metrics", {}):
        citations = result["citations"]
        for citation in citations:
            if "original_score" in citation:
                hybrid_score = (
                        weight * citation["score"] +
                        (1 - weight) * citation["original_score"]
                )
                citation["hybrid_score"] = round(hybrid_score, 6)

        citations.sort(key=lambda x: x.get("hybrid_score", x["score"]), reverse=True)

        result["citations"] = citations[:k_final]
        result["metrics"]["scoring_method"] = "hybrid"
        result["metrics"]["reranker_weight"] = weight

    return result