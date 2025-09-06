from typing import Dict, List
from config import settings
from normalize import normalize_markdown, NormalizationConfig
from chunking import chunk_student_markdown, ChunkerConfig
from embeddings import embed_texts
from qdrant_store import ensure_collection, upsert_chunks, search
from llm import generate_answer


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


def ask_dataset(dataset_id: str, version: int, question: str, k: int | None, min_score: float | None,
                max_ctx_chars: int | None) -> Dict:
    """
    Поиск по датасету и генерация ответа с цитатами.
    """
    k = k or settings.k_top
    min_score = settings.min_score if min_score is None else min_score
    max_ctx_chars = max_ctx_chars or settings.max_ctx_chars

    q_vec = embed_texts([question])[0] if question else []
    if not q_vec:
        return {"answer": "Ошибка: не удалось обработать вопрос", "citations": [], "metrics": {}}

    hits = search(dataset_id, version, q_vec, k)

    filtered = [h for h in hits if h[0] >= min_score]
    contexts: List[dict] = [
        {"chunk_id": int(p.get("chunk_id", idx)), "text": p["text"], "score": s}
        for idx, (s, p) in enumerate(filtered)
    ]

    total_chars = 0
    trimmed_contexts = []
    for ctx in contexts:
        ctx_len = len(ctx["text"])
        if total_chars + ctx_len > max_ctx_chars:
            remaining = max_ctx_chars - total_chars
            if remaining > 100:
                ctx_copy = ctx.copy()
                ctx_copy["text"] = ctx["text"][:remaining] + "..."
                trimmed_contexts.append(ctx_copy)
            break
        trimmed_contexts.append(ctx)
        total_chars += ctx_len

    system = (
        "Ты отвечаешь на вопрос пользователя **ТОЛЬКО** по данному контексту."
        "Если информации нехватает скажи что недостаточно данных"
    )

    if trimmed_contexts:
        answer = generate_answer(question, trimmed_contexts, system_prompt=system)
    else:
        answer = "Не найдено релевантной информации для ответа на вопрос."

    metrics = {
        "chunks_used": len(trimmed_contexts),
        "top_scores": [round(s, 6) for s, _ in hits[:5]] if hits else [],
        "total_chars": total_chars,
        "insufficient_data": "Недостаточно данных" in answer
    }

    citations = [{"chunk_id": c["chunk_id"], "score": round(c["score"], 6)} for c in trimmed_contexts]

    return {"answer": answer, "citations": citations, "metrics": metrics}