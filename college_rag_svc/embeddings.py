from typing import List
from config import settings

try:
    from FlagEmbedding import BGEM3FlagModel
except Exception as e:
    BGEM3FlagModel = None
    _import_error = e
else:
    _import_error = None

_model = None


def _load_model():
    global _model

    if _model is not None:
        return _model

    if BGEM3FlagModel is None:
        raise ImportError(
            "FlagEmbedding is required. Install with: pip install -U FlagEmbedding\n"
            f"Original import error: {_import_error}"
        )

    model_name = getattr(settings, "embedding_model", "BAAI/bge-m3")
    use_fp16 = bool(getattr(settings, "embedding_use_fp16", True))

    _model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not isinstance(texts, list):
        raise TypeError("texts must be a List[str]")
    if not texts:
        return []

    model = _load_model()

    batch_size = int(getattr(settings, "embedding_batch_size", 12))
    max_length = int(getattr(settings, "embedding_max_length", 8192))

    outputs = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    dense = outputs["dense_vecs"]

    try:
        return dense.tolist()
    except AttributeError:
        return [list(map(float, vec)) for vec in dense]


def unload_model() -> None:
    global _model
    _model = None
