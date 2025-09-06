import hashlib
from typing import List, Dict, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from config import settings

_client = QdrantClient(url=settings.qdrant_url)

def _distance() -> Distance:
    return Distance.COSINE

def ensure_collection(vector_size: int):
    existing = [c.name for c in _client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        _client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=vector_size, distance=_distance()),
        )

def _point_id(dataset_id: str, version: int, chunk_id: int) -> int:
    h = hashlib.md5(f"{dataset_id}:{version}:{chunk_id}".encode()).hexdigest()[:12]
    return int(h, 16)

def upsert_chunks(dataset_id: str, version: int, title: str | None, chunks: List[Dict], vectors: List[List[float]]) -> int:
    points: List[PointStruct] = []
    for ch, vec in zip(chunks, vectors):
        points.append(PointStruct(
            id=_point_id(dataset_id, version, ch["idx"]),
            vector=vec,
            payload={
                "dataset_id": dataset_id,
                "version": version,
                "chunk_id": ch["idx"],
                "title": title or "",
                "text": ch["text"],
            }
        ))
    _client.upsert(collection_name=settings.qdrant_collection, points=points)
    return len(points)

def search(dataset_id: str, version: int, query_vector: List[float], k: int) -> List[Tuple[float, Dict]]:
    flt = Filter(must=[
        FieldCondition(key="dataset_id", match=MatchValue(value=dataset_id)),
        FieldCondition(key="version", match=MatchValue(value=version)),
    ])
    hits = _client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        limit=k,
        query_filter=flt
    )
    return [(float(h.score), h.payload) for h in hits]
