from pydantic import BaseModel, Field

class IndexReq(BaseModel):
    dataset_id: str = Field(..., description="UUID датасета")
    version: int = Field(1, description="Версия датасета")
    title: str | None = None
    text: str = Field(..., description="Плэйн-текст студента")
    overwrite: bool = True  # TODO: политика переиндексации


class IndexResp(BaseModel):
    ok: bool
    chunks: int


class AskReq(BaseModel):
    dataset_id: str
    version: int = 1
    question: str
    k: int | None = None
    min_score: float | None = None
    max_ctx_chars: int | None = None
    use_reranking: bool | None = None
    debug_reranking: bool = False


class AskResp(BaseModel):
    answer: str
    citations: list[dict]
    metrics: dict
