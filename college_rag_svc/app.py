from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from models import IndexReq, IndexResp, AskReq, AskResp
from middleware import RequestIDMiddleware, ServiceAuthMiddleware
from metrics import INDEX_REQUESTS, INDEX_DURATION, ASK_REQUESTS, ASK_DURATION
from rag import index_dataset, ask_dataset
import time
from typing import Optional

app = FastAPI(title="college-rag-svc", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(ServiceAuthMiddleware)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/readyz")
def readyz():
    # TODO: проверка соединения к Qdrant (ping)
    return {"ready": True}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Старый endpoint для обратной совместимости
@app.post("/index", response_model=IndexResp)
def index(req: IndexReq, request: Request):
    INDEX_REQUESTS.inc()
    t0 = time.time()
    try:
        chunks = index_dataset(
            dataset_id=req.dataset_id,
            version=req.version,
            title=req.title,
            text=req.text,
            overwrite=req.overwrite,
        )
        return {"ok": True, "chunks": chunks}
    finally:
        INDEX_DURATION.observe(time.time() - t0)


@app.post("/index/file", response_model=IndexResp)
async def index_file(
        request: Request,
        file: UploadFile = File(..., description="Markdown файл с материалом"),
        dataset_id: str = Form(..., description="UUID датасета"),
        version: int = Form(1, description="Версия датасета"),
        title: Optional[str] = Form(None, description="Заголовок документа (если не указан в файле)"),
        overwrite: bool = Form(True, description="Перезаписать существующие данные")
):
    """
    Индексация markdown файла.
    Файл должен быть в формате .md или .markdown
    """
    INDEX_REQUESTS.inc()
    t0 = time.time()

    try:
        if not file.filename.endswith(('.md', '.markdown')):
            raise HTTPException(
                status_code=400,
                detail="Файл должен быть в формате Markdown (.md или .markdown)"
            )

        content = await file.read()
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Не удалось декодировать файл. Убедитесь, что файл в кодировке UTF-8"
            )

        chunks = index_dataset(
            dataset_id=dataset_id,
            version=version,
            title=title,
            text=text,
            overwrite=overwrite,
        )

        return {"ok": True, "chunks": chunks}

    finally:
        INDEX_DURATION.observe(time.time() - t0)


@app.post("/ask", response_model=AskResp)
def ask(req: AskReq, request: Request):
    ASK_REQUESTS.inc()
    t0 = time.time()
    try:
        res = ask_dataset(
            dataset_id=req.dataset_id,
            version=req.version,
            question=req.question,
            k=req.k,
            min_score=req.min_score,
            max_ctx_chars=req.max_ctx_chars
        )
        return AskResp(**res)
    finally:
        ASK_DURATION.observe(time.time() - t0)