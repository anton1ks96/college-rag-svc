from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from models import IndexReq, IndexResp, AskReq, AskResp
from middleware import RequestIDMiddleware, ServiceAuthMiddleware
from metrics import INDEX_REQUESTS, INDEX_DURATION, ASK_REQUESTS, ASK_DURATION
from rag import index_dataset, ask_dataset_async
import time
from typing import Optional
import logging
from contextlib import asynccontextmanager
from config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения.
    Предзагружаем модели при старте сервиса.
    """
    logger.info("Starting model preloading...")

    try:
        if getattr(settings, "preload_embeddings", True):
            logger.info("Loading embedding model...")
            from embeddings import _load_model as load_embedding_model
            load_embedding_model()
            logger.info(f"Embedding model loaded: {getattr(settings, 'embedding_model', 'BAAI/bge-m3')}")

        if getattr(settings, "reranker_enabled", False) and getattr(settings, "preload_reranker", True):
            logger.info("Loading reranker model...")
            from reranking import _load_reranker_model
            _load_reranker_model()
            logger.info(f"Reranker model loaded: {getattr(settings, 'reranker_model', 'BAAI/bge-reranker-v2-m3')}")

        if getattr(settings, "preload_llm_check", True):
            provider = settings.llm_provider.lower()
            if provider == "openai":
                logger.info("Checking OpenAI API availability...")
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=settings.openai_api_key)
                    client.chat.completions.create(
                        model=settings.openai_model,
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=1
                    )
                    logger.info(f"OpenAI API available, model: {settings.openai_model}")
                except Exception as e:
                    logger.warning(f"OpenAI API check failed: {e}")

            elif provider == "ollama":
                logger.info("Checking Ollama availability...")
                try:
                    import ollama
                    models = ollama.list()
                    model_names = [m['name'] for m in models.get('models', [])]
                    if settings.ollama_model in model_names or any(settings.ollama_model in m for m in model_names):
                        logger.info(f"Ollama model available: {settings.ollama_model}")
                    else:
                        logger.warning(f"Ollama model {settings.ollama_model} not found. Available: {model_names}")
                        logger.info(f"Attempting to pull Ollama model {settings.ollama_model}...")
                        ollama.pull(settings.ollama_model)
                        logger.info(f"Ollama model pulled: {settings.ollama_model}")
                except Exception as e:
                    logger.warning(f"Ollama check failed: {e}")

        logger.info("All models preloaded successfully")

    except Exception as e:
        logger.error(f"Error during model preloading: {e}")

    yield

    logger.info("Shutting down, unloading models...")
    try:
        from embeddings import unload_model
        from reranking import unload_reranker_model
        unload_model()
        unload_reranker_model()
        logger.info("Models unloaded")
    except Exception as e:
        logger.error(f"Error during model unloading: {e}")

app = FastAPI(title="college-rag-svc", version="0.1.0", lifespan=lifespan)

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
    """Проверка готовности сервиса.
    Проверяет загрузку моделей и доступность Qdrant.
    """
    ready_status = {"ready": True, "models": {}}

    try:
        from embeddings import _model
        ready_status["models"]["embeddings"] = _model is not None
    except:
        ready_status["models"]["embeddings"] = False

    if getattr(settings, "reranker_enabled", False):
        try:
            from reranking import _reranker_model
            ready_status["models"]["reranker"] = _reranker_model is not None
        except:
            ready_status["models"]["reranker"] = False

    try:
        from qdrant_store import get_qdrant_client
        client = get_qdrant_client()
        client.get_collections()
        ready_status["qdrant"] = True
    except:
        ready_status["qdrant"] = False
        ready_status["ready"] = False

    if not all(ready_status["models"].values()):
        ready_status["ready"] = False

    return ready_status


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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
async def ask(req: AskReq, request: Request):
    ASK_REQUESTS.inc()
    t0 = time.time()
    try:
        res = await ask_dataset_async(
            dataset_id=req.dataset_id,
            version=req.version,
            question=req.question,
            k=req.k,
            min_score=req.min_score,
            max_ctx_chars=req.max_ctx_chars,
            use_reranking=req.use_reranking,
            debug_reranking=req.debug_reranking,
        )
        return AskResp(**res)
    finally:
        ASK_DURATION.observe(time.time() - t0)