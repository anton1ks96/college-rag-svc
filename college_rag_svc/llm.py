import httpx
from openai import AsyncOpenAI
from config import settings
import logging

logger = logging.getLogger(__name__)

_openai_client: AsyncOpenAI | None = None
_vllm_client: AsyncOpenAI | None = None


def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai_client


def _get_vllm_client() -> AsyncOpenAI:
    global _vllm_client
    if _vllm_client is None:
        _vllm_client = AsyncOpenAI(
            base_url=f"{settings.vllm_base_url}/v1",
            api_key=settings.vllm_api_key,
        )
    return _vllm_client


async def generate_answer_async(
    question: str,
    contexts: list[dict],
    system_prompt: str
) -> str:
    ctx = "\n\n".join([
        f"<chunk id=\"{c['chunk_id']}\">\n{c['text']}\n</chunk>"
        for c in contexts
    ])

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Вопрос: {question}\n\n<context>\n{ctx}\n</context>"}
    ]

    provider = settings.llm_provider.lower()

    if provider == "openai":
        client = _get_openai_client()
        resp = await client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens
        )
        return resp.choices[0].message.content

    elif provider == "vllm":
        client = _get_vllm_client()
        resp = await client.chat.completions.create(
            model=settings.vllm_model,
            messages=messages,
            temperature=settings.vllm_temperature,
            max_tokens=settings.vllm_max_tokens
        )
        return resp.choices[0].message.content

    elif provider == "ollama":
        async with httpx.AsyncClient(timeout=500.0) as client:
            resp = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": settings.ollama_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": getattr(settings, "ollama_temperature", 0.7)
                    }
                }
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    raise RuntimeError(f"Unknown LLM provider: {provider}")
