import ollama

from config import settings

def generate_answer(question: str, contexts: list[dict], system_prompt: str) -> str:
    provider = settings.llm_provider.lower()
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        ctx = "\n\n".join([f"[Chunk {c['chunk_id']}]\n{c['text']}" for c in contexts])
        messages = [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": f"Вопрос: {question}\n\nКонтекст:\n{ctx}"}
        ]
        resp = client.chat.completions.create(
          model=settings.openai_model,
          messages=messages,
          temperature=settings.openai_temperature,
          max_tokens=settings.openai_max_tokens
        )
        return resp.choices[0].message.content

    elif provider == "ollama":
        ctx = "\n\n".join([f"[Chunk {c['chunk_id']}]\n{c['text']}" for c in contexts])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Вопрос: {question}\n\nКонтекст:\n{ctx}"}
        ]
        response = ollama.chat(
            model=settings.ollama_model,
            messages=messages,
            options={
                "temperature": getattr(settings, "ollama_temperature", 0.7),
            }
        )
        return response["message"]["content"]

    raise RuntimeError(f"Unknown LLM provider: {provider}")
