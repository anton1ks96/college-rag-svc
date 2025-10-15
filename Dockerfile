FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --user --no-cache-dir --upgrade pip

COPY requirements.txt* ./
RUN pip install --user --no-cache-dir -r requirements.txt

RUN pip install --user --no-cache-dir \
    torch \
    transformers \
    sentence-transformers \
    FlagEmbedding \
    qdrant-client \
    fastapi \
    uvicorn

FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash app

COPY --from=builder /root/.local /home/app/.local

ENV PATH=/home/app/.local/bin:$PATH
ENV PYTHONPATH=/app:/app/college_rag_svc

WORKDIR /app

COPY . .

RUN mkdir -p /app/cache && \
    chown -R app:app /app

USER app

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "college_rag_svc.app:app", "--host", "0.0.0.0", "--port", "8001"]