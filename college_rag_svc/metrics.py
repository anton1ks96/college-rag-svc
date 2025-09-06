from prometheus_client import Counter, Histogram

INDEX_REQUESTS = Counter("rag_index_requests_total", "Total index requests")
INDEX_DURATION = Histogram("rag_index_duration_seconds", "Index duration seconds")

ASK_REQUESTS = Counter("rag_ask_requests_total", "Total ask requests")
ASK_DURATION = Histogram("rag_ask_duration_seconds", "Ask duration seconds")

# TODO: добавить ошибочные метрики/исключения
