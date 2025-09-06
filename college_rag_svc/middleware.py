from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import uuid

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

class ServiceAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # # TODO: Ограничить доступ сервисным токеном для /index и /ask
        # token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        # if request.url.path in ("/index", "/ask"):
        #     if not token or token != settings.rag_service_token:
        #         raise HTTPException(status_code=401, detail="Unauthorized")
        return await call_next(request)
