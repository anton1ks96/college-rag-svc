from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
import uuid
from config import settings


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

class ServiceAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/index/file", "/ask") and settings.rag_service_token:
            auth = request.headers.get("Authorization", "")
            parts = auth.split(None, 1)
            token = parts[1].strip() if parts and parts[0].lower() == "bearer" and len(parts) > 1 else auth.strip()
            if not token or token != settings.rag_service_token:
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)
