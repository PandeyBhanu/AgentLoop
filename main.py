"""
Main entry point for the ReAct Agent API server.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file BEFORE importing routes
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.routes import router as routes_router
from app.api.sse import router as sse_router
import uvicorn


app = FastAPI(
    title="ReAct Agent API",
    description="Production-quality minimal ReAct agent framework",
    version="1.0.0"
)

# CORS — explicit origins only. Never use "*" with credentials.
# Configure via CORS_ORIGINS="https://a.com,https://b.com"
_cors_origins = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# Optional shared-secret auth. When AGENTLOOP_API_KEY is set, every request
# except / and /api/health must carry the matching X-API-Key header.
# When unset the API is open (dev mode) — set it before exposing the server.
_api_key = os.getenv("AGENTLOOP_API_KEY")
if _api_key:

    @app.middleware("http")
    async def api_key_middleware(request, call_next):
        if request.url.path in ("/", "/api/health") or request.method == "OPTIONS":
            return await call_next(request)
        if request.headers.get("x-api-key") != _api_key:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        return await call_next(request)


# Include routers
app.include_router(routes_router, prefix="/api")
app.include_router(sse_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ReAct Agent API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/api/chat",
            "trace": "/api/trace/{run_id}",
            "stream": "/api/stream",
            "health": "/api/health"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=True
    )
