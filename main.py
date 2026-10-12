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
import asyncio
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app):
    from app.storage import runs
    runs.mark_interrupted_runs()
    yield
    from app.api.sse import live_runs
    from app.api import routes
    current_loop = asyncio.get_running_loop()
    active_tasks = [run.task for run in live_runs.values()
                    if run.task and not run.finished and run.task.get_loop() is current_loop]
    for task in active_tasks:
        task.cancel()
    await asyncio.gather(*active_tasks, return_exceptions=True)
    if hasattr(routes.llm_client, "close"):
        await routes.llm_client.close()


app = FastAPI(
    title="ReAct Agent API",
    description="ReAct agent framework",
    version="1.0.0",
    lifespan=lifespan
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


# API identity and server-side tool policy. Production requires credentials.
# Without credentials, development mode is limited to loopback/test clients.
_api_key = os.getenv("AGENTLOOP_API_KEY")
_users = os.getenv("AGENTLOOP_USERS_JSON")
if _users:
    import json
    _identities = json.loads(_users)
    if not isinstance(_identities, dict) or not all(
        isinstance(identity, str) and isinstance(config, dict) and
        isinstance(config.get("api_key"), str) and config["api_key"] and
        isinstance(config.get("allowed_tools"), list) and
        all(isinstance(tool, str) for tool in config["allowed_tools"])
        for identity, config in _identities.items()
    ) or len({config["api_key"] for config in _identities.values()}) != len(_identities):
        raise RuntimeError("Invalid AGENTLOOP_USERS_JSON configuration")
else:
    _identities = {"default": {"api_key": _api_key, "allowed_tools": [
        name for name in os.getenv("AGENTLOOP_ALLOWED_TOOLS", "Calculator,WebSearch,FileReader").split(",") if name
    ]}} if _api_key else {}
if os.getenv("AGENTLOOP_MODE", "development").lower() == "production" and not _identities:
    raise RuntimeError("An API key is required in production mode")

@app.middleware("http")
async def api_key_middleware(request, call_next):
    if request.url.path in ("/", "/api/health") or request.method == "OPTIONS":
        return await call_next(request)
    key = request.headers.get("x-api-key")
    if _identities:
        import hmac
        user = next((name for name, config in _identities.items()
                     if key and hmac.compare_digest(key, config["api_key"])), None)
        if user is None:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        policy = _identities[user]["allowed_tools"]
    else:
        if request.client and request.client.host not in ("127.0.0.1", "::1", "testclient"):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        user = "development"
        policy = [name for name in os.getenv("AGENTLOOP_ALLOWED_TOOLS", "Calculator,WebSearch,FileReader").split(",") if name]
    request.state.user_id = user
    request.state.allowed_tools = policy
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
