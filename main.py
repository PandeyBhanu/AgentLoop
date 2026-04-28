"""
Main entry point for the ReAct Agent API server.
"""
from dotenv import load_dotenv

# Load environment variables from .env file BEFORE importing routes
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as routes_router
from app.api.sse import router as sse_router
import uvicorn


app = FastAPI(
    title="ReAct Agent API",
    description="Production-quality minimal ReAct agent framework",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        host="0.0.0.0",
        port=8000,
        reload=True
    )
