"""
FastAPI routes for the agent API.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict
from app.schemas.messages import ChatRequest, ChatResponse
from app.agent.loop import AgentLoop
from app.llm import LLMClient, LLMConfig
from app.tools.registry import ToolRegistry
from app.tools.calculator import Calculator
from app.tools.web_search import WebSearch
from app.tools.file_reader import FileReader
from app.tools.python_exec import PythonExec
from app.agent.replay import replay, replay_with_analysis
from app.storage import runs
import asyncio
import os


def setup_tool_registry() -> ToolRegistry:
    """Set up and register all tools."""
    registry = ToolRegistry()
    registry.register(Calculator())
    registry.register(WebSearch())
    registry.register(FileReader())
    registry.register(PythonExec())
    return registry


def get_llm_client() -> LLMClient:
    """
    Get the LLM client instance with provider configuration.
    
    Provider selection is controlled via LLM_PROVIDER environment variable.
    Defaults to 'openai' if not specified.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    config = LLMConfig(
        provider=provider,
        # API key will be picked up from environment variables in LLMConfig
        temperature=0.7,
        max_tokens=1024
    )
    
    return LLMClient(config)


# Initialize components
tool_registry = setup_tool_registry()
llm_client = get_llm_client()

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a user message through the agent loop.
    
    Args:
        request: ChatRequest with user message, max_steps, allowed_tools, and max_token_budget
        
    Returns:
        ChatResponse with run_id, final_answer, steps, total_tokens, and error if any
    """
    try:
        # Create agent loop with new parameters
        agent_loop = AgentLoop(
            llm_client=llm_client,
            tool_registry=tool_registry,
            max_steps=request.max_steps,
            max_token_budget=request.max_token_budget,
            allowed_tools=request.allowed_tools
        )
        
        # Run the agent
        state = await agent_loop.run(request.message)
        
        # Store run state
        runs[state.run_id] = state.to_dict()
        
        # Return response
        if state.error:
            return ChatResponse(
                run_id=state.run_id,
                final_answer=f"Error: {state.error}",
                steps=state.current_step,
                total_tokens=state.total_tokens,
                error=state.error
            )
        
        return ChatResponse(
            run_id=state.run_id,
            final_answer=state.final_answer or "No answer provided",
            steps=state.current_step,
            total_tokens=state.total_tokens
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trace/{run_id}")
async def get_trace(run_id: str) -> Dict:
    """
    Get the execution trace for a specific run.
    
    Args:
        run_id: Unique identifier for the run
        
    Returns:
        Dictionary with full execution trace
    """
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return runs[run_id]


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        Dictionary with health status
    """
    return {"status": "healthy", "service": "react-agent"}


@router.get("/replay/{run_id}")
async def replay_run(run_id: str) -> Dict:
    """
    Replay a previous agent execution.
    
    Args:
        run_id: Unique identifier for the run to replay
        
    Returns:
        Dictionary with replay results
    """
    try:
        replay_result = replay(run_id)
        return {
            "success": replay_result.success,
            "steps": replay_result.steps,
            "final_answer": replay_result.final_answer,
            "error": replay_result.error
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/replay/{run_id}/analysis")
async def replay_with_analysis_endpoint(run_id: str) -> Dict:
    """
    Replay a run and provide detailed analysis.
    
    Args:
        run_id: Unique identifier for the run to replay
        
    Returns:
        Dictionary with replay results and detailed analysis
    """
    try:
        analysis = replay_with_analysis(run_id)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
