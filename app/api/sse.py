"""
Server-Sent Events (SSE) streaming endpoint for real-time agent execution.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import json
import asyncio
from app.storage import runs


router = APIRouter()


@router.get("/stream/{run_id}")
async def stream_run(run_id: str) -> StreamingResponse:
    """
    Stream a completed agent run's trace via Server-Sent Events.
    
    The frontend expects event types:
    - 'step': trace entry (thought, action, observation, error, finish)
    - 'tool_call': tool execution details
    - 'metrics': total tokens and cost
    - 'finish': signals completion
    
    Args:
        run_id: The run ID to stream
        
    Returns:
        StreamingResponse with SSE events
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        if run_id not in runs:
            error_data = {"type": "error", "error": f"Run {run_id} not found"}
            yield f"data: {json.dumps(error_data)}\n\n"
            return
        
        run_data = runs[run_id]
        trace = run_data.get("trace", [])
        
        # Send each trace entry as a 'step' event
        for entry in trace:
            step_data = {
                "type": "step",
                "step": {
                    "type": entry.get("type", "unknown"),
                    "content": entry.get("content", ""),
                    "step_number": entry.get("step_number", 0),
                    "latency_ms": entry.get("latency_ms", 0),
                    "tool_name": entry.get("tool_name"),
                    "tokens_input": entry.get("tokens_input", 0),
                    "tokens_output": entry.get("tokens_output", 0),
                    "error_details": entry.get("error_details"),
                }
            }
            yield f"data: {json.dumps(step_data)}\n\n"
            
            # Send tool_call event for action/observation entries
            if entry.get("type") in ("action", "observation") and entry.get("tool_name"):
                # Find the observation that follows this action
                tool_call_data = {
                    "type": "tool_call",
                    "tool_call": {
                        "tool_name": entry.get("tool_name", ""),
                        "arguments": {},  # We don't store arguments in trace, but we could
                        "success": entry.get("type") == "observation",
                        "execution_time": entry.get("execution_time", 0),
                        "error": entry.get("content", "") if entry.get("type") == "error" else None,
                    }
                }
                yield f"data: {json.dumps(tool_call_data)}\n\n"
            
            await asyncio.sleep(0.3)  # Small delay to simulate live execution
        
        # Send metrics
        total_tokens = run_data.get("total_tokens", 0)
        metrics_data = {
            "type": "metrics",
            "metrics": {
                "total_tokens": total_tokens,
                "total_cost": total_tokens * 0.00003,  # Rough estimate
            }
        }
        yield f"data: {json.dumps(metrics_data)}\n\n"
        
        # Send finish event
        finish_data = {
            "type": "finish",
            "final_answer": run_data.get("final_answer", ""),
            "error": run_data.get("error"),
            "steps": run_data.get("current_step", 0),
            "run_id": run_id,
        }
        yield f"data: {json.dumps(finish_data)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/stream")
async def stream_chat(message: str, max_steps: int = 10) -> StreamingResponse:
    """
    Stream agent execution in real-time using Server-Sent Events.
    (Legacy endpoint - kept for direct streaming use)
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        error_data = {
            "type": "error",
            "error": "Direct streaming is not supported. Use POST /chat first, then GET /stream/{run_id}"
        }
        yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
