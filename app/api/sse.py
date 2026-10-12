"""
Server-Sent Events (SSE) streaming endpoint for real-time agent execution.
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import json
import asyncio
import uuid
from collections import deque
from app.storage import runs
from app.schemas.messages import ChatRequest
from app.agent.loop import AgentLoop
import app.api.routes as routes


router = APIRouter()


class LiveRun:
    def __init__(self, run_id: str, owner_id: str):
        self.run_id = run_id
        self.owner_id = owner_id
        self.events: deque[dict] = deque(maxlen=500)
        self.listeners: set[asyncio.Queue] = set()
        self.task: asyncio.Task | None = None
        self.finished = False
        self.state = None
        self.trace_offset = 0

    def publish(self, event: dict) -> None:
        self.events.append(event)
        stale = []
        for listener in self.listeners:
            try:
                listener.put_nowait(event)
            except asyncio.QueueFull:
                stale.append(listener)
        for listener in stale:
            self.listeners.discard(listener)


live_runs: dict[str, LiveRun] = {}


@router.post("/runs", status_code=202)
async def create_run(request: ChatRequest, http_request: Request) -> dict:
    if not routes.acquire_run_slot():
        raise HTTPException(status_code=429, detail="Too many concurrent runs")
    for run_id, run in list(live_runs.items()):
        if run.finished and len(live_runs) > 100:
            del live_runs[run_id]
    run = LiveRun(str(uuid.uuid4()), http_request.state.user_id)
    import time
    try:
        runs[run.run_id] = {
            "run_id": run.run_id, "owner_id": run.owner_id, "input": request.message,
            "status": "running", "created_at": time.time(), "provider": routes.llm_client.config.provider
            if hasattr(routes.llm_client, "config") else "test", "configuration": request.model_dump(),
            "trace": [], "total_tokens": 0, "total_cost": 0.0,
        }
    except OverflowError:
        routes.release_run_slot()
        raise HTTPException(status_code=503, detail="Run storage capacity reached")
    live_runs[run.run_id] = run

    async def execute():
        agent = AgentLoop(
            llm_client=routes.llm_client,
            tool_registry=routes.tool_registry,
            max_steps=request.max_steps,
            max_token_budget=request.max_token_budget,
            allowed_tools=[name for name in http_request.state.allowed_tools
                           if request.allowed_tools is None or name in request.allowed_tools],
            max_tool_calls=request.max_tool_calls,
            max_run_seconds=request.max_run_seconds,
            max_cost_budget=request.max_cost_budget,
        )

        async def on_step(state):
            run.publish({"type": "llm", "step": state.current_step, "total_tokens": state.total_tokens})
            for entry in state.trace[run.trace_offset:]:
                step = entry.model_dump(mode="json")
                run.publish({"type": "step", "step": step})
                if entry.type == "action":
                    run.publish({"type": "tool_call", "tool_name": entry.tool_name, "step": entry.step_number})
                elif entry.type == "observation":
                    run.publish({"type": "tool_result", "tool_name": entry.tool_name, "result": entry.content})
            run.trace_offset = len(state.trace)
            run.publish({"type": "metrics", "total_tokens": state.total_tokens, "total_cost": state.total_cost})
            runs[run.run_id] = {**runs[run.run_id], **state.to_dict(), "owner_id": run.owner_id,
                                "status": "running" if not state.is_finished else "finished"}

        try:
            state = await agent.run(request.message, on_step=on_step, run_id=run.run_id)
            run.state = state
            runs[run.run_id] = {**runs[run.run_id], **state.to_dict(), "owner_id": run.owner_id,
                                "status": "failed" if state.error else "finished"}
            if state.error:
                run.publish({"type": "error", "error": state.error, "termination_reason": state.termination_reason})
            run.publish({"type": "finish", "run_id": run.run_id, "final_answer": state.final_answer,
                         "error": state.error, "termination_reason": state.termination_reason,
                         "steps": state.current_step})
        except asyncio.CancelledError:
            runs[run.run_id] = {**runs[run.run_id], "status": "cancelled", "termination_reason": "cancelled"}
            run.publish({"type": "error", "error": "Run cancelled", "termination_reason": "cancelled"})
            run.publish({"type": "finish", "run_id": run.run_id, "error": "Run cancelled",
                         "final_answer": None, "termination_reason": "cancelled"})
            raise
        except Exception:
            runs[run.run_id] = {**runs[run.run_id], "status": "failed", "termination_reason": "error"}
            run.publish({"type": "error", "error": "Run failed", "termination_reason": "error"})
            run.publish({"type": "finish", "run_id": run.run_id, "error": "Run failed",
                         "final_answer": None, "termination_reason": "error"})
        finally:
            run.finished = True
            routes.release_run_slot()

    run.task = asyncio.create_task(execute())
    return {"run_id": run.run_id, "status": "running"}


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str, request: Request) -> dict:
    run = live_runs.get(run_id)
    if not run or run.owner_id != request.state.user_id:
        raise HTTPException(status_code=404, detail="Run not found")
    if not run.finished and run.task is not None:
        run.task.cancel()
    return {"run_id": run_id, "status": "cancelled" if not run.finished else "finished"}


async def live_events(run: LiveRun, request: Request) -> AsyncGenerator[str, None]:
    listener: asyncio.Queue = asyncio.Queue(maxsize=500)
    run.listeners.add(listener)
    try:
        for event in list(run.events):
            yield f"data: {json.dumps(event)}\n\n"
        if run.finished:
            return
        disconnect_wait = asyncio.create_task(request.is_disconnected())
        event_wait = asyncio.create_task(listener.get())
        try:
            while True:
                done, _ = await asyncio.wait(
                    {disconnect_wait, event_wait}, timeout=15,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    yield ": heartbeat\n\n"
                    continue
                if disconnect_wait in done:
                    if disconnect_wait.result():
                        event_wait.cancel()
                        await asyncio.gather(event_wait, return_exceptions=True)
                        return
                    disconnect_wait = asyncio.create_task(request.is_disconnected())
                if event_wait in done:
                    event = event_wait.result()
                    yield f"data: {json.dumps(event)}\n\n"
                    if event["type"] == "finish":
                        disconnect_wait.cancel()
                        await asyncio.gather(disconnect_wait, return_exceptions=True)
                        return
                    event_wait = asyncio.create_task(listener.get())
        finally:
            disconnect_wait.cancel()
            event_wait.cancel()
            await asyncio.gather(disconnect_wait, event_wait, return_exceptions=True)
    finally:
        run.listeners.discard(listener)


@router.get("/stream/{run_id}")
async def stream_run(run_id: str, request: Request) -> StreamingResponse:
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
    if run_id in live_runs:
        if live_runs[run_id].owner_id != request.state.user_id:
            raise HTTPException(status_code=404, detail="Run not found")
        return StreamingResponse(live_events(live_runs[run_id], request), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if run_id in runs and runs[run_id].get("owner_id") != request.state.user_id:
        raise HTTPException(status_code=404, detail="Run not found")

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
            
        
        # Send metrics
        total_tokens = run_data.get("total_tokens", 0)
        metrics_data = {
            "type": "metrics",
            "metrics": {
                "total_tokens": total_tokens,
                "total_cost": run_data.get("total_cost", 0),
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
