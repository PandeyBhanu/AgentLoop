"""
Deterministic replay function for agent execution traces.
"""
from typing import Dict, Any, Optional
from app.agent.state import AgentState
from app.storage import runs


class ReplayResult:
    """Result of a replay operation."""
    
    def __init__(self, success: bool, steps: int, final_answer: Optional[str] = None, error: Optional[str] = None):
        self.success = success
        self.steps = steps
        self.final_answer = final_answer
        self.error = error


def replay(run_id: str) -> ReplayResult:
    """
    Replay a previous agent execution using the stored trace.
    
    This function simulates agent decisions without calling the LLM,
    allowing for deterministic replay of previous runs for debugging
    and analysis.
    
    Args:
        run_id: Unique identifier of the run to replay
        
    Returns:
        ReplayResult with the replay outcome
    """
    # Retrieve the stored run state
    if run_id not in runs:
        return ReplayResult(success=False, steps=0, error=f"Run {run_id} not found")
    
    run_data = runs[run_id]
    trace = run_data.get("trace", [])
    
    if not trace:
        return ReplayResult(success=False, steps=0, error="No trace found for this run")
    
    # Simulate the execution by stepping through the trace
    steps_replayed = 0
    final_answer = None
    error = None
    
    for entry in trace:
        steps_replayed += 1
        entry_type = entry.get("type")
        
        # Simulate different entry types
        if entry_type == "thought":
            # Simulate thought processing
            pass
        elif entry_type == "action":
            # Simulate tool call validation
            tool_name = entry.get("tool_name")
            content = entry.get("content", "")
            # In a real implementation, this would validate tool availability
            # and arguments without actually executing
        elif entry_type == "observation":
            # Simulate observation handling
            pass
        elif entry_type == "error":
            # Check if error was recoverable
            error_details = entry.get("error_details", {})
            if not error_details.get("recoverable", True):
                error = entry.get("content")
                return ReplayResult(success=False, steps=steps_replayed, error=error)
        elif entry_type == "finish":
            # Extract final answer
            final_answer = entry.get("content")
    
    # Check if the original run finished successfully
    if run_data.get("is_finished") and run_data.get("final_answer"):
        return ReplayResult(success=True, steps=steps_replayed, final_answer=run_data.get("final_answer"))
    elif run_data.get("error"):
        return ReplayResult(success=False, steps=steps_replayed, error=run_data.get("error"))
    else:
        return ReplayResult(success=False, steps=steps_replayed, error="Run did not complete")


def replay_with_analysis(run_id: str) -> Dict[str, Any]:
    """
    Replay a run and provide detailed analysis.
    
    Args:
        run_id: Unique identifier of the run to replay
        
    Returns:
        Dictionary with replay results and analysis
    """
    if run_id not in runs:
        return {"error": f"Run {run_id} not found"}
    
    run_data = runs[run_id]
    trace = run_data.get("trace", [])
    
    # Analyze the trace
    analysis = {
        "run_id": run_id,
        "total_steps": len(trace),
        "steps_by_type": {},
        "total_tokens": run_data.get("total_tokens", 0),
        "total_cost": run_data.get("total_cost", 0.0),
        "errors": [],
        "tool_calls": [],
        "latency_stats": {}
    }
    
    # Count steps by type
    for entry in trace:
        entry_type = entry.get("type")
        analysis["steps_by_type"][entry_type] = analysis["steps_by_type"].get(entry_type, 0) + 1
        
        # Track tool calls
        if entry_type == "action":
            analysis["tool_calls"].append({
                "tool_name": entry.get("tool_name"),
                "step": entry.get("step_number")
            })
        
        # Track errors
        if entry_type == "error":
            analysis["errors"].append({
                "stage": entry.get("error_details", {}).get("stage"),
                "message": entry.get("content"),
                "recoverable": entry.get("error_details", {}).get("recoverable")
            })
        
        # Track latency
        latency = entry.get("latency_ms")
        if latency:
            if "latency_stats" not in analysis:
                analysis["latency_stats"] = {"min": latency, "max": latency, "total": 0, "count": 0}
            stats = analysis["latency_stats"]
            stats["min"] = min(stats["min"], latency)
            stats["max"] = max(stats["max"], latency)
            stats["total"] += latency
            stats["count"] += 1
    
    # Calculate average latency
    if analysis["latency_stats"] and analysis["latency_stats"]["count"] > 0:
        analysis["latency_stats"]["avg"] = analysis["latency_stats"]["total"] / analysis["latency_stats"]["count"]
    
    # Perform replay
    replay_result = replay(run_id)
    analysis["replay_success"] = replay_result.success
    analysis["replay_steps"] = replay_result.steps
    analysis["replay_final_answer"] = replay_result.final_answer
    analysis["replay_error"] = replay_result.error
    
    return analysis
