"""
Pydantic schemas for agent messages and responses.
"""
from pydantic import BaseModel, Field
from typing import Literal, Optional, Any
from datetime import datetime


class Message(BaseModel):
    """Base message structure for conversation history."""
    role: str = Field(..., description="Role of the message sender (user/assistant/system)")
    content: Any = Field(..., description="Message content")


class ToolCall(BaseModel):
    """Represents a tool call from the LLM."""
    tool_name: str = Field(..., description="Name of the tool to call")
    arguments: dict[str, Any] = Field(..., description="Arguments for the tool")


class LLMResponse(BaseModel):
    """Structured response from the LLM."""
    type: Literal["thought", "action", "finish"] = Field(..., description="Type of response")
    thought: Optional[str] = Field(None, description="Current reasoning/thought")
    tool_name: Optional[str] = Field(None, description="Tool name if action type")
    arguments: Optional[dict[str, Any]] = Field(None, description="Tool arguments if action type")
    final_answer: Optional[Any] = Field(None, description="Final answer if finish type")


class ToolExecutionResult(BaseModel):
    """Result of a tool execution."""
    success: bool = Field(..., description="Whether the tool execution succeeded")
    result: Any = Field(..., description="Result of the tool execution")
    error: Optional[str] = Field(None, description="Error message if failed")


class TraceEntry(BaseModel):
    """Single entry in the execution trace."""
    type: str = Field(..., description="Type of entry (thought/action/observation/error)")
    content: Any = Field(..., description="Content of the entry")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the entry")
    step_number: Optional[int] = Field(None, description="Step number in the execution")
    tool_name: Optional[str] = Field(None, description="Tool name if applicable")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    tokens_input: Optional[int] = Field(None, description="Input tokens used")
    tokens_output: Optional[int] = Field(None, description="Output tokens used")
    latency_ms: Optional[float] = Field(None, description="Latency in milliseconds")
    token_usage: Optional[dict[str, int]] = Field(None, description="Token usage if available (legacy)")
    error_details: Optional[dict[str, Any]] = Field(None, description="Error details if type is error")


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""
    message: str = Field(..., description="User message")
    max_steps: int = Field(default=10, description="Maximum number of agent steps")
    allowed_tools: Optional[list[str]] = Field(None, description="List of allowed tools (empty = all tools)")
    max_token_budget: Optional[int] = Field(None, description="Maximum token budget for the run")


class ChatResponse(BaseModel):
    """Response body for the /chat endpoint."""
    run_id: str = Field(..., description="Unique identifier for this run")
    final_answer: Any = Field(..., description="Final answer from the agent")
    steps: int = Field(..., description="Number of steps taken")
    total_tokens: Optional[int] = Field(None, description="Total tokens used")
    error: Optional[str] = Field(None, description="Error message if run failed")


class StructuredError(BaseModel):
    """Structured error format for consistent error handling."""
    type: Literal["error"] = Field(default="error", description="Error type")
    stage: Literal["tool_execution", "parsing", "llm", "guardrail", "timeout"] = Field(..., description="Stage where error occurred")
    message: str = Field(..., description="Error message")
    recoverable: bool = Field(..., description="Whether the error is recoverable")
    details: Optional[dict[str, Any]] = Field(None, description="Additional error details")
