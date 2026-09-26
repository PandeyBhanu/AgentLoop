"""
Agent state management for tracking conversation history and execution state.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from app.schemas.messages import Message, TraceEntry
import uuid


class AgentState(BaseModel):
    """
    Manages the state of an agent run including conversation history and trace.
    """
    
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique run identifier")
    conversation_history: List[Message] = Field(default_factory=list, description="Conversation history")
    trace: List[TraceEntry] = Field(default_factory=list, description="Execution trace")
    current_step: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=10, description="Maximum allowed steps")
    is_finished: bool = Field(default=False, description="Whether the run is finished")
    final_answer: Optional[str] = Field(None, description="Final answer when finished")
    error: Optional[str] = Field(None, description="Error message if run failed")
    termination_reason: Optional[str] = Field(
        None,
        description="Why the run ended: completed | error | max_steps | budget_exceeded | loop_detected"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Run creation timestamp")
    total_tokens: int = Field(default=0, description="Total tokens used in the run")
    total_cost: float = Field(default=0.0, description="Total estimated cost in USD")
    
    def add_message(self, role: str, content: Any) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: Role of the message sender
            content: Message content (will be converted to string)
        """
        self.conversation_history.append(Message(role=role, content=str(content)))
    
    def add_trace_entry(
        self,
        entry_type: str,
        content: str,
        step_number: Optional[int] = None,
        tool_name: Optional[str] = None,
        execution_time: Optional[float] = None,
        tokens_input: Optional[int] = None,
        tokens_output: Optional[int] = None,
        latency_ms: Optional[float] = None,
        token_usage: Optional[Dict[str, int]] = None,
        error_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an entry to the execution trace.
        
        Args:
            entry_type: Type of entry (thought/action/observation/error)
            content: Content of the entry
            step_number: Step number in the execution
            tool_name: Tool name if applicable
            execution_time: Execution time in seconds
            tokens_input: Input tokens used
            tokens_output: Output tokens used
            latency_ms: Latency in milliseconds
            token_usage: Token usage dictionary (legacy)
            error_details: Error details if type is error
        """
        entry = TraceEntry(
            type=entry_type,
            content=content,
            timestamp=datetime.utcnow(),
            step_number=step_number,
            tool_name=tool_name,
            execution_time=execution_time,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
            token_usage=token_usage,
            error_details=error_details
        )
        self.trace.append(entry)

    def record_usage(self, token_metadata: Dict[str, Any]) -> None:
        """
        Accumulate token usage and cost for one LLM call.

        This is the single writer for total_tokens / total_cost — trace
        entries only annotate per-step usage, they do not accumulate it.

        Args:
            token_metadata: metadata dict returned by the LLM client
        """
        self.total_tokens += token_metadata.get("total_tokens", 0) or 0
        self.total_cost += token_metadata.get("estimated_cost", 0.0) or 0.0
    
    def increment_step(self) -> None:
        """Increment the current step counter."""
        self.current_step += 1
    
    def should_continue(self) -> bool:
        """
        Check if the agent should continue running.
        
        Returns:
            True if should continue, False otherwise
        """
        if self.is_finished:
            return False
        if self.current_step >= self.max_steps:
            return False
        if self.error is not None:
            return False
        return True
    
    def finish(self, final_answer: Any) -> None:
        """
        Mark the run as finished with a final answer.
        
        Args:
            final_answer: The final answer from the agent (will be converted to string)
        """
        self.is_finished = True
        self.final_answer = str(final_answer)
        self.termination_reason = "completed"
    
    def fail(self, error: str, reason: str = "error") -> None:
        """
        Mark the run as failed with an error and a termination reason.
        
        Args:
            error: Error message
            reason: Termination reason (error | max_steps | budget_exceeded | loop_detected)
        """
        self.is_finished = True
        self.error = error
        self.termination_reason = reason
    
    def get_last_n_messages(self, n: int) -> List[Message]:
        """
        Get the last n messages from conversation history.
        
        Args:
            n: Number of messages to retrieve
            
        Returns:
            List of last n messages
        """
        return self.conversation_history[-n:]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert state to dictionary.
        
        Returns:
            Dictionary representation of the state
        """
        # Use model_dump for Pydantic v2 compatibility
        def _dump(obj):
            return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
        
        return {
            "run_id": self.run_id,
            "conversation_history": [_dump(msg) for msg in self.conversation_history],
            "trace": [_dump(entry) for entry in self.trace],
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "is_finished": self.is_finished,
            "final_answer": self.final_answer,
            "error": self.error,
            "termination_reason": self.termination_reason,
            "created_at": self.created_at.isoformat(),
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
        }
