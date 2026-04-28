"""
Logging utilities for the agent system with token and cost tracking.
"""
import logging
from typing import Optional
from datetime import datetime


class AgentLogger:
    """
    Custom logger for agent execution.
    """
    
    def __init__(self, name: str = "agent", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)
    
    def log_step(self, step: int, message: str) -> None:
        """Log a step with its number."""
        self.logger.info(f"Step {step}: {message}")
    
    def log_tool_call(self, tool_name: str, arguments: dict) -> None:
        """Log a tool call."""
        self.logger.info(f"Tool call: {tool_name} with args {arguments}")
    
    def log_observation(self, tool_name: str, result: str) -> None:
        """Log an observation from a tool."""
        self.logger.info(f"Observation from {tool_name}: {result[:200]}...")
    
    def log_token_usage(self, step: int, prompt_tokens: int, completion_tokens: int, 
                        total_tokens: int, latency_ms: float, cost: float,
                        provider: Optional[str] = None, model: Optional[str] = None) -> None:
        """Log token usage and cost for a step."""
        provider_str = f" [{provider}/{model}]" if provider and model else ""
        self.logger.info(
            f"Step {step}{provider_str} tokens: input={prompt_tokens}, output={completion_tokens}, "
            f"total={total_tokens}, latency={latency_ms:.2f}ms, cost=${cost:.6f}"
        )
    
    def log_error(self, stage: str, message: str, recoverable: bool) -> None:
        """Log a structured error."""
        self.logger.error(f"[{stage}] {message} (recoverable: {recoverable})")


# Global logger instance
logger = AgentLogger()
