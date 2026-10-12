"""
Logging utilities for the agent system with token and cost tracking.
"""
import logging
import threading
from collections import defaultdict, deque
from typing import Optional


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
        self.logger.info("Tool call: %s", tool_name)
    
    def log_observation(self, tool_name: str, result: str) -> None:
        """Log an observation from a tool."""
        self.logger.info("Observation from %s: %d characters", tool_name, len(result))
    
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


class MetricsRegistry:
    def __init__(self, max_observations: int = 2048):
        self._counters = defaultdict(int)
        self._observations = defaultdict(lambda: deque(maxlen=max_observations))
        self._lock = threading.Lock()

    @staticmethod
    def _key(name: str, labels: Optional[dict] = None) -> tuple:
        return name, tuple(sorted((labels or {}).items()))

    def increment(self, name: str, value: int = 1, labels: Optional[dict] = None) -> None:
        with self._lock:
            self._counters[self._key(name, labels)] += value

    def observe(self, name: str, value: float, labels: Optional[dict] = None) -> None:
        with self._lock:
            self._observations[self._key(name, labels)].append(float(value))

    def snapshot(self) -> dict:
        with self._lock:
            counters = {
                ".".join([name, *(f"{key}={value}" for key, value in labels)]): count
                for (name, labels), count in self._counters.items()
            }
            observations = {}
            for (name, labels), values in self._observations.items():
                data = list(values)
                if data:
                    key = ".".join([name, *(f"{k}={v}" for k, v in labels)])
                    observations[key] = {
                        "count": len(data), "total": sum(data),
                        "avg": sum(data) / len(data), "max": max(data),
                    }
            return {"counters": counters, "observations": observations}


logger = AgentLogger()
metrics = MetricsRegistry()
