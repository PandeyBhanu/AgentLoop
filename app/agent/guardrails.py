"""
Guardrails for agent execution including loop detection and step limits.
"""
from typing import Dict, Any, Optional, List
from collections import deque
import hashlib
import json
from app.utils.hashing import create_fingerprint


class Guardrails:
    """
    Implements safety checks for agent execution.
    """
    
    def __init__(
        self,
        max_steps: int = 10,
        tool_timeout: int = 30,
        max_thought_history: int = 5,
        action_history_size: int = 12,
        max_tool_frequency: int = 4,
    ):
        self.max_steps = max_steps
        self.tool_timeout = tool_timeout
        self.fingerprints: set[str] = set()
        self.action_signatures: deque[str] = deque(maxlen=action_history_size)
        self.max_tool_frequency = max_tool_frequency
        self.thought_history: List[str] = []  # Track last N thoughts
        self.max_thought_history = max_thought_history
    
    def check_step_limit(self, current_step: int) -> tuple[bool, Optional[str]]:
        """
        Check if the agent has exceeded the maximum step limit.
        
        Args:
            current_step: Current step number
            
        Returns:
            Tuple of (should_continue, error_message)
        """
        if current_step >= self.max_steps:
            return False, f"Maximum step limit ({self.max_steps}) reached"
        return True, None
    
    @staticmethod
    def _normalized_argument(value: Any, selector: bool = False) -> Any:
        """Keep stable operation selectors while ignoring changing payload values."""
        if isinstance(value, dict):
            return {
                key: Guardrails._normalized_argument(
                    item, selector=key.lower() in {"operation", "mode", "type", "action", "command", "method"}
                )
                for key, item in sorted(value.items())
            }
        if isinstance(value, (list, tuple)):
            return [Guardrails._normalized_argument(item) for item in value[:8]]
        if selector and isinstance(value, str) and len(value) <= 64:
            return value
        if selector and (isinstance(value, bool) or value is None):
            return value
        return type(value).__name__

    def _action_signature(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        normalized = self._normalized_argument(arguments)
        return f"{tool_name}:{json.dumps(normalized, sort_keys=True, default=str)}"

    def check_loop(self, tool_name: str, arguments: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Check for exact repeats, alternating action cycles, and excessive tool frequency.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool

        Returns:
            Tuple of (should_continue, error_message)
        """
        fingerprint = create_fingerprint(tool_name, arguments)
        signature = self._action_signature(tool_name, arguments)

        if fingerprint in self.fingerprints:
            return False, f"Loop detected: tool '{tool_name}' with same arguments called twice"

        history = list(self.action_signatures)
        if signature and history.count(signature) + 1 > self.max_tool_frequency:
            return False, f"Loop detected: tool '{tool_name}' invoked too frequently"

        candidate = history + [signature]
        for period in range(2, min(4, len(candidate) // 2) + 1):
            if candidate[-period:] == candidate[-2 * period:-period]:
                return False, f"Loop detected: repeated action sequence of length {period}"

        self.fingerprints.add(fingerprint)
        self.action_signatures.append(signature)
        return True, None
    
    def reset_fingerprints(self) -> None:
        """Reset the action fingerprint and sequence tracking."""
        self.fingerprints.clear()
        self.action_signatures.clear()
    
    def add_thought(self, thought: str) -> None:
        """
        Add a thought to the history for loop detection.
        
        Args:
            thought: The thought content to track
        """
        self.thought_history.append(thought)
        # Keep only the last N thoughts
        if len(self.thought_history) > self.max_thought_history:
            self.thought_history.pop(0)
    
    def check_thought_loop(self, thought: str) -> tuple[bool, Optional[str]]:
        """
        Check for repeated reasoning patterns in thoughts.
        
        Allows one repeat before triggering to avoid false positives
        on models that may restate their reasoning.
        
        Args:
            thought: Current thought to check
            
        Returns:
            Tuple of (should_continue, error_message)
        """
        # Hash the current thought
        thought_hash = hashlib.sha256(thought.encode()).hexdigest()
        
        # Count exact matches in history
        exact_matches = 0
        max_similarity = 0.0
        
        for i, historical_thought in enumerate(self.thought_history):
            historical_hash = hashlib.sha256(historical_thought.encode()).hexdigest()

            # Check for exact match — one repeat is tolerated, a second
            # identical repeat (3rd occurrence) is a loop.
            if thought_hash == historical_hash:
                exact_matches += 1
                if exact_matches >= 2:
                    return False, f"Thought loop detected: identical thought repeated {exact_matches} times"
                continue  # identical is handled by the counter above

            # Check for high similarity (simple check: if they share >90% of words)
            thought_words = set(thought.lower().split())
            historical_words = set(historical_thought.lower().split())
            
            if thought_words and historical_words:
                intersection = thought_words & historical_words
                union = thought_words | historical_words
                similarity = len(intersection) / len(union)
                max_similarity = max(max_similarity, similarity)
                
                if similarity > 0.9:
                    return False, f"Thought loop detected: highly similar thought (similarity: {similarity:.2f})"
        
        # Add thought to history
        self.add_thought(thought)
        
        return True, None
    
    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate a tool call against all guardrails.
        
        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for loops
        should_continue, error = self.check_loop(tool_name, arguments)
        if not should_continue:
            return False, error
        
        return True, None
    
    def get_fingerprint_count(self) -> int:
        """
        Get the number of unique fingerprints tracked.
        
        Returns:
            Number of unique fingerprints
        """
        return len(self.fingerprints)
