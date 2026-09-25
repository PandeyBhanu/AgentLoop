"""
Shared fixtures and fakes for the AgentLoop test suite.

No test in this suite is allowed to call a real LLM API. All LLM
interactions go through ScriptedLLM, a deterministic stand-in for
app.llm.LLMClient.
"""
import json
import pytest

from app.tools.registry import ToolRegistry
from app.tools.calculator import Calculator


def make_metadata(
    prompt_tokens=10,
    completion_tokens=5,
    total_tokens=None,
    latency_ms=1.0,
    cost=0.001,
    provider="mock",
    model="mock-model",
):
    """Build a token-metadata dict in the shape returned by LLMClient."""
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_ms": latency_ms,
        "estimated_cost": cost,
        "provider": provider,
        "model": model,
    }


def thought_response(text="Thinking..."):
    return json.dumps({"type": "thought", "thought": text})


def action_response(tool_name="Calculator", arguments=None):
    if arguments is None:
        arguments = {"operation": "add", "a": 2, "b": 3}
    return json.dumps(
        {"type": "action", "tool_name": tool_name, "arguments": arguments}
    )


def finish_response(answer="The answer is 42"):
    return json.dumps({"type": "finish", "final_answer": answer})


class ScriptedLLM:
    """
    Deterministic stand-in for app.llm.LLMClient.

    `responses` is a list consumed in order. Each item is either:
      - a str            -> returned as (text, metadata)
      - an Exception     -> raised (to simulate API failures/timeouts)

    Once the script is exhausted the last item repeats, so an agent
    that keeps looping keeps getting the same final response.
    Every call is recorded in self.calls for assertions.
    """

    def __init__(self, responses, metadata=None):
        assert responses, "ScriptedLLM needs at least one scripted response"
        self.responses = list(responses)
        self.metadata = metadata or make_metadata()
        self.calls = []

    async def send_messages(self, messages, tool_schemas=None, json_mode=True, max_retries=2):
        idx = len(self.calls)
        self.calls.append(
            {
                "messages": list(messages),
                "tool_schemas": tool_schemas,
                "json_mode": json_mode,
            }
        )
        item = self.responses[min(idx, len(self.responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item, dict(self.metadata)


@pytest.fixture
def registry():
    """Tool registry with only the Calculator registered."""
    reg = ToolRegistry()
    reg.register(Calculator())
    return reg


@pytest.fixture(autouse=True)
def clean_runs():
    """Isolate the global run store between tests."""
    from app.storage import runs

    runs.clear()
    yield
    runs.clear()
