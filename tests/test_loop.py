"""
Tests for app.agent.loop — AgentLoop driven by a deterministic ScriptedLLM.

No real LLM API is ever called; ScriptedLLM replays a fixed script of
responses (or raises exceptions) and records every call.
"""
import asyncio

import pytest

from app.agent.loop import AgentLoop
from app.tools.base import BaseTool
from app.tools.calculator import Calculator
from app.tools.registry import ToolRegistry

from conftest import (
    ScriptedLLM,
    action_response,
    finish_response,
    make_metadata,
    thought_response,
)


def make_registry(*tools):
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def make_loop(responses, registry=None, metadata=None, **kwargs):
    llm = ScriptedLLM(responses, metadata=metadata)
    loop = AgentLoop(
        llm_client=llm,
        tool_registry=registry or make_registry(Calculator()),
        **kwargs,
    )
    return loop, llm


class SlowTool(BaseTool):
    def get_description(self):
        return "sleeps"

    def get_input_schema(self):
        return {"type": "object", "properties": {}}

    async def execute(self, arguments):
        await asyncio.sleep(5)
        return {"ok": True}


class EchoTool(BaseTool):
    """No required arguments — used for the allowed_tools bypass test."""

    def get_description(self):
        return "echoes"

    def get_input_schema(self):
        return {"type": "object", "properties": {}}

    async def execute(self, arguments):
        return {"echo": arguments}


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

class TestHappyPath:
    async def test_thought_action_observation_finish(self):
        loop, llm = make_loop([
            thought_response("I need to add numbers"),
            action_response("Calculator", {"operation": "add", "a": 2, "b": 3}),
            finish_response("The answer is 5"),
        ])
        state = await loop.run("What is 2 + 3?")

        assert state.is_finished
        assert state.error is None
        assert state.final_answer == "The answer is 5"
        assert state.current_step == 3

    async def test_direct_finish(self):
        loop, _ = make_loop([finish_response("immediate")])
        state = await loop.run("hi")
        assert state.is_finished
        assert state.final_answer == "immediate"
        assert state.current_step == 1

    async def test_multiple_sequential_tool_calls(self):
        loop, _ = make_loop([
            action_response("Calculator", {"operation": "add", "a": 1, "b": 1}),
            action_response("Calculator", {"operation": "multiply", "a": 2, "b": 5}),
            finish_response("1+1=2 and 2*5=10"),
        ])
        state = await loop.run("compute things")

        assert state.is_finished
        types = [e.type for e in state.trace]
        assert types == ["action", "observation", "action", "observation", "finish"]

    async def test_observation_content_in_trace(self):
        loop, _ = make_loop([
            action_response("Calculator", {"operation": "multiply", "a": 6, "b": 7}),
            finish_response("42"),
        ])
        state = await loop.run("6*7?")

        obs = [e for e in state.trace if e.type == "observation"]
        assert len(obs) == 1
        assert "'result': 42" in obs[0].content or '"result": 42' in obs[0].content
        assert obs[0].tool_name == "Calculator"
        assert obs[0].execution_time is not None

    async def test_observation_added_to_conversation(self):
        loop, _ = make_loop([
            action_response("Calculator", {"operation": "add", "a": 1, "b": 2}),
            finish_response("3"),
        ])
        state = await loop.run("add")
        contents = [m.content for m in state.conversation_history]
        assert any("Observation:" in c and "result" in c for c in contents)

    async def test_trace_ordering(self):
        loop, _ = make_loop([
            thought_response("first thought"),
            action_response("Calculator"),
            finish_response("fin"),
        ])
        state = await loop.run("go")
        assert [e.type for e in state.trace] == [
            "thought", "action", "observation", "finish",
        ]
        # Step numbers are monotonically non-decreasing
        steps = [e.step_number for e in state.trace]
        assert steps == sorted(steps)

    @pytest.mark.xfail(
        reason="BUG: on_step is never invoked for the step that produces "
               "'finish' — loop.py breaks out of the while loop before the "
               "on_step call, so streaming consumers miss the final step",
        strict=False,
    )
    async def test_on_step_callback_async(self):
        loop, _ = make_loop([finish_response("done")])
        seen_steps = []

        async def cb(state):
            seen_steps.append(state.current_step)

        await loop.run("go", on_step=cb)
        assert seen_steps == [1]

    async def test_tokens_and_cost_tracked(self):
        meta = make_metadata(prompt_tokens=100, completion_tokens=50, cost=0.01)
        loop, _ = make_loop([finish_response("x")], metadata=meta)
        state = await loop.run("go")
        assert state.total_tokens >= 150
        assert state.total_cost == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Conversation construction
# ---------------------------------------------------------------------------

class TestConversation:
    async def test_system_prompt_is_first_message(self):
        loop, llm = make_loop([finish_response("x")])
        await loop.run("hello")
        roles = [m.role for m in llm.calls[0]["messages"]]
        assert roles[0] == "system"

    async def test_user_message_present(self):
        loop, llm = make_loop([finish_response("x")])
        await loop.run("hello world")
        contents = [m.content for m in llm.calls[0]["messages"]]
        assert "hello world" in contents

    async def test_tool_schemas_sent_to_llm(self):
        loop, llm = make_loop([finish_response("x")])
        await loop.run("go")
        schemas = llm.calls[0]["tool_schemas"]
        assert schemas and schemas[0]["name"] == "Calculator"

    async def test_allowed_tools_filters_schemas(self):
        reg = make_registry(Calculator(), EchoTool())
        loop, llm = make_loop(
            [finish_response("x")], registry=reg, allowed_tools=["Calculator"]
        )
        await loop.run("go")
        names = [s["name"] for s in llm.calls[0]["tool_schemas"]]
        assert names == ["Calculator"]


# ---------------------------------------------------------------------------
# Error / recovery paths
# ---------------------------------------------------------------------------

class TestErrorPaths:
    async def test_unknown_tool_recovers(self):
        # NB: arguments must be non-empty — {} is rejected by the parser
        loop, _ = make_loop([
            action_response("NoSuchTool", {"q": 1}),
            finish_response("recovered"),
        ])
        state = await loop.run("go")

        assert state.is_finished
        assert state.final_answer == "recovered"
        errors = [e for e in state.trace if e.type == "error"]
        assert errors and "Unknown tool" in errors[0].content
        assert errors[0].error_details["stage"] == "tool_execution"
        assert errors[0].error_details["recoverable"] is False

    async def test_invalid_arguments_recovers(self):
        loop, _ = make_loop([
            action_response("Calculator", {"operation": "add"}),  # missing 'a'
            finish_response("recovered"),
        ])
        state = await loop.run("go")

        assert state.is_finished
        errors = [e for e in state.trace if e.type == "error"]
        assert errors and "Invalid arguments" in errors[0].content
        assert errors[0].error_details["recoverable"] is True

    async def test_tool_execution_exception(self):
        class BoomTool(BaseTool):
            def get_description(self):
                return "boom"

            def get_input_schema(self):
                return {"type": "object", "properties": {}}

            async def execute(self, arguments):
                raise RuntimeError("tool exploded")

        reg = make_registry(BoomTool())
        loop, _ = make_loop(
            [action_response("BoomTool", {"x": 1}), finish_response("recovered")],
            registry=reg,
        )
        state = await loop.run("go")

        assert state.is_finished
        errors = [e for e in state.trace if e.type == "error"]
        assert errors and "tool exploded" in errors[0].content
        assert errors[0].error_details["stage"] == "tool_execution"
        assert errors[0].error_details["recoverable"] is True

    async def test_tool_timeout(self):
        reg = make_registry(SlowTool())
        loop, _ = make_loop(
            [action_response("SlowTool", {"x": 1}), finish_response("after timeout")],
            registry=reg,
            tool_timeout=1,
        )
        state = await loop.run("go")

        assert state.is_finished
        errors = [e for e in state.trace if e.type == "error"]
        assert errors and "timed out" in errors[0].content
        assert errors[0].error_details["stage"] == "timeout"

    async def test_tool_error_result_becomes_observation(self):
        # Calculator returns {"error": ...} for div-by-zero — an observation,
        # not an error trace entry.
        loop, _ = make_loop([
            action_response("Calculator", {"operation": "divide", "a": 1, "b": 0}),
            finish_response("handled"),
        ])
        state = await loop.run("go")

        assert state.is_finished
        obs = [e for e in state.trace if e.type == "observation"]
        assert obs and "Division by zero" in obs[0].content

    async def test_llm_api_failure_fails_run(self):
        loop, llm = make_loop([RuntimeError("API down")])
        state = await loop.run("go")

        assert state.is_finished
        assert state.error is not None
        assert len(llm.calls) == 3  # 1 attempt + 2 retries

    async def test_llm_timeout_fails_run(self):
        loop, llm = make_loop([asyncio.TimeoutError("llm timeout")])
        state = await loop.run("go")

        assert state.is_finished
        assert state.error is not None
        assert len(llm.calls) == 3

    async def test_invalid_llm_json_then_recovery(self):
        # _call_llm_with_retry retries on invalid JSON (3 identical attempts),
        # then parse_with_retry produces a thought fallback; next step finishes.
        loop, _ = make_loop([
            "garbage not json",
            "garbage not json",
            "garbage not json",
            finish_response("recovered"),
        ])
        state = await loop.run("go")

        assert state.is_finished
        assert state.final_answer == "recovered"
        errors = [e for e in state.trace if e.type == "error"]
        assert errors and errors[0].error_details["stage"] == "parsing"

    async def test_llm_missing_type_retries(self):
        loop, llm = make_loop([
            '{"thought": "no type field"}',
            finish_response("recovered"),
        ])
        state = await loop.run("go")

        # First call missing "type" triggers retry; second call finishes.
        assert len(llm.calls) == 2
        assert state.final_answer == "recovered"


# ---------------------------------------------------------------------------
# Termination conditions
# ---------------------------------------------------------------------------

class TestTermination:
    async def test_max_steps_termination(self):
        # Distinct thoughts so thought-loop detection doesn't fire first
        loop, _ = make_loop(
            [thought_response(f"unique thought {i}") for i in range(5)],
            max_steps=3,
        )
        state = await loop.run("go")

        assert state.is_finished
        assert state.error is not None
        assert "max steps" in state.error.lower()
        assert state.current_step == 3

    async def test_token_budget_termination(self):
        meta = make_metadata(total_tokens=100)
        loop, _ = make_loop(
            [thought_response("t"), finish_response("x")],
            metadata=meta,
            max_token_budget=50,
        )
        state = await loop.run("go")

        assert state.is_finished
        assert "budget" in state.error.lower()

    async def test_tool_loop_detection_terminates(self):
        # Same tool + same args twice in a row -> fingerprint loop
        repeated = action_response("Calculator", {"operation": "add", "a": 1, "b": 1})
        loop, _ = make_loop([repeated, repeated, finish_response("unreachable")])
        state = await loop.run("go")

        assert state.is_finished
        assert state.error is not None
        assert "loop" in state.error.lower()

    async def test_thought_loop_terminates(self):
        same = thought_response("identical reasoning")
        loop, _ = make_loop([same] * 5)
        state = await loop.run("go")

        assert state.is_finished
        assert "loop" in state.error.lower()


# ---------------------------------------------------------------------------
# Tool access control
# ---------------------------------------------------------------------------

class TestAllowedTools:
    async def test_disallowed_tool_not_executed(self):
        reg = make_registry(Calculator(), EchoTool())
        loop, _ = make_loop(
            [
                action_response("EchoTool", {"a": 1}),
                finish_response("done"),
            ],
            registry=reg,
            allowed_tools=["Calculator"],
        )
        state = await loop.run("go")

        # EchoTool call must be rejected as an error, never executed.
        assert not any(
            e.type == "observation" and e.tool_name == "EchoTool"
            for e in state.trace
        )
        errors = [e for e in state.trace if e.type == "error"]
        assert errors and "not allowed" in errors[0].content.lower()

    async def test_allowed_tool_executes(self):
        reg = make_registry(Calculator(), EchoTool())
        loop, _ = make_loop(
            [
                action_response("Calculator", {"operation": "add", "a": 1, "b": 1}),
                finish_response("done"),
            ],
            registry=reg,
            allowed_tools=["Calculator"],
        )
        state = await loop.run("go")
        assert state.is_finished
        assert any(e.type == "observation" for e in state.trace)
