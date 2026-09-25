"""
Tests for app.agent.guardrails — step limits, loop detection, thought loops.
"""
import pytest

from app.agent.guardrails import Guardrails


@pytest.fixture
def guardrails():
    return Guardrails(max_steps=3, tool_timeout=5)


class TestStepLimit:
    def test_below_limit(self, guardrails):
        ok, err = guardrails.check_step_limit(0)
        assert ok
        assert err is None

    def test_at_limit(self, guardrails):
        ok, err = guardrails.check_step_limit(3)
        assert not ok
        assert "step limit" in err.lower()

    def test_above_limit(self, guardrails):
        ok, err = guardrails.check_step_limit(99)
        assert not ok


class TestToolCallLoopDetection:
    def test_unique_calls_pass(self, guardrails):
        ok1, _ = guardrails.check_loop("Calculator", {"operation": "add", "a": 1, "b": 2})
        ok2, _ = guardrails.check_loop("Calculator", {"operation": "add", "a": 3, "b": 4})
        ok3, _ = guardrails.check_loop("WebSearch", {"query": "x"})
        assert ok1 and ok2 and ok3

    def test_identical_call_detected(self, guardrails):
        guardrails.check_loop("Calculator", {"operation": "add", "a": 1, "b": 2})
        ok, err = guardrails.check_loop("Calculator", {"operation": "add", "a": 1, "b": 2})
        assert not ok
        assert "loop detected" in err.lower()

    def test_argument_order_insensitive(self, guardrails):
        guardrails.check_loop("Calculator", {"a": 1, "operation": "add", "b": 2})
        ok, _ = guardrails.check_loop("Calculator", {"b": 2, "operation": "add", "a": 1})
        assert not ok

    def test_same_tool_different_args_passes(self, guardrails):
        guardrails.check_loop("Calculator", {"operation": "add", "a": 1, "b": 2})
        ok, _ = guardrails.check_loop("Calculator", {"operation": "subtract", "a": 1, "b": 2})
        assert ok

    def test_reset_fingerprints(self, guardrails):
        guardrails.check_loop("Calculator", {"operation": "add", "a": 1, "b": 2})
        guardrails.reset_fingerprints()
        ok, _ = guardrails.check_loop("Calculator", {"operation": "add", "a": 1, "b": 2})
        assert ok

    def test_validate_tool_call_delegates(self, guardrails):
        ok, _ = guardrails.validate_tool_call("Calculator", {"a": 1})
        assert ok
        ok, err = guardrails.validate_tool_call("Calculator", {"a": 1})
        assert not ok


class TestThoughtLoopDetection:
    def test_unique_thoughts_pass(self, guardrails):
        for i in range(5):
            ok, _ = guardrails.check_thought_loop(f"completely different idea {i}")
            assert ok

    @pytest.mark.xfail(
        reason="BUG: docstring says 'allows one repeat' but the similarity check "
               "(>0.9) fires on the first identical repeat since identical "
               "thoughts score 1.0 — one repeat is never actually allowed",
        strict=False,
    )
    def test_single_repeat_allowed(self, guardrails):
        # Documented behavior: first repeat is tolerated to avoid false positives
        assert guardrails.check_thought_loop("same thought")[0]
        assert guardrails.check_thought_loop("same thought")[0]

    def test_third_identical_thought_detected(self, guardrails):
        guardrails.check_thought_loop("same thought")
        guardrails.check_thought_loop("same thought")
        ok, err = guardrails.check_thought_loop("same thought")
        assert not ok
        assert "loop" in err.lower()

    def test_similar_thought_detected(self, guardrails):
        # 10 shared words, 1 extra -> Jaccard similarity 10/11 ≈ 0.91 > 0.9
        guardrails.check_thought_loop("one two three four five six seven eight nine ten")
        ok, err = guardrails.check_thought_loop(
            "one two three four five six seven eight nine ten extra"
        )
        assert not ok
        assert "similar" in err.lower()

    def test_history_is_bounded(self):
        g = Guardrails(max_thought_history=3)
        for i in range(10):
            g.add_thought(f"thought {i}")
        assert len(g.thought_history) == 3
        assert g.thought_history[-1] == "thought 9"

    def test_dissimilar_thoughts_pass(self, guardrails):
        guardrails.check_thought_loop("alpha beta gamma delta epsilon zeta")
        ok, _ = guardrails.check_thought_loop("completely unrelated content about other topics")
        assert ok
