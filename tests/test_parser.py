"""
Tests for app.agent.parser — ResponseParser and repair_llm_output.
"""
import pytest

from app.agent.parser import ResponseParser, repair_llm_output


# ---------------------------------------------------------------------------
# ResponseParser.parse — valid responses
# ---------------------------------------------------------------------------

class TestParseValid:
    def test_thought(self):
        r = ResponseParser.parse('{"type": "thought", "thought": "I need to think"}')
        assert r.type == "thought"
        assert r.thought == "I need to think"
        assert r.tool_name is None
        assert r.arguments is None

    def test_action(self):
        r = ResponseParser.parse(
            '{"type": "action", "tool_name": "Calculator", '
            '"arguments": {"operation": "add", "a": 1, "b": 2}}'
        )
        assert r.type == "action"
        assert r.tool_name == "Calculator"
        assert r.arguments == {"operation": "add", "a": 1, "b": 2}

    def test_finish(self):
        r = ResponseParser.parse('{"type": "finish", "final_answer": "done"}')
        assert r.type == "finish"
        assert r.final_answer == "done"

    def test_ignores_extra_fields(self):
        r = ResponseParser.parse(
            '{"type": "thought", "thought": "t", "extra": "ignored"}'
        )
        assert r.type == "thought"
        assert r.thought == "t"

    def test_surrounding_whitespace(self):
        r = ResponseParser.parse('   \n{"type": "finish", "final_answer": "x"}\n  ')
        assert r.type == "finish"


# ---------------------------------------------------------------------------
# ResponseParser.parse — invalid responses
# ---------------------------------------------------------------------------

class TestParseInvalid:
    def test_malformed_json(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            ResponseParser.parse("{not valid json")

    def test_missing_type(self):
        with pytest.raises(ValueError, match="missing 'type'"):
            ResponseParser.parse('{"thought": "hello"}')

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid response type"):
            ResponseParser.parse('{"type": "bogus"}')

    def test_action_missing_tool_name(self):
        with pytest.raises(ValueError, match="missing 'tool_name'"):
            ResponseParser.parse('{"type": "action", "arguments": {"a": 1}}')

    def test_action_missing_arguments(self):
        with pytest.raises(ValueError, match="missing 'arguments'"):
            ResponseParser.parse('{"type": "action", "tool_name": "Calculator"}')

    def test_finish_missing_final_answer(self):
        with pytest.raises(ValueError, match="missing 'final_answer'"):
            ResponseParser.parse('{"type": "finish"}')

    def test_thought_missing_thought(self):
        with pytest.raises(ValueError, match="missing 'thought'"):
            ResponseParser.parse('{"type": "thought"}')

    def test_empty_string(self):
        with pytest.raises(ValueError):
            ResponseParser.parse("")

    @pytest.mark.xfail(
        reason="BUG: valid non-object JSON (list/scalar) raises AttributeError "
               "instead of ValueError — data.get() is called on a list",
        strict=False,
    )
    def test_json_array_not_object(self):
        # A valid JSON array has no "type" field
        with pytest.raises(ValueError):
            ResponseParser.parse('["thought", "action"]')

    @pytest.mark.xfail(
        reason="BUG: valid non-object JSON (list/scalar) raises AttributeError "
               "instead of ValueError — data.get() is called on a list",
        strict=False,
    )
    def test_json_scalar_not_object(self):
        with pytest.raises(ValueError):
            ResponseParser.parse('42')

    @pytest.mark.xfail(
        reason="BUG: parser rejects empty arguments dict — `if not data.get('arguments')` "
               "treats {} as missing; tools with no required inputs can never be called",
        strict=False,
    )
    def test_action_with_empty_arguments(self):
        r = ResponseParser.parse('{"type": "action", "tool_name": "Ping", "arguments": {}}')
        assert r.arguments == {}


# ---------------------------------------------------------------------------
# Markdown code fences
# ---------------------------------------------------------------------------

class TestMarkdownFences:
    def test_json_fence(self):
        r = ResponseParser.parse(
            '```json\n{"type": "finish", "final_answer": "x"}\n```'
        )
        assert r.type == "finish"
        assert r.final_answer == "x"

    def test_plain_fence(self):
        r = ResponseParser.parse(
            '```\n{"type": "thought", "thought": "t"}\n```'
        )
        assert r.type == "thought"

    def test_text_around_json(self):
        # Strict parse fails, but fence extraction recovers the JSON
        r = ResponseParser.parse(
            'Here is my response:\n```json\n'
            '{"type": "finish", "final_answer": "done"}\n```\nDone.'
        )
        assert r.type == "finish"


# ---------------------------------------------------------------------------
# repair_llm_output
# ---------------------------------------------------------------------------

class TestRepair:
    def test_trailing_comma(self):
        repaired = repair_llm_output('{"type": "finish", "final_answer": "x",}')
        r = ResponseParser.parse(repaired)
        assert r.type == "finish"

    def test_unquoted_keys(self):
        repaired = repair_llm_output('{type: "finish", final_answer: "x"}')
        r = ResponseParser.parse(repaired)
        assert r.type == "finish"

    def test_unbalanced_brace(self):
        repaired = repair_llm_output('{"type": "finish", "final_answer": "x"')
        r = ResponseParser.parse(repaired)
        assert r.type == "finish"

    def test_comment_removal(self):
        repaired = repair_llm_output(
            '// leading comment\n{"type": "finish", "final_answer": "x"}'
        )
        r = ResponseParser.parse(repaired)
        assert r.type == "finish"

    def test_is_deterministic(self):
        raw = '{type: "finish",}'
        assert repair_llm_output(raw) == repair_llm_output(raw)


# ---------------------------------------------------------------------------
# parse_with_retry / parse_with_fallback
# ---------------------------------------------------------------------------

class TestParseWithRetry:
    def test_strict_parse_succeeds_first(self):
        r, ok, err = ResponseParser.parse_with_retry(
            '{"type": "finish", "final_answer": "x"}'
        )
        assert ok is True
        assert err == ""
        assert r.type == "finish"

    def test_repair_path_succeeds(self):
        r, ok, err = ResponseParser.parse_with_retry(
            '{"type": "finish", "final_answer": "x",}'  # trailing comma
        )
        assert ok is True
        assert r.type == "finish"

    def test_unrepairable_returns_thought_fallback(self):
        r, ok, err = ResponseParser.parse_with_retry("this is not json at all")
        assert ok is False
        assert err != ""
        assert r.type == "thought"
        assert "Parsing error" in r.thought

    def test_semantically_invalid_json_fails(self):
        # Valid JSON, but missing required fields — repair cannot fix this
        r, ok, err = ResponseParser.parse_with_retry('{"type": "action"}')
        assert ok is False
        assert r.type == "thought"

    def test_parse_with_fallback(self):
        r = ResponseParser.parse_with_fallback("garbage!!")
        assert r.type == "thought"
        assert r.thought == "garbage!!"

    def test_parse_with_fallback_valid(self):
        r = ResponseParser.parse_with_fallback(
            '{"type": "finish", "final_answer": "x"}'
        )
        assert r.type == "finish"
