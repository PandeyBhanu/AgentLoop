"""
API tests for the FastAPI surface: /chat, /trace, /health, /replay, /stream.

All agent runs use ScriptedLLM patched into app.api.routes.llm_client —
no real LLM calls are ever made.
"""
import json

import pytest
from fastapi.testclient import TestClient

import app.api.routes as routes
from conftest import (
    ScriptedLLM,
    action_response,
    finish_response,
    thought_response,
)


@pytest.fixture
def app():
    from main import app as fastapi_app

    return fastapi_app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def mock_llm(monkeypatch):
    """Patch the module-level llm_client used by the routes."""

    def install(responses, metadata=None):
        fake = ScriptedLLM(responses, metadata=metadata)
        monkeypatch.setattr(routes, "llm_client", fake)
        return fake

    return install


def parse_sse(body: str):
    """Parse an SSE response body into a list of JSON event dicts."""
    events = []
    for block in body.split("\n\n"):
        block = block.strip()
        if block.startswith("data:"):
            events.append(json.loads(block[len("data:"):].strip()))
    return events


def run_chat(client, message="hello", **kwargs):
    return client.post("/api/chat", json={"message": message, **kwargs})


# ---------------------------------------------------------------------------
# POST /api/chat
# ---------------------------------------------------------------------------

class TestChat:
    def test_happy_path(self, client, mock_llm):
        mock_llm([finish_response("all done")])
        r = run_chat(client, "say hi")

        assert r.status_code == 200
        data = r.json()
        assert data["final_answer"] == "all done"
        assert data["steps"] == 1
        assert data["run_id"]
        assert data["error"] is None

    def test_run_stored_for_trace(self, client, mock_llm):
        mock_llm([finish_response("stored")])
        run_id = run_chat(client).json()["run_id"]

        r = client.get(f"/api/trace/{run_id}")
        assert r.status_code == 200
        trace = r.json()
        assert trace["run_id"] == run_id
        assert trace["final_answer"] == "stored"
        assert len(trace["trace"]) == 1

    def test_full_agent_run_via_api(self, client, mock_llm):
        mock_llm([
            thought_response("need math"),
            action_response("Calculator", {"operation": "multiply", "a": 6, "b": 7}),
            finish_response("42"),
        ])
        r = run_chat(client, "what is 6*7?")
        data = r.json()
        assert data["final_answer"] == "42"
        assert data["steps"] == 3

    def test_llm_failure_returns_error_field(self, client, mock_llm):
        mock_llm([RuntimeError("provider exploded")])
        r = run_chat(client)

        assert r.status_code == 200  # error is reported in-band
        data = r.json()
        assert data["error"] is not None
        assert "Error:" in data["final_answer"]

    def test_missing_message_422(self, client):
        r = client.post("/api/chat", json={})
        assert r.status_code == 422

    def test_wrong_type_422(self, client):
        r = client.post("/api/chat", json={"message": "hi", "max_steps": "abc"})
        assert r.status_code == 422

    def test_allowed_tools_filtering(self, client, mock_llm):
        fake = mock_llm([finish_response("x")])
        run_chat(client, allowed_tools=["Calculator"])

        names = [s["name"] for s in fake.calls[0]["tool_schemas"]]
        assert names == ["Calculator"]

    def test_token_budget_enforced_via_api(self, client, mock_llm):
        from conftest import make_metadata

        mock_llm(
            [thought_response("t"), finish_response("x")],
            metadata=make_metadata(total_tokens=500),
        )
        r = run_chat(client, max_token_budget=100)
        data = r.json()
        assert data["error"] is not None
        assert "budget" in data["error"].lower()


# ---------------------------------------------------------------------------
# GET /api/trace/{run_id}
# ---------------------------------------------------------------------------

class TestTrace:
    def test_unknown_run_404(self, client):
        r = client.get("/api/trace/definitely-not-a-run")
        assert r.status_code == 404

    def test_trace_contains_conversation(self, client, mock_llm):
        mock_llm([finish_response("x")])
        run_id = run_chat(client, "my question").json()["run_id"]

        trace = client.get(f"/api/trace/{run_id}").json()
        contents = [m["content"] for m in trace["conversation_history"]]
        assert "my question" in contents


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"


# ---------------------------------------------------------------------------
# Replay endpoints
# ---------------------------------------------------------------------------

class TestReplay:
    def test_replay_completed_run(self, client, mock_llm):
        mock_llm([
            action_response("Calculator", {"operation": "add", "a": 1, "b": 1}),
            finish_response("two"),
        ])
        run_id = run_chat(client).json()["run_id"]

        r = client.get(f"/api/replay/{run_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["final_answer"] == "two"
        assert data["steps"] > 0

    def test_replay_unknown_run(self, client):
        r = client.get("/api/replay/nope")
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    @pytest.mark.xfail(
        reason="BUG: replay_with_analysis 500s whenever any trace entry has "
               "latency_ms — analysis['latency_stats'] is initialized to {} "
               "(replay.py:109) so the lazy-init 'not in' check at line 135 "
               "never fires and stats['min'] raises KeyError",
        strict=False,
    )
    def test_replay_analysis(self, client, mock_llm):
        mock_llm([
            action_response("Calculator", {"operation": "add", "a": 1, "b": 1}),
            finish_response("two"),
        ])
        run_id = run_chat(client).json()["run_id"]

        r = client.get(f"/api/replay/{run_id}/analysis")
        assert r.status_code == 200
        data = r.json()
        assert data["run_id"] == run_id
        assert data["replay_success"] is True
        assert "steps_by_type" in data
        assert data["steps_by_type"].get("finish") == 1
        assert len(data["tool_calls"]) == 1

    def test_replay_analysis_unknown_run(self, client):
        r = client.get("/api/replay/nope/analysis")
        assert r.status_code == 200
        assert "error" in r.json()

    @pytest.mark.xfail(
        reason="BUG: replay_with_analysis response does not include a 'trace' "
               "key — frontend ReplayControls reads analysis.trace and gets "
               "undefined, crashing the replay UI",
        strict=False,
    )
    def test_replay_analysis_includes_trace(self, client, mock_llm):
        mock_llm([finish_response("x")])
        run_id = run_chat(client).json()["run_id"]
        data = client.get(f"/api/replay/{run_id}/analysis").json()
        assert "trace" in data


# ---------------------------------------------------------------------------
# SSE endpoints
# ---------------------------------------------------------------------------

class TestSSE:
    def test_stream_replays_trace(self, client, mock_llm):
        mock_llm([
            action_response("Calculator", {"operation": "add", "a": 1, "b": 1}),
            finish_response("two"),
        ])
        run_id = run_chat(client).json()["run_id"]

        r = client.get(f"/api/stream/{run_id}")
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]

        events = parse_sse(r.text)
        types = [e["type"] for e in events]
        assert "step" in types
        assert "metrics" in types
        assert types[-1] == "finish"
        assert events[-1]["run_id"] == run_id
        assert events[-1]["final_answer"] == "two"

    def test_stream_emits_tool_call_events(self, client, mock_llm):
        mock_llm([
            action_response("Calculator", {"operation": "add", "a": 1, "b": 1}),
            finish_response("two"),
        ])
        run_id = run_chat(client).json()["run_id"]

        events = parse_sse(client.get(f"/api/stream/{run_id}").text)
        tool_calls = [e for e in events if e["type"] == "tool_call"]
        assert tool_calls
        assert tool_calls[0]["tool_call"]["tool_name"] == "Calculator"

    def test_stream_unknown_run(self, client):
        r = client.get("/api/stream/does-not-exist")
        events = parse_sse(r.text)
        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert "not found" in events[0]["error"].lower()

    def test_stream_direct_stub_returns_error(self, client):
        r = client.get("/api/stream", params={"message": "hi"})
        events = parse_sse(r.text)
        assert events[0]["type"] == "error"
        assert "not supported" in events[0]["error"].lower()
