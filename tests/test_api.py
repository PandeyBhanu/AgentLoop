"""
API tests for the FastAPI surface: /chat, /trace, /health, /replay, /stream.

All agent runs use ScriptedLLM patched into app.api.routes.llm_client —
no real LLM calls are ever made.
"""
import json
import asyncio

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
        assert data["termination_reason"] == "completed"
        assert data["total_cost"] is not None

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
        assert data["termination_reason"] == "budget_exceeded"

    def test_shared_concurrency_limit_applies_to_chat_and_live_runs(self, client, monkeypatch):
        monkeypatch.setattr(routes, "MAX_ACTIVE_RUNS", 0)
        assert run_chat(client).status_code == 429
        assert client.post("/api/runs", json={"message": "hi"}).status_code == 429


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

    def test_metrics_are_exposed_after_run(self, client, mock_llm):
        mock_llm([finish_response("done")])
        run_chat(client)
        r = client.get("/api/metrics")
        assert r.status_code == 200
        data = r.json()
        assert data["counters"]["agent.runs.started"] >= 1
        assert any(key.startswith("agent.run.duration_seconds") for key in data["observations"])


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


class TestLiveSSE:
    async def test_step_published_before_run_finishes(self, monkeypatch):
        from app.api.sse import create_run, live_runs
        from app.schemas.messages import ChatRequest
        from types import SimpleNamespace
        http_request = SimpleNamespace(state=SimpleNamespace(user_id="development", allowed_tools=["Calculator"]))
        gate = asyncio.Event()

        class PausedLLM:
            def __init__(self):
                self.calls = 0

            async def send_messages(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return thought_response("first step"), {"total_tokens": 5}
                await gate.wait()
                return finish_response("done"), {"total_tokens": 5}

        monkeypatch.setattr(routes, "llm_client", PausedLLM())
        run_id = (await create_run(ChatRequest(message="go"), http_request))["run_id"]
        run = live_runs[run_id]
        try:
            await asyncio.sleep(0.2)
            assert not run.finished
            assert any(event["type"] == "step" for event in run.events)
            gate.set()
            await run.task
            assert run.state.run_id == run_id
            assert run.events[-1]["type"] == "finish"
        finally:
            if not run.finished:
                run.task.cancel()
                await asyncio.gather(run.task, return_exceptions=True)
            live_runs.pop(run_id, None)

    async def test_cancel_running_run(self, monkeypatch):
        from app.api.sse import create_run, cancel_run, live_runs
        from app.schemas.messages import ChatRequest
        from types import SimpleNamespace
        http_request = SimpleNamespace(state=SimpleNamespace(user_id="development", allowed_tools=["Calculator"]))

        class PausedLLM:
            async def send_messages(self, **kwargs):
                await asyncio.Event().wait()

        monkeypatch.setattr(routes, "llm_client", PausedLLM())
        run_id = (await create_run(ChatRequest(message="go"), http_request))["run_id"]
        run = live_runs[run_id]
        await asyncio.sleep(0)
        await cancel_run(run_id, http_request)
        await asyncio.gather(run.task, return_exceptions=True)
        assert run.finished
        assert run.events[-1]["termination_reason"] == "cancelled"
        live_runs.pop(run_id, None)

    def test_live_endpoint_has_run_id(self, client, mock_llm):
        mock_llm([finish_response("done")])
        with TestClient(client.app) as browser:
            r = browser.post("/api/runs", json={"message": "hello"})
            assert r.status_code == 202
            run_id = r.json()["run_id"]
            events = parse_sse(browser.get(f"/api/stream/{run_id}").text)
            assert events[-1]["type"] == "finish"
            assert events[-1]["run_id"] == run_id

    async def test_live_generator_emits_event_before_run_finishes(self, monkeypatch):
        from app.api.sse import create_run, live_events, live_runs
        from app.schemas.messages import ChatRequest
        from types import SimpleNamespace

        http_request = SimpleNamespace(
            state=SimpleNamespace(user_id="development", allowed_tools=["Calculator"])
        )
        gate = asyncio.Event()

        class PausedLLM:
            def __init__(self):
                self.calls = 0

            async def send_messages(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return thought_response("first"), {"total_tokens": 5}
                await gate.wait()
                return finish_response("done"), {"total_tokens": 5}

        async def not_disconnected():
            return False

        monkeypatch.setattr(routes, "llm_client", PausedLLM())
        run_id = (await create_run(ChatRequest(message="go"), http_request))["run_id"]
        run = live_runs[run_id]
        stream = live_events(run, SimpleNamespace(is_disconnected=not_disconnected))
        try:
            first_frame = await asyncio.wait_for(anext(stream), 2)
            first_event = json.loads(first_frame[5:].strip())
            assert first_event["type"] == "llm"
            assert not run.finished
            gate.set()
            last_event = None
            while last_event is None or last_event["type"] != "finish":
                frame = await asyncio.wait_for(anext(stream), 2)
                last_event = json.loads(frame[5:].strip())
            assert last_event["run_id"] == run_id
        finally:
            gate.set()
            if not run.finished:
                run.task.cancel()
                await asyncio.gather(run.task, return_exceptions=True)
            await stream.aclose()
            live_runs.pop(run_id, None)
