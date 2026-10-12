"""
Focused security tests for AgentLoop.

Covers: FileReader workspace jail (traversal, absolute paths, symlinks,
hidden files, extensions, size), PythonExec disable-by-default and AST
whitelist, tool authorization (allowed_tools enforced at execution),
request validation bounds, optional API-key auth, and leakage of
sensitive filesystem details in error responses.
"""
import asyncio
import importlib
import json
import math
import os
import sys
import subprocess
import shutil

import pytest
from fastapi.testclient import TestClient

from app.tools.file_reader import FileReader
from app.tools.python_exec import PythonExec
from app.tools.base import BaseTool
from app.tools.registry import ToolRegistry
from app.agent.loop import AgentLoop

from conftest import (
    ScriptedLLM,
    action_response,
    finish_response,
)


# ---------------------------------------------------------------------------
# FileReader — workspace jail
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path):
    """A workspace root containing an innocent file and a 'secret' outside."""
    (tmp_path / "ok.txt").write_text("safe content", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "inner.txt").write_text("inner", encoding="utf-8")
    return tmp_path


@pytest.fixture
def reader(workspace):
    return FileReader(workspace_root=str(workspace))


class TestFileReaderJail:
    async def test_reads_inside_root(self, reader):
        r = await reader.execute({"file_path": "ok.txt"})
        assert r["content"] == "safe content"

    @pytest.mark.parametrize("path", [
        "../outside.txt",
        "../../outside.txt",
        "..\\outside.txt",                 # Windows separator
        "subdir/../../outside.txt",
        "subdir/../..",
    ])
    async def test_traversal_variants_blocked(self, reader, path):
        r = await reader.execute({"file_path": path})
        assert "error" in r

    async def test_traversal_never_returns_content(self, reader, workspace):
        secret = workspace.parent / f"secret_{os.getpid()}.txt"
        secret.write_text("TOPSECRET", encoding="utf-8")
        try:
            r = await reader.execute({"file_path": f"../{secret.name}"})
            assert "error" in r
            assert "TOPSECRET" not in json.dumps(r)
        finally:
            secret.unlink()

    async def test_absolute_path_outside_root_blocked(self, reader, tmp_path):
        # tmp_path sibling — absolute path outside the workspace root.
        outside = tmp_path.parent / "abs_secret.txt"
        outside.write_text("SECRET", encoding="utf-8")
        try:
            r = await reader.execute({"file_path": str(outside)})
            assert "error" in r
        finally:
            outside.unlink()

    async def test_symlink_escape_blocked(self, reader, workspace, tmp_path):
        target = tmp_path.parent / "symlink_target.txt"
        target.write_text("SECRET", encoding="utf-8")
        link = workspace / "link.txt"
        try:
            os.symlink(str(target), str(link))
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this platform")
        try:
            r = await reader.execute({"file_path": "link.txt"})
            assert "error" in r
            assert "SECRET" not in json.dumps(r)
        finally:
            target.unlink()

    async def test_hidden_files_blocked(self, reader, workspace):
        (workspace / ".env").write_text("API_KEY=SECRET", encoding="utf-8")
        r = await reader.execute({"file_path": ".env"})
        assert "error" in r
        assert "SECRET" not in json.dumps(r)

    async def test_hidden_dir_blocked(self, reader, workspace):
        (workspace / ".git").mkdir()
        (workspace / ".git" / "config").write_text("secret", encoding="utf-8")
        r = await reader.execute({"file_path": ".git/config"})
        assert "error" in r

    async def test_blocked_extension(self, reader, workspace):
        (workspace / "evil.exe").write_bytes(b"MZ")
        r = await reader.execute({"file_path": "evil.exe"})
        assert "error" in r

    async def test_oversized_file_rejected(self, reader, workspace, monkeypatch):
        monkeypatch.setattr("app.tools.file_reader.MAX_FILE_BYTES", 10)
        (workspace / "big.txt").write_text("x" * 100, encoding="utf-8")
        r = await reader.execute({"file_path": "big.txt"})
        assert "error" in r
        assert "large" in r["error"].lower()

    async def test_errors_do_not_leak_resolved_paths(self, reader, workspace):
        r = await reader.execute({"file_path": "../outside.txt"})
        assert "error" in r
        # The resolved absolute path / workspace root must not appear.
        assert str(workspace.resolve()) not in json.dumps(r)

    async def test_empty_path_rejected(self, reader):
        r = await reader.execute({"file_path": "   "})
        assert "error" in r


# ---------------------------------------------------------------------------
# PythonExec — disabled by default + AST whitelist
# ---------------------------------------------------------------------------

class TestPythonExecDisabled:
    async def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("AGENTLOOP_ENABLE_PYTHON_EXEC", raising=False)
        r = await PythonExec().execute({"code": "2 + 2"})
        assert r["success"] is False
        assert "disabled" in r["error"].lower()

    async def test_disabled_response_is_structured(self, monkeypatch):
        monkeypatch.delenv("AGENTLOOP_ENABLE_PYTHON_EXEC", raising=False)
        r = await PythonExec().execute({"code": "2 + 2"})
        assert set(r.keys()) == {"error", "details", "success"}


@pytest.fixture
def pyexec(monkeypatch):
    monkeypatch.setenv("AGENTLOOP_ENABLE_PYTHON_EXEC", "1")
    return PythonExec()


class TestPythonExecWhitelist:
    @pytest.mark.parametrize("escape", [
        "().__class__.__base__.__subclasses__()",
        "().__class__.__mro__",
        "''.__class__",
        "(1).__class__.__init__.__globals__",
        "getattr((), '__class__')",
        "__builtins__",
        "__import__('os')",
        "open('x')",
        "exec('1')",
        "eval('1')",
        "[x for x in ()][0].__class__",
    ])
    async def test_escape_attempts_blocked(self, pyexec, escape):
        r = await pyexec.execute({"code": escape})
        assert r["success"] is False, f"escape not blocked: {escape}"

    @pytest.mark.parametrize("stmt", [
        "import os",
        "x = 1",
        "def f(): pass",
        "lambda x: x",
        "while True: pass",
        "for i in []: pass",
        "class A: pass",
    ])
    async def test_statements_rejected(self, pyexec, stmt):
        r = await pyexec.execute({"code": stmt})
        assert r["success"] is False

    @pytest.mark.parametrize("expr,expected", [
        ("2 + 2", "4"),
        ("cos(0)", "1.0"),
        ("math.sqrt(16)", "4.0"),
        ("sum(x*x for x in range(4))", "14"),
        ("sorted([3,1,2])", "[1, 2, 3]"),
        ("abs(-5)", "5"),
        ("pi", str(math.pi)),
    ])
    async def test_legitimate_expressions_work(self, pyexec, expr, expected):
        if not shutil.which("docker") or subprocess.run(["docker", "image", "inspect", "agentloop-python-worker:local"], capture_output=True).returncode:
            pytest.skip("Docker worker image unavailable")
        r = await pyexec.execute({"code": expr})
        assert r["success"] is True
        assert r["output"] == expected

    async def test_output_capped(self, pyexec, monkeypatch):
        async def oversized(code, timeout):
            return "Output limit exceeded"
        monkeypatch.setattr(pyexec, "_run_container", oversized)
        r = await pyexec.execute({"code": "'x' * 5000"})
        assert r["success"] is False
        assert "limit" in r["details"].lower()

    async def test_timeout_rejected(self, pyexec):
        # Excessive timeout values must be rejected, not trusted.
        r = await pyexec.execute({"code": "1 + 1", "timeout": 999999})
        assert r["success"] is False

    async def test_no_stdout_leak(self, pyexec, capfd):
        if not shutil.which("docker") or subprocess.run(["docker", "image", "inspect", "agentloop-python-worker:local"], capture_output=True).returncode:
            pytest.skip("Docker worker image unavailable")
        await pyexec.execute({"code": 'print("SHOULD_BE_CAPTURED")'})
        out, _ = capfd.readouterr()
        assert "SHOULD_BE_CAPTURED" not in out


# ---------------------------------------------------------------------------
# Tool authorization — allowed_tools enforced at execution time
# ---------------------------------------------------------------------------

class DangerousTool(BaseTool):
    """Registered tool that must NOT run when not in allowed_tools."""

    executed = False

    def get_description(self):
        return "dangerous"

    def get_input_schema(self):
        return {"type": "object", "properties": {}}

    async def execute(self, arguments):
        DangerousTool.executed = True
        return {"pwned": True}


class TestToolAuthorization:
    async def test_registered_but_unauthorized_tool_cannot_execute(self):
        DangerousTool.executed = False
        reg = ToolRegistry()
        reg.register(DangerousTool())

        llm = ScriptedLLM([
            action_response("DangerousTool", {"x": 1}),
            finish_response("done"),
        ])
        loop = AgentLoop(
            llm_client=llm, tool_registry=reg, allowed_tools=["Calculator"]
        )
        state = await loop.run("go")

        assert DangerousTool.executed is False
        assert not any(e.type == "observation" for e in state.trace)
        errors = [e for e in state.trace if e.type == "error"]
        assert errors and "not allowed" in errors[0].content.lower()

    async def test_unknown_tool_rejected(self):
        reg = ToolRegistry()
        llm = ScriptedLLM([
            action_response("GhostTool", {"x": 1}),
            finish_response("done"),
        ])
        loop = AgentLoop(llm_client=llm, tool_registry=reg)
        state = await loop.run("go")
        errors = [e for e in state.trace if e.type == "error"]
        assert errors and "unknown tool" in errors[0].content.lower()

    async def test_malformed_arguments_rejected(self):
        reg = ToolRegistry()
        reg.register(DangerousTool())
        # Schema requires object; arguments are validated before execution.
        llm = ScriptedLLM([
            json.dumps({"type": "action", "tool_name": "DangerousTool",
                        "arguments": "not-a-dict"}),
            finish_response("done"),
        ])
        loop = AgentLoop(llm_client=llm, tool_registry=reg)
        state = await loop.run("go")
        assert DangerousTool.executed is False


# ---------------------------------------------------------------------------
# API hardening — request bounds, auth, CORS
# ---------------------------------------------------------------------------

class TestRequestBounds:
    @pytest.fixture
    def client(self):
        import main
        importlib.reload(main)  # ensure clean module state (no API key)
        return TestClient(main.app)

    def test_max_steps_too_high(self, client):
        r = client.post("/api/chat", json={"message": "hi", "max_steps": 999})
        assert r.status_code == 422

    def test_max_steps_zero(self, client):
        r = client.post("/api/chat", json={"message": "hi", "max_steps": 0})
        assert r.status_code == 422

    def test_message_too_long(self, client):
        r = client.post("/api/chat", json={"message": "x" * 20_001})
        assert r.status_code == 422

    def test_empty_message(self, client):
        r = client.post("/api/chat", json={"message": ""})
        assert r.status_code == 422

    def test_cors_allows_configured_origin(self, client):
        r = client.options(
            "/api/chat",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_cors_rejects_unknown_origin(self, client):
        r = client.options(
            "/api/chat",
            headers={
                "Origin": "http://evil.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert r.headers.get("access-control-allow-origin") != "http://evil.example.com"


class TestApiKeyAuth:
    @pytest.fixture
    def protected_client(self, monkeypatch):
        monkeypatch.setenv("AGENTLOOP_API_KEY", "test-secret-key")
        import main
        importlib.reload(main)
        yield TestClient(main.app)
        # Restore open app for subsequent tests
        monkeypatch.delenv("AGENTLOOP_API_KEY", raising=False)
        importlib.reload(main)

    def test_health_is_public(self, protected_client):
        assert protected_client.get("/api/health").status_code == 200

    def test_chat_requires_key(self, protected_client):
        r = protected_client.post("/api/chat", json={"message": "hi"})
        assert r.status_code == 401

    def test_wrong_key_rejected(self, protected_client):
        r = protected_client.post(
            "/api/chat", json={"message": "hi"}, headers={"X-API-Key": "wrong"}
        )
        assert r.status_code == 401

    def test_trace_requires_key(self, protected_client):
        r = protected_client.get("/api/trace/whatever")
        assert r.status_code == 401

    def test_401_does_not_leak_details(self, protected_client):
        r = protected_client.post("/api/chat", json={"message": "hi"})
        assert r.json() == {"detail": "Unauthorized"}


class TestRunOwnership:
    def test_users_cannot_read_or_cancel_each_others_runs(self, monkeypatch):
        monkeypatch.setenv("AGENTLOOP_USERS_JSON", json.dumps({
            "alice": {"api_key": "alice-test-key", "allowed_tools": ["Calculator"]},
            "bob": {"api_key": "bob-test-key", "allowed_tools": []},
        }))
        import main
        import app.api.routes as routes
        from conftest import ScriptedLLM
        monkeypatch.setattr(routes, "llm_client", ScriptedLLM([finish_response("ok")]))
        importlib.reload(main)
        with TestClient(main.app) as client:
            alice = {"X-API-Key": "alice-test-key"}
            bob = {"X-API-Key": "bob-test-key"}
            run_id = client.post("/api/chat", json={"message": "hi"}, headers=alice).json()["run_id"]
            assert client.get(f"/api/trace/{run_id}", headers=alice).status_code == 200
            assert client.get(f"/api/trace/{run_id}", headers=bob).status_code == 404
            assert client.get(f"/api/stream/{run_id}", headers=bob).status_code == 404
            assert client.get(f"/api/replay/{run_id}", headers=bob).status_code == 404
            live_id = client.post("/api/runs", json={"message": "hi"}, headers=alice).json()["run_id"]
            assert client.post(f"/api/runs/{live_id}/cancel", headers=bob).status_code == 404
        monkeypatch.delenv("AGENTLOOP_USERS_JSON")
        importlib.reload(main)

    def test_server_policy_does_not_grant_client_requested_tool(self, monkeypatch):
        monkeypatch.setenv("AGENTLOOP_USERS_JSON", json.dumps({
            "bob": {"api_key": "bob-test-key", "allowed_tools": []}
        }))
        import main
        import app.api.routes as routes
        from conftest import ScriptedLLM
        fake = ScriptedLLM([finish_response("ok")])
        monkeypatch.setattr(routes, "llm_client", fake)
        importlib.reload(main)
        with TestClient(main.app) as client:
            r = client.post("/api/chat", json={"message": "hi", "allowed_tools": ["Calculator"]},
                            headers={"X-API-Key": "bob-test-key"})
            assert r.status_code == 200
            assert fake.calls[0]["tool_schemas"] == []
        monkeypatch.delenv("AGENTLOOP_USERS_JSON")
        importlib.reload(main)


class TestPythonContainerIsolation:
    @pytest.fixture
    def worker(self):
        if not shutil.which("docker") or subprocess.run(
            ["docker", "image", "inspect", "agentloop-python-worker:local"], capture_output=True
        ).returncode:
            pytest.skip("Docker worker image unavailable")
        return PythonExec()

    async def probe(self, worker, script):
        import uuid
        proc = await asyncio.create_subprocess_exec(
            *worker._docker_args(f"agentloop-test-{uuid.uuid4().hex}"), "python", "-c", script,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(b""), 15)
        assert proc.returncode == 0, stderr.decode()
        return stdout.decode().strip()

    async def test_no_host_secrets_and_read_only_root(self, worker, monkeypatch):
        monkeypatch.setenv("AGENTLOOP_TEST_SECRET", "host-secret")
        output = await self.probe(worker,
            "import os; print(os.getenv('AGENTLOOP_TEST_SECRET', 'absent')); "
            "print(os.geteuid()); "
            "print('blocked' if not os.access('/etc', os.W_OK) else 'writable')")
        assert output.splitlines() == ["absent", "65534", "blocked"]

    async def test_network_disabled_and_cgroup_limits(self, worker):
        output = await self.probe(worker,
            "import socket; s=socket.socket(); "
            "print('blocked' if s.connect_ex(('198.51.100.1', 80)) != 0 else 'connected'); "
            "print(open('/sys/fs/cgroup/pids.max').read().strip()); "
            "print(open('/sys/fs/cgroup/memory.max').read().strip())")
        network, pids, memory = output.splitlines()
        assert network == "blocked"
        assert int(pids) <= 16
        assert int(memory) <= 128 * 1024 * 1024

    async def test_timeout_kills_worker_and_output_bounded(self, worker, monkeypatch):
        monkeypatch.setenv("AGENTLOOP_ENABLE_PYTHON_EXEC", "1")
        result = await worker.execute({"code": "'x' * 20000"})
        assert not result["success"] and "limit" in result["details"].lower()
        with pytest.raises(asyncio.TimeoutError):
            await worker._run_container("sum(x for x in range(1000000000))", 1)
        await asyncio.sleep(0.2)
        proc = await asyncio.create_subprocess_exec(
            "docker", "ps", "--filter", "name=agentloop-", "--format", "{{.Names}}",
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        import re
        assert not any(re.fullmatch(r"agentloop-[0-9a-f]{32}", name)
                       for name in stdout.decode().splitlines())
