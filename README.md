# ReAct Agent Framework

A minimal ReAct agent framework built from scratch in Python — no LangChain, LlamaIndex, or other agent frameworks. Implements the Thought → Action → Observation loop with validated tool calling, guardrails, full execution traces, and replay.

## Features

- **ReAct Pattern**: Thought → Action → Observation loop with explicit state machine and termination reasons
- **Tool System**: Registry with JSON Schema argument validation, per-run `allowed_tools` authorization, and execution timeouts
- **Guardrails**: Repeated-tool-call and repeated-thought detection, step limits, token budgets
- **Trace Logging**: Complete execution trace with timestamps, latency, and token usage
- **Replay**: Re-run analysis of completed executions with latency statistics
- **SSE Streaming**: Replays a completed run's trace as Server-Sent Events
- **Structured Errors**: Typed error entries (`tool_execution`, `parsing`, `llm`, `guardrail`, `timeout`) that don't corrupt agent state
- **Async Design**: Fully async/await throughout
- **Type Safety**: Pydantic v2 models with full type hints

See `AGENT_LOOP.md` for the state-machine diagram, transition invariants, and termination semantics.

## Project Structure

```
/app
  /agent
    loop.py          # Core ReAct agent loop
    parser.py        # LLM response parser (strict JSON + repair)
    state.py         # Agent state, trace entries, termination reasons
    guardrails.py    # Step limits, loop detection, timeouts
    memory.py        # Conversation memory
    replay.py        # Run replay and analysis
  /tools
    base.py          # Base tool class
    registry.py      # Tool registry
    calculator.py    # Calculator tool
    web_search.py    # Web search tool (mock)
    file_reader.py   # File reader (workspace-jailed)
    python_exec.py   # Python executor (disabled by default)
  /api
    routes.py        # FastAPI routes
    sse.py           # Server-Sent Events streaming
  /schemas
    messages.py      # Message and response schemas
    tools.py         # Tool validation schemas
  llm.py             # Multi-provider LLM client (OpenAI/Groq/Gemini)
  storage.py         # In-memory run store
main.py              # FastAPI app entry point
/frontend            # React 18 + Vite + TailwindCSS UI
/tests               # pytest suite (~210 tests)
```

## Installation

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # for running tests
```

## Configuration

All configuration is via environment variables (loaded from `.env`):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai`, `groq`, or `gemini` |
| `OPENAI_API_KEY` | — | Required when provider is `openai` |
| `GROQ_API_KEY` | — | Required when provider is `groq` |
| `GEMINI_API_KEY` | — | Required when provider is `gemini` |
| `AGENTLOOP_API_KEY` | unset | If set, all endpoints except `/` and `/api/health` require the `X-API-Key` header |
| `AGENTLOOP_ENABLE_PYTHON_EXEC` | unset | Set to `1` to enable the PythonExec tool (see Security Notes) |
| `CORS_ORIGINS` | `localhost:3000` origins | Comma-separated allowed origins |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Server bind address |

Defaults per provider: OpenAI `gpt-4`, Groq `llama-3.1-8b-instant`, Gemini `gemini-pro`.

## Running

Backend:

```bash
python main.py    # http://127.0.0.1:8000
```

Frontend:

```bash
cd frontend && npm install && npm run dev
```

## API Endpoints

### POST /api/chat

Process a user message through the agent loop.

**Request:**
```json
{
  "message": "What is 15 * 7?",
  "max_steps": 10,
  "allowed_tools": ["Calculator"],
  "max_token_budget": 50000
}
```

- `message` (required): 1–20,000 chars
- `max_steps`: 1–50, default 10
- `allowed_tools`: optional whitelist; omit to allow all registered tools
- `max_token_budget`: optional cap; run terminates with `budget_exceeded` when exceeded

**Response:**
```json
{
  "run_id": "uuid-here",
  "final_answer": "The answer is 105",
  "steps": 2,
  "total_tokens": 842,
  "total_cost": 0.012,
  "error": null,
  "termination_reason": "completed"
}
```

`termination_reason` is one of `completed`, `error`, `max_steps`, `budget_exceeded`, `loop_detected`.

### GET /api/trace/{run_id}

Full execution trace for a run: conversation history, trace entries (thought/action/observation/error), step count, token usage, termination reason. Returns 404 for unknown run IDs.

### GET /api/replay/{run_id}

Replays a stored run's trace. Returns `success`, `steps`, `final_answer`, `error`.

### GET /api/replay/{run_id}/analysis

Replay plus analysis: per-step latency stats, tool-call counts, trace data, termination reason.

### GET /api/stream/{run_id}

Streams a **completed** run's trace as Server-Sent Events (step, tool_call, metrics, finish). This replays stored results — it is not live execution streaming.

### GET /api/health

Health check. Always open, even when `AGENTLOOP_API_KEY` is set.

## Available Tools

1. **Calculator**: Basic arithmetic (add, subtract, multiply, divide)
2. **WebSearch**: Web search (mock implementation)
3. **FileReader**: Reads text files, jailed to a workspace root — blocks `..` traversal, absolute-path escapes, symlinks, hidden files (`.env`, `.git`), and files over 1 MB
4. **PythonExec**: Executes whitelisted Python expressions. **Disabled by default** — set `AGENTLOOP_ENABLE_PYTHON_EXEC=1` to enable. See Security Notes before enabling.

## Example Usage

```bash
# Chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Calculate 25 * 4", "max_steps": 10}'

# Stream the completed run's trace
curl http://localhost:8000/api/stream/<run_id>

# With API key auth enabled
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"message": "Hello"}'
```

## Testing

```bash
python -m pytest tests/ -q                                              # full suite
python -m pytest tests/ -q --cov=app --cov=main --cov-report=term-missing # with coverage
python -m pytest tests/test_security.py -q                              # security tests only
```

The suite (~210 tests, ~92% coverage) uses a scripted mock LLM and mocked httpx — it never calls real LLM APIs. Covers parser, tools/registry, guardrails, agent loop, API/SSE/replay, provider-specific message assembly, and security regressions.

## Security Notes

- **PythonExec is not a sandbox.** Even with its AST whitelist, it runs in-process. Keep it disabled for untrusted input; move execution to a subprocess/container if you need real isolation.
- **FileReader** is jailed to its workspace but prompt injection can steer reads anywhere *inside* it.
- **Auth is opt-in** — without `AGENTLOOP_API_KEY`, trace/replay endpoints expose full run contents to anyone with a run_id.
- **Run storage is unbounded in-memory** — restarts clear it; there's no persistence or eviction.
- CORS uses an explicit origin allowlist with credentials disabled; `max_steps` and message length are bounded at the schema level.

## Design Principles

- **No Frameworks**: Built without LangChain, LlamaIndex, or similar
- **Modular**: Clean separation of concerns with minimal coupling
- **Debuggable**: Full trace logging and replayable runs
- **Extensible**: Register new tools via `ToolRegistry` with a JSON Schema
