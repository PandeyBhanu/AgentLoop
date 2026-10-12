# AgentLoop

AgentLoop is a lightweight Python ReAct-style agent framework built from scratch. It implements a `Thought -> Action -> Observation` loop with strict JSON parsing, tool validation, guardrails, tracing, replay, and SSE streaming. The project is intentionally framework-free: there is no LangChain, no LlamaIndex, and no hidden orchestration layer.

The current codebase contains the core agent runtime, a FastAPI API, a React frontend, SQLite-backed run persistence, and a small tool system for arithmetic, web search, file reads, and optionally isolated Python execution.

## What the project does

AgentLoop can:

- accept a user request and reason through multiple steps using an LLM
- restrict tool use based on server-side policies and per-request `allowed_tools`
- validate tool inputs against JSON Schema before execution
- stream progress events while a run is in flight
- persist full run traces and replay them after completion
- enforce guardrails such as max-step limits, token budgets, cost budgets, timeouts, and repeated-thought/tool loops
- expose a simple browser UI for chatting with the agent

## Current implementation

The implementation is organized like this:

```text
app/
  agent/
    loop.py          # ReAct execution loop and run lifecycle
    parser.py        # strict JSON parser with repair logic
    state.py         # agent state and termination reasons
    guardrails.py    # step limits, loop detection, timeouts
    memory.py        # conversation memory
    replay.py        # run replay and analysis helpers
  api/
    routes.py        # FastAPI endpoints for chat, trace, replay, health, metrics
    sse.py           # SSE stream manager and live run subscribers
  tools/
    base.py          # base tool contract
    registry.py      # tool registration and lookup
    calculator.py    # arithmetic tool
    web_search.py    # configurable SearXNG-compatible search tool
    file_reader.py   # workspace-jail file reader
    python_exec.py   # optional isolated Python executor
  schemas/
    messages.py      # request/response schema models
    tools.py         # tool parameter validation helpers
  llm.py             # provider adapters for OpenAI, Groq, and Gemini
  storage.py         # SQLite-backed run store
  utils/
    logger.py        # metrics and logging helpers
frontend/            # React + Vite UI for interacting with the API
main.py              # FastAPI app entry point and auth middleware
tests/               # deterministic regression and security tests
```

## Core runtime behavior

The runtime is built around `AgentLoop` and `AgentState`:

- each run starts with a `system` message and a `user` message
- the model emits either `thought`, `action`, or `finish`
- `action` calls are checked in a fixed order: tool exists, tool is allowed, schema is valid, fingerprint is not a repeated loop, then tool executes
- every step is recorded in a structured trace with token/cost/latency metadata
- a run terminates with a defined reason such as `completed`, `error`, `max_steps`, `budget_exceeded`, `cost_exceeded`, `run_timeout`, or `loop_detected`

The project also intentionally preserves runtime observability via stored traces, replay, and metrics snapshots.

## Features

- ReAct agent loop with explicit state transitions
- JSON schema validation for tool arguments
- Tool allowlists and per-user API-key policies
- Guardrails for repeated tool/thought loops, step caps, token caps, and max runtime
- Full run traces stored in SQLite and replayable after completion
- Live event streaming through SSE for active runs
- Async LLM provider support across OpenAI, Groq, and Gemini
- Server-side CORS protection and loopback-only development mode
- Optional Python execution runner that is disabled by default and isolated behind a separate worker image

## Installation

Create a Python environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

The repo also includes a Docker setup for the API and optional worker image.

## Configuration

Configuration is loaded from environment variables through `.env` files and the process environment.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai`, `groq`, or `gemini` |
| `OPENAI_API_KEY` | unset | API key for OpenAI |
| `GROQ_API_KEY` | unset | API key for Groq |
| `GEMINI_API_KEY` | unset | API key for Gemini |
| `AGENTLOOP_MODE` | `development` | `production` requires configured API keys |
| `AGENTLOOP_API_KEY` | unset | Shared API key for single-key mode |
| `AGENTLOOP_USERS_JSON` | unset | Mapping of user name to `{ "api_key": "...", "allowed_tools": [...] }` |
| `AGENTLOOP_ALLOWED_TOOLS` | `Calculator,WebSearch,FileReader` | Default server-side tool policy |
| `AGENTLOOP_DB_PATH` | `.agentloop/runs.sqlite3` | SQLite DB path for runs |
| `AGENTLOOP_MAX_ACTIVE_RUNS` | `20` | Max concurrently active runs |
| `SEARCH_PROVIDER` | unset | e.g. `searxng` |
| `SEARCH_BASE_URL` | unset | Base URL for the configured search service |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Allowed origins |
| `HOST` | `127.0.0.1` | API bind host |
| `PORT` | `8000` | API bind port |
| `FILE_READER_ROOT` | current working directory | Root directory for `FileReader` |
| `AGENTLOOP_ENABLE_PYTHON_EXEC` | unset | Enables optional Python execution path |

Example `.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
AGENTLOOP_MODE=development
AGENTLOOP_ALLOWED_TOOLS=Calculator,WebSearch,FileReader
SEARCH_PROVIDER=searxng
SEARCH_BASE_URL=https://search.example.com
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

## Running the project

Start the backend:

```bash
python main.py
```

The API runs at `http://127.0.0.1:8000`.

Start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Run the API via Docker:

```bash
docker build -t agentloop-api:local .
docker compose up --build -d
curl http://127.0.0.1:8000/api/health
```

The Compose setup keeps SQLite on a named volume and runs the API in a non-root container. Python execution is intentionally disabled in the standard Compose API service; it requires a separate host-side worker setup.

## API overview

### `POST /api/chat`

Starts a synchronous run and returns the final result.

Request body:

```json
{
  "message": "What is 15 * 7?",
  "max_steps": 10,
  "allowed_tools": ["Calculator"],
  "max_token_budget": 50000,
  "max_tool_calls": 20,
  "max_run_seconds": 120,
  "max_cost_budget": 0.05
}
```

Response includes:

```json
{
  "run_id": "uuid",
  "final_answer": "The answer is 105",
  "steps": 2,
  "total_tokens": 842,
  "total_cost": 0.012,
  "error": null,
  "termination_reason": "completed"
}
```

### `GET /api/trace/{run_id}`

Returns the full trace for a stored run.

### `GET /api/replay/{run_id}`

Replays a stored run and returns `success`, `steps`, `final_answer`, and `error`.

### `GET /api/replay/{run_id}/analysis`

Returns replay results plus latency statistics, traces, and termination metadata.

### `POST /api/runs`

Creates a background run and returns a `run_id` immediately. Use `GET /api/stream/{run_id}` to subscribe to live step events.

### `POST /api/runs/{run_id}/cancel`

Requests cancellation of a live run owned by the caller.

### `GET /api/stream/{run_id}`

Streams step, tool, LLM, metrics, and finish events for a run.

### `GET /api/metrics`

Returns a JSON snapshot of in-memory counters and run metrics.

### `GET /api/health`

Basic health check endpoint.

## Available tools

1. `Calculator`
   - basic arithmetic operations
2. `WebSearch`
   - configured search provider via `SEARCH_PROVIDER` and `SEARCH_BASE_URL`
   - intentionally fails cleanly when not configured rather than fabricating results
3. `FileReader`
   - reads text files inside a configured workspace root
   - blocks traversal outside the root, hidden files/directories, symlinks, and unsupported binary types
4. `PythonExec`
   - disabled by default
   - available only when explicitly enabled and when the environment supports an isolated worker image

## Security model

The project includes several protections:

- `FileReader` enforces workspace containment and blocks `..`, absolute escapes, hidden paths, and large files
- `WebSearch` validates result URLs and rejects unsafe or non-global hosts
- API auth is enforced in production and loopback-only access is allowed in development when no credentials are configured
- tool execution is gated by allowlists and JSON Schema validation
- Python execution is not performed in-process by default; it runs through an isolated worker if enabled
- CORS is restricted to an explicit allowlist rather than `*`

## Testing

The project includes regression, API, parser, storage, security, and tool tests.

```bash
python -m pytest tests/ -q
python -m pytest tests/ -q --cov=app --cov=main --cov-report=term-missing
python -m pytest tests/test_security.py -q
```

The suite uses mocked LLM/provider responses and includes checks for SSE streaming, cancellation, replay, persistence, and security boundaries. Docker-dependent behavior is skipped when Docker is not available.

## Design principles

- no framework lock-in
- explicit state transitions instead of hidden orchestration
- modular runtime boundaries between agent, tools, API, and storage
- replayability and traceability as first-class features
- safe tool execution through validation and hard limits

## Related notes

For deeper implementation details, see:

- `decisions.md` for the state machine and design notes
- `AGENT_LOOP.md` for the formal loop semantics and invariant checks
- `SECURITY_FIXES.md` and `TECHNICAL_AUDIT.md` for security and implementation follow-up notes
