# ReAct Agent Framework

A production-quality minimal ReAct agent framework built from scratch in Python. Implements the Thought → Action → Observation loop with tool calling, guardrails, and real-time streaming.

## Features

- **ReAct Pattern**: Full implementation of Thought → Action → Observation loop
- **Tool System**: Modular tool registry with JSON Schema validation
- **Guardrails**: Loop detection, step limits, and timeout protection
- **Trace Logging**: Complete execution trace with timestamps and token usage
- **SSE Streaming**: Real-time streaming of agent execution
- **Async Design**: Fully async/await throughout
- **Type Safety**: Pydantic v2 models with full type hints
- **FastAPI Backend**: RESTful API with health checks

## Project Structure

```
/app
  /agent
    loop.py          # Core ReAct agent loop
    parser.py        # LLM response parser
    state.py         # Agent state management
    guardrails.py    # Safety checks and loop detection
  /tools
    base.py          # Base tool class
    registry.py      # Tool registry
    calculator.py    # Calculator tool
    web_search.py    # Web search tool (mock)
    file_reader.py   # File reader tool
    python_exec.py   # Python code executor
  /api
    routes.py        # FastAPI routes
    sse.py           # Server-Sent Events streaming
  /schemas
    messages.py      # Message and response schemas
    tools.py         # Tool validation schemas
  /utils
    logger.py        # Logging utilities
    hashing.py       # Hashing for loop detection
llm.py              # LLM client wrapper
main.py             # Application entry point
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export OPENAI_API_KEY="your-api-key"
```

## Running the Server

Start the FastAPI server:
```bash
python main.py
```

The server will start on `http://localhost:8000`

## API Endpoints

### POST /api/chat
Process a user message through the agent loop.

**Request:**
```json
{
  "message": "What is 15 * 7?",
  "max_steps": 10
}
```

**Response:**
```json
{
  "run_id": "uuid-here",
  "final_answer": "The answer is 105",
  "steps": 2
}
```

### GET /api/trace/{run_id}
Get the full execution trace for a run.

**Response:**
```json
{
  "run_id": "uuid-here",
  "conversation_history": [...],
  "trace": [
    {
      "type": "thought",
      "content": "I need to calculate 15 * 7",
      "timestamp": "2024-01-01T00:00:00",
      "tool_name": null
    },
    {
      "type": "action",
      "content": "Calling Calculator",
      "tool_name": "Calculator",
      "timestamp": "2024-01-01T00:00:01"
    },
    {
      "type": "observation",
      "content": "{\"result\": 105}",
      "tool_name": "Calculator",
      "timestamp": "2024-01-01T00:00:01",
      "execution_time": 0.05
    }
  ],
  "current_step": 2,
  "is_finished": true
}
```

### GET /api/stream
Stream agent execution in real-time using Server-Sent Events.

**Query Parameters:**
- `message`: User message to process
- `max_steps`: Maximum steps (default: 10)

**Events:**
```json
{"type": "thought", "content": "...", "step": 1}
{"type": "action", "content": "...", "tool_name": "...", "step": 2}
{"type": "observation", "content": "...", "tool_name": "...", "step": 2}
{"type": "final", "final_answer": "...", "steps": 2}
```

### GET /api/health
Health check endpoint.

## Available Tools

1. **Calculator**: Perform basic arithmetic (add, subtract, multiply, divide)
2. **WebSearch**: Search the web (mock implementation)
3. **FileReader**: Read text files from the local filesystem
4. **PythonExec**: Execute Python code in a sandboxed environment

## Example Usage

### Using curl

```bash
# Chat endpoint
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Calculate 25 * 4", "max_steps": 10}'

# Stream endpoint
curl "http://localhost:8000/api/stream?message=Calculate%2025%20*%204&max_steps=10"
```

### Using Python

```python
import httpx

async def chat(message: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/chat",
            json={"message": message, "max_steps": 10}
        )
        return response.json()

result = await chat("What is 100 / 5?")
print(result)
```

## Configuration

### LLM Configuration

Edit `app/api/routes.py` to configure the LLM client:

```python
def get_llm_client() -> LLMClient:
    return LLMClient(
        api_key="your-key",
        base_url="https://api.openai.com/v1",
        model="gpt-4",
        temperature=0.7
    )
```

### Guardrails Configuration

Configure guardrails in `app/agent/loop.py`:

```python
agent_loop = AgentLoop(
    llm_client=llm_client,
    tool_registry=tool_registry,
    max_steps=10,        # Maximum agent steps
    tool_timeout=30      # Tool execution timeout (seconds)
)
```

## Design Principles

- **No Frameworks**: Built without LangChain, LlamaIndex, or similar frameworks
- **Modular**: Clean separation of concerns with minimal coupling
- **Debuggable**: Full trace logging for every execution step
- **Production-Ready**: Error handling, guardrails, and async design throughout
- **Extensible**: Easy to add new tools and modify behavior

## License

MIT
