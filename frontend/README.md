# ReAct Agent Frontend Dashboard

A production-quality React frontend for visualizing ReAct agent execution in real-time.

## Features

- **Real-time Trace Visualization**: Stream agent execution steps via Server-Sent Events (SSE)
- **Chat Interface**: User input and conversation history
- **Tool Execution Log**: Detailed view of tool calls, arguments, and results
- **Metrics Tracking**: Live token usage, cost estimation, and step count
- **Replay Functionality**: Step-by-step replay of previous runs with play/pause controls
- **Tool Access Control**: Select which tools the agent can use before sending a request
- **Error Visualization**: Structured error display with stage information
- **Color-coded Steps**: Thought (blue), Action (yellow), Observation (green), Error (red)

## Tech Stack

- React 18 with Vite
- Functional components + hooks
- Tailwind CSS
- Native EventSource (SSE)

## Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will run on `http://localhost:3000` with API proxying to the backend on `http://localhost:8000`.

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ChatPanel.jsx          # Chat interface
│   │   ├── TracePanel.jsx        # Real-time trace viewer
│   │   ├── ToolLog.jsx           # Tool execution details
│   │   ├── MetricsBar.jsx        # Token/cost metrics
│   │   └── ReplayControls.jsx    # Replay functionality
│   ├── services/
│   │   └── api.js                # API service module
│   ├── App.jsx                   # Main application
│   ├── main.jsx                  # Entry point
│   └── index.css                 # Global styles
├── index.html
├── vite.config.js
├── tailwind.config.js
└── package.json
```

## Component Overview

### ChatPanel
- User input with send button
- Chat history with user/assistant messages
- Disabled state while agent is running

### TracePanel
- Real-time streaming of agent steps
- Color-coded by step type
- Shows step_number, tool_name, latency_ms, tokens
- Auto-scrolls to latest step
- Error details display with stage

### ToolLog
- Tool execution history
- Shows tool_name, arguments, execution_time
- Success/failure status
- Error messages

### MetricsBar
- Total tokens used
- Estimated cost
- Current step count
- Running indicator

### ReplayControls
- Load previous run by run_id
- Play/pause controls
- Step progress bar
- Reset functionality

## API Integration

The frontend integrates with the backend APIs:

- `POST /api/chat` - Send user message
- `GET /api/stream/{run_id}` - SSE streaming
- `GET /api/trace/{run_id}` - Get full trace
- `GET /api/replay/{run_id}` - Replay run
- `GET /api/replay/{run_id}/analysis` - Replay with analysis

## Usage

### Sending a Message
1. Type your message in the chat input
2. Optionally select allowed tools using checkboxes
3. Optionally set a max token budget
4. Click "Send"

### Viewing Real-time Trace
- The trace panel automatically updates as the agent executes
- Steps are color-coded by type
- Scroll to see all steps

### Replaying a Run
1. Click "Show Replay" button
2. Enter a run ID from a previous execution
3. Click "Load"
4. Use Play/Pause/Reset controls to step through

### Tool Access Control
- Check/uncheck tools in the header
- Only selected tools will be available to the agent
- Empty selection = all tools available

## Visual Semantics

| Type | Color | Description |
|------|-------|-------------|
| Thought | Blue | Agent reasoning |
| Action | Yellow | Tool call |
| Observation | Green | Tool result |
| Error | Red | Failure |
| Finish | White/Bold | Final answer |

## Error Handling

Errors are displayed with:
- Type (error)
- Stage (tool_execution, parsing, llm, guardrail, timeout)
- Message
- Recoverable flag

## Backend Integration

The frontend requires the backend to be running on `http://localhost:8000`. The Vite proxy configuration handles CORS automatically.

## Development

Build for production:
```bash
npm run build
```

Preview production build:
```bash
npm run preview
```

## Notes

- Tailwind CSS warnings during development are expected and will resolve after `npm install`
- SSE streaming requires the backend to support the `/api/stream/{run_id}` endpoint
- Run IDs are returned in the chat response and can be used for replay
