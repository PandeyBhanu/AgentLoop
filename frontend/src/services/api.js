const API_BASE = '/api';
const API_KEY = import.meta.env.VITE_AGENTLOOP_API_KEY || localStorage.getItem('agentloop-api-key') || '';

const authHeaders = (headers = {}) => (
  API_KEY ? { ...headers, 'X-API-Key': API_KEY } : headers
);

export const api = {
  async createRun(message, maxSteps = 10, allowedTools = null, maxTokenBudget = null) {
    const payload = { message, max_steps: maxSteps };
    if (allowedTools && allowedTools.length > 0) payload.allowed_tools = allowedTools;
    if (maxTokenBudget) payload.max_token_budget = maxTokenBudget;
    const response = await fetch(`${API_BASE}/runs`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return response.json();
  },

  async cancelRun(runId) {
    const response = await fetch(`${API_BASE}/runs/${runId}/cancel`, {
      method: 'POST',
      headers: authHeaders(),
    });
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return response.json();
  },

  async chat(message, maxSteps = 10, allowedTools = null, maxTokenBudget = null) {
    const payload = {
      message,
      max_steps: maxSteps,
    };

    if (allowedTools && allowedTools.length > 0) {
      payload.allowed_tools = allowedTools;
    }

    if (maxTokenBudget) {
      payload.max_token_budget = maxTokenBudget;
    }

    const response = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  },

  async getTrace(runId) {
    const response = await fetch(`${API_BASE}/trace/${runId}`, {
      headers: authHeaders(),
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  },

  async replay(runId) {
    const response = await fetch(`${API_BASE}/replay/${runId}`, {
      headers: authHeaders(),
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  },

  async replayWithAnalysis(runId) {
    const response = await fetch(`${API_BASE}/replay/${runId}/analysis`, {
      headers: authHeaders(),
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  },

  createEventSource(runId, onMessage, onError) {
    const controller = new AbortController();
    const decoder = new TextDecoder();
    let buffer = '';

    const readEvents = async () => {
      const response = await fetch(`${API_BASE}/stream/${runId}`, {
        headers: authHeaders({ Accept: 'text/event-stream' }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const reader = response.body.getReader();

      while (true) {
        const { done, value } = await reader.read();
        if (done) return;
        buffer += decoder.decode(value, { stream: true });
        let boundary;
        while ((boundary = buffer.indexOf('\n\n')) !== -1) {
          const frame = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          const data = frame
            .split('\n')
            .filter((line) => line.startsWith('data:'))
            .map((line) => line.slice(5).trim())
            .join('\n');
          if (data) onMessage(JSON.parse(data));
        }
      }
    };

    readEvents().catch((error) => {
      if (!controller.signal.aborted && onError) onError(error);
    });

    return {
      close: () => controller.abort(),
      get readyState() {
        return controller.signal.aborted ? 2 : 0;
      },
    };
  },
};
