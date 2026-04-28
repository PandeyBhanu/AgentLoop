const API_BASE = '/api';

export const api = {
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
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  },

  async getTrace(runId) {
    const response = await fetch(`${API_BASE}/trace/${runId}`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  },

  async replay(runId) {
    const response = await fetch(`${API_BASE}/replay/${runId}`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  },

  async replayWithAnalysis(runId) {
    const response = await fetch(`${API_BASE}/replay/${runId}/analysis`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  },

  createEventSource(runId, onMessage, onError) {
    const eventSource = new EventSource(`${API_BASE}/stream/${runId}`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (e) {
        console.error('Failed to parse SSE message:', e);
      }
    };

    eventSource.onerror = (error) => {
      console.error('SSE error:', error);
      if (onError) onError(error);
    };

    return eventSource;
  },
};
