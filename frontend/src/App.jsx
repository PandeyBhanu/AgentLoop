import React, { useState, useEffect, useRef } from 'react';
import ChatPanel from './components/ChatPanel';
import TracePanel from './components/TracePanel';
import ToolLog from './components/ToolLog';
import MetricsBar from './components/MetricsBar';
import ReplayControls from './components/ReplayControls';
import { api } from './services/api';

const AVAILABLE_TOOLS = ['Calculator', 'PythonExec', 'WebSearch', 'FileReader'];

function App() {
  const [messages, setMessages] = useState([]);
  const [traceSteps, setTraceSteps] = useState([]);
  const [toolLogs, setToolLogs] = useState([]);
  const [metrics, setMetrics] = useState({
    totalTokens: 0,
    totalCost: 0.0,
    currentStep: 0,
    isRunning: false,
  });
  const [isRunning, setIsRunning] = useState(false);
  const [currentRunId, setCurrentRunId] = useState(null);
  const [allowedTools, setAllowedTools] = useState(AVAILABLE_TOOLS);
  const [maxTokenBudget, setMaxTokenBudget] = useState('');
  const [showReplay, setShowReplay] = useState(false);
  
  const traceRef = useRef(null);
  const eventSourceRef = useRef(null);
  const runCompletedRef = useRef(false);

  useEffect(() => {
    if (!showReplay && currentRunId && !runCompletedRef.current) {
      // Close any existing connection first
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }

      const eventSource = api.createEventSource(
        currentRunId,
        (data) => {
          // Handle SSE message
          if (data.type === 'step') {
            setTraceSteps((prev) => {
              // Deduplicate by step_number + type
              const exists = prev.some(
                (s) => s.step_number === data.step.step_number && s.type === data.step.type
              );
              if (exists) return prev;
              return [...prev, data.step];
            });
            setMetrics((prev) => ({
              ...prev,
              currentStep: data.step.step_number || prev.currentStep + 1,
            }));
          }
          if (data.type === 'tool_call') {
            setToolLogs((prev) => [...prev, data.tool_call]);
          }
          if (data.type === 'metrics') {
            setMetrics((prev) => ({
              ...prev,
              totalTokens: data.metrics.total_tokens,
              totalCost: data.metrics.total_cost,
            }));
          }
          if (data.type === 'finish') {
            setIsRunning(false);
            setMetrics((prev) => ({ ...prev, isRunning: false }));
            runCompletedRef.current = true;
            eventSource.close();
          }
        },
        (error) => {
          console.error('SSE error:', error);
          setIsRunning(false);
          setMetrics((prev) => ({ ...prev, isRunning: false }));
        }
      );

      eventSourceRef.current = eventSource;

      return () => {
        eventSource.close();
      };
    }
  }, [currentRunId, showReplay]);

  const handleSendMessage = async (message) => {
    if (isRunning) return;

    // Close any active SSE connection and clear previous run
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    runCompletedRef.current = false;
    setCurrentRunId(null);

    // Add user message to chat
    setMessages((prev) => [...prev, { role: 'user', content: message }]);
    
    // Clear previous trace and logs
    setTraceSteps([]);
    setToolLogs([]);
    setMetrics({
      totalTokens: 0,
      totalCost: 0.0,
      currentStep: 0,
      isRunning: true,
    });
    setIsRunning(true);
    setShowReplay(false);

    try {
      const response = await api.chat(
        message,
        10,
        allowedTools,
        maxTokenBudget ? parseInt(maxTokenBudget) : null
      );

      setCurrentRunId(response.run_id);
      setMetrics((prev) => ({
        ...prev,
        totalTokens: response.total_tokens || 0,
        totalCost: response.total_tokens ? response.total_tokens * 0.00003 : 0,
      }));

      if (response.error) {
        setMessages((prev) => [...prev, { role: 'assistant', content: response.error }]);
      } else {
        setMessages((prev) => [...prev, { role: 'assistant', content: response.final_answer }]);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages((prev) => [...prev, { role: 'assistant', content: `Error: ${error.message}` }]);
      setIsRunning(false);
      setMetrics((prev) => ({ ...prev, isRunning: false }));
    }
  };

  const handleToolToggle = (tool) => {
    setAllowedTools((prev) =>
      prev.includes(tool)
        ? prev.filter((t) => t !== tool)
        : [...prev, tool]
    );
  };

  const handleReplaySelect = (replayData) => {
    setTraceSteps(replayData.trace);
    setMetrics((prev) => ({
      ...prev,
      totalTokens: replayData.total_tokens || 0,
      totalCost: replayData.total_cost || 0,
      currentStep: 0,
      isRunning: false,
    }));
    setShowReplay(true);
  };

  return (
    <div className="h-screen flex flex-col bg-gray-900 text-white">
      {/* Header */}
      <div className="p-4 border-b border-gray-700 bg-gray-800">
        <h1 className="text-2xl font-bold text-gray-100">ReAct Agent Dashboard</h1>
      </div>

      {/* Tool Access Control */}
      <div className="p-4 border-b border-gray-700 bg-gray-800">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400 font-medium">Allowed Tools:</span>
            {AVAILABLE_TOOLS.map((tool) => (
              <label key={tool} className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={allowedTools.includes(tool)}
                  onChange={() => handleToolToggle(tool)}
                  className="w-4 h-4 rounded bg-gray-700 border-gray-600 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-sm">{tool}</span>
              </label>
            ))}
          </div>
          
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-400 font-medium">Max Token Budget:</label>
            <input
              type="number"
              value={maxTokenBudget}
              onChange={(e) => setMaxTokenBudget(e.target.value)}
              placeholder="Optional"
              className="w-32 px-3 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-white focus:outline-none focus:border-blue-500"
            />
          </div>

          <button
            onClick={() => setShowReplay(!showReplay)}
            className="ml-auto px-4 py-1 bg-purple-600 hover:bg-purple-700 rounded text-sm font-medium transition-colors"
          >
            {showReplay ? 'Hide Replay' : 'Show Replay'}
          </button>
        </div>
      </div>

      {/* Metrics Bar */}
      <MetricsBar metrics={{ ...metrics, isRunning }} />

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - Chat */}
        <div className="w-1/3 border-r border-gray-700 flex flex-col">
          <ChatPanel
            onSendMessage={handleSendMessage}
            disabled={isRunning}
            messages={messages}
          />
        </div>

        {/* Middle Panel - Trace or Replay */}
        <div className="w-1/3 border-r border-gray-700 flex flex-col">
          {showReplay ? (
            <ReplayControls onReplaySelect={handleReplaySelect} />
          ) : (
            <TracePanel traceSteps={traceSteps} traceRef={traceRef} />
          )}
        </div>

        {/* Right Panel - Tool Log */}
        <div className="w-1/3 flex flex-col">
          <ToolLog toolLogs={toolLogs} />
        </div>
      </div>
    </div>
  );
}

export default App;
