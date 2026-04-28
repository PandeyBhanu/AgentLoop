import React, { useEffect, useRef } from 'react';

const TracePanel = ({ traceSteps, traceRef }) => {
  useEffect(() => {
    if (traceRef.current) {
      traceRef.current.scrollTop = traceRef.current.scrollHeight;
    }
  }, [traceSteps, traceRef]);

  const getStepColor = (type) => {
    switch (type) {
      case 'thought':
        return 'bg-blue-900 border-blue-700 text-blue-100';
      case 'action':
        return 'bg-yellow-900 border-yellow-700 text-yellow-100';
      case 'observation':
        return 'bg-green-900 border-green-700 text-green-100';
      case 'error':
        return 'bg-red-900 border-red-700 text-red-100';
      case 'finish':
        return 'bg-gray-800 border-gray-600 text-white font-bold';
      default:
        return 'bg-gray-800 border-gray-600 text-gray-100';
    }
  };

  return (
    <div className="flex flex-col h-full bg-gray-900 text-white">
      <div className="p-4 border-b border-gray-700">
        <h2 className="text-lg font-bold text-gray-100">Live Trace</h2>
      </div>
      
      <div 
        ref={traceRef}
        className="flex-1 overflow-y-auto p-4 space-y-3 trace-scroll"
      >
        {traceSteps.length === 0 ? (
          <div className="text-gray-500 text-sm">No trace steps yet. Start an agent run!</div>
        ) : (
          traceSteps.map((step, index) => (
            <div
              key={index}
              className={`p-3 rounded-lg border ${getStepColor(step.type)}`}
            >
              <div className="flex justify-between items-start mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs uppercase font-bold">{step.type}</span>
                  {step.step_number && (
                    <span className="text-xs opacity-70">Step {step.step_number}</span>
                  )}
                </div>
                {step.latency_ms && (
                  <span className="text-xs opacity-70">{step.latency_ms.toFixed(0)}ms</span>
                )}
              </div>
              
              {step.tool_name && (
                <div className="text-xs opacity-80 mb-1">Tool: {step.tool_name}</div>
              )}
              
              <div className="text-sm whitespace-pre-wrap break-words">
                {step.content}
              </div>

              {(step.tokens_input || step.tokens_output) && (
                <div className="mt-2 pt-2 border-t border-current opacity-60 text-xs">
                  Tokens: In: {step.tokens_input} | Out: {step.tokens_output}
                </div>
              )}

              {step.error_details && (
                <div className="mt-2 pt-2 border-t border-current">
                  <div className="text-xs font-bold">Error Stage: {step.error_details.stage}</div>
                  <div className="text-xs opacity-80">Recoverable: {step.error_details.recoverable ? 'Yes' : 'No'}</div>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default TracePanel;
