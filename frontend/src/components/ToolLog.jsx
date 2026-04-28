import React from 'react';

const ToolLog = ({ toolLogs }) => {
  return (
    <div className="flex flex-col h-full bg-gray-900 text-white">
      <div className="p-4 border-b border-gray-700">
        <h2 className="text-lg font-bold text-gray-100">Tool Execution Log</h2>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {toolLogs.length === 0 ? (
          <div className="text-gray-500 text-sm">No tool executions yet.</div>
        ) : (
          toolLogs.map((log, index) => (
            <div
              key={index}
              className={`p-3 rounded-lg border ${
                log.success
                  ? 'bg-gray-800 border-gray-600'
                  : 'bg-red-900 border-red-700'
              }`}
            >
              <div className="flex justify-between items-start mb-2">
                <div className="font-bold text-sm">{log.tool_name}</div>
                <div className={`text-xs px-2 py-1 rounded ${
                  log.success ? 'bg-green-700' : 'bg-red-700'
                }`}>
                  {log.success ? 'Success' : 'Failed'}
                </div>
              </div>
              
              <div className="text-xs text-gray-300 mb-1">
                <span className="font-medium">Arguments:</span> {JSON.stringify(log.arguments)}
              </div>
              
              {log.execution_time && (
                <div className="text-xs text-gray-300">
                  <span className="font-medium">Time:</span> {log.execution_time.toFixed(3)}s
                </div>
              )}
              
              {log.error && (
                <div className="mt-2 pt-2 border-t border-red-700 text-xs text-red-200">
                  <span className="font-medium">Error:</span> {log.error}
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default ToolLog;
