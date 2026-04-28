import React from 'react';

const MetricsBar = ({ metrics }) => {
  return (
    <div className="flex items-center gap-6 px-4 py-3 bg-gray-800 border-t border-gray-700 text-white text-sm">
      <div className="flex items-center gap-2">
        <span className="text-gray-400">Total Tokens:</span>
        <span className="font-mono font-bold text-blue-400">{metrics.totalTokens}</span>
      </div>
      
      <div className="flex items-center gap-2">
        <span className="text-gray-400">Estimated Cost:</span>
        <span className="font-mono font-bold text-green-400">${metrics.totalCost.toFixed(4)}</span>
      </div>
      
      <div className="flex items-center gap-2">
        <span className="text-gray-400">Current Step:</span>
        <span className="font-mono font-bold text-yellow-400">{metrics.currentStep}</span>
      </div>
      
      {metrics.isRunning && (
        <div className="flex items-center gap-2 ml-auto">
          <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
          <span className="text-blue-400 text-xs">Agent Running</span>
        </div>
      )}
    </div>
  );
};

export default MetricsBar;
