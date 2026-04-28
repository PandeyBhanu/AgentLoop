import React, { useState, useEffect } from 'react';
import { api } from '../services/api';

const ReplayControls = ({ onReplaySelect }) => {
  const [runId, setRunId] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [replayData, setReplayData] = useState(null);
  const [error, setError] = useState(null);

  const handleLoadReplay = async () => {
    if (!runId.trim()) return;
    
    try {
      setError(null);
      const analysis = await api.replayWithAnalysis(runId);
      setReplayData(analysis);
      setTotalSteps(analysis.total_steps);
      setCurrentStep(0);
      setIsPlaying(false);
      setIsPaused(false);
      
      onReplaySelect({
        runId,
        trace: analysis.trace,
        isReplay: true
      });
    } catch (err) {
      setError(err.message);
    }
  };

  const handlePlay = () => {
    if (!replayData) return;
    setIsPlaying(true);
    setIsPaused(false);
    
    // Simulate step-by-step playback
    let step = currentStep;
    const interval = setInterval(() => {
      if (step < totalSteps) {
        setCurrentStep(step + 1);
        step++;
      } else {
        clearInterval(interval);
        setIsPlaying(false);
      }
    }, 1000); // 1 second per step
    
    return () => clearInterval(interval);
  };

  const handlePause = () => {
    setIsPaused(true);
    setIsPlaying(false);
  };

  const handleReset = () => {
    setCurrentStep(0);
    setIsPlaying(false);
    setIsPaused(false);
  };

  return (
    <div className="flex flex-col h-full bg-gray-900 text-white">
      <div className="p-4 border-b border-gray-700">
        <h2 className="text-lg font-bold text-gray-100">Replay Controls</h2>
      </div>
      
      <div className="p-4 space-y-4">
        <div>
          <label className="block text-sm text-gray-400 mb-2">Run ID</label>
          <div className="flex gap-2">
            <input
              type="text"
              value={runId}
              onChange={(e) => setRunId(e.target.value)}
              placeholder="Enter run ID..."
              className="flex-1 px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:border-blue-500"
            />
            <button
              onClick={handleLoadReplay}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-medium transition-colors"
            >
              Load
            </button>
          </div>
        </div>

        {error && (
          <div className="p-3 bg-red-900 border border-red-700 rounded-lg text-sm text-red-100">
            {error}
          </div>
        )}

        {replayData && (
          <div className="space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Total Steps:</span>
              <span className="font-mono">{totalSteps}</span>
            </div>
            
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Current Step:</span>
              <span className="font-mono">{currentStep} / {totalSteps}</span>
            </div>

            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all"
                style={{ width: `${(currentStep / totalSteps) * 100}%` }}
              />
            </div>

            <div className="flex gap-2">
              <button
                onClick={handlePlay}
                disabled={isPlaying || currentStep >= totalSteps}
                className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg text-sm font-medium transition-colors"
              >
                {isPlaying ? 'Playing...' : 'Play'}
              </button>
              
              <button
                onClick={handlePause}
                disabled={!isPlaying}
                className="flex-1 px-4 py-2 bg-yellow-600 hover:bg-yellow-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg text-sm font-medium transition-colors"
              >
                Pause
              </button>
              
              <button
                onClick={handleReset}
                className="flex-1 px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors"
              >
                Reset
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ReplayControls;
