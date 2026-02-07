import React, { useState, useEffect } from 'react';
import VideoUpload from './components/VideoUpload';
import AnalysisResults from './components/AnalysisResults';
import { getStatus, getResults } from './services/api';
import './App.css';

function App() {
  const [taskId, setTaskId] = useState(null);
  const [status, setStatus] = useState(null);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);

  // Poll for status updates
  useEffect(() => {
    if (!taskId || status === 'completed' || status === 'failed') {
      return;
    }

    const pollInterval = setInterval(async () => {
      try {
        const statusData = await getStatus(taskId);
        setStatus(statusData.status);
        setProgress(statusData.progress || 0);

        if (statusData.status === 'completed') {
          // Fetch results
          const resultsData = await getResults(taskId);
          setResults(resultsData);
          clearInterval(pollInterval);
        } else if (statusData.status === 'failed') {
          setError(statusData.message || 'Processing failed');
          clearInterval(pollInterval);
        }
      } catch (err) {
        console.error('Error polling status:', err);
        setError(err.response?.data?.detail || err.message || 'Failed to get status');
        clearInterval(pollInterval);
      }
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(pollInterval);
  }, [taskId, status]);

  const handleUploadSuccess = (response) => {
    setTaskId(response.task_id);
    setStatus(response.status);
    setError(null);
    setResults(null);
    setProgress(0);
  };

  const handleUploadError = (errorMessage) => {
    setError(errorMessage);
  };

  const handleReset = () => {
    setTaskId(null);
    setStatus(null);
    setResults(null);
    setError(null);
    setProgress(0);
  };

  const getStatusMessage = () => {
    switch (status) {
      case 'queued':
        return 'Video queued for processing...';
      case 'processing':
        return 'Processing video...';
      case 'completed':
        return 'Analysis complete!';
      case 'failed':
        return 'Processing failed';
      default:
        return '';
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>🏈 NFL Video Analysis</h1>
          <p className="tagline">AI-Powered Quarterback Decision Analysis</p>
        </div>
      </header>

      <main className="app-main">
        <div className="container">
          {!taskId && !results && (
            <>
              <div className="intro-section">
                <h2>Analyze Your Game Footage</h2>
                <p>
                  Upload your NFL game video to get detailed analysis on player tracking,
                  receiver openness scores, and quarterback decision-making feedback.
                </p>
              </div>
              <VideoUpload
                onUploadSuccess={handleUploadSuccess}
                onUploadError={handleUploadError}
              />
            </>
          )}

          {taskId && status !== 'completed' && (
            <div className="processing-section">
              <div className="processing-card">
                <div className="processing-icon">
                  <div className="spinner-large"></div>
                </div>
                <h2>{getStatusMessage()}</h2>
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
                <p className="progress-text">{progress}% complete</p>
                <p className="processing-info">
                  This may take several minutes depending on video length.
                  The video is being analyzed with YOLOv8 detection, player tracking, and OpenScore calculations.
                </p>
              </div>
            </div>
          )}

          {results && status === 'completed' && (
            <>
              <AnalysisResults results={results} taskId={taskId} />
              <div className="reset-section">
                <button onClick={handleReset} className="reset-button">
                  Analyze Another Video
                </button>
              </div>
            </>
          )}

          {error && (
            <div className="error-section">
              <div className="error-card">
                <svg className="error-icon-large" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
                </svg>
                <h3>Error</h3>
                <p>{error}</p>
                <button onClick={handleReset} className="reset-button">
                  Try Again
                </button>
              </div>
            </div>
          )}
        </div>
      </main>

      <footer className="app-footer">
        <p>Powered by YOLOv8, FastAPI, and React | ByteTrack Player Tracking</p>
      </footer>
    </div>
  );
}

export default App;
