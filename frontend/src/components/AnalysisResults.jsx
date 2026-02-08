import React, { useMemo, useRef, useState } from 'react';
import { getDownloadUrl } from '../services/api';

const AnalysisResults = ({ results, taskId }) => {
  if (!results || !results.results) {
    return null;
  }

  const { feedback, openscore_data, tracking_data } = results.results;
  const fps = results.results.fps || 30;
  const frameData = results.results.frame_data || [];
  const sourceWidth = results.results.video_width || 0;
  const sourceHeight = results.results.video_height || 0;
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const [activePlayers, setActivePlayers] = useState([]);
  const [hoveredPlayer, setHoveredPlayer] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  const framePlayerMap = useMemo(() => {
    const map = new Map();
    frameData.forEach((frame) => {
      map.set(frame.frame_id, frame.players || []);
    });
    return map;
  }, [frameData]);

  const updateActivePlayersForCurrentTime = () => {
    const videoEl = videoRef.current;
    if (!videoEl) return;
    const frameId = Math.max(0, Math.floor(videoEl.currentTime * fps));
    setActivePlayers(framePlayerMap.get(frameId) || []);
  };

  const handleOverlayMouseMove = (event) => {
    const videoEl = videoRef.current;
    const overlayEl = overlayRef.current;
    if (!videoEl || !overlayEl || !videoEl.paused) {
      setHoveredPlayer(null);
      return;
    }

    const rect = overlayEl.getBoundingClientRect();
    const relX = event.clientX - rect.left;
    const relY = event.clientY - rect.top;

    const nativeWidth = sourceWidth || videoEl.videoWidth || 1;
    const nativeHeight = sourceHeight || videoEl.videoHeight || 1;
    const px = (relX / rect.width) * nativeWidth;
    const py = (relY / rect.height) * nativeHeight;

    const hitPlayer = activePlayers.find((player) => {
      const [x1, y1, x2, y2] = player.bbox;
      return px >= x1 && px <= x2 && py >= y1 && py <= y2;
    });

    if (!hitPlayer) {
      setHoveredPlayer(null);
      return;
    }

    setHoveredPlayer(hitPlayer);
    setTooltipPos({ x: relX, y: relY });
  };

  const getGradeColor = (grade) => {
    switch (grade) {
      case 'A': return '#22c55e';
      case 'B': return '#3b82f6';
      case 'C': return '#eab308';
      case 'D': return '#f97316';
      case 'F': return '#ef4444';
      default: return '#6b7280';
    }
  };

  return (
    <div className="analysis-results">
      <h2>Analysis Complete</h2>

      {/* Overall Grade */}
      <div className="grade-section">
        <div className="grade-card" style={{ borderColor: getGradeColor(feedback.overall_grade) }}>
          <div className="grade-label">Overall Grade</div>
          <div className="grade-value" style={{ color: getGradeColor(feedback.overall_grade) }}>
            {feedback.overall_grade}
          </div>
          <div className="grade-score">Score: {feedback.overall_score}/100</div>
        </div>
        <div className="grade-summary">
          <p>{feedback.summary}</p>
        </div>
      </div>

      {/* Statistics */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-label">Receivers Tracked</div>
          <div className="stat-value">{feedback.statistics.total_receivers_tracked}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Avg OpenScore</div>
          <div className="stat-value">{feedback.statistics.avg_openscore_all_receivers.toFixed(1)}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Players Detected</div>
          <div className="stat-value">{feedback.statistics.total_players_detected}</div>
        </div>
      </div>

      {/* Strengths and Improvements */}
      <div className="feedback-grid">
        {feedback.strengths.length > 0 && (
          <div className="feedback-section strengths">
            <h3>💪 Strengths</h3>
            <ul>
              {feedback.strengths.map((strength, idx) => (
                <li key={idx}>{strength}</li>
              ))}
            </ul>
          </div>
        )}

        {feedback.areas_for_improvement.length > 0 && (
          <div className="feedback-section improvements">
            <h3>📈 Areas for Improvement</h3>
            <ul>
              {feedback.areas_for_improvement.map((area, idx) => (
                <li key={idx}>{area}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Recommendations */}
      {feedback.recommendations.length > 0 && (
        <div className="recommendations-section">
          <h3>💡 Recommendations</h3>
          <ul className="recommendations-list">
            {feedback.recommendations.map((rec, idx) => (
              <li key={idx}>{rec}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Best Options */}
      {feedback.best_options.length > 0 && (
        <div className="best-options-section">
          <h3>🎯 Best Passing Options</h3>
          <div className="options-grid">
            {feedback.best_options.map((option, idx) => (
              <div key={idx} className="option-card">
                <div className="option-rank">#{idx + 1}</div>
                <div className="option-receiver">{option.receiver}</div>
                <div className="option-stats">
                  <div className="option-stat">
                    <span className="label">Avg OpenScore:</span>
                    <span className="value">{option.avg_openscore}</span>
                  </div>
                  <div className="option-stat">
                    <span className="label">Max OpenScore:</span>
                    <span className="value">{option.max_openscore}</span>
                  </div>
                  <div className="option-stat">
                    <span className="label">Consistency:</span>
                    <span className="value">{option.consistency}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Missed Opportunities */}
      {feedback.missed_opportunities.length > 0 && (
        <div className="missed-section">
          <h3>⚠️ Missed Opportunities</h3>
          <div className="missed-list">
            {feedback.missed_opportunities.map((missed, idx) => (
              <div key={idx} className="missed-card">
                <div className="missed-receiver">{missed.receiver}</div>
                <div className="missed-stats">
                  <span>Peak: {missed.peak_openscore}</span>
                  <span>Avg: {missed.avg_openscore}</span>
                </div>
                <div className="missed-note">{missed.note}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Key Moments */}
      {feedback.key_moments.length > 0 && (
        <div className="key-moments-section">
          <h3>⭐ Key Moments</h3>
          <div className="moments-list">
            {feedback.key_moments.map((moment, idx) => (
              <div key={idx} className="moment-card">
                <div className="moment-frame">Frame {moment.frame}</div>
                <div className="moment-description">{moment.description}</div>
                <div className="moment-score">OpenScore: {moment.openscore}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Annotated Video Player */}
      <div className="video-player-section">
        <h3>🎬 Annotated Video</h3>
        <div className="video-player-card">
          <div
            className="video-overlay-container"
            ref={overlayRef}
            onMouseMove={handleOverlayMouseMove}
            onMouseLeave={() => setHoveredPlayer(null)}
          >
          <video
            className="analysis-video-player"
            src={getDownloadUrl(taskId)}
            controls
            preload="metadata"
            ref={videoRef}
            onLoadedMetadata={updateActivePlayersForCurrentTime}
            onTimeUpdate={updateActivePlayersForCurrentTime}
            onSeeked={updateActivePlayersForCurrentTime}
            onPause={updateActivePlayersForCurrentTime}
            onPlay={() => setHoveredPlayer(null)}
          >
            Your browser does not support the video tag.
          </video>
            {videoRef.current?.paused && activePlayers.map((player) => {
              const [x1, y1, x2, y2] = player.bbox;
              const width = sourceWidth || videoRef.current?.videoWidth || 1;
              const height = sourceHeight || videoRef.current?.videoHeight || 1;

              return (
                <div
                  key={player.track_id}
                  className="player-hover-box"
                  style={{
                    left: `${(x1 / width) * 100}%`,
                    top: `${(y1 / height) * 100}%`,
                    width: `${((x2 - x1) / width) * 100}%`,
                    height: `${((y2 - y1) / height) * 100}%`,
                  }}
                />
              );
            })}
            {hoveredPlayer && (
              <div
                className="player-tooltip"
                style={{
                  left: `${tooltipPos.x + 12}px`,
                  top: `${tooltipPos.y + 12}px`,
                }}
              >
                <div className="tooltip-title">Receiver #{hoveredPlayer.track_id}</div>
                <div>OpenScore: {Number(hoveredPlayer.openscore || 0).toFixed(1)}</div>
                <div className="tooltip-placeholder">
                  AI placeholder: Add LLM explanation here for route, leverage, and passing risk.
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisResults;
