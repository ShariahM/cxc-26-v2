import React, { useMemo, useRef, useState } from 'react';
import { getDownloadUrl } from '../services/api';

const AnalysisResults = ({ results, taskId }) => {
  if (!results || !results.results) {
    console.log("no ai results found")
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

  // AI-enhanced data from Gemini
  const aiOpenscoreExplanations = feedback.ai_openscore_explanations || {};
  const playerContexts = feedback.player_contexts || {};
  const aiSummary = feedback.ai_summary || '';
  const aiStrengthsAnalysis = feedback.ai_strengths_analysis || '';
  const aiImprovementAnalysis = feedback.ai_improvement_analysis || '';
  const aiPlayReading = feedback.ai_play_reading || '';

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

  const getScoreColor = (score) => {
    if (score >= 70) return '#22c55e';
    if (score >= 50) return '#eab308';
    if (score >= 30) return '#f97316';
    return '#ef4444';
  };

  const getScoreLabel = (score) => {
    if (score >= 80) return 'Wide Open';
    if (score >= 60) return 'Open';
    if (score >= 40) return 'Contested';
    if (score >= 20) return 'Covered';
    return 'Locked Down';
  };

  // Get AI explanation for a player's tooltip
  const getPlayerTooltipExplanation = (player) => {
    const playerId = `player_${player.track_id}`;
    const aiExplanation = aiOpenscoreExplanations[playerId];
    if (aiExplanation) return aiExplanation;

    // Fallback: build from context data
    const ctx = playerContexts[playerId];
    if (ctx) {
      const pxPerYard = 1920 / 53.3;
      const yards = (ctx.nearest_defender_distance / pxPerYard).toFixed(1);
      const nearby = ctx.num_nearby_defenders;
      const closing = ctx.closing_speed;

      let explanation = `Nearest defender: ~${yards} yds. `;
      if (nearby === 0) explanation += 'No defenders in coverage zone. ';
      else if (nearby === 1) explanation += '1 defender nearby. ';
      else explanation += `${nearby} defenders converging. `;

      if (closing > 100) explanation += 'Defender closing fast.';
      else if (closing < -50) explanation += 'Defender moving away.';
      else explanation += 'Defender holding position.';

      return explanation;
    }

    return null;
  };

  return (
    <div className="analysis-results">
      <h2>🏈 Analysis Complete</h2>

      {/* Annotated Video Player - Moved to Top */}
      <div className="video-player-section">
        <h3>🎬 Annotated Video</h3>
        <p className="video-hint">Pause the video and hover over players to see AI-powered OpenScore analysis.</p>
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
            {hoveredPlayer && (
              <div
                className="player-tooltip"
                style={{
                  left: `${tooltipPos.x + 12}px`,
                  top: `${tooltipPos.y + 12}px`,
                }}
              >
                <div className="tooltip-title">Receiver</div>
                <div className="tooltip-score" style={{ color: getScoreColor(hoveredPlayer.openscore || 0) }}>
                  OpenScore: {Number(hoveredPlayer.openscore || 0).toFixed(1)}
                  <span className="tooltip-score-label"> ({getScoreLabel(hoveredPlayer.openscore || 0)})</span>
                </div>
                {(() => {
                  const explanation = getPlayerTooltipExplanation(hoveredPlayer);
                  if (explanation) {
                    return (
                      <div className="tooltip-ai-explanation">
                        {explanation}
                      </div>
                    );
                  }
                  return null;
                })()}
              </div>
            )}
          </div>
        </div>
      </div>

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
          {/* Show AI-enhanced summary if available, otherwise fall back to original */}
          {aiSummary ? (
            <div className="ai-summary-block">
              <div className="ai-badge">🤖 AI Analysis</div>
              <p>{aiSummary.replace(/Receiver #\d+/g, 'receiver')}</p>
            </div>
          ) : (
            <p>{feedback.summary}</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default AnalysisResults;
