import React from 'react';
import { getDownloadUrl } from '../services/api';

const AnalysisResults = ({ results, taskId }) => {
  if (!results || !results.results) {
    return null;
  }

  const { feedback, openscore_data, tracking_data } = results.results;

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

      {/* Download Video */}
      <div className="download-section">
        <a
          href={getDownloadUrl(taskId)}
          download
          className="download-button"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" className="download-icon">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          Download Annotated Video
        </a>
      </div>
    </div>
  );
};

export default AnalysisResults;
