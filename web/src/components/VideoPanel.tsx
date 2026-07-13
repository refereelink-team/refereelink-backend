import React from 'react';
import { useDashboardStore } from '../store/dashboardStore';

const VideoPanel: React.FC = () => {
  const wsConnected = useDashboardStore((s) => s.wsConnected);
  const sourceStatus = useDashboardStore((s) => s.sourceStatus);

  return (
    <div className="panel video-panel">
      <div className="panel-header">
        <span>LIVE FEED</span>
        <span className={`status-badge ${sourceStatus}`}>{sourceStatus}</span>
      </div>
      <div className="video-container">
        {wsConnected ? (
          <img
            src="/video/stream"
            alt="Live feed"
            className="video-feed"
          />
        ) : (
          <div className="video-placeholder">
            <span>Waiting for connection...</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoPanel;
