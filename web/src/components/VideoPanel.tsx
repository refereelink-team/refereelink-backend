import React, { useEffect, useRef, useState } from 'react';
import { useDashboardStore } from '../store/dashboardStore';

const VideoPanel: React.FC = () => {
  const wsConnected = useDashboardStore((s) => s.wsConnected);
  const sourceStatus = useDashboardStore((s) => s.sourceStatus);
  const pipelineRunning = useDashboardStore((s) => s.pipelineRunning);
  const [streamKey, setStreamKey] = useState<number>(0);

  // Bump the stream src whenever the pipeline starts so the browser
  // reconnects to the MJPEG endpoint immediately.
  useEffect(() => {
    if (pipelineRunning) {
      setStreamKey(Date.now());
    }
  }, [pipelineRunning]);

  return (
    <div className="panel video-panel">
      <div className="panel-header">
        <span>LIVE FEED</span>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span className={`status-badge ${wsConnected ? 'connected' : 'disconnected'}`}>
            {wsConnected ? 'WS OK' : 'WS OFF'}
          </span>
          <span className={`status-badge ${sourceStatus}`}>{sourceStatus}</span>
        </div>
      </div>
      <div className="video-container">
        <img
          key={streamKey}
          src={`/video/stream?ts=${streamKey}`}
          alt="Live feed"
          className="video-feed"
          onError={(e) => {
            (e.target as HTMLImageElement).style.opacity = '0.3';
          }}
          onLoad={(e) => {
            (e.target as HTMLImageElement).style.opacity = '1';
          }}
        />
        {(!pipelineRunning || sourceStatus === 'disconnected') && (
          <div className="video-overlay">
            {!wsConnected
              ? 'Waiting for backend connection...'
              : !pipelineRunning
              ? 'Pipeline idle — configure a source and press START'
              : 'Video source disconnected — check RTSP / file path'}
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoPanel;