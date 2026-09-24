import React from 'react';
import { useDashboardStore } from '../store/dashboardStore';

const StatusCards: React.FC = () => {
  const metrics = useDashboardStore((s) => s.metrics);
  const frameState = useDashboardStore((s) => s.frameState);
  const wsConnected = useDashboardStore((s) => s.wsConnected);

  if (!metrics) {
    return (
      <div className="panel status-panel">
        <div className="panel-header">STATUS</div>
        <div className="status-empty">Waiting for data...</div>
      </div>
    );
  }

  const cards = [
    { label: 'Proc FPS', value: metrics.processing_fps.toFixed(1), unit: 'fps' },
    { label: 'Input FPS', value: metrics.input_fps.toFixed(1), unit: 'fps' },
    { label: 'Inf Latency', value: metrics.inference_latency_ms.toFixed(0), unit: 'ms' },
    { label: 'E2E Latency', value: metrics.end_to_end_latency_ms.toFixed(0), unit: 'ms' },
    { label: 'Dropped', value: metrics.dropped_frames, unit: 'frames' },
    { label: 'Players', value: metrics.player_count, unit: '' },
    { label: 'Mem', value: metrics.memory_mb.toFixed(0), unit: 'MB' },
    ...(metrics.gpu_memory_mb != null
      ? [{ label: 'GPU Mem', value: metrics.gpu_memory_mb.toFixed(0), unit: 'MB' }]
      : []),
    { label: 'Queue', value: metrics.queue_length, unit: '' },
    { label: 'WS', value: wsConnected ? 'ON' : 'OFF', unit: '' },
    { label: 'Homo', value: frameState?.homography_status ?? '—', unit: '' },
    { label: 'Decode', value: metrics.decoded_frames, unit: 'frames' },
    { label: 'Join Miss', value: metrics.join_missing_frames, unit: '' },
    { label: 'Field', value: metrics.field_session_id ? 'ON' : 'OFF', unit: '' },
  ];

  return (
    <div className="panel status-panel">
      <div className="panel-header">STATUS</div>
      <div className="status-grid">
        {cards.map((c) => (
          <div key={c.label} className="status-card">
            <div className="status-label">{c.label}</div>
            <div className="status-value">
              {c.value}
              {c.unit && <span className="status-unit">{c.unit}</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default StatusCards;
