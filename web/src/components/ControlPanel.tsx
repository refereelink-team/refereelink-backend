import React, { useState, useEffect } from 'react';
import { useDashboardStore } from '../store/dashboardStore';
import type { useWebSocket } from '../hooks/useWebSocket';

interface Props {
  sendCommand: ReturnType<typeof useWebSocket>['sendCommand'];
}

const ControlPanel: React.FC<Props> = ({ sendCommand }) => {
  const pipelineRunning = useDashboardStore((s) => s.pipelineRunning);
  const teamCalibration = useDashboardStore((s) => s.teamCalibration);
  const sourceStatus = useDashboardStore((s) => s.sourceStatus);
  const config = useDashboardStore((s) => s.config);
  const setConfig = useDashboardStore((s) => s.setConfig);
  const setPipelineRunning = useDashboardStore((s) => s.setPipelineRunning);
  const addLog = useDashboardStore((s) => s.addLog);

  const [videoSource, setVideoSource] = useState<string>('');
  const [device, setDevice] = useState<string>('cpu');
  const [enableFoul, setEnableFoul] = useState<boolean>(false);
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    if (config.video_source) setVideoSource(config.video_source);
    if (config.device) setDevice(config.device);
    if (config.enable_foul_detection != null) setEnableFoul(config.enable_foul_detection);
  }, [config.video_source, config.device, config.enable_foul_detection]);

  const startPipeline = async () => {
    if (!videoSource.trim()) {
      setError('Video source is required');
      return;
    }
    setBusy(true);
    setError('');
    try {
      const resp = await fetch('/api/pipeline/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_source: videoSource.trim(),
          device,
          enable_foul_detection: enableFoul,
        }),
      });
      const data = await resp.json();
      if (data.status === 'error') {
        setError(data.detail || 'Failed to start');
        addLog(`[Pipeline] start error: ${data.detail}`);
      } else {
        setConfig({ video_source: videoSource.trim(), device, enable_foul_detection: enableFoul });
        addLog(`[Pipeline] started (${data.mode}) source=${videoSource.trim()}`);
        // Force the MJPEG <img> to reconnect by bumping its src
        const img = document.querySelector<HTMLImageElement>('.video-feed');
        if (img) {
          img.src = '/video/stream?ts=' + Date.now();
        }
      }
    } catch (e) {
      setError(String(e));
      addLog(`[Pipeline] start exception: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const stopPipeline = async () => {
    setBusy(true);
    setError('');
    try {
      const resp = await fetch('/api/pipeline/stop', { method: 'POST' });
      const data = await resp.json();
      setPipelineRunning(false);
      addLog(`[Pipeline] stopped (${data.mode || 'cleared'})`);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const onToggleFoul = () => {
    setEnableFoul((v) => {
      const nv = !v;
      sendCommand('update_config', { enable_foul_detection: nv });
      return nv;
    });
  };

  const onToggleKeypoints = () => {
    sendCommand('update_config', {
      show_keypoints: !config.show_keypoints,
    });
  };

  const sourceType = videoSource.trim().startsWith('rtsp') ? 'RTSP' : 'FILE';
  const teamReady = teamCalibration.ready || teamCalibration.state === 'running';

  return (
    <div className="panel control-panel">
      <div className="panel-header">
        <span>CONTROL</span>
        <span className={`status-badge ${sourceStatus}`}>
          {sourceStatus}
        </span>
      </div>

      <div className="control-body">
        <div className="control-row">
          <label className="control-label">
            VIDEO SOURCE
            <span className="control-hint">
              ({sourceType}) — full path or rtsp://
            </span>
          </label>
          <input
            type="text"
            className="control-input"
            placeholder="e.g. assets/data/sample.mp4  or  rtsp://192.168.1.100/stream"
            value={videoSource}
            onChange={(e) => {
              const value = e.target.value;
              setVideoSource(value);
              setConfig({ video_source: value });
            }}
            disabled={pipelineRunning}
          />
        </div>

        <div className="control-row">
          <label className="control-label">DEVICE</label>
          <select
            className="control-select"
            value={device}
            onChange={(e) => {
              const value = e.target.value;
              setDevice(value);
              setConfig({ device: value });
            }}
            disabled={pipelineRunning}
          >
            <option value="cpu">cpu</option>
            <option value="cuda">cuda</option>
            <option value="cuda:0">cuda:0</option>
          </select>

          <label className="control-checkbox">
            <input
              type="checkbox"
              checked={enableFoul}
              onChange={onToggleFoul}
            />
            Foul Detect
          </label>

          <label className="control-checkbox">
            <input
              type="checkbox"
              checked={config.show_keypoints}
              onChange={onToggleKeypoints}
            />
            Show Keypoints
          </label>
        </div>

        <div className="control-row">
          <button
            className={`btn ${pipelineRunning ? 'btn-stop' : 'btn-start'}`}
            onClick={pipelineRunning ? stopPipeline : startPipeline}
            disabled={busy || (!pipelineRunning && config.require_team_calibration && !teamReady)}
          >
            {busy ? '...' : pipelineRunning ? 'STOP' : teamReady ? 'START' : 'CALIBRATE FIRST'}
          </button>
          {!teamReady && !pipelineRunning && config.require_team_calibration && (
            <span className="control-error">TEAM_CALIBRATION_REQUIRED</span>
          )}
          {error && <span className="control-error">{error}</span>}
        </div>
      </div>
    </div>
  );
};

export default ControlPanel;
