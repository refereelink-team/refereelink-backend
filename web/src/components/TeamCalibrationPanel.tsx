import React, { useMemo, useState } from 'react';
import { useDashboardStore } from '../store/dashboardStore';
import type { CalibrationState as CalibrationPhase } from '../types/messages';

const LABELS = [
  ['home_outfield', 'HOME'],
  ['away_outfield', 'AWAY'],
  ['home_goalkeeper', 'HOME GK'],
  ['away_goalkeeper', 'AWAY GK'],
  ['referee', 'REF'],
  ['ignore', 'IGNORE'],
] as const;

const TeamCalibrationPanel: React.FC = () => {
  const calibration = useDashboardStore((s) => s.teamCalibration);
  const frameState = useDashboardStore((s) => s.frameState);
  const config = useDashboardStore((s) => s.config);
  const setTeamCalibration = useDashboardStore((s) => s.setTeamCalibration);
  const addLog = useDashboardStore((s) => s.addLog);
  const [matchId, setMatchId] = useState('match-1');
  const [cameraId, setCameraId] = useState('camera-1');
  const [assigned, setAssigned] = useState<Record<number, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const trackSamples = useMemo(
    () => new Map(calibration.tracks.map((track) => [track.track_id, track])),
    [calibration.tracks]
  );
  const players = frameState?.players ?? [];

  const request = async (path: string, body?: Record<string, unknown>) => {
    const response = await fetch(path, {
      method: 'POST',
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || data.code || 'request failed');
    return data;
  };

  const start = async () => {
    if (!config.video_source.trim()) {
      setError('先在 CONTROL 中填写视频源');
      return;
    }
    setBusy(true);
    setError('');
    try {
      const data = await request('/api/team-calibration/start', {
        match_id: matchId.trim() || 'match-1',
        camera_id: cameraId.trim() || 'default',
        video_source: config.video_source,
        device: config.device,
      });
      setTeamCalibration(data);
      addLog(`[Calibration] started match=${matchId}`);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const labelTrack = async (trackId: number, label: string) => {
    setBusy(true);
    setError('');
    try {
      const data = await request('/api/team-calibration/label', {
        track_id: trackId,
        label,
      });
      setAssigned((current) => ({ ...current, [trackId]: label }));
      setTeamCalibration(data);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const validate = async () => {
    setBusy(true);
    setError('');
    try {
      const data = await request('/api/team-calibration/validate');
      setTeamCalibration(data);
      addLog(`[Calibration] validation=${data.state}`);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const reset = async () => {
    setBusy(true);
    setError('');
    try {
      const data = await request('/api/team-calibration/reset');
      setTeamCalibration(data);
      setAssigned({});
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const phase: CalibrationPhase = calibration.state;
  const validation = calibration.validation_report;
  return (
    <div className="panel team-calibration-panel">
      <div className="panel-header">
        <span>TEAM CALIBRATION</span>
        <span className={`status-badge calibration-${phase}`}>{phase}</span>
      </div>
      <div className="calibration-body">
        <div className="calibration-config">
          <input
            className="control-input"
            value={matchId}
            onChange={(event) => setMatchId(event.target.value)}
            placeholder="match id"
            disabled={phase === 'running'}
          />
          <input
            className="control-input"
            value={cameraId}
            onChange={(event) => setCameraId(event.target.value)}
            placeholder="camera id"
            disabled={phase === 'running'}
          />
          <button className="btn btn-action" onClick={start} disabled={busy || phase === 'running'}>
            CALIBRATE
          </button>
          <button className="btn btn-action" onClick={validate} disabled={busy || !['calibrating', 'recalibration_required'].includes(phase)}>
            VALIDATE
          </button>
          <button className="btn" onClick={reset} disabled={busy}>RESET</button>
        </div>

        <div className="calibration-summary">
          <span>HOME {validation?.home_track_count ?? 0} tracks</span>
          <span>AWAY {validation?.away_track_count ?? 0} tracks</span>
          <span>{calibration.observed_frames} frames</span>
          <span className={calibration.ready ? 'ready-text' : 'warning-text'}>
            {calibration.ready ? 'TEAM READY' : 'NOT READY'}
          </span>
        </div>

        <div className="calibration-tracks">
          {players.length === 0 && <div className="calibration-empty">等待检测到 Track...</div>}
          {players.map((player) => {
            const sample = trackSamples.get(player.track_id);
            const currentLabel = assigned[player.track_id] || sample?.label || 'unlabelled';
            return (
              <div className="calibration-track" key={player.track_id}>
                <div className="calibration-track-meta">
                  <strong>#{player.track_id}</strong>
                  <span>{currentLabel}</span>
                  <span>{sample?.sample_count ?? 0} samples</span>
                </div>
                <div className="calibration-labels">
                  {LABELS.map(([value, text]) => (
                    <button
                      key={value}
                      className={`btn calibration-label-btn ${currentLabel === value ? 'selected' : ''}`}
                      onClick={() => labelTrack(player.track_id, value)}
                      disabled={busy || !['calibrating', 'recalibration_required'].includes(phase)}
                    >
                      {text}
                    </button>
                  ))}
                </div>
              </div>
            );
          })}
        </div>

        {validation && validation.reasons.length > 0 && (
          <div className="calibration-errors">
            {validation.reasons.map((reason) => <span key={reason}>{reason}</span>)}
          </div>
        )}
        {error && <div className="control-error">{error}</div>}
      </div>
    </div>
  );
};

export default TeamCalibrationPanel;

