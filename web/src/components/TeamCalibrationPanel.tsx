import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useDashboardStore } from '../store/dashboardStore';
import type { CalibrationState as CalibrationPhase, CalibrationTrack } from '../types/messages';

const LABELS = [
  ['home_outfield', 'HOME'],
  ['away_outfield', 'AWAY'],
  ['home_goalkeeper', 'HOME GK'],
  ['away_goalkeeper', 'AWAY GK'],
  ['referee', 'REF'],
  ['ignore', 'IGNORE'],
] as const;

type ClipFrame = {
  frame_index: number;
  timestamp_ms: number;
  players: Array<{
    track_id: number;
    bbox: [number, number, number, number];
    confidence: number;
    quality_accepted: boolean;
    quality_score: number;
  }>;
};

type ClipMetadata = {
  clip_id: string;
  fps: number;
  width: number;
  height: number;
  frames: ClipFrame[];
};

const formatTime = (milliseconds: number | null | undefined) => {
  const totalSeconds = Math.max(0, Math.round((milliseconds ?? 0) / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
};

const TeamCalibrationPanel: React.FC = () => {
  const calibration = useDashboardStore((s) => s.teamCalibration);
  const config = useDashboardStore((s) => s.config);
  const setTeamCalibration = useDashboardStore((s) => s.setTeamCalibration);
  const addLog = useDashboardStore((s) => s.addLog);
  const [matchId, setMatchId] = useState('match-1');
  const [cameraId, setCameraId] = useState('camera-1');
  const [currentTime, setCurrentTime] = useState(0);
  const [metadata, setMetadata] = useState<ClipMetadata | null>(null);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);
  const [search, setSearch] = useState('');
  const [onlyUnlabelled, setOnlyUnlabelled] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const sourceVideoRef = useRef<HTMLVideoElement>(null);
  const reviewVideoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);

  const phase: CalibrationPhase = calibration.state;
  const validation = calibration.validation_report;
  const tracks = useMemo(() => {
    const query = search.trim().toLowerCase();
    return calibration.tracks.filter((track) => {
      if (onlyUnlabelled && track.label) return false;
      if (query && !String(track.track_id).includes(query)) return false;
      return true;
    });
  }, [calibration.tracks, onlyUnlabelled, search]);

  const request = async (path: string, method = 'POST', body?: Record<string, unknown>) => {
    const response = await fetch(path, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || data.code || 'request failed');
    return data;
  };

  const previewSource = async () => {
    if (!config.video_source.trim()) {
      setError('先在 CONTROL 中填写本地视频路径');
      return;
    }
    setBusy(true);
    setError('');
    try {
      const data = await request('/api/team-calibration/source/preview', 'POST', {
        video_source: config.video_source,
        match_id: matchId.trim() || 'match-1',
        camera_id: cameraId.trim() || 'camera-1',
        device: config.device,
      });
      setMetadata(null);
      setCurrentTime(0);
      setTeamCalibration(data);
      addLog(`[Calibration] source preview ${data.width}x${data.height} ${formatTime(data.duration_ms)}`);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const beginClip = async () => {
    setBusy(true);
    setError('');
    try {
      const data = await request('/api/team-calibration/clip/start', 'POST', {
        start_ms: Math.round(currentTime * 1000),
      });
      setTeamCalibration(data);
      addLog(`[Calibration] clip start ${formatTime(data.clip_start_ms)}`);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const finishClip = async () => {
    const endMs = Math.round(currentTime * 1000);
    if (calibration.clip_start_ms === null || endMs <= calibration.clip_start_ms) {
      setError('结束时间必须晚于开始时间');
      return;
    }
    setBusy(true);
    setError('');
    try {
      const data = await request('/api/team-calibration/clip/finish', 'POST', { end_ms: endMs });
      setTeamCalibration(data);
      addLog(`[Calibration] processing job=${data.job_id}`);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (phase !== 'processing') return undefined;
    const timer = window.setInterval(async () => {
      try {
        const data = await request('/api/team-calibration/clip/status', 'GET');
        setTeamCalibration(data);
      } catch (e) {
        setError(String(e));
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [phase, setTeamCalibration]);

  useEffect(() => {
    if (phase !== 'review' || !calibration.metadata_url || metadata?.clip_id === calibration.clip_id) {
      return undefined;
    }
    let cancelled = false;
    request(calibration.metadata_url, 'GET')
      .then((data) => {
        if (!cancelled) setMetadata(data as ClipMetadata);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [calibration.clip_id, calibration.metadata_url, metadata?.clip_id, phase]);

  const drawOverlay = useCallback(() => {
    const video = reviewVideoRef.current;
    const canvas = overlayRef.current;
    if (!video || !canvas || !metadata || !metadata.width || !metadata.height) return;
    const width = video.clientWidth;
    const height = video.clientHeight;
    if (!width || !height) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const context = canvas.getContext('2d');
    if (!context) return;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    const timestamp = video.currentTime * 1000;
    let frame = metadata.frames[0];
    for (const candidate of metadata.frames) {
      if (candidate.timestamp_ms <= timestamp) frame = candidate;
      else break;
    }
    const scaleX = width / metadata.width;
    const scaleY = height / metadata.height;
    for (const player of frame?.players ?? []) {
      const [x1, y1, x2, y2] = player.bbox;
      const left = x1 * scaleX;
      const top = y1 * scaleY;
      const boxWidth = (x2 - x1) * scaleX;
      const boxHeight = (y2 - y1) * scaleY;
      context.strokeStyle = '#ffe600';
      context.lineWidth = 3;
      context.strokeRect(left, top, boxWidth, boxHeight);
      const label = `ID ${player.track_id}`;
      context.font = '700 14px ui-monospace, monospace';
      const labelWidth = context.measureText(label).width + 10;
      context.fillStyle = '#090d14';
      context.fillRect(left, Math.max(0, top - 22), labelWidth, 22);
      context.fillStyle = '#ffffff';
      context.fillText(label, left + 5, Math.max(15, top - 7));
    }
  }, [metadata]);

  useEffect(() => {
    if (phase !== 'review') return undefined;
    let frameHandle = 0;
    const render = () => {
      drawOverlay();
      frameHandle = window.requestAnimationFrame(render);
    };
    frameHandle = window.requestAnimationFrame(render);
    return () => window.cancelAnimationFrame(frameHandle);
  }, [drawOverlay, phase]);

  const labelTrack = async (trackId: number, label: string) => {
    setSelectedTrackId(trackId);
    setBusy(true);
    setError('');
    try {
      const data = await request('/api/team-calibration/label', 'POST', {
        track_id: trackId,
        label,
      });
      setTeamCalibration(data);
      addLog(`[Calibration] Track #${trackId} -> ${label}`);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const selectTrack = (track: CalibrationTrack) => {
    setSelectedTrackId(track.track_id);
    const video = reviewVideoRef.current;
    if (video && track.representative_timestamp_ms !== undefined) {
      video.currentTime = track.representative_timestamp_ms / 1000;
      setCurrentTime(video.currentTime);
      video.pause();
    }
  };

  const validate = async () => {
    setBusy(true);
    setError('');
    try {
      const data = await request('/api/team-calibration/validate', 'POST');
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
      const data = await request('/api/team-calibration/reset', 'POST');
      setMetadata(null);
      setSelectedTrackId(null);
      setTeamCalibration(data);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const sourceIsReady = phase === 'source_preview' || phase === 'clip_selecting';
  const canStartClip = phase === 'source_preview';
  const canFinishClip = phase === 'clip_selecting';
  const canLabel = phase === 'review' || phase === 'recalibration_required';

  return (
    <div className="panel team-calibration-panel">
      <div className="panel-header">
        <span>TEAM CALIBRATION</span>
        <span className={`status-badge calibration-${phase}`}>{phase}</span>
      </div>
      <div className="calibration-body">
        <div className="calibration-config">
          <input className="control-input" value={matchId} onChange={(event) => setMatchId(event.target.value)} placeholder="match id" disabled={phase === 'running'} />
          <input className="control-input" value={cameraId} onChange={(event) => setCameraId(event.target.value)} placeholder="camera id" disabled={phase === 'running'} />
          <button className="btn btn-action" onClick={previewSource} disabled={busy || phase !== 'idle'}>LOAD SOURCE</button>
          <button className="btn btn-action" onClick={validate} disabled={busy || !['review', 'calibrating', 'recalibration_required'].includes(phase)}>VALIDATE</button>
          <button className="btn" onClick={reset} disabled={busy || phase === 'processing'}>RESET</button>
        </div>

        <div className="calibration-summary">
          <span>HOME {validation?.home_track_count ?? 0} tracks</span>
          <span>AWAY {validation?.away_track_count ?? 0} tracks</span>
          <span>{calibration.observed_frames} frames</span>
          <span className={calibration.ready ? 'ready-text' : 'warning-text'}>{calibration.ready ? 'TEAM READY' : 'NOT READY'}</span>
        </div>

        {phase === 'idle' && <div className="calibration-help">先点击 LOAD SOURCE，再在视频时间轴暂停并选择一段 30 秒以内的清晰片段。</div>}

        {sourceIsReady && calibration.source_url && (
          <div className="calibration-source-stage">
            <video
              ref={sourceVideoRef}
              className="calibration-video"
              src={calibration.source_url}
              controls
              preload="metadata"
              onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
            />
            <div className="calibration-time-row">
              <span>当前 {formatTime(currentTime * 1000)}</span>
              <span>片段开始 {formatTime(calibration.clip_start_ms)}</span>
              <span>最长 60 秒</span>
              {canStartClip && <button className="btn btn-action" onClick={beginClip} disabled={busy}>开始截取</button>}
              {canFinishClip && <button className="btn btn-action" onClick={finishClip} disabled={busy}>结束截取并处理</button>}
            </div>
          </div>
        )}

        {phase === 'processing' && (
          <div className="calibration-progress">
            <div>CUDA 逐帧处理：{Math.round(calibration.processing_progress * 100)}%</div>
            <div className="progress-track"><div className="progress-value" style={{ width: `${Math.round(calibration.processing_progress * 100)}%` }} /></div>
            <div className="calibration-help">处理期间不会丢帧，也不能重复开始新的标注任务。</div>
          </div>
        )}

        {phase === 'review' && calibration.review_video_url && (
          <div className="calibration-review-layout">
            <div className="calibration-review-video">
              <div className="review-video-stage">
                <video
                  ref={reviewVideoRef}
                  className="calibration-video"
                  src={calibration.review_video_url}
                  controls
                  preload="metadata"
                  onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
                  onLoadedMetadata={drawOverlay}
                />
                <canvas ref={overlayRef} className="review-overlay" />
              </div>
              <div className="calibration-time-row">
                <span>回放 {formatTime(currentTime * 1000)}</span>
                <span>片段 {formatTime(calibration.clip_duration_ms)}</span>
                <span>暂停后点击 Track 即可跳转代表帧</span>
              </div>
            </div>
            <div className="calibration-track-browser">
              <div className="track-browser-toolbar">
                <input className="control-input" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索 Track ID" />
                <label className="track-filter"><input type="checkbox" checked={onlyUnlabelled} onChange={(event) => setOnlyUnlabelled(event.target.checked)} /> 未标注</label>
              </div>
              <div className="calibration-tracks stable-track-list">
                {tracks.length === 0 && <div className="calibration-empty">没有符合条件的片段 Track</div>}
                {tracks.map((track) => {
                  const currentLabel = track.label || 'unlabelled';
                  return (
                    <div className={`calibration-track ${selectedTrackId === track.track_id ? 'active' : ''}`} key={track.track_id} onClick={() => selectTrack(track)}>
                      <div className="calibration-track-meta">
                        <strong>#{track.track_id}</strong>
                        <span>{currentLabel}</span>
                        <span>{track.sample_count} samples</span>
                      </div>
                      <div className="calibration-track-stats">
                        <span>{formatTime(track.first_timestamp_ms)}—{formatTime(track.last_timestamp_ms)}</span>
                        <span>{track.observation_count ?? 0} obs / {track.quality_observation_count ?? 0} good</span>
                      </div>
                      <div className="calibration-labels">
                        {LABELS.map(([value, text]) => (
                          <button key={value} className={`btn calibration-label-btn ${currentLabel === value ? 'selected' : ''}`} onClick={(event) => { event.stopPropagation(); labelTrack(track.track_id, value); }} disabled={busy || !canLabel}>{text}</button>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {calibration.processing_error && <div className="calibration-errors"><span>{calibration.processing_error}</span></div>}
        {validation && validation.reasons.length > 0 && <div className="calibration-errors">{validation.reasons.map((reason) => <span key={reason}>{reason}</span>)}</div>}
        {error && <div className="control-error">{error}</div>}
      </div>
    </div>
  );
};

export default TeamCalibrationPanel;
