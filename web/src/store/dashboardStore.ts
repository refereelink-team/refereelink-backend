import { create } from 'zustand';
import type { FrameState, MetricsSnapshot, GameEvent, PipelineConfig } from '../types/messages';

interface DashboardState {
  frameState: FrameState | null;
  metrics: MetricsSnapshot | null;
  events: GameEvent[];
  logLines: string[];
  config: PipelineConfig;
  wsConnected: boolean;
  sourceStatus: string;
  pipelineRunning: boolean;

  setFrameState: (fs: FrameState) => void;
  setMetrics: (m: MetricsSnapshot) => void;
  addEvent: (e: GameEvent) => void;
  addLog: (msg: string) => void;
  setConfig: (c: Partial<PipelineConfig>) => void;
  setWsConnected: (c: boolean) => void;
  setSourceStatus: (s: string) => void;
  setPipelineRunning: (r: boolean) => void;
}

export const useDashboardStore = create<DashboardState>((set) => ({
  frameState: null,
  metrics: null,
  events: [],
  logLines: [],
  config: {
    mode: 'realtime',
    video_source: '',
    device: 'cpu',
    inference_backend: 'auto',
    enable_foul_detection: false,
    enable_recording: false,
    show_keypoints: true,
    show_tracking_boxes: true,
    show_2d_projection: true,
    foul_confidence_threshold: 0.48,
    player_model_path: 'assets/weights/yolo11s.pt',
    pitch_model_path: 'assets/weights/football-pitch-detection.pt',
    role_model_path: 'assets/weights/player-role-yolo11n.pt',
    team_classifier_path: null,
    ball_model_path: 'assets/weights/football-ball-detection.pt',
    enable_ball: true,
    role_detection_interval: 3,
    team_classification_interval: 5,
    ball_detection_interval: 2,
    ball_max_prediction_frames: 8,
    camera_calibration_path: 'assets/calibration/camera.npz',
    enable_undistortion: true,
    calibration_alpha: 0,
    pitch_detection_interval: 5,
    imgsz: 640,
  },
  wsConnected: false,
  sourceStatus: 'disconnected',
  pipelineRunning: false,

  setFrameState: (fs) => set({ frameState: fs }),
  setMetrics: (m) => set({ metrics: m }),
  addEvent: (e) => set((s) => ({ events: [e, ...s.events].slice(0, 500) })),
  addLog: (msg) =>
    set((s) => ({
      logLines: [...s.logLines, msg].slice(-200),
    })),
  setConfig: (c) => set((s) => ({ config: { ...s.config, ...c } })),
  setWsConnected: (c) => set({ wsConnected: c }),
  setSourceStatus: (s) => set({ sourceStatus: s }),
  setPipelineRunning: (r) => set({ pipelineRunning: r }),
}));
