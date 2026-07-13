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
    enable_foul_detection: false,
    enable_recording: false,
    show_keypoints: true,
    show_tracking_boxes: true,
    show_2d_projection: true,
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
