export type HomographyStatus = 'fresh' | 'stale' | 'unavailable';
export type PlayerRole = 'player' | 'goalkeeper' | 'referee';
export type SourceStatus = 'connected' | 'disconnected' | 'reconnecting' | 'error';

export interface PlayerState {
  track_id: number;
  role: PlayerRole;
  team_id: number;
  field_x: number | null;
  field_y: number | null;
  confidence: number;
}

export interface GameEvent {
  id: string;
  event_type: string;
  confidence: number;
  severity: string;
  timestamp: number;
  frame_id: number;
  field_x: number | null;
  field_y: number | null;
  reviewed: boolean;
  foul_details?: Record<string, string | number>;
}

export interface FrameState {
  type: 'frame_state';
  frame_id: number;
  capture_timestamp_ms: number;
  processed_timestamp_ms: number;
  processing_fps: number;
  homography_status: HomographyStatus;
  players: PlayerState[];
  events: GameEvent[];
}

export interface MetricsSnapshot {
  type: 'metrics';
  processing_fps: number;
  input_fps: number;
  inference_latency_ms: number;
  end_to_end_latency_ms: number;
  dropped_frames: number;
  queue_length: number;
  player_count: number;
  source_status: SourceStatus;
  memory_mb: number;
  gpu_memory_mb: number | null;
}

export type WSMessage = FrameState | MetricsSnapshot;

export interface PipelineConfig {
  mode: string;
  video_source: string;
  enable_foul_detection: boolean;
  enable_recording: boolean;
  show_keypoints: boolean;
  show_tracking_boxes: boolean;
  show_2d_projection: boolean;
}

export interface PitchData {
  players: PlayerState[];
  homography_status: HomographyStatus;
}
