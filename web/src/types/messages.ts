export type HomographyStatus =
  | 'fresh'
  | 'reused'
  | 'stale'
  | 'unavailable'
  | 'relocalized'
  | 'corrected'
  | 'tracked'
  | 'predicted'
  | 'lost';
export type BallStatus = 'fresh' | 'predicted' | 'stale' | 'unavailable';
export type PlayerRole = 'outfield' | 'player' | 'goalkeeper' | 'referee' | 'staff' | 'unknown';
export type TeamLabel = 'home' | 'away' | 'none' | 'unknown';
export type SourceStatus = 'connected' | 'disconnected' | 'reconnecting' | 'error';

export interface PlayerState {
  track_id: number;
  entity_id: number | null;
  track_status: string;
  missing_frames: number;
  role: PlayerRole;
  team: TeamLabel;
  team_label: TeamLabel;
  team_id: number;
  field_x: number | null;
  field_y: number | null;
  field_x_m: number | null;
  field_y_m: number | null;
  field_sigma_m: number | null;
  field_coordinate_source: string;
  field_coordinate_usable: boolean;
  field_camera_status: HomographyStatus | null;
  confidence: number;
  role_confidence: number;
  team_confidence: number;
  team_rejection_reason: string | null;
  bbox: [number, number, number, number] | null;
  semantic_status: string;
  velocity_x: number | null;
  velocity_y: number | null;
}

export interface BallState {
  status: BallStatus;
  image_x: number | null;
  image_y: number | null;
  field_x: number | null;
  field_y: number | null;
  velocity_x: number | null;
  velocity_y: number | null;
  confidence: number;
  age_frames: number;
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
  involved_track_ids: number[];
  evidence: Record<string, unknown>;
}

export interface FrameState {
  type: 'frame_state';
  frame_id: number;
  capture_timestamp_ms: number;
  processed_timestamp_ms: number;
  processing_fps: number;
  homography_status: HomographyStatus;
  camera_confidence: number;
  camera_pan_rad: number | null;
  camera_measurement_usable: boolean;
  players: PlayerState[];
  ball: BallState | null;
  possession_track_id: number | null;
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
  player_inference_latency_ms: number;
  pitch_inference_latency_ms: number;
  pitch_detection_count: number;
  homography_reuse_ratio: number;
  homography_available_ratio: number;
  camera_tracking_status: string;
  camera_tracking_confidence: number;
  camera_pan_rad: number | null;
  camera_pan_velocity_rad_s: number | null;
  field_flow_update_count: number;
  field_prediction_count: number;
  field_lost_count: number;
  field_relocalization_count: number;
  track_id_interruptions: number;
  track_occlusion_events: number;
  track_predicted_frames: number;
  track_recovered_count: number;
  track_reactivated_count: number;
  track_id_switches: number;
  track_fragmentations: number;
  track_max_missing_frames: number;
  track_entity_rebinds: number;
  track_entity_fragmentations: number;
  track_lifecycle_counts: Record<string, number>;
  semantic_inference_count: number;
  semantic_label_switches: number;
  team_inference_count: number;
  team_unknown_rate: number;
  team_label_switches: number;
  ball_detection_count: number;
  ball_predicted_frames: number;
  ball_available_ratio: number;
  jpeg_frames_encoded: number;
  jpeg_encode_latency_ms: number;
  foul_inference_count: number;
}

export type CalibrationState =
  | 'idle'
  | 'source_preview'
  | 'clip_selecting'
  | 'processing'
  | 'review'
  | 'calibrating'
  | 'validating'
  | 'ready'
  | 'running'
  | 'recalibration_required';

export interface CalibrationTrack {
  track_id: number;
  label: string | null;
  team: TeamLabel;
  role: PlayerRole;
  sample_count: number;
  quality_score: number;
  last_update_frame: number | null;
  first_timestamp_ms?: number;
  last_timestamp_ms?: number;
  observation_count?: number;
  quality_observation_count?: number;
  representative_frame_index?: number;
  representative_timestamp_ms?: number;
  representative_bbox?: [number, number, number, number];
  representative_quality_score?: number;
}

export interface CalibrationValidationReport {
  passed: boolean;
  reasons: string[];
  home_track_count: number;
  away_track_count: number;
  home_sample_count: number;
  away_sample_count: number;
  home_intra_class_dispersion: number;
  away_intra_class_dispersion: number;
  inter_class_separation: number;
  leave_one_track_out_accuracy: number | null;
  goalkeeper_mapping_ready: boolean;
  referee_mapping_ready: boolean;
}

export interface TeamCalibrationState {
  type: 'team_calibration';
  state: CalibrationState;
  match_id: string;
  camera_id: string;
  bundle_path: string | null;
  ready: boolean;
  goalkeeper_mapping_ready: boolean;
  referee_mapping_ready: boolean;
  observed_frames: number;
  last_frame_index: number | null;
  tracks: CalibrationTrack[];
  validation_report: CalibrationValidationReport | null;
  source_url: string | null;
  clip_id: string | null;
  clip_start_ms: number | null;
  clip_end_ms: number | null;
  clip_duration_ms: number | null;
  review_video_url: string | null;
  metadata_url: string | null;
  job_id: string | null;
  processing_progress: number;
  processing_error: string | null;
}

export type WSMessage = FrameState | MetricsSnapshot | TeamCalibrationState;

export interface PipelineConfig {
  mode: string;
  video_source: string;
  device: string;
  inference_backend: 'auto' | 'pytorch' | 'onnx' | 'tensorrt' | string;
  enable_foul_detection: boolean;
  enable_recording: boolean;
  show_keypoints: boolean;
  show_tracking_boxes: boolean;
  show_2d_projection: boolean;
  foul_confidence_threshold: number;
  player_model_path: string;
  pitch_model_path: string;
  role_model_path: string;
  team_classifier_path: string | null;
  team_calibration_path: string | null;
  require_team_calibration: boolean;
  ball_model_path: string;
  enable_ball: boolean;
  role_detection_interval: number;
  team_classification_interval: number;
  track_activation_threshold: number;
  track_lost_buffer: number;
  track_matching_threshold: number;
  track_minimum_consecutive_frames: number;
  ball_detection_interval: number;
  ball_max_prediction_frames: number;
  camera_calibration_path: string;
  camera_rig_profile_path: string | null;
  enable_field_registration_v2: boolean;
  enable_undistortion: boolean;
  calibration_alpha: number;
  pitch_detection_interval: number;
  imgsz: number;
  player_confidence: number;
  player_iou: number;
  max_prediction_gap_frames: number;
  track_reactivation_window_frames: number;
}

export interface PitchData {
  players: PlayerState[];
  homography_status: HomographyStatus;
}
