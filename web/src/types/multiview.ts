export type ReviewState = 'pending' | 'reviewed' | 'archived' | 'uncertain';
export type RiskLevel = 'high' | 'medium' | 'low';

export interface EvidenceView {
  camera_id: string;
  display_name: string;
  role: 'main' | 'side' | 'replay' | 'other';
  sync_offset_ms: number;
  quality: string;
  media_url: string | null;
  media_kind: 'image' | 'video';
}

export interface MultiviewCase {
  case_id: string;
  title: string;
  match_name: string;
  match_clock: string;
  description: string;
  event_time_s: number;
  review_state: ReviewState;
  risk_level: RiskLevel;
  zone: string;
  videos: EvidenceView[];
  evidence_notes: string[];
}

export interface LocalizationBox {
  rect: [number, number, number, number];
  score: number;
  source: 'gradcam' | 'optical_flow' | 'scripted' | string;
}

export interface MultiviewDecision {
  event_id: string;
  case_id: string;
  timestamp: number;
  decision: string;
  decision_zh: string;
  action: string;
  severity: string;
  confidence: number;
  card: 'none' | 'yellow' | 'red';
  mode: 'model' | 'scripted';
  model: string | null;
  device: string | null;
  inference_ms: number | null;
  preprocess_ms: number | null;
  gradcam_ms: number | null;
  gpu_mem_mb: number | null;
  localization: Record<string, LocalizationBox>;
  localization_source: string | null;
  view_attention: number[];
  detail: Record<string, unknown>;
}

export interface MultiviewStatus {
  ready: boolean;
  cuda_available: boolean;
  gpu_name: string | null;
  model_code_ready: boolean;
  weights_ready: boolean;
  missing: string[];
  mode: 'model' | 'scripted_fallback';
  fallback_available: boolean;
}
