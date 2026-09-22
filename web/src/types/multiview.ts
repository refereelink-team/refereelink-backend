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
  review_revision: number;
  capture_state?: 'capture_ready' | 'capture_failed';
}

export type EvidenceSource = 'model' | 'human' | 'geometry' | 'rule';
export type TeamLabel = 'home' | 'away' | 'unknown';
export type DefendsSide = 'left' | 'right' | 'unknown';
export type ReviewAssessmentStatus = 'complete' | 'incomplete' | 'unsupported';
export type RestartType = 'play_on' | 'direct_free_kick' | 'indirect_free_kick' | 'penalty' | 'previous_restart' | 'unknown';
export type SanctionType = 'none' | 'yellow_card' | 'red_card' | 'pending';

export interface EvidenceValue<T = unknown> {
  value: T | null;
  source: EvidenceSource;
  confidence: number | null;
  confirmed: boolean;
}

export interface FoulLocation {
  x_m: number;
  y_m: number;
  source: 'human' | 'vision';
  confirmed: boolean;
}

export interface LocationGeometry {
  half: 'left' | 'right' | 'center';
  zone: string;
  penalty_area_side: 'left' | 'right' | null;
  in_penalty_area: boolean;
  in_offender_own_penalty_area: boolean | null;
  distance_to_left_goal_m: number;
  distance_to_right_goal_m: number;
}

export interface FoulFacts {
  offence_confirmed: EvidenceValue<boolean>;
  action: EvidenceValue<string>;
  offender_team: EvidenceValue<TeamLabel>;
  victim_team: EvidenceValue<TeamLabel>;
  ball_in_play: EvidenceValue<boolean>;
  contact: EvidenceValue<boolean>;
  contact_region: EvidenceValue<string>;
  intensity: EvidenceValue<string>;
  attempt_to_play_ball: EvidenceValue<boolean>;
  tactical_impact: EvidenceValue<string>;
  location: FoulLocation | null;
  home_defends_side: EvidenceValue<DefendsSide>;
}

export interface RuleTraceEntry {
  rule_id: string;
  law: string;
  section: string;
  facts_used: string[];
  result: string;
  priority: number;
  law_excerpt: string;
}

export interface RuleAssessment {
  status: ReviewAssessmentStatus;
  restart: RestartType;
  sanction: SanctionType;
  ruleset_version: string;
  rule_trace: RuleTraceEntry[];
  missing_facts: string[];
  conflicts: string[];
  geometry: LocationGeometry | null;
  explanation_template: string;
}

export interface ReviewRecord {
  case_id: string;
  analysis_id: string | null;
  revision: number;
  facts: FoulFacts;
  assessment: RuleAssessment;
  review_state: ReviewState;
  created_at: string;
  updated_at: string;
}

export interface CandidateScore {
  label: string;
  confidence: number;
}

export interface TemporalBin {
  start_s: number;
  end_s: number;
  score: number;
}

export interface LocalizationBox {
  rect: [number, number, number, number];
  score: number;
  source: 'gradcam' | 'optical_flow' | 'scripted' | string;
  active_start_s?: number | null;
  active_end_s?: number | null;
  peak_s?: number | null;
  temporal_bins?: TemporalBin[];
  temporal_source?: 'gradcam' | 'event_prior' | null;
  display_tier?: 'normal' | 'caution' | 'hidden';
  reliable?: boolean;
  reliability_score?: number;
  reliability_reasons?: string[];
}

export interface MultiviewDecision {
  analysis_id: string;
  event_id: string;
  case_id: string;
  timestamp: number;
  decision: string;
  decision_zh: string;
  action: string;
  severity: string;
  confidence: number;
  action_candidates: CandidateScore[];
  severity_candidates: CandidateScore[];
  checkpoint_hash: string | null;
  ruleset_compatible: boolean;
  card: 'none' | 'yellow' | 'red';
  suggested_intensity: string | null;
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

export interface ExplanationResponse {
  case_id: string;
  revision: number;
  source: 'template' | 'local_llm';
  summary: string;
  restart: RestartType;
  sanction: SanctionType;
  rule_ids: string[];
  fallback_reason: string | null;
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

export interface LiveCameraStatus {
  camera_id: string;
  display_name: string;
  role: EvidenceView['role'];
  online: boolean;
  latest_segment_age_s: number | null;
  buffer_seconds: number;
  reconnect_count: number;
  last_error: string | null;
}

export interface LiveMultiviewStatus {
  configured: boolean;
  ready: boolean;
  trigger_ready: boolean;
  buffer_target_s: number;
  min_buffer_s: number;
  cameras: LiveCameraStatus[];
  message: string;
}
