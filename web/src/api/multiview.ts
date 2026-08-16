import type {
  ExplanationResponse,
  FoulFacts,
  MultiviewCase,
  MultiviewDecision,
  MultiviewStatus,
  ReviewRecord,
  ReviewState,
} from '../types/multiview';

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const detail = typeof payload.detail === 'string'
      ? payload.detail
      : payload.detail
        ? JSON.stringify(payload.detail)
        : `HTTP ${response.status}`;
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export async function fetchMultiviewReview(
  caseId: string,
): Promise<{ review: ReviewRecord | null; analysis: MultiviewDecision | null }> {
  return readJson(await fetch(`/api/multiview/cases/${encodeURIComponent(caseId)}/review`));
}

export async function updateMultiviewReview(
  caseId: string,
  payload: {
    expected_revision: number;
    analysis_id: string | null;
    facts: FoulFacts;
    review_state: ReviewState;
  },
): Promise<{ review: ReviewRecord }> {
  return readJson(
    await fetch(`/api/multiview/cases/${encodeURIComponent(caseId)}/review`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  );
}

export async function explainMultiviewReview(
  caseId: string,
  revision: number,
  useLlm = true,
): Promise<ExplanationResponse> {
  return readJson(
    await fetch(`/api/multiview/cases/${encodeURIComponent(caseId)}/explanation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ revision, use_llm: useLlm }),
    }),
  );
}

export async function fetchMultiviewCases(): Promise<MultiviewCase[]> {
  const payload = await readJson<{ cases: MultiviewCase[] }>(
    await fetch('/api/multiview/cases'),
  );
  return payload.cases;
}

export async function fetchMultiviewStatus(): Promise<MultiviewStatus> {
  return readJson<MultiviewStatus>(await fetch('/api/multiview/status'));
}

export async function analyzeMultiviewCase(
  caseId: string,
): Promise<{ status: 'ok' | 'error'; message: string; decision: MultiviewDecision | null }> {
  return readJson(
    await fetch('/api/multiview/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ case_id: caseId, device: 'auto' }),
    }),
  );
}
