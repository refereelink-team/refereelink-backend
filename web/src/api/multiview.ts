import type { MultiviewCase, MultiviewDecision, MultiviewStatus } from '../types/multiview';

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
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
