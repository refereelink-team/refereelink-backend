import type { CandidateScore, MultiviewDecision } from '../../types/multiview';

function CandidateList({ title, candidates }: { title: string; candidates: CandidateScore[] }) {
  return (
    <div className="mv-candidate-group">
      <span>{title}</span>
      {candidates.map((candidate) => (
        <div key={candidate.label} className="mv-candidate-row">
          <b>{candidate.label}</b>
          <i><em style={{ width: `${candidate.confidence * 100}%` }} /></i>
          <strong>{Math.round(candidate.confidence * 100)}%</strong>
        </div>
      ))}
    </div>
  );
}

export default function ModelEvidencePanel({ decision }: { decision: MultiviewDecision | null }) {
  if (!decision) {
    return <div className="mv-decision-empty"><i />运行分析后显示原始视觉建议</div>;
  }
  return (
    <div className="mv-model-evidence">
      <div className="mv-model-warning">模型建议不是规则结论；Grad-CAM 仅表示模型关注区域。</div>
      <CandidateList title="动作 Top-K" candidates={decision.action_candidates} />
      <CandidateList title="犯规 / 牌级 Top-K" candidates={decision.severity_candidates} />
      <dl className="mv-record-grid">
        <div><dt>模型</dt><dd>{decision.mode === 'model' ? decision.model : 'SCRIPTED DEMO'}</dd></div>
        <div><dt>前向耗时</dt><dd>{decision.inference_ms ? `${decision.inference_ms} ms` : '—'}</dd></div>
        <div><dt>解释来源</dt><dd>{decision.localization_source ?? '无定位'}</dd></div>
        <div><dt>权重哈希</dt><dd title={decision.checkpoint_hash ?? undefined}>{decision.checkpoint_hash?.slice(0, 10) ?? '—'}</dd></div>
      </dl>
    </div>
  );
}
