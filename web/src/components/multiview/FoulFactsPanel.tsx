import type {
  EvidenceValue,
  FoulFacts,
} from '../../types/multiview';

interface Props {
  facts: FoulFacts;
  onChange: (facts: FoulFacts) => void;
}

const ACTIONS = [
  'Tackle',
  'Standing Tackle',
  'High Leg',
  'Holding',
  'Pushing',
  'Elbowing',
  'Challenge',
  'Dive',
];

const sourceLabels: Record<string, string> = {
  human: '人工',
  model: '模型',
  geometry: '几何',
  rule: '推导',
};

function humanValue<T>(value: T | null): EvidenceValue<T> {
  return {
    value,
    source: 'human',
    confidence: null,
    confirmed: value !== null && value !== 'unknown',
  };
}

function selectValue(value: EvidenceValue): string {
  return value.confirmed && value.value !== null ? String(value.value) : '';
}

function SourceTag({ value }: { value: EvidenceValue }) {
  if (!value.confirmed) return <span className="mv-source-tag">待确认</span>;
  const label = sourceLabels[value.source] ?? value.source;
  const withConfidence = value.source === 'model' && typeof value.confidence === 'number'
    ? `${label} ${Math.round(value.confidence * 100)}%`
    : label;
  return <span className={`mv-source-tag ${value.source}`}>{withConfidence}</span>;
}

function Segmented({
  ariaLabel,
  value,
  options,
  onSelect,
}: {
  ariaLabel: string;
  value: string;
  options: { value: string; label: string }[];
  onSelect: (value: string | null) => void;
}) {
  return (
    <div className="mv-segmented" role="group" aria-label={ariaLabel}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          className={option.value === value ? 'active' : undefined}
          aria-pressed={option.value === value}
          onClick={() => onSelect(option.value === value ? null : option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

const YES_NO = [
  { value: 'true', label: '是' },
  { value: 'false', label: '否' },
];

function FactRow({
  label,
  value,
  children,
}: {
  label: string;
  value: EvidenceValue;
  children: React.ReactNode;
}) {
  return (
    <div className="mv-fact-row">
      <span>{label}<SourceTag value={value} /></span>
      {children}
    </div>
  );
}

export default function FoulFactsPanel({ facts, onChange }: Props) {
  function setFact(name: keyof FoulFacts, value: unknown | null) {
    onChange({ ...facts, [name]: humanValue(value) } as FoulFacts);
  }

  function setBool(name: keyof FoulFacts) {
    return (raw: string | null) => setFact(name, raw === null ? null : raw === 'true');
  }

  const offence = facts.offence_confirmed.confirmed ? facts.offence_confirmed.value : null;
  const action = selectValue(facts.action);
  const isDive = action.toLowerCase() === 'dive';
  const showAttempt = selectValue(facts.tactical_impact) === 'dogso'
    || selectValue(facts.tactical_impact) === 'spa';
  const hasModelPrefill = [facts.offence_confirmed, facts.action, facts.intensity]
    .some((item) => item.source === 'model' && item.confirmed);

  return (
    <div className="mv-facts-form">
      {hasModelPrefill ? (
        <p className="mv-facts-prefill-note">模型已完成预填，仅在有误时修改</p>
      ) : null}
      <FactRow
        label="确认犯规"
        value={facts.offence_confirmed}
      >
        <Segmented
          ariaLabel="是否确认发生犯规"
          value={offence === null ? '' : String(offence)}
          options={[
            { value: 'true', label: '犯规' },
            { value: 'false', label: '不犯规' },
          ]}
          onSelect={(raw) => setFact('offence_confirmed', raw === null ? null : raw === 'true')}
        />
      </FactRow>

      {offence === true ? (
        <>
          <FactRow label="动作类型" value={facts.action}>
            <select aria-label="动作类型" value={action} onChange={(event) => setFact('action', event.target.value || null)}>
              <option value="">请选择</option>
              {ACTIONS.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </FactRow>
          <FactRow label="犯规方" value={facts.offender_team}>
            <Segmented
              ariaLabel="犯规方"
              value={selectValue(facts.offender_team)}
              options={[
                { value: 'home', label: 'HOME' },
                { value: 'away', label: 'AWAY' },
              ]}
              onSelect={(raw) => setFact('offender_team', raw)}
            />
          </FactRow>
          <div className="mv-fact-row-pair">
            <FactRow label="球在比赛中" value={facts.ball_in_play}>
              <Segmented
                ariaLabel="球是否在比赛中"
                value={selectValue(facts.ball_in_play)}
                options={YES_NO}
                onSelect={setBool('ball_in_play')}
              />
            </FactRow>
            <FactRow label="HOME 防守" value={facts.home_defends_side}>
              <Segmented
                ariaLabel="HOME 防守球门方向"
                value={selectValue(facts.home_defends_side)}
                options={[
                  { value: 'left', label: '左侧' },
                  { value: 'right', label: '右侧' },
                ]}
                onSelect={(raw) => setFact('home_defends_side', raw)}
              />
            </FactRow>
          </div>

          {!isDive ? (
            <>
              <FactRow label="动作强度" value={facts.intensity}>
                <select aria-label="动作强度" value={selectValue(facts.intensity)} onChange={(event) => setFact('intensity', event.target.value || null)}>
                  <option value="">请选择</option>
                  <option value="careless">草率</option>
                  <option value="reckless">鲁莽</option>
                  <option value="excessive_force">使用过分力量</option>
                </select>
              </FactRow>
              <FactRow label="战术影响" value={facts.tactical_impact}>
                <Segmented
                  ariaLabel="战术影响"
                  value={selectValue(facts.tactical_impact)}
                  options={[
                    { value: 'none', label: '无' },
                    { value: 'spa', label: 'SPA' },
                    { value: 'dogso', label: 'DOGSO' },
                  ]}
                  onSelect={(raw) => setFact('tactical_impact', raw)}
                />
              </FactRow>
              {showAttempt ? (
                <FactRow label="尝试争抢球" value={facts.attempt_to_play_ball}>
                  <Segmented
                    ariaLabel="是否尝试或争抢球"
                    value={selectValue(facts.attempt_to_play_ball)}
                    options={YES_NO}
                    onSelect={setBool('attempt_to_play_ball')}
                  />
                </FactRow>
              ) : null}
            </>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
