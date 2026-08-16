import type {
  EvidenceValue,
  FoulFacts,
  MultiviewDecision,
  TeamLabel,
} from '../../types/multiview';

interface Props {
  facts: FoulFacts;
  decision: MultiviewDecision | null;
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
  return <span className={`mv-source-tag ${value.source}`}>{value.confirmed ? value.source.toUpperCase() : '待确认'}</span>;
}

function FactRow({
  label,
  value,
  children,
  modelHint,
}: {
  label: string;
  value: EvidenceValue;
  children: React.ReactNode;
  modelHint?: string;
}) {
  return (
    <label className="mv-fact-row">
      <span>{label}<SourceTag value={value} /></span>
      {children}
      {modelHint ? <small>模型建议：{modelHint}</small> : null}
    </label>
  );
}

export default function FoulFactsPanel({ facts, decision, onChange }: Props) {
  function setFact(name: keyof FoulFacts, value: unknown | null) {
    onChange({ ...facts, [name]: humanValue(value) } as FoulFacts);
  }

  function setOffender(value: TeamLabel | null) {
    const victim = value === 'home' ? 'away' : value === 'away' ? 'home' : null;
    onChange({
      ...facts,
      offender_team: humanValue(value),
      victim_team: facts.victim_team.confirmed ? facts.victim_team : humanValue(victim),
    });
  }

  const offence = facts.offence_confirmed.confirmed ? facts.offence_confirmed.value : null;
  const action = selectValue(facts.action);
  const isDive = action.toLowerCase() === 'dive';
  const showAttempt = selectValue(facts.tactical_impact) === 'dogso'
    || selectValue(facts.tactical_impact) === 'spa';

  return (
    <div className="mv-facts-form">
      <FactRow
        label="是否确认发生犯规"
        value={facts.offence_confirmed}
        modelHint={decision ? decision.decision_zh : undefined}
      >
        <select
          aria-label="是否确认发生犯规"
          value={offence === null ? '' : String(offence)}
          onChange={(event) => setFact('offence_confirmed', event.target.value === '' ? null : event.target.value === 'true')}
        >
          <option value="">请选择</option>
          <option value="true">确认犯规</option>
          <option value="false">确认不犯规</option>
        </select>
      </FactRow>

      {offence === true ? (
        <>
          <FactRow label="动作类型" value={facts.action} modelHint={decision?.action}>
            <select aria-label="动作类型" value={action} onChange={(event) => setFact('action', event.target.value || null)}>
              <option value="">请选择</option>
              {ACTIONS.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </FactRow>
          <div className="mv-fact-row-pair">
            <FactRow label="犯规方" value={facts.offender_team}>
              <select
                aria-label="犯规方"
                value={selectValue(facts.offender_team)}
                onChange={(event) => setOffender((event.target.value || null) as TeamLabel | null)}
              >
                <option value="">请选择</option>
                <option value="home">HOME</option>
                <option value="away">AWAY</option>
                <option value="unknown">UNKNOWN</option>
              </select>
            </FactRow>
            <FactRow label="受害方" value={facts.victim_team}>
              <select aria-label="受害方" value={selectValue(facts.victim_team)} onChange={(event) => setFact('victim_team', event.target.value || null)}>
                <option value="">请选择</option>
                <option value="home">HOME</option>
                <option value="away">AWAY</option>
                <option value="unknown">UNKNOWN</option>
              </select>
            </FactRow>
          </div>
          <FactRow label="球是否在比赛中" value={facts.ball_in_play}>
            <select aria-label="球是否在比赛中" value={selectValue(facts.ball_in_play)} onChange={(event) => setFact('ball_in_play', event.target.value === '' ? null : event.target.value === 'true')}>
              <option value="">请选择</option>
              <option value="true">是</option>
              <option value="false">否</option>
            </select>
          </FactRow>
          <FactRow label="HOME 防守球门" value={facts.home_defends_side}>
            <select aria-label="HOME 防守球门" value={selectValue(facts.home_defends_side)} onChange={(event) => setFact('home_defends_side', event.target.value || null)}>
              <option value="">请选择</option>
              <option value="left">左侧</option>
              <option value="right">右侧</option>
              <option value="unknown">未知</option>
            </select>
          </FactRow>

          {!isDive ? (
            <>
              <div className="mv-fact-row-pair">
                <FactRow label="发生身体接触" value={facts.contact}>
                  <select aria-label="发生身体接触" value={selectValue(facts.contact)} onChange={(event) => setFact('contact', event.target.value === '' ? null : event.target.value === 'true')}>
                    <option value="">请选择</option>
                    <option value="true">是</option>
                    <option value="false">否</option>
                  </select>
                </FactRow>
                <FactRow label="动作强度" value={facts.intensity} modelHint={decision?.severity}>
                  <select aria-label="动作强度" value={selectValue(facts.intensity)} onChange={(event) => setFact('intensity', event.target.value || null)}>
                    <option value="">请选择</option>
                    <option value="careless">草率</option>
                    <option value="reckless">鲁莽</option>
                    <option value="excessive_force">使用过分力量</option>
                    <option value="unknown">未知</option>
                  </select>
                </FactRow>
              </div>
              {facts.contact.value === true && facts.contact.confirmed ? (
                <FactRow label="接触部位" value={facts.contact_region}>
                  <select aria-label="接触部位" value={selectValue(facts.contact_region)} onChange={(event) => setFact('contact_region', event.target.value || null)}>
                    <option value="">请选择</option>
                    <option value="upper_body">上身</option>
                    <option value="lower_body">下身</option>
                    <option value="head">头部</option>
                    <option value="unknown">未知</option>
                  </select>
                </FactRow>
              ) : null}
              <FactRow label="战术影响" value={facts.tactical_impact}>
                <select aria-label="战术影响" value={selectValue(facts.tactical_impact)} onChange={(event) => setFact('tactical_impact', event.target.value || null)}>
                  <option value="">请选择</option>
                  <option value="none">无</option>
                  <option value="spa">SPA · 阻止有希望进攻</option>
                  <option value="dogso">DOGSO · 破坏明显得分机会</option>
                  <option value="unknown">未知</option>
                </select>
              </FactRow>
              {showAttempt ? (
                <FactRow label="尝试或争抢球" value={facts.attempt_to_play_ball}>
                  <select aria-label="尝试或争抢球" value={selectValue(facts.attempt_to_play_ball)} onChange={(event) => setFact('attempt_to_play_ball', event.target.value === '' ? null : event.target.value === 'true')}>
                    <option value="">请选择</option>
                    <option value="true">是</option>
                    <option value="false">否</option>
                  </select>
                </FactRow>
              ) : null}
            </>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
