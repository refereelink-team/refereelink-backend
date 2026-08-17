import type { ExplanationResponse, RuleAssessment } from '../../types/multiview';

const restartLabels: Record<string, string> = {
  play_on: '继续比赛',
  direct_free_kick: '直接任意球',
  indirect_free_kick: '间接任意球',
  penalty: '点球',
  previous_restart: '维持原恢复方式',
  unknown: '待确认',
};

const sanctionLabels: Record<string, string> = {
  none: '不出牌',
  yellow_card: '黄牌',
  red_card: '红牌',
  pending: '待确认',
};

const factLabels: Record<string, string> = {
  offence_confirmed: '是否确认犯规',
  action: '动作类型',
  offender_team: '犯规方',
  victim_team: '受害方',
  ball_in_play: '球是否在比赛中',
  contact: '身体接触',
  intensity: '动作强度',
  tactical_impact: '战术影响',
  attempt_to_play_ball: '是否争抢球',
  location: '犯规位置',
  home_defends_side: 'HOME 防守方向',
};

export default function RuleAssessmentPanel({
  assessment,
  explanation,
  explaining,
  draftChanged,
}: {
  assessment: RuleAssessment | null;
  explanation: ExplanationResponse | null;
  explaining?: boolean;
  draftChanged?: boolean;
}) {
  if (draftChanged) {
    return (
      <div className="mv-decision-empty stale">
        <i />
        <strong>人工事实已修改</strong>
        <span>保存后重新计算</span>
      </div>
    );
  }
  if (!assessment) {
    return <div className="mv-decision-empty"><i />保存事实后计算规则结论</div>;
  }
  return (
    <div className="mv-rule-assessment">
      <div className={`mv-rule-hero ${assessment.sanction}`}>
        <div><span>比赛重启</span><strong>{restartLabels[assessment.restart]}</strong></div>
        <div><span>纪律处罚</span><strong>{sanctionLabels[assessment.sanction]}</strong></div>
        <em>{assessment.status === 'complete' ? '完整结论' : assessment.status === 'unsupported' ? '超出范围' : '部分结论'}</em>
      </div>
      {assessment.missing_facts.length ? (
        <div className="mv-rule-alert missing">
          仍需确认：{assessment.missing_facts.map((item) => factLabels[item] ?? item).join('、')}
        </div>
      ) : null}
      {assessment.conflicts.map((conflict) => <div className="mv-rule-alert conflict" key={conflict}>{conflict}</div>)}
      {assessment.geometry ? (
        <div className="mv-geometry-result">
          <span>区域</span>
          {assessment.geometry.zone} ·
          {assessment.geometry.in_offender_own_penalty_area === null
            ? ' 本方禁区条件无法确定'
            : assessment.geometry.in_offender_own_penalty_area
              ? ' 犯规方本方禁区内'
              : ' 犯规方本方禁区外'}
        </div>
      ) : null}
      <p className="mv-rule-summary">{explanation?.summary ?? assessment.explanation_template}</p>
      <div className="mv-explanation-controls">
        <span>
          {explanation
            ? `${explanation.source === 'local_llm' ? 'AI 生成' : '模板'} · REV ${explanation.revision}`
            : explaining
              ? '正在生成 AI 判罚说明…'
              : assessment.ruleset_version}
        </span>
      </div>
      <div className="mv-rule-trace">
        {assessment.rule_trace.map((item) => (
          <details key={item.rule_id}>
            <summary><b>{item.law}</b><span>{item.rule_id}</span></summary>
            <p>{item.result}</p>
            {item.law_excerpt ? <blockquote className="mv-law-quote">{item.law_excerpt}</blockquote> : null}
            <small>{item.section}</small>
          </details>
        ))}
      </div>
    </div>
  );
}
