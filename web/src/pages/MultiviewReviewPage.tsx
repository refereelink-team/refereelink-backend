import { useEffect, useMemo, useState } from 'react';
import {
  analyzeMultiviewCase,
  fetchMultiviewCases,
  fetchMultiviewStatus,
} from '../api/multiview';
import type {
  EvidenceView,
  LocalizationBox,
  MultiviewCase,
  MultiviewDecision,
  MultiviewStatus,
  ReviewState,
} from '../types/multiview';
import './multiview-review.css';

type Filter = 'all' | ReviewState;

const stateLabels: Record<ReviewState, string> = {
  pending: '待复核',
  reviewed: '已复核',
  archived: '已归档',
  uncertain: '暂不确定',
};

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function modeLabel(decision: MultiviewDecision | null, status: MultiviewStatus | null): string {
  if (decision?.mode === 'model') return 'CUDA 模型结果';
  if (decision?.mode === 'scripted') return '演示数据';
  return status?.ready ? '模型待命' : '演示模式';
}

function MediaFrame({
  view,
  box,
  primary,
  attention,
  onSelect,
}: {
  view: EvidenceView;
  box?: LocalizationBox;
  primary?: boolean;
  attention?: number;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      className={`mv-camera ${primary ? 'primary' : 'secondary'}`}
      onClick={onSelect}
      aria-label={`查看${view.display_name}`}
    >
      <div className="mv-camera-media">
        {view.media_url ? (
          view.media_kind === 'video' ? (
            <video src={view.media_url} muted preload="metadata" />
          ) : (
            <img src={view.media_url} alt={`${view.display_name}证据帧`} />
          )
        ) : (
          <div className="mv-media-empty">NO EVIDENCE MEDIA</div>
        )}
        {box && (
          <div
            className={`mv-focus-box ${box.source}`}
            style={{
              left: `${box.rect[0]}%`,
              top: `${box.rect[1]}%`,
              width: `${box.rect[2]}%`,
              height: `${box.rect[3]}%`,
            }}
          >
            <span>{box.source === 'gradcam' ? 'GRAD-CAM' : box.source === 'optical_flow' ? 'FLOW' : 'DEMO'}</span>
          </div>
        )}
        {attention !== undefined && (
          <div className="mv-attention-badge">ATTN {percent(attention)}</div>
        )}
      </div>
      <div className="mv-camera-meta">
        <strong>{view.camera_id.toUpperCase()} · {view.display_name}</strong>
        <span>{view.quality} · SYNC {view.sync_offset_ms >= 0 ? '+' : ''}{view.sync_offset_ms}ms</span>
      </div>
    </button>
  );
}

function PitchEvidence({ activeCase }: { activeCase: MultiviewCase }) {
  return (
    <div className="mv-pitch-wrap">
      <svg viewBox="0 0 360 210" role="img" aria-label="二维球场证据示意图">
        <rect x="8" y="8" width="344" height="194" />
        <line x1="180" y1="8" x2="180" y2="202" />
        <circle cx="180" cy="105" r="34" />
        <rect x="8" y="54" width="58" height="102" />
        <rect x="294" y="54" width="58" height="102" />
        <circle className="home" cx="228" cy="87" r="6" />
        <circle className="away" cx="248" cy="105" r="6" />
        <circle className="away" cx="274" cy="118" r="6" />
        <circle className="ref" cx="216" cy="128" r="5" />
        <path className="motion" d="M230 92 C246 93, 254 98, 267 110" />
      </svg>
      <div className="mv-pitch-caption">
        <span>事件区域</span>
        <strong>{activeCase.zone}</strong>
        <span>空间证据仅作辅助</span>
      </div>
    </div>
  );
}

export default function MultiviewReviewPage() {
  const [cases, setCases] = useState<MultiviewCase[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>('all');
  const [status, setStatus] = useState<MultiviewStatus | null>(null);
  const [decision, setDecision] = useState<MultiviewDecision | null>(null);
  const [analysisMessage, setAnalysisMessage] = useState('选择事件后运行多视角分析');
  const [analyzing, setAnalyzing] = useState(false);
  const [reviewState, setReviewState] = useState<ReviewState | null>(null);
  const [playhead, setPlayhead] = useState(50);
  const [playing, setPlaying] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([fetchMultiviewCases(), fetchMultiviewStatus()])
      .then(([loadedCases, loadedStatus]) => {
        setCases(loadedCases);
        setStatus(loadedStatus);
        if (loadedCases[0]) {
          setSelectedId(loadedCases[0].case_id);
          setSelectedCameraId(loadedCases[0].videos[0]?.camera_id ?? null);
        }
      })
      .catch((error) => setLoadError(String(error)));
  }, []);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setPlayhead((current) => (current >= 100 ? 0 : current + 1));
    }, 120);
    return () => window.clearInterval(timer);
  }, [playing]);

  const activeCase = useMemo(
    () => cases.find((item) => item.case_id === selectedId) ?? null,
    [cases, selectedId],
  );
  const filteredCases = useMemo(
    () => cases.filter((item) => filter === 'all' || item.review_state === filter),
    [cases, filter],
  );
  const activeView = activeCase?.videos.find((view) => view.camera_id === selectedCameraId)
    ?? activeCase?.videos[0]
    ?? null;
  const sideViews = activeCase?.videos.filter((view) => view.camera_id !== activeView?.camera_id) ?? [];
  const frameNumber = Math.round(12120 + playhead * 2.56);
  const pendingCount = cases.filter((item) => item.review_state === 'pending').length;
  const archivedCount = cases.filter((item) => item.review_state === 'archived').length;

  function selectCase(item: MultiviewCase) {
    setSelectedId(item.case_id);
    setSelectedCameraId(item.videos[0]?.camera_id ?? null);
    setDecision(null);
    setAnalysisMessage('证据已载入，等待运行模型');
    setReviewState(null);
    setPlayhead(50);
    setPlaying(false);
  }

  async function runAnalysis() {
    if (!activeCase || analyzing) return;
    setAnalyzing(true);
    setDecision(null);
    setAnalysisMessage('正在解码多机位片段并执行推理…');
    try {
      const response = await analyzeMultiviewCase(activeCase.case_id);
      if (response.status !== 'ok' || !response.decision) {
        throw new Error(response.message);
      }
      setDecision(response.decision);
      setAnalysisMessage(response.message);
      setPlayhead(50);
      setPlaying(false);
    } catch (error) {
      setAnalysisMessage(`分析失败：${String(error)}`);
    } finally {
      setAnalyzing(false);
    }
  }

  function applyReview(next: ReviewState) {
    setReviewState(next);
    setAnalysisMessage(`人工复核已在本次会话标记为“${stateLabels[next]}”`);
  }

  if (loadError) {
    return (
      <main className="mv-page mv-centered">
        <div className="mv-load-error">多视角页面加载失败：{loadError}</div>
        <a href="/">返回实时分析</a>
      </main>
    );
  }

  return (
    <main className="mv-page">
      <header className="mv-topbar">
        <div className="mv-brand">
          <div className="mv-brand-mark"><span /></div>
          <div>
            <div className="mv-eyebrow">SOCCER ASSISTED OFFICIATING · EVIDENCE DESK</div>
            <h1>多视角犯规判罚中心</h1>
            <p>{activeCase ? `${activeCase.match_name} · ${activeCase.match_clock}` : '正在载入案例'}</p>
          </div>
        </div>
        <div className="mv-top-stats">
          <span className={`mv-system-pill ${status?.ready ? 'ready' : 'demo'}`}>
            <i />{status?.ready ? 'MViT 模型就绪' : '演示模式'}
          </span>
          <span>机位 <b>{activeCase?.videos.length ?? 0}</b></span>
          <span>待复核 <b>{pendingCount}</b></span>
          <span>已归档 <b>{archivedCount}</b></span>
          <a className="mv-back-link" href="/">返回实时分析</a>
        </div>
      </header>

      <div className="mv-ticker">
        <span>多机位证据同步 · 模型推理与人工结论分层记录 · Grad-CAM 仅解释模型关注区域</span>
        <strong className={decision?.mode === 'model' ? 'model' : 'demo'}>{modeLabel(decision, status)}</strong>
      </div>

      <section className="mv-workspace">
        <aside className="mv-panel mv-queue">
          <div className="mv-panel-title">
            <div><span>EVENT QUEUE</span><h2>争议事件队列</h2></div>
            <b>{filteredCases.length}</b>
          </div>
          <div className="mv-filters">
            {([
              ['all', '全部'],
              ['pending', '待复核'],
              ['reviewed', '已复核'],
              ['archived', '已归档'],
              ['uncertain', '暂不确定'],
            ] as [Filter, string][]).map(([value, label]) => (
              <button key={value} className={filter === value ? 'active' : ''} onClick={() => setFilter(value)}>
                {label}
              </button>
            ))}
          </div>
          <div className="mv-event-list">
            {filteredCases.map((item, index) => (
              <button
                key={item.case_id}
                className={`mv-event-card ${item.case_id === selectedId ? 'active' : ''}`}
                onClick={() => selectCase(item)}
              >
                <div className="mv-event-card-top">
                  <span>{item.case_id.toUpperCase()}</span>
                  <em className={item.risk_level}>{item.risk_level === 'high' ? '高风险' : item.risk_level === 'medium' ? '中风险' : '低风险'}</em>
                </div>
                <strong>{item.title}</strong>
                <p>{item.match_clock} · {item.zone} · {item.videos.length} 机位</p>
                <div><i>{String(index + 1).padStart(2, '0')}</i><span className={item.review_state}>{stateLabels[item.review_state]}</span></div>
              </button>
            ))}
          </div>
        </aside>

        <section className="mv-center-column">
          <div className="mv-panel mv-event-heading">
            <div>
              <span>{activeCase?.case_id.toUpperCase() ?? 'NO CASE'}</span>
              <h2>{activeCase?.title ?? '等待案例'}</h2>
            </div>
            <div className="mv-heading-tags">
              <span className={`risk ${activeCase?.risk_level ?? 'low'}`}>{activeCase?.risk_level === 'high' ? '高风险' : '常规复核'}</span>
              <span>{activeCase?.videos.length ?? 0} 路证据</span>
              <span className={reviewState ?? activeCase?.review_state ?? 'pending'}>
                {stateLabels[reviewState ?? activeCase?.review_state ?? 'pending']}
              </span>
            </div>
          </div>

          <div className="mv-panel mv-camera-grid">
            {activeView && (
              <MediaFrame
                view={activeView}
                box={decision?.localization[activeView.camera_id]}
                attention={decision?.view_attention[activeCase?.videos.findIndex((view) => view.camera_id === activeView.camera_id) ?? 0]}
                primary
                onSelect={() => undefined}
              />
            )}
            <div className="mv-side-cameras">
              {sideViews.map((view) => (
                <MediaFrame
                  key={view.camera_id}
                  view={view}
                  box={decision?.localization[view.camera_id]}
                  attention={decision?.view_attention[activeCase?.videos.findIndex((item) => item.camera_id === view.camera_id) ?? 0]}
                  onSelect={() => setSelectedCameraId(view.camera_id)}
                />
              ))}
            </div>
          </div>

          <div className="mv-panel mv-timeline-panel">
            <div className="mv-timeline-head">
              <div><span>SYNCHRONIZED CLIP</span><strong>多机位同期证据时间轴</strong></div>
              <div className="mv-frame-readout">FRAME {frameNumber} · {playhead.toFixed(0)}%</div>
            </div>
            <div className="mv-timeline-rows">
              {activeCase?.videos.map((view, index) => (
                <div className="mv-timeline-row" key={view.camera_id}>
                  <label>{view.camera_id.toUpperCase()}</label>
                  <div className="mv-timeline-track"><i style={{ width: `${86 - index * 4}%` }} /><b style={{ left: `${playhead}%` }} /></div>
                </div>
              ))}
            </div>
            <div className="mv-timeline-controls">
              <button onClick={() => setPlayhead((value) => Math.max(0, value - 2))}>−1 帧</button>
              <button className="play" onClick={() => setPlaying((value) => !value)}>{playing ? '暂停' : '播放'}</button>
              <button onClick={() => setPlayhead((value) => Math.min(100, value + 2))}>+1 帧</button>
              <span><i className="key" /> 关键帧</span><span><i className="alert" /> 模型关注</span>
              <input aria-label="证据时间轴" type="range" min="0" max="100" value={playhead} onChange={(event) => setPlayhead(Number(event.target.value))} />
            </div>
          </div>

          <div className="mv-panel mv-actions">
            <button className="analyze" disabled={!activeCase || analyzing} onClick={runAnalysis}>
              {analyzing ? '正在分析…' : '运行多视角分析'}
            </button>
            <span className="mv-action-status">{analysisMessage}</span>
            <div className="mv-review-actions">
              <button disabled={!decision} onClick={() => applyReview('uncertain')}>暂不确定</button>
              <button disabled={!decision} onClick={() => applyReview('reviewed')}>确认建议</button>
              <button disabled={!decision} onClick={() => applyReview('archived')}>完成归档</button>
            </div>
          </div>
        </section>

        <aside className="mv-right-column">
          {activeCase && (
            <div className="mv-panel mv-pitch-panel">
              <div className="mv-panel-title compact"><div><span>SPATIAL EVIDENCE</span><h2>二维球场空间证据</h2></div><b>辅助</b></div>
              <PitchEvidence activeCase={activeCase} />
            </div>
          )}

          <div className="mv-panel mv-decision-panel">
            <div className="mv-panel-title compact"><div><span>MODEL DECISION</span><h2>结构化判罚记录</h2></div></div>
            {decision ? (
              <>
                <div className={`mv-decision-hero ${decision.card}`}>
                  <div><span>辅助建议</span><strong>{decision.decision_zh}</strong></div>
                  <b>{percent(decision.confidence)}</b>
                </div>
                <dl className="mv-record-grid">
                  <div><dt>动作类型</dt><dd>{decision.action}</dd></div>
                  <div><dt>严重程度</dt><dd>{decision.severity}</dd></div>
                  <div><dt>解释来源</dt><dd>{decision.localization_source ?? '无定位'}</dd></div>
                  <div><dt>推理模式</dt><dd className={decision.mode}>{decision.mode === 'model' ? decision.model : 'SCRIPTED DEMO'}</dd></div>
                  <div><dt>前向耗时</dt><dd>{decision.inference_ms ? `${decision.inference_ms} ms` : '—'}</dd></div>
                  <div><dt>Grad-CAM</dt><dd>{decision.gradcam_ms ? `${decision.gradcam_ms} ms` : '—'}</dd></div>
                </dl>
              </>
            ) : (
              <div className="mv-decision-empty"><i />运行分析后显示模型建议、置信度与解释来源</div>
            )}
          </div>

          <div className="mv-panel mv-chain-panel">
            <div className="mv-panel-title compact"><div><span>CHAIN OF CUSTODY</span><h2>事件证据链</h2></div></div>
            <ol>
              <li className="done"><b>01</b><div><strong>多机位同步</strong><span>偏差检查与片段对齐</span></div></li>
              <li className={decision ? 'done' : 'active'}><b>02</b><div><strong>模型联合分析</strong><span>MViT 动作与严重程度</span></div></li>
              <li className={decision ? 'done' : ''}><b>03</b><div><strong>Grad-CAM 解释</strong><span>定位模型关注区域</span></div></li>
              <li className={reviewState ? 'done' : ''}><b>04</b><div><strong>人工复核</strong><span>裁判确认或拒绝建议</span></div></li>
            </ol>
          </div>

          <div className="mv-panel mv-notes-panel">
            <div className="mv-panel-title compact"><div><span>EVIDENCE NOTES</span><h2>证据质量</h2></div></div>
            <ul>{activeCase?.evidence_notes.map((note) => <li key={note}>{note}</li>)}</ul>
            {!status?.ready && (
              <p className="mv-runtime-note">真实模型未就绪：{status?.missing.join('；') || '正在检查运行资产'}</p>
            )}
          </div>
        </aside>
      </section>
    </main>
  );
}
