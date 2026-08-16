import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  analyzeMultiviewCase,
  explainMultiviewReview,
  fetchMultiviewCases,
  fetchMultiviewReview,
  fetchMultiviewStatus,
  updateMultiviewReview,
} from '../api/multiview';
import FoulFactsPanel from '../components/multiview/FoulFactsPanel';
import FoulLocationPitch from '../components/multiview/FoulLocationPitch';
import ModelEvidencePanel from '../components/multiview/ModelEvidencePanel';
import RuleAssessmentPanel from '../components/multiview/RuleAssessmentPanel';
import type {
  DefendsSide,
  EvidenceView,
  ExplanationResponse,
  FoulFacts,
  LocalizationBox,
  MultiviewCase,
  MultiviewDecision,
  MultiviewStatus,
  ReviewRecord,
  ReviewState,
  TeamLabel,
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

interface LocalizationWindow {
  startS: number;
  endS: number;
  peakS: number;
  source: 'gradcam' | 'event_prior';
}

function localizationWindow(box: LocalizationBox, eventTimeS: number): LocalizationWindow {
  const hasModelWindow = box.active_start_s !== null
    && box.active_start_s !== undefined
    && box.active_end_s !== null
    && box.active_end_s !== undefined;
  if (hasModelWindow) {
    const startS = Math.max(0, box.active_start_s as number);
    const endS = Math.max(startS, box.active_end_s as number);
    return {
      startS,
      endS,
      peakS: Math.min(endS, Math.max(startS, box.peak_s ?? (startS + endS) / 2)),
      source: box.temporal_source === 'event_prior' ? 'event_prior' : 'gradcam',
    };
  }
  const peakS = Math.max(0, eventTimeS);
  return {
    startS: Math.max(0, peakS - 0.5),
    endS: peakS + 0.5,
    peakS,
    source: 'event_prior',
  };
}

function seconds(value: number): string {
  return `${value.toFixed(2)}s`;
}

function modeLabel(decision: MultiviewDecision | null, status: MultiviewStatus | null): string {
  if (decision?.mode === 'model') return 'CUDA 模型结果';
  if (decision?.mode === 'scripted') return '演示数据';
  return status?.ready ? '模型待命' : '演示模式';
}

function unconfirmed<T>(value: T | null = null, source: 'human' | 'model' = 'human', confidence: number | null = null) {
  return { value, source, confidence, confirmed: false };
}

function emptyFacts(): FoulFacts {
  return {
    offence_confirmed: unconfirmed<boolean>(),
    action: unconfirmed<string>(),
    offender_team: unconfirmed<TeamLabel>(),
    victim_team: unconfirmed<TeamLabel>(),
    ball_in_play: unconfirmed<boolean>(),
    contact: unconfirmed<boolean>(),
    contact_region: unconfirmed<string>(),
    intensity: unconfirmed<string>(),
    attempt_to_play_ball: unconfirmed<boolean>(),
    tactical_impact: unconfirmed<string>(),
    location: null,
    home_defends_side: unconfirmed<DefendsSide>(),
  };
}

function factsFromDecision(decision: MultiviewDecision): FoulFacts {
  const facts = emptyFacts();
  facts.offence_confirmed = unconfirmed(
    decision.severity !== 'No Offence',
    'model',
    decision.severity_candidates[0]?.confidence ?? decision.confidence,
  );
  facts.action = unconfirmed(
    decision.action,
    'model',
    decision.action_candidates[0]?.confidence ?? decision.confidence,
  );
  const intensity = decision.card === 'red'
    ? 'excessive_force'
    : decision.card === 'yellow'
      ? 'reckless'
      : 'careless';
  facts.intensity = unconfirmed(intensity, 'model', decision.confidence);
  return facts;
}

function MediaFrame({
  view,
  box,
  primary,
  attention,
  onSelect,
  onVideoRef,
  onVideoReady,
  currentTimeS,
  eventTimeS,
}: {
  view: EvidenceView;
  box?: LocalizationBox;
  primary?: boolean;
  attention?: number;
  onSelect: () => void;
  onVideoRef: (cameraId: string, element: HTMLVideoElement | null) => void;
  onVideoReady: (cameraId: string) => void;
  currentTimeS: number;
  eventTimeS: number;
}) {
  const window = box ? localizationWindow(box, eventTimeS) : null;
  const showBox = Boolean(
    box
    && window
    && currentTimeS >= window.startS
    && currentTimeS <= window.endS,
  );
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
            <video
              ref={(element) => onVideoRef(view.camera_id, element)}
              src={view.media_url}
              muted
              playsInline
              preload="auto"
              onLoadedMetadata={() => onVideoReady(view.camera_id)}
            />
          ) : (
            <img src={view.media_url} alt={`${view.display_name}证据帧`} />
          )
        ) : (
          <div className="mv-media-empty">NO EVIDENCE MEDIA</div>
        )}
        {box && window && (
          <div
            className={`mv-focus-box ${box.source} ${showBox ? 'active' : ''}`}
            aria-hidden={!showBox}
            style={{
              left: `${box.rect[0]}%`,
              top: `${box.rect[1]}%`,
              width: `${box.rect[2]}%`,
              height: `${box.rect[3]}%`,
            }}
          >
            <span>
              {box.source === 'gradcam' ? 'GRAD-CAM' : box.source === 'optical_flow' ? 'FLOW' : 'DEMO'}
              {' · '}{seconds(window.startS)}–{seconds(window.endS)}
            </span>
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

export default function MultiviewReviewPage() {
  const [cases, setCases] = useState<MultiviewCase[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>('all');
  const [status, setStatus] = useState<MultiviewStatus | null>(null);
  const [decision, setDecision] = useState<MultiviewDecision | null>(null);
  const [review, setReview] = useState<ReviewRecord | null>(null);
  const [facts, setFacts] = useState<FoulFacts>(() => emptyFacts());
  const [reviewDirty, setReviewDirty] = useState(false);
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [analysisMessage, setAnalysisMessage] = useState('选择事件后运行多视角分析');
  const [analyzing, setAnalyzing] = useState(false);
  const [savingReview, setSavingReview] = useState(false);
  const [explaining, setExplaining] = useState(false);
  const [reviewState, setReviewState] = useState<ReviewState | null>(null);
  const [playhead, setPlayhead] = useState(50);
  const [playing, setPlaying] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const videoRefs = useRef(new Map<string, HTMLVideoElement>());
  const playheadRef = useRef(playhead);
  const playingRef = useRef(playing);

  useEffect(() => {
    playheadRef.current = playhead;
  }, [playhead]);

  useEffect(() => {
    playingRef.current = playing;
  }, [playing]);

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
    if (!selectedId) return;
    let cancelled = false;
    setReview(null);
    setDecision(null);
    setFacts(emptyFacts());
    setReviewDirty(false);
    setExplanation(null);
    fetchMultiviewReview(selectedId)
      .then((payload) => {
        if (cancelled) return;
        setReview(payload.review);
        setDecision(payload.analysis);
        setFacts(payload.review?.facts ?? (payload.analysis ? factsFromDecision(payload.analysis) : emptyFacts()));
        setReviewDirty(false);
        setReviewState(payload.review?.review_state ?? null);
        if (payload.review) {
          setAnalysisMessage(`已恢复审核修订 REV ${payload.review.revision}`);
        }
      })
      .catch((error) => {
        if (!cancelled) setAnalysisMessage(`审核记录加载失败：${String(error)}`);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

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

  const pauseAllVideos = useCallback(() => {
    videoRefs.current.forEach((video) => video.pause());
  }, []);

  const registerVideo = useCallback((cameraId: string, element: HTMLVideoElement | null) => {
    if (element) {
      videoRefs.current.set(cameraId, element);
    } else {
      videoRefs.current.delete(cameraId);
    }
  }, []);

  function viewOffsetSeconds(cameraId: string): number {
    const view = activeCase?.videos.find((item) => item.camera_id === cameraId);
    return (view?.sync_offset_ms ?? 0) / 1000;
  }

  function targetTime(video: HTMLVideoElement, cameraId: string, percentValue: number): number {
    if (!Number.isFinite(video.duration) || video.duration <= 0) return 0;
    const base = video.duration * percentValue / 100;
    return Math.min(video.duration, Math.max(0, base + viewOffsetSeconds(cameraId)));
  }

  function fallbackDuration(view: EvidenceView): number {
    return Math.max(5, (activeCase?.event_time_s ?? 3) + 2, Math.abs(view.sync_offset_ms) / 1000 + 5);
  }

  function viewDuration(view: EvidenceView): number {
    const video = videoRefs.current.get(view.camera_id);
    return video && Number.isFinite(video.duration) && video.duration > 0
      ? video.duration
      : fallbackDuration(view);
  }

  function viewCurrentTime(view: EvidenceView): number {
    const video = videoRefs.current.get(view.camera_id);
    if (video && Number.isFinite(video.currentTime)) return video.currentTime;
    return Math.max(0, viewDuration(view) * playhead / 100 + view.sync_offset_ms / 1000);
  }

  function viewEventTime(view: EvidenceView): number {
    return Math.max(0, (activeCase?.event_time_s ?? 3) + view.sync_offset_ms / 1000);
  }

  function timelinePosition(timeS: number, durationS: number): number {
    if (!Number.isFinite(durationS) || durationS <= 0) return 0;
    return Math.min(100, Math.max(0, timeS / durationS * 100));
  }

  function seekTo(percentValue: number) {
    const bounded = Math.min(100, Math.max(0, percentValue));
    playheadRef.current = bounded;
    setPlayhead(bounded);
    videoRefs.current.forEach((video, cameraId) => {
      if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
        video.currentTime = targetTime(video, cameraId, bounded);
      }
    });
  }

  const handleVideoReady = useCallback((cameraId: string) => {
    const video = videoRefs.current.get(cameraId);
    if (!video) return;
    const percentValue = playheadRef.current;
    if (Number.isFinite(video.duration) && video.duration > 0) {
      const offset = activeCase?.videos.find((item) => item.camera_id === cameraId)?.sync_offset_ms ?? 0;
      const base = video.duration * percentValue / 100 + offset / 1000;
      video.currentTime = Math.min(video.duration, Math.max(0, base));
    }
    if (playingRef.current) {
      void video.play().catch(() => undefined);
    }
  }, [activeCase]);

  function togglePlayback() {
    if (playing) {
      pauseAllVideos();
      setPlaying(false);
      return;
    }
    const startPercent = playheadRef.current >= 99.5 ? 0 : playheadRef.current;
    seekTo(startPercent);
    setPlaying(true);
    videoRefs.current.forEach((video) => {
      void video.play().catch(() => undefined);
    });
  }

  function stepFrame(direction: -1 | 1) {
    pauseAllVideos();
    setPlaying(false);
    const masterId = activeCase?.videos[0]?.camera_id;
    const master = masterId ? videoRefs.current.get(masterId) : undefined;
    const percentPerFrame = master && Number.isFinite(master.duration) && master.duration > 0
      ? 100 / (master.duration * 25)
      : 0.5;
    seekTo(playheadRef.current + direction * percentPerFrame);
  }

  useEffect(() => {
    if (!playing || !activeCase) return;
    let animationFrame = 0;
    const masterId = activeCase.videos[0]?.camera_id;

    const update = () => {
      const master = masterId ? videoRefs.current.get(masterId) : undefined;
      if (master && Number.isFinite(master.duration) && master.duration > 0) {
        const nextPlayhead = Math.min(100, Math.max(0, master.currentTime / master.duration * 100));
        playheadRef.current = nextPlayhead;
        setPlayhead(nextPlayhead);

        videoRefs.current.forEach((video, cameraId) => {
          if (video === master || video.paused || !Number.isFinite(video.duration)) return;
          const expected = targetTime(video, cameraId, nextPlayhead);
          if (Math.abs(video.currentTime - expected) > 0.18) {
            video.currentTime = expected;
          }
        });

        if (master.ended || nextPlayhead >= 99.9) {
          pauseAllVideos();
          setPlaying(false);
          return;
        }
      }
      animationFrame = window.requestAnimationFrame(update);
    };

    animationFrame = window.requestAnimationFrame(update);
    return () => window.cancelAnimationFrame(animationFrame);
  }, [activeCase, pauseAllVideos, playing]);

  function selectCase(item: MultiviewCase) {
    pauseAllVideos();
    setSelectedId(item.case_id);
    setSelectedCameraId(item.videos[0]?.camera_id ?? null);
    setDecision(null);
    setReview(null);
    setFacts(emptyFacts());
    setReviewDirty(false);
    setExplanation(null);
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
      if (!review) setFacts(factsFromDecision(response.decision));
      setAnalysisMessage(response.message);
      const mainView = activeCase.videos[0];
      const mainBox = mainView ? response.decision.localization[mainView.camera_id] : undefined;
      const eventTimeS = mainView ? viewEventTime(mainView) : activeCase.event_time_s;
      const peakS = mainBox ? localizationWindow(mainBox, eventTimeS).peakS : eventTimeS;
      const offsetS = mainView ? mainView.sync_offset_ms / 1000 : 0;
      const durationS = mainView ? viewDuration(mainView) : fallbackDuration(activeCase.videos[0]);
      seekTo(timelinePosition(peakS - offsetS, durationS));
      setPlaying(false);
    } catch (error) {
      setAnalysisMessage(`分析失败：${String(error)}`);
    } finally {
      setAnalyzing(false);
    }
  }

  function invalidateSavedAssessment() {
    setReviewDirty(true);
    setExplanation(null);
    setAnalysisMessage('人工事实已修改；旧规则结论已失效，请保存并更新判罚');
  }

  function updateDraftFacts(nextFacts: FoulFacts) {
    setFacts(nextFacts);
    invalidateSavedAssessment();
  }

  function updateDraftLocation(location: FoulFacts['location']) {
    setFacts((current) => ({ ...current, location }));
    invalidateSavedAssessment();
  }

  async function saveReview(nextState: ReviewState = reviewState ?? 'pending') {
    if (!activeCase || savingReview) return;
    setSavingReview(true);
    setExplanation(null);
    try {
      const payload = await updateMultiviewReview(activeCase.case_id, {
        expected_revision: review?.revision ?? 0,
        analysis_id: decision?.analysis_id ?? review?.analysis_id ?? null,
        facts,
        review_state: nextState,
      });
      setReview(payload.review);
      setFacts(payload.review.facts);
      setReviewDirty(false);
      setReviewState(payload.review.review_state);
      setCases((current) => current.map((item) => item.case_id === activeCase.case_id
        ? { ...item, review_state: payload.review.review_state, review_revision: payload.review.revision }
        : item));
      setAnalysisMessage(`人工事实与规则结论已保存为 REV ${payload.review.revision}`);
    } catch (error) {
      setAnalysisMessage(`保存失败：${String(error)}；如有版本冲突请重新载入案例`);
    } finally {
      setSavingReview(false);
    }
  }

  async function generateExplanation() {
    if (!activeCase || !review || explaining) return;
    if (reviewDirty) {
      setAnalysisMessage('请先保存当前人工事实，再基于新规则结论整理解释');
      return;
    }
    setExplaining(true);
    try {
      const result = await explainMultiviewReview(activeCase.case_id, review.revision, true);
      setExplanation(result);
      setAnalysisMessage(result.source === 'local_llm' ? '本地 LLM 已在规则边界内整理文案' : '已使用确定性模板生成解释');
    } catch (error) {
      setAnalysisMessage(`解释生成失败：${String(error)}`);
    } finally {
      setExplaining(false);
    }
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
                onVideoRef={registerVideo}
                onVideoReady={handleVideoReady}
                currentTimeS={viewCurrentTime(activeView)}
                eventTimeS={viewEventTime(activeView)}
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
                  onVideoRef={registerVideo}
                  onVideoReady={handleVideoReady}
                  currentTimeS={viewCurrentTime(view)}
                  eventTimeS={viewEventTime(view)}
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
              {activeCase?.videos.map((view, index) => {
                const durationS = viewDuration(view);
                const currentS = viewCurrentTime(view);
                const box = decision?.localization[view.camera_id];
                const window = box ? localizationWindow(box, viewEventTime(view)) : null;
                const windowLeft = window ? timelinePosition(window.startS, durationS) : 0;
                const windowRight = window ? timelinePosition(window.endS, durationS) : 0;
                const peakLeft = window ? timelinePosition(window.peakS, durationS) : 0;
                return (
                  <div className="mv-timeline-row" key={view.camera_id}>
                    <label>{view.camera_id.toUpperCase()}</label>
                    <div className="mv-timeline-track">
                      <i style={{ width: `${86 - index * 4}%` }} />
                      {window && (
                        <>
                          <span
                            className={`mv-temporal-window ${window.source}`}
                            style={{ left: `${windowLeft}%`, width: `${Math.max(0.8, windowRight - windowLeft)}%` }}
                          />
                          <em className="mv-temporal-peak" style={{ left: `${peakLeft}%` }} />
                        </>
                      )}
                      <b style={{ left: `${timelinePosition(currentS, durationS)}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="mv-timeline-controls">
              <button onClick={() => stepFrame(-1)}>−1 帧</button>
              <button className="play" onClick={togglePlayback}>{playing ? '暂停' : '播放'}</button>
              <button onClick={() => stepFrame(1)}>+1 帧</button>
              <span><i className="key" /> 关键帧</span><span><i className="alert" /> 模型关注</span>
              <input aria-label="证据时间轴" type="range" min="0" max="100" value={playhead} onChange={(event) => seekTo(Number(event.target.value))} />
            </div>
          </div>

          <div className="mv-panel mv-actions">
            <button className="analyze" disabled={!activeCase || analyzing} onClick={runAnalysis}>
              {analyzing ? '正在分析…' : '运行多视角分析'}
            </button>
            <span className="mv-action-status">{analysisMessage}</span>
            <div className="mv-review-actions">
              <button disabled={!activeCase || savingReview} onClick={() => saveReview()}>保存事实</button>
              <button disabled={!activeCase || savingReview} onClick={() => saveReview('uncertain')}>暂不确定</button>
              <button disabled={!activeCase || savingReview} onClick={() => saveReview('reviewed')}>完成复核</button>
              <button disabled={!review || savingReview} onClick={() => saveReview('archived')}>完成归档</button>
            </div>
          </div>
        </section>

        <aside className="mv-right-column">
          {activeCase && (
            <div className="mv-panel mv-pitch-panel">
              <div className="mv-panel-title compact"><div><span>HUMAN SPATIAL FACT</span><h2>人工确认犯规位置</h2></div><b>105×68</b></div>
              <FoulLocationPitch
                location={facts.location}
                geometry={reviewDirty ? null : review?.assessment.geometry ?? null}
                offenderTeam={facts.offender_team.confirmed ? facts.offender_team.value : null}
                homeDefendsSide={facts.home_defends_side.confirmed ? facts.home_defends_side.value : null}
                dirty={reviewDirty}
                saving={savingReview}
                onChange={updateDraftLocation}
                onSave={() => saveReview()}
              />
            </div>
          )}

          <div className="mv-panel mv-facts-panel">
            <div className="mv-panel-title compact"><div><span>CONFIRMED EVENT FACTS</span><h2>渐进式事实确认</h2></div><b>HUMAN</b></div>
            <FoulFactsPanel facts={facts} decision={decision} onChange={updateDraftFacts} />
          </div>

          <div className="mv-panel mv-rule-panel">
            <div className="mv-panel-title compact"><div><span>DETERMINISTIC RULES</span><h2>规则辅助判罚</h2></div><b>{review?.assessment.ruleset_version ?? 'IFAB'}</b></div>
            <RuleAssessmentPanel
              assessment={reviewDirty ? null : review?.assessment ?? null}
              explanation={reviewDirty ? null : explanation}
              explaining={explaining}
              draftChanged={reviewDirty}
              onExplain={generateExplanation}
            />
          </div>

          <div className="mv-panel mv-decision-panel">
            <div className="mv-panel-title compact"><div><span>MODEL EVIDENCE</span><h2>视觉模型原始建议</h2></div><b>{decision ? percent(decision.confidence) : '—'}</b></div>
            <ModelEvidencePanel decision={decision} />
          </div>

          <div className="mv-panel mv-chain-panel">
            <div className="mv-panel-title compact"><div><span>CHAIN OF CUSTODY</span><h2>事件证据链</h2></div></div>
            <ol>
              <li className="done"><b>01</b><div><strong>多机位同步</strong><span>偏差检查与片段对齐</span></div></li>
              <li className={decision ? 'done' : 'active'}><b>02</b><div><strong>模型联合分析</strong><span>MViT 动作与严重程度</span></div></li>
              <li className={decision ? 'done' : ''}><b>03</b><div><strong>Grad-CAM 解释</strong><span>定位模型关注区域</span></div></li>
              <li className={review ? 'done' : ''}><b>04</b><div><strong>事实与规则</strong><span>人工确认后确定性计算</span></div></li>
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
