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

interface MediaBounds {
  left: number;
  top: number;
  width: number;
  height: number;
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

function containedMediaBounds(
  containerWidth: number,
  containerHeight: number,
  mediaWidth: number,
  mediaHeight: number,
): MediaBounds | null {
  if (containerWidth <= 0 || containerHeight <= 0 || mediaWidth <= 0 || mediaHeight <= 0) {
    return null;
  }
  const scale = Math.min(containerWidth / mediaWidth, containerHeight / mediaHeight);
  const width = mediaWidth * scale;
  const height = mediaHeight * scale;
  return {
    left: (containerWidth - width) / 2,
    top: (containerHeight - height) / 2,
    width,
    height,
  };
}

function expandedFocusRect(rect: LocalizationBox['rect']): LocalizationBox['rect'] {
  const [left, top, width, height] = rect;
  const padX = Math.max(3.5, width * 0.25);
  const padY = Math.max(3.5, height * 0.25);
  const expandedLeft = Math.max(0, left - padX);
  const expandedTop = Math.max(0, top - padY);
  const expandedRight = Math.min(100, left + width + padX);
  const expandedBottom = Math.min(100, top + height + padY);
  return [
    expandedLeft,
    expandedTop,
    expandedRight - expandedLeft,
    expandedBottom - expandedTop,
  ];
}

function temporalFocusStrength(
  box: LocalizationBox,
  currentTimeS: number,
  window: LocalizationWindow,
): number {
  const fadeS = Math.max(0.12, Math.min(0.2, (window.endS - window.startS) * 0.35));
  let gate = 1;
  if (currentTimeS < window.startS) {
    gate = (currentTimeS - (window.startS - fadeS)) / fadeS;
  } else if (currentTimeS > window.endS) {
    gate = ((window.endS + fadeS) - currentTimeS) / fadeS;
  }
  gate = Math.min(1, Math.max(0, gate));
  if (gate === 0) return 0;

  const currentBin = box.temporal_bins?.find(
    (bin) => currentTimeS >= bin.start_s && currentTimeS <= bin.end_s,
  );
  const evidenceStrength = currentBin ? 0.65 + 0.35 * currentBin.score : 0.82;
  return gate * evidenceStrength;
}

function unconfirmed<T>(value: T | null = null, source: 'human' | 'model' = 'human', confidence: number | null = null) {
  return { value, source, confidence, confirmed: false };
}

function modelFact<T>(value: T, confidence: number | null) {
  return { value, source: 'model' as const, confidence, confirmed: true };
}

function defaultFact<T>(value: T) {
  return { value, source: 'human' as const, confidence: null, confirmed: true };
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
  facts.offence_confirmed = modelFact(
    decision.severity !== 'No Offence',
    decision.severity_candidates[0]?.confidence ?? decision.confidence,
  );
  facts.action = modelFact(
    decision.action,
    decision.action_candidates[0]?.confidence ?? decision.confidence,
  );
  const fallbackIntensity = decision.card === 'red'
    ? 'excessive_force'
    : decision.card === 'yellow'
      ? 'reckless'
      : 'careless';
  facts.intensity = modelFact(
    decision.suggested_intensity ?? fallbackIntensity,
    decision.severity_candidates[0]?.confidence ?? decision.confidence,
  );
  // Smart defaults the reviewer only touches when wrong.
  facts.ball_in_play = defaultFact(true);
  facts.tactical_impact = defaultFact('none');
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
  const mediaContainerRef = useRef<HTMLDivElement | null>(null);
  const mediaElementRef = useRef<HTMLVideoElement | HTMLImageElement | null>(null);
  const [mediaBounds, setMediaBounds] = useState<MediaBounds | null>(null);

  const updateMediaBounds = useCallback(() => {
    const container = mediaContainerRef.current;
    const media = mediaElementRef.current;
    if (!container || !media) return;
    const mediaWidth = media instanceof HTMLVideoElement ? media.videoWidth : media.naturalWidth;
    const mediaHeight = media instanceof HTMLVideoElement ? media.videoHeight : media.naturalHeight;
    const next = containedMediaBounds(
      container.clientWidth,
      container.clientHeight,
      mediaWidth,
      mediaHeight,
    );
    if (!next) return;
    setMediaBounds((current) => (
      current
      && Math.abs(current.left - next.left) < 0.25
      && Math.abs(current.top - next.top) < 0.25
      && Math.abs(current.width - next.width) < 0.25
      && Math.abs(current.height - next.height) < 0.25
        ? current
        : next
    ));
  }, []);

  useEffect(() => {
    const container = mediaContainerRef.current;
    if (!container) return undefined;
    const observer = typeof ResizeObserver === 'undefined'
      ? null
      : new ResizeObserver(updateMediaBounds);
    observer?.observe(container);
    window.addEventListener('resize', updateMediaBounds);
    updateMediaBounds();
    return () => {
      observer?.disconnect();
      window.removeEventListener('resize', updateMediaBounds);
    };
  }, [updateMediaBounds]);

  const setVideoElement = useCallback((element: HTMLVideoElement | null) => {
    mediaElementRef.current = element;
    onVideoRef(view.camera_id, element);
    if (element) window.requestAnimationFrame(updateMediaBounds);
  }, [onVideoRef, updateMediaBounds, view.camera_id]);

  const setImageElement = useCallback((element: HTMLImageElement | null) => {
    mediaElementRef.current = element;
    if (element) window.requestAnimationFrame(updateMediaBounds);
  }, [updateMediaBounds]);

  const activeWindow = box ? localizationWindow(box, eventTimeS) : null;
  const focusRect = box ? expandedFocusRect(box.rect) : null;
  const focusStrength = box && activeWindow
    ? temporalFocusStrength(box, currentTimeS, activeWindow)
    : 0;
  const displayTier = box?.display_tier ?? (box?.reliable === false ? 'hidden' : 'normal');
  const showFocus = displayTier !== 'hidden' && focusStrength > 0.01;
  const displayedFocusStrength = displayTier === 'caution'
    ? focusStrength * 0.64
    : focusStrength;
  return (
    <button
      type="button"
      className={`mv-camera ${primary ? 'primary' : 'secondary'}`}
      onClick={onSelect}
      aria-label={`查看${view.display_name}`}
    >
      <div className="mv-camera-media" ref={mediaContainerRef}>
        {view.media_url ? (
          view.media_kind === 'video' ? (
            <video
              ref={setVideoElement}
              src={view.media_url}
              muted
              playsInline
              preload="auto"
              onLoadedMetadata={() => {
                updateMediaBounds();
                onVideoReady(view.camera_id);
              }}
            />
          ) : (
            <img
              ref={setImageElement}
              src={view.media_url}
              alt={`${view.display_name}证据帧`}
              onLoad={updateMediaBounds}
            />
          )
        ) : (
          <div className="mv-media-empty">无媒体</div>
        )}
        {mediaBounds && (
          <div className="mv-media-coordinate-layer" style={mediaBounds}>
            {box && activeWindow && focusRect && (
              <div
                className={`mv-focus-region ${box.source} ${displayTier} ${showFocus ? 'active' : ''}`}
                aria-hidden={!showFocus}
                style={{
                  left: `${focusRect[0]}%`,
                  top: `${focusRect[1]}%`,
                  width: `${focusRect[2]}%`,
                  height: `${focusRect[3]}%`,
                  opacity: showFocus ? displayedFocusStrength : 0,
                }}
              />
            )}
          </div>
        )}
        {attention !== undefined && (
          <div className="mv-attention-badge">注意力 {percent(attention)}</div>
        )}
      </div>
      <div className="mv-camera-meta">
        <strong>{view.display_name}</strong>
        <span>{view.camera_id.toUpperCase()} · {view.sync_offset_ms >= 0 ? '+' : ''}{view.sync_offset_ms}ms</span>
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
  const selectedIdRef = useRef<string | null>(null);
  const playheadRef = useRef(playhead);
  const playingRef = useRef(playing);

  useEffect(() => {
    selectedIdRef.current = selectedId;
  }, [selectedId]);

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
    () => cases
      .filter((item) => filter === 'all' || item.review_state === filter)
      .sort((a, b) => a.match_clock.localeCompare(b.match_clock)),
    [cases, filter],
  );
  const activeView = activeCase?.videos.find((view) => view.camera_id === selectedCameraId)
    ?? activeCase?.videos[0]
    ?? null;
  const sideViews = activeCase?.videos.filter((view) => view.camera_id !== activeView?.camera_id) ?? [];
  const frameNumber = Math.round(12120 + playhead * 2.56);
  const pendingCount = cases.filter((item) => item.review_state === 'pending').length;

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
    const requestedCaseId = activeCase.case_id;
    setAnalyzing(true);
    setDecision(null);
    setAnalysisMessage('正在解码多机位片段并执行推理…');
    try {
      const response = await analyzeMultiviewCase(requestedCaseId);
      // 分析期间若已切换到其他案例，丢弃过期响应，避免跨案例污染
      if (selectedIdRef.current !== requestedCaseId) return;
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
    setAnalysisMessage('人工事实已修改，请保存后重新计算');
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
      // 保存成功后自动调用模型整理解释，无需再手动触发
      void loadExplanation(activeCase.case_id, payload.review.revision);
    } catch (error) {
      setAnalysisMessage(`保存失败：${String(error)}；如有版本冲突请重新载入案例`);
    } finally {
      setSavingReview(false);
    }
  }

  async function loadExplanation(caseId: string, revision: number) {
    setExplaining(true);
    try {
      const result = await explainMultiviewReview(caseId, revision, true);
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
        <div className="mv-load-error">页面加载失败：{loadError}</div>
        <a href="/">返回实时分析</a>
      </main>
    );
  }

  return (
    <main className="mv-page">
      <header className="mv-topbar">
        <div className="mv-brand">
          <h1>多视角复核平台</h1>
        </div>
        <div className="mv-top-stats">
          <span className={`mv-system-pill ${status?.ready ? 'ready' : 'demo'}`}>
            <i />{status?.ready ? '模型就绪' : '演示模式'}
          </span>
          <span className="mv-stat">机位 <b>{activeCase?.videos.length ?? 0}</b></span>
          <span className="mv-stat">待复核 <b>{pendingCount}</b></span>
          <a className="mv-back-link" href="/">返回实时分析</a>
        </div>
      </header>

      <section className="mv-workspace">
        <aside className="mv-panel mv-queue">
          <div className="mv-panel-title">
            <h2>事件队列</h2>
            <b>{filteredCases.length}</b>
          </div>
          <div className="mv-filters" role="tablist" aria-label="按状态筛选">
            {([
              ['all', '全部'],
              ['pending', '待复核'],
              ['reviewed', '已复核'],
              ['archived', '已归档'],
              ['uncertain', '不确定'],
            ] as [Filter, string][]).map(([value, label]) => (
              <button
                key={value}
                role="tab"
                aria-selected={filter === value}
                className={filter === value ? 'active' : ''}
                onClick={() => setFilter(value)}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="mv-event-list">
            {filteredCases.map((item) => (
              <button
                key={item.case_id}
                className={`mv-event-card ${item.case_id === selectedId ? 'active' : ''}`}
                onClick={() => selectCase(item)}
              >
                <div className="mv-event-card-top">
                  <strong>{item.match_clock} · {item.title}</strong>
                  <em className={item.risk_level}>{item.risk_level === 'high' ? '高风险' : item.risk_level === 'medium' ? '中风险' : '低风险'}</em>
                </div>
                <p>{item.videos.length} 机位 · {item.zone}</p>
                <div><span className={item.review_state}>{stateLabels[item.review_state]}</span></div>
              </button>
            ))}
          </div>
        </aside>

        <section className="mv-center-column">
          <div className="mv-panel mv-event-heading">
            <div>
              <h2>{activeCase ? `${activeCase.match_clock} · ${activeCase.title}` : '等待案例'}</h2>
              <span>{activeCase ? `${activeCase.videos.length} 机位 · ${activeCase.zone}` : ''}</span>
            </div>
            <div className="mv-heading-tags">
              <span className={`risk ${activeCase?.risk_level ?? 'low'}`}>{activeCase?.risk_level === 'high' ? '高风险' : '常规复核'}</span>
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
              <h3>同期时间轴</h3>
              <div className="mv-frame-readout">FRAME {frameNumber}</div>
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
                            className={`mv-temporal-window ${window.source} ${box?.display_tier ?? (box?.reliable === false ? 'hidden' : 'normal')}`}
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
              <button aria-label="上一帧" onClick={() => stepFrame(-1)}>−1 帧</button>
              <button className="play" onClick={togglePlayback}>{playing ? '暂停' : '播放'}</button>
              <button aria-label="下一帧" onClick={() => stepFrame(1)}>+1 帧</button>
              <span><i className="key" /> 片段</span><span><i className="alert" /> 模型关注</span>
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
              <button disabled={!review || savingReview} onClick={() => saveReview('archived')}>归档</button>
            </div>
          </div>
        </section>

        <aside className="mv-right-column">
          {activeCase && (
            <div className="mv-panel mv-pitch-panel">
              <div className="mv-panel-title compact"><h2>犯规位置</h2></div>
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
            <div className="mv-panel-title compact"><h2>事实确认</h2></div>
            <FoulFactsPanel facts={facts} onChange={updateDraftFacts} />
          </div>

          <div className="mv-panel mv-rule-panel">
            <div className="mv-panel-title compact"><h2>规则判罚</h2><b>{review?.assessment.ruleset_version ?? 'IFAB'}</b></div>
            <RuleAssessmentPanel
              assessment={reviewDirty ? null : review?.assessment ?? null}
              explanation={reviewDirty ? null : explanation}
              explaining={explaining}
              draftChanged={reviewDirty}
            />
          </div>
        </aside>
      </section>
    </main>
  );
}
