import { useRef } from 'react';
import type { PointerEvent } from 'react';
import type {
  DefendsSide,
  FoulLocation,
  LocationGeometry,
  TeamLabel,
} from '../../types/multiview';

interface Props {
  location: FoulLocation | null;
  geometry: LocationGeometry | null;
  offenderTeam: TeamLabel | null;
  homeDefendsSide: DefendsSide | null;
  dirty: boolean;
  saving: boolean;
  onChange: (location: FoulLocation | null) => void;
  onSave: () => void;
}

function pointerLocation(event: PointerEvent<SVGSVGElement>): FoulLocation {
  const bounds = event.currentTarget.getBoundingClientRect();
  const x = Math.min(105, Math.max(0, (event.clientX - bounds.left) / bounds.width * 105));
  const y = Math.min(68, Math.max(0, (event.clientY - bounds.top) / bounds.height * 68));
  return {
    x_m: Math.round(x * 100) / 100,
    y_m: Math.round(y * 100) / 100,
    source: 'human',
    confirmed: true,
  };
}

function approximateZone(location: FoulLocation | null): string {
  if (!location) return '尚未设置犯规点';
  const inPenaltyWidth = location.y_m >= 13.84 && location.y_m <= 54.16;
  if (inPenaltyWidth && location.x_m <= 16.5) return '左侧禁区';
  if (inPenaltyWidth && location.x_m >= 88.5) return '右侧禁区';
  if (Math.abs(location.x_m - 52.5) < 0.25) return '中线区域';
  return location.x_m < 52.5 ? '左半场' : '右半场';
}

export default function FoulLocationPitch({
  location,
  geometry,
  offenderTeam,
  homeDefendsSide,
  dirty,
  saving,
  onChange,
  onSave,
}: Props) {
  const previousLocation = useRef<FoulLocation | null>(null);
  const dragging = useRef(false);

  function begin(event: PointerEvent<SVGSVGElement>) {
    previousLocation.current = location;
    dragging.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
    onChange(pointerLocation(event));
  }

  function move(event: PointerEvent<SVGSVGElement>) {
    if (dragging.current) onChange(pointerLocation(event));
  }

  function end(event: PointerEvent<SVGSVGElement>) {
    dragging.current = false;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function clear() {
    previousLocation.current = location;
    onChange(null);
  }

  return (
    <div className="mv-pitch-editor">
      <svg
        viewBox="0 0 105 68"
        role="application"
        aria-label="点击或拖动以设置犯规位置"
        onPointerDown={begin}
        onPointerMove={move}
        onPointerUp={end}
        onPointerCancel={end}
      >
        <rect className="field" x="0.6" y="0.6" width="103.8" height="66.8" />
        <line x1="52.5" y1="0.6" x2="52.5" y2="67.4" />
        <circle cx="52.5" cy="34" r="9.15" />
        <circle className="spot" cx="52.5" cy="34" r="0.45" />
        <rect x="0.6" y="13.84" width="16.5" height="40.32" />
        <rect x="87.9" y="13.84" width="16.5" height="40.32" />
        <rect x="0.6" y="24.84" width="5.5" height="18.32" />
        <rect x="98.9" y="24.84" width="5.5" height="18.32" />
        <circle className="spot" cx="11" cy="34" r="0.45" />
        <circle className="spot" cx="94" cy="34" r="0.45" />
        {location ? (
          <g className="foul-marker" transform={`translate(${location.x_m} ${location.y_m})`}>
            <circle r="2.15" />
            <line x1="-3.5" y1="0" x2="3.5" y2="0" />
            <line x1="0" y1="-3.5" x2="0" y2="3.5" />
          </g>
        ) : null}
      </svg>
      <div className="mv-location-readout">
        <div>
          <span>人工空间事实</span>
          <strong>{geometry?.zone ?? approximateZone(location)}</strong>
          <small>
            {location ? `x ${location.x_m.toFixed(2)} m · y ${location.y_m.toFixed(2)} m` : '点击球场设置位置'}
          </small>
        </div>
        <div className="mv-location-actions">
          <button type="button" disabled={!location} onClick={clear}>清除</button>
          <button
            type="button"
            disabled={previousLocation.current === location}
            onClick={() => onChange(previousLocation.current)}
          >撤销</button>
          <button
            type="button"
            className="save"
            disabled={!dirty || saving}
            onClick={onSave}
          >{saving ? '保存中…' : '保存并更新判罚'}</button>
        </div>
      </div>
      <p className="mv-location-context">
        犯规方 {offenderTeam?.toUpperCase() ?? '未确认'} · HOME 防守
        {homeDefendsSide === 'left' ? '左侧' : homeDefendsSide === 'right' ? '右侧' : '方向未确认'}
        {geometry?.in_offender_own_penalty_area === true ? ' · 犯规方本方禁区内' : ''}
        {dirty ? ' · 尚未保存' : ''}
      </p>
    </div>
  );
}
