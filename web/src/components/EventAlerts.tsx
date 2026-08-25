import React from 'react';
import { useDashboardStore } from '../store/dashboardStore';
import type { GameEvent } from '../types/messages';

function foulSummary(e: GameEvent): string | null {
  const details = e.foul_details;
  if (details?.summary && typeof details.summary === 'string') {
    return details.summary;
  }
  const evidenceSummary = e.evidence?.summary;
  if (typeof evidenceSummary === 'string') {
    return evidenceSummary;
  }
  if (details?.label_a && details?.label_b) {
    const action = details.action ?? 'contact';
    return `${details.label_a} · ${action} · ${details.label_b}`;
  }
  if (e.involved_track_ids.length >= 2) {
    const action = details?.action ?? 'contact';
    return `T${e.involved_track_ids[0]} · ${action} · T${e.involved_track_ids[1]}`;
  }
  return null;
}

const EventAlerts: React.FC = () => {
  const events = useDashboardStore((s) => s.events);

  return (
    <div className="panel events-panel">
      <div className="panel-header">
        <span>EVENTS &amp; ALERTS</span>
        <span className="event-count">{events.length}</span>
      </div>
      <div className="events-list">
        {events.length === 0 && (
          <div className="events-empty">No events detected</div>
        )}
        {events.slice(0, 20).map((e) => {
          const summary = e.event_type === 'foul_candidate' ? foulSummary(e) : null;
          const source =
            typeof e.evidence?.source === 'string' ? String(e.evidence.source) : null;
          return (
            <div
              key={e.id}
              className={`event-item severity-${e.severity}${
                e.event_type === 'foul_candidate' ? ' event-foul' : ''
              }`}
            >
              <div className="event-header">
                <span className="event-type">
                  {e.event_type === 'foul_candidate' ? 'FOUL' : e.event_type.toUpperCase()}
                </span>
                <span className="event-confidence">
                  {(e.confidence * 100).toFixed(0)}%
                </span>
                <span className="event-severity">{e.severity}</span>
                {source && <span className="event-source">{source}</span>}
                {e.reviewed && <span className="event-reviewed">REVIEWED</span>}
              </div>
              {summary && <div className="event-summary">{summary}</div>}
              <div className="event-details">
                t={e.timestamp.toFixed(1)}s
                {e.field_x != null && e.field_y != null && (
                  <span>
                    &nbsp;pos=({e.field_x.toFixed(0)},{e.field_y.toFixed(0)})
                  </span>
                )}
                &nbsp;frame={e.frame_id}
                {e.foul_details && !summary && (
                  <span>
                    &nbsp;offence={e.foul_details.offence}
                    &nbsp;action={e.foul_details.action}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default EventAlerts;
