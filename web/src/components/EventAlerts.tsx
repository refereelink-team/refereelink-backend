import React from 'react';
import { useDashboardStore } from '../store/dashboardStore';

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
        {events.slice(0, 20).map((e) => (
          <div key={e.id} className={`event-item severity-${e.severity}`}>
            <div className="event-header">
              <span className="event-type">{e.event_type.toUpperCase()}</span>
              <span className="event-confidence">
                {(e.confidence * 100).toFixed(0)}%
              </span>
              <span className="event-severity">{e.severity}</span>
              {e.reviewed && <span className="event-reviewed">REVIEWED</span>}
            </div>
            <div className="event-details">
              t={e.timestamp.toFixed(1)}s
              {e.field_x != null && e.field_y != null && (
                <span>
                  &nbsp;pos=({e.field_x.toFixed(0)},{e.field_y.toFixed(0)})
                </span>
              )}
              &nbsp;frame={e.frame_id}
              {e.foul_details && (
                <span>
                  &nbsp;offence={e.foul_details.offence}
                  &nbsp;action={e.foul_details.action}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default EventAlerts;
