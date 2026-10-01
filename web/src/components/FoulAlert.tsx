import React, { useEffect, useState } from 'react';
import { useDashboardStore } from '../store/dashboardStore';

const FoulAlert: React.FC = () => {
  const events = useDashboardStore((s) => s.events);
  const latestFoul = events.find((e) => e.event_type === 'foul_candidate');
  const latestFoulId = latestFoul?.id;
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (!latestFoulId) {
      setVisible(false);
      return;
    }
    setVisible(true);
    const timer = setTimeout(() => setVisible(false), 5000);
    return () => clearTimeout(timer);
  }, [latestFoulId]);

  if (!visible || !latestFoul) return null;

  const action = latestFoul.foul_details?.action ?? 'Unknown';
  const confidence = (latestFoul.confidence * 100).toFixed(0);

  return (
    <div className="foul-alert-overlay">
      <div className="foul-alert-badge">
        <span className="foul-alert-title">⚠ FOUL CANDIDATE</span>
        <span className="foul-alert-action">{action}</span>
        <span className="foul-alert-confidence">{confidence}%</span>
      </div>
    </div>
  );
};

export default FoulAlert;
