import React, { useEffect, useRef, useState } from 'react';
import { useDashboardStore } from '../store/dashboardStore';

const FoulAlert: React.FC = () => {
  const events = useDashboardStore((s) => s.events);
  const [latestFoul, setLatestFoul] = useState<any>(null);
  const [visible, setVisible] = useState(false);
  const lastIdRef = useRef<string | null>(null);

  useEffect(() => {
    const foul = events.find((e) => e.event_type === 'foul_candidate');
    if (foul && foul.id !== lastIdRef.current) {
      lastIdRef.current = foul.id;
      setLatestFoul(foul);
      setVisible(true);
      const timer = setTimeout(() => setVisible(false), 5000);
      return () => clearTimeout(timer);
    }
  }, [events]);

  if (!visible || !latestFoul) return null;

  const action = (latestFoul.foul_details as any)?.action ?? 'Unknown';
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