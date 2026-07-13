import React, { useRef, useEffect } from 'react';
import { useDashboardStore } from '../store/dashboardStore';

const LOG_LEVEL_STYLES: Record<string, React.CSSProperties> = {
  ERROR: { color: '#ff4444' },
  WARNING: { color: '#ffaa00' },
  INFO: { color: '#aaccee' },
};

function getLogStyle(line: string): React.CSSProperties {
  if (line.includes('[ERROR]') || line.includes('error')) return LOG_LEVEL_STYLES.ERROR;
  if (line.includes('[WARNING]') || line.includes('warn')) return LOG_LEVEL_STYLES.WARNING;
  return LOG_LEVEL_STYLES.INFO;
}

const LogPanel: React.FC = () => {
  const logLines = useDashboardStore((s) => s.logLines);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logLines]);

  return (
    <div className="panel log-panel">
      <div className="panel-header">
        <span>LOG</span>
        <span className="log-count">{logLines.length}/200</span>
      </div>
      <div className="log-container">
        {logLines.length === 0 && (
          <div className="log-empty">No log entries</div>
        )}
        {logLines.map((line, i) => (
          <div key={i} className="log-line" style={getLogStyle(line)}>
            {line}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
};

export default LogPanel;
