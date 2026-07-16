import { useEffect, useRef, useCallback } from 'react';
import { useDashboardStore } from '../store/dashboardStore';
import type {
  WSMessage,
  FrameState,
  MetricsSnapshot,
  TeamCalibrationState,
} from '../types/messages';

const WS_URL = `ws://${window.location.hostname}:8000/ws/state`;
const RECONNECT_DELAY = 2000;

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const {
    setFrameState,
    setMetrics,
    addEvent,
    setWsConnected,
    setSourceStatus,
    setPipelineRunning,
    setTeamCalibration,
    addLog,
  } = useDashboardStore();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsConnected(true);
      addLog(`[WS] Connected to ${WS_URL}`);
    };

    ws.onclose = () => {
      setWsConnected(false);
      addLog('[WS] Disconnected, reconnecting...');
      reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY);
    };

    ws.onerror = () => {
      ws.close();
    };

    ws.onmessage = (event) => {
      try {
        const msg: WSMessage = JSON.parse(event.data);
        if (msg.type === 'frame_state') {
          setFrameState(msg as FrameState);
          setPipelineRunning(true);  // receiving frames means pipeline is running
          for (const ev of (msg as FrameState).events) {
            addEvent(ev);
          }
        } else if (msg.type === 'metrics') {
          setMetrics(msg as MetricsSnapshot);
          setSourceStatus((msg as MetricsSnapshot).source_status);
        } else if (msg.type === 'team_calibration') {
          setTeamCalibration(msg as TeamCalibrationState);
        }
      } catch {
        // ignore malformed messages
      }
    };
  }, [setFrameState, setMetrics, addEvent, setWsConnected, setSourceStatus, setTeamCalibration, addLog]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const sendCommand = useCallback((command: string, params: Record<string, unknown> = {}) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command, params }));
    }
  }, []);

  return { sendCommand };
}
