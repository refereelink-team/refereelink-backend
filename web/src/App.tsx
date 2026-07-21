import React from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { useDashboardStore } from './store/dashboardStore';
import VideoPanel from './components/VideoPanel';
import Pitch2D from './components/Pitch2D';
import StatusCards from './components/StatusCards';
import EventAlerts from './components/EventAlerts';
import ControlPanel from './components/ControlPanel';
import LogPanel from './components/LogPanel';
import TeamCalibrationPanel from './components/TeamCalibrationPanel';
import './styles/global.css';

const App: React.FC = () => {
  const { sendCommand } = useWebSocket();
  const calibrationPhase = useDashboardStore((s) => s.teamCalibration.state);
  const calibrationWorkspace = [
    'source_preview',
    'clip_selecting',
    'processing',
    'review',
    'validating',
    'recalibration_required',
  ].includes(calibrationPhase);

  return (
    <div className={`dashboard-layout${calibrationWorkspace ? ' calibration-workspace-mode' : ''}`}>
      <VideoPanel />
      <Pitch2D />
      <StatusCards />
      <EventAlerts />
      <TeamCalibrationPanel />
      <ControlPanel sendCommand={sendCommand} />
      <LogPanel />
    </div>
  );
};

export default App;
