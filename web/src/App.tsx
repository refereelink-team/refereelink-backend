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
import MultiviewReviewPage from './pages/MultiviewReviewPage';
import './styles/global.css';
import FoulAlert from './components/FoulAlert';

const Dashboard: React.FC = () => {
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
      {!calibrationWorkspace && (
        <a className="multiview-entry" href="/multiview" aria-label="进入多视角犯规判罚中心">
          <span>MULTI-VIEW</span>
          多视角判罚
        </a>
      )}
      <VideoPanel />
      <FoulAlert />
      <Pitch2D />
      <StatusCards />
      <EventAlerts />
      <TeamCalibrationPanel />
      <ControlPanel sendCommand={sendCommand} />
      <LogPanel />
    </div>
  );
};

const App: React.FC = () => {
  const normalizedPath = window.location.pathname.replace(/\/+$/, '') || '/';
  if (normalizedPath === '/multiview') {
    return <MultiviewReviewPage />;
  }
  return <Dashboard />;
};

export default App;
