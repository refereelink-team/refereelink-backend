import React from 'react';
import { useWebSocket } from './hooks/useWebSocket';
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

  return (
    <>
      <VideoPanel />
      <Pitch2D />
      <StatusCards />
      <EventAlerts />
      <TeamCalibrationPanel />
      <ControlPanel sendCommand={sendCommand} />
      <LogPanel />
    </>
  );
};

export default App;
