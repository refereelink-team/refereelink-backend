import React from 'react';
import { useDashboardStore } from '../store/dashboardStore';
import type { useWebSocket } from '../hooks/useWebSocket';

interface Props {
  sendCommand: ReturnType<typeof useWebSocket>['sendCommand'];
}

const ControlPanel: React.FC<Props> = ({ sendCommand }) => {
  const pipelineRunning = useDashboardStore((s) => s.pipelineRunning);

  return (
    <div className="panel control-panel">
      <div className="panel-header">CONTROL</div>
      <div className="control-buttons">
        <button
          className={`btn ${pipelineRunning ? 'btn-stop' : 'btn-start'}`}
          onClick={() => sendCommand(pipelineRunning ? 'stop' : 'start')}
        >
          {pipelineRunning ? 'STOP' : 'START'}
        </button>
        <button
          className="btn btn-action"
          onClick={() =>
            sendCommand('update_config', {
              enable_foul_detection: !useDashboardStore.getState().config.enable_foul_detection,
            })
          }
        >
          Toggle Foul
        </button>
        <button
          className="btn btn-action"
          onClick={() =>
            sendCommand('update_config', {
              show_keypoints: !useDashboardStore.getState().config.show_keypoints,
            })
          }
        >
          Toggle Keypoints
        </button>
      </div>
    </div>
  );
};

export default ControlPanel;
