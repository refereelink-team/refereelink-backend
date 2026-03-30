import argparse
from enum import Enum
from typing import Optional

import cv2
import supervision as sv

from app.runtime import normalize_proxy_env


class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'
    RADAR_DASHBOARD = 'RADAR_DASHBOARD'
    FOUL_DETECTION = 'FOUL_DETECTION'


def main(
    source_video_path: str,
    target_video_path: str,
    device: str,
    mode: Mode,
    foul_checkpoint_path: Optional[str] = None,
) -> None:
    normalize_proxy_env()

    if mode == Mode.RADAR_DASHBOARD:
        from app.modes.radar_dashboard import run_radar_dashboard

        run_radar_dashboard(
            source_video_path=source_video_path,
            target_video_path=target_video_path,
            device=device,
            foul_checkpoint_path=foul_checkpoint_path,
        )
        return

    if mode == Mode.PITCH_DETECTION:
        from app.modes.pitch_detection import run_pitch_detection

        frame_generator = run_pitch_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_DETECTION:
        from app.modes.player_detection import run_player_detection

        frame_generator = run_player_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.BALL_DETECTION:
        from app.modes.ball_detection import run_ball_detection

        frame_generator = run_ball_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_TRACKING:
        from app.modes.player_tracking import run_player_tracking

        frame_generator = run_player_tracking(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.TEAM_CLASSIFICATION:
        from app.modes.team_classification import run_team_classification

        frame_generator = run_team_classification(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.RADAR:
        from app.modes.radar import run_radar

        frame_generator = run_radar(
            source_video_path=source_video_path,
            device=device,
            foul_checkpoint_path=foul_checkpoint_path,
        )
    elif mode == Mode.FOUL_DETECTION:
        from app.modes.foul_detection import run_foul_detection

        if foul_checkpoint_path is None:
            from app.constants.paths import FOUL_MODEL_PATH
            foul_checkpoint_path = FOUL_MODEL_PATH

        frame_generator = run_foul_detection(
            source_video_path=source_video_path,
            device=device,
            foul_checkpoint_path=foul_checkpoint_path,
        )
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_DETECTION)
    parser.add_argument(
        '--foul_checkpoint_path',
        type=str,
        default=None,
        help=(
            'Path to MVFoul checkpoint (.pth.tar). '
            'Required for FOUL_DETECTION mode (falls back to assets/weights/mvfoul.pth.tar). '
            'Optional for RADAR / RADAR_DASHBOARD — enables the foul HUD overlay when provided.'
        ),
    )
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode,
        foul_checkpoint_path=args.foul_checkpoint_path,
    )
