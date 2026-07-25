"""Unified CLI entry for Soccer Analysis (SC-live).

Subcommands:
  live         Low-latency RTSP web live with detection/tracking overlays
  field-live   Field-plane calibration + realtime 2D pitch projection web
  record-live  Pure RTSP→MP4 recording (no YOLO / Torch / web)
  tracking     Offline player tracking (writes annotated video)
  demo-vision  Offline MP4 detection+track+dynamic recalibration+2D projection
  demo-tracking Offline MP4 detection + ByteTrack only (no pitch/projection)
  demo-offside Semi-automatic offside assist (discover / judge)
  pipeline     Full FastAPI + React dashboard server
  server       Alias of pipeline

Legacy:
  python -m app.main --mode ... still works unchanged.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence


DEFAULT_RTSP = "rtsp://172.32.0.93:554/live/0"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Soccer Analysis CLI (live / field-live / record-live / tracking / pipeline)",
    )
    sub = parser.add_subparsers(dest="command")

    live = sub.add_parser("live", help="RTSP web live with YOLO + ByteTrack overlays")
    live.add_argument("--source", default=DEFAULT_RTSP, help="RTSP or local video path")
    live.add_argument("--host", default="127.0.0.1")
    live.add_argument("--port", type=int, default=8000)
    live.add_argument("--device", default="auto", help="auto|cpu|cuda")
    live.add_argument("--model", default=None, help="Player detection .pt path")
    live.add_argument("--max-width", type=int, default=1280)
    live.add_argument("--jpeg-quality", type=int, default=70)
    live.add_argument("--process-every", type=int, default=1)
    live.add_argument("--conf-threshold", type=float, default=0.25)
    live.add_argument(
        "--disable-team-classification",
        action="store_true",
        default=True,
        help="Keep team labels as UNKNOWN (default for live)",
    )
    live.add_argument(
        "--enable-team-classification",
        action="store_true",
        help="Enable team labels when a calibration bundle is provided",
    )
    live.add_argument("--team-warmup-frames", type=int, default=150)
    live.add_argument("--team-min-crops", type=int, default=64)
    live.add_argument("--team-calibration-path", default=None)
    live.add_argument("--rtsp-transport", default="tcp", choices=("tcp", "udp"))
    live.add_argument("--reconnect-delay", type=float, default=1.0)
    live.add_argument("--enable-pitch", action="store_true")
    live.add_argument("--enable-undistortion", action="store_true")
    live.add_argument("--imgsz", type=int, default=640)

    field = sub.add_parser(
        "field-live",
        help="RTSP field-plane calibration + realtime 2D pitch projection web",
    )
    field.add_argument("--source", default=DEFAULT_RTSP)
    field.add_argument("--camera-id", default="cam01")
    field.add_argument("--host", default="127.0.0.1")
    field.add_argument("--port", type=int, default=8000)
    field.add_argument("--device", default="auto")
    field.add_argument("--model", default=None)
    field.add_argument("--max-width", type=int, default=1280)
    field.add_argument("--jpeg-quality", type=int, default=70)
    field.add_argument("--process-every", type=int, default=1)
    field.add_argument("--conf-threshold", type=float, default=0.25)
    field.add_argument("--disable-team-classification", action="store_true", default=True)
    field.add_argument("--rtsp-transport", default="tcp", choices=("tcp", "udp"))
    field.add_argument("--reconnect-delay", type=float, default=1.0)
    field.add_argument("--pitch-detection-interval", type=int, default=5)
    field.add_argument("--imgsz", type=int, default=640)
    field.add_argument("--calibration-path", default=None)
    field.add_argument("--no-auto-load-calibration", action="store_true")
    field.add_argument(
        "--mode",
        default="dashboard",
        choices=("dashboard", "both"),
        help="dashboard=大屏不自动录像; both=大屏+自动开始录像",
    )
    field.add_argument(
        "--record-duration",
        type=int,
        default=600,
        help="both 模式下自动录像秒数；0=直到手动停止",
    )
    field.add_argument("--output-dir", default="recordings")
    field.add_argument(
        "--recording-backend",
        default="auto",
        choices=("auto", "ffmpeg-copy", "ffmpeg-transcode"),
    )

    record = sub.add_parser(
        "record-live",
        help="Pure RTSP→MP4 recording without loading detection models",
    )
    record.add_argument("--source", default=DEFAULT_RTSP)
    record.add_argument("--camera-id", default="cam01")
    record.add_argument("--output-dir", default="recordings")
    record.add_argument("--duration", type=int, default=600)
    record.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "ffmpeg-copy", "ffmpeg-transcode"),
    )
    record.add_argument("--rtsp-transport", default="tcp", choices=("tcp", "udp"))
    record.add_argument("--min-free-disk-gb", type=float, default=5.0)

    preview = sub.add_parser(
        "record-preview",
        help="Preview RTSP in browser + record controls (no YOLO)",
    )
    preview.add_argument("--source", default=DEFAULT_RTSP)
    preview.add_argument("--camera-id", default="cam01")
    preview.add_argument("--output-dir", default="recordings")
    preview.add_argument("--host", default="127.0.0.1")
    preview.add_argument("--port", type=int, default=8000)
    preview.add_argument("--max-width", type=int, default=1280)
    preview.add_argument("--jpeg-quality", type=int, default=70)
    preview.add_argument("--rtsp-transport", default="tcp", choices=("tcp", "udp"))

    lite = sub.add_parser(
        "lite-detect",
        help="Lightweight YOLO11n detection web (no pitch, simple track)",
    )
    lite.add_argument("--source", default=DEFAULT_RTSP)
    lite.add_argument("--host", default="127.0.0.1")
    lite.add_argument("--port", type=int, default=8000)
    lite.add_argument("--model", default=None, help="Default: assets/weights/yolo11n.pt")
    lite.add_argument("--max-width", type=int, default=640)
    lite.add_argument("--imgsz", type=int, default=416)
    lite.add_argument("--process-every", type=int, default=3)
    lite.add_argument("--conf", type=float, default=0.35)
    lite.add_argument("--device", default="cpu")
    lite.add_argument("--no-track", action="store_true", help="Draw boxes only, no IDs")
    lite.add_argument("--rtsp-transport", default="tcp", choices=("tcp", "udp"))

    tracking = sub.add_parser(
        "tracking",
        help="Offline PLAYER_TRACKING mode (annotated MP4 + optional imshow)",
    )
    tracking.add_argument("--source_video_path", "--source", dest="source_video_path", required=True)
    tracking.add_argument("--target_video_path", "--target", dest="target_video_path", required=True)
    tracking.add_argument("--device", default="cpu")
    tracking.add_argument("--inference_backend", default="auto")

    demo = sub.add_parser(
        "demo-vision",
        help="Offline MP4: detection + ByteTrack + dynamic recalibration + 2D projection",
    )
    demo.add_argument("--input", required=True, help="Local MP4 path")
    demo.add_argument("--output-dir", required=True, help="Demo output directory")
    demo.add_argument("--device", default="cuda", help="cuda|cpu|auto")
    demo.add_argument("--camera-id", default="CAM_DETECT")
    demo.add_argument("--pitch-interval", type=int, default=5)
    demo.add_argument("--imgsz", type=int, default=640)
    demo.add_argument("--conf-threshold", type=float, default=0.25)
    demo.add_argument("--player-model", default=None)
    demo.add_argument("--pitch-model", default=None)
    demo.add_argument("--ball-model", default=None)
    demo.add_argument("--start-seconds", type=float, default=0.0)
    demo.add_argument("--end-seconds", type=float, default=None)
    demo.add_argument("--max-frames", type=int, default=None)
    demo.add_argument("--stable-accept-count", type=int, default=1)
    demo.add_argument("--max-stale-seconds", type=float, default=5.0)
    demo.add_argument("--stale-after-seconds", type=float, default=2.0)
    demo.add_argument("--smoke", action="store_true", help="Process ~3s and write smoke artifacts")
    demo.add_argument("--no-overwrite", action="store_true")

    tracking_only = sub.add_parser(
        "demo-tracking",
        help="Offline MP4: detection + ByteTrack only (no pitch/projection/ball)",
    )
    tracking_only.add_argument("--input", required=True)
    tracking_only.add_argument("--output-dir", required=True)
    tracking_only.add_argument("--device", default="cuda")
    tracking_only.add_argument("--camera-id", default="CAM_DETECT")
    tracking_only.add_argument("--model", default="assets/weights/yolo11s.pt")
    tracking_only.add_argument("--conf", type=float, default=0.20)
    tracking_only.add_argument("--iou", type=float, default=0.50)
    tracking_only.add_argument("--imgsz", type=int, default=960)
    tracking_only.add_argument("--track-activation-threshold", type=float, default=0.20)
    tracking_only.add_argument("--lost-track-buffer", type=int, default=60)
    tracking_only.add_argument("--minimum-matching-threshold", type=float, default=0.75)
    tracking_only.add_argument("--minimum-consecutive-frames", type=int, default=2)
    tracking_only.add_argument("--trail-length", type=int, default=20)

    offside = sub.add_parser(
        "demo-offside",
        help="Semi-automatic offside assist: discover candidates or judge after human confirm",
    )
    offside_sub = offside.add_subparsers(dest="offside_command", required=True)
    off_disc = offside_sub.add_parser("discover", help="Propose human-review pass/touch candidates")
    off_disc.add_argument("--input", required=True)
    off_disc.add_argument("--output-dir", required=True)
    off_disc.add_argument("--device", default="cuda")
    off_disc.add_argument("--player-model", default="assets/weights/yolo11s.pt")
    off_disc.add_argument("--ball-model", default=None)
    off_disc.add_argument("--pitch-model", default=None)
    off_disc.add_argument("--conf", type=float, default=0.20)
    off_disc.add_argument("--iou", type=float, default=0.50)
    off_disc.add_argument("--imgsz", type=int, default=960)
    off_disc.add_argument("--pitch-interval", type=int, default=5)
    off_disc.add_argument("--thumb-interval-s", type=float, default=0.5)

    off_judge = offside_sub.add_parser("judge", help="Judge only after human-confirmed critical instant")
    off_judge.add_argument("--config", required=True)
    off_judge.add_argument("--device", default="cuda")
    off_judge.add_argument("--output-dir", default="/root/autodl-tmp/demo_outputs/offside")

    off_review = offside_sub.add_parser(
        "review",
        help="Human review UI (reads existing Discover/review cache; no YOLO preprocess)",
    )
    off_review.add_argument("--input", required=True)
    off_review.add_argument("--discovery-dir", required=True)
    off_review.add_argument("--host", default="0.0.0.0")
    off_review.add_argument("--port", type=int, default=6006)
    off_review.add_argument("--device", default="cuda")
    off_review.add_argument("--config", default="configs/offside_demo.yaml")

    dashboard = sub.add_parser(
        "demo-dashboard",
        help="Local MP4 multiview dashboard (preprocess-only, shared clock)",
    )
    dashboard.add_argument("--config", default="configs/demo_dashboard.yaml")
    dashboard.add_argument("--host", default="0.0.0.0")
    dashboard.add_argument("--port", type=int, default=6008)

    pipeline = sub.add_parser(
        "pipeline",
        help="Start the full FastAPI dashboard server (app.server.main)",
    )
    pipeline.add_argument("--video_source", "--source", dest="video_source", default=None)
    pipeline.add_argument("--device", default="cpu")
    pipeline.add_argument("--host", default="0.0.0.0")
    pipeline.add_argument("--port", type=int, default=8000)
    pipeline.add_argument("--inference_backend", default="auto")

    server = sub.add_parser("server", help="Alias of pipeline")
    server.add_argument("--video_source", "--source", dest="video_source", default=None)
    server.add_argument("--device", default="cpu")
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=8000)
    server.add_argument("--inference_backend", default="auto")

    return parser


def _run_live(args: argparse.Namespace) -> None:
    from app.constants.paths import PLAYER_DETECTION_MODEL_PATH
    from live.pipeline import LivePipelineConfig
    from live.web_app import run_live_server

    disable_team = True
    if args.enable_team_classification:
        disable_team = False
    if args.disable_team_classification and not args.enable_team_classification:
        disable_team = True

    config = LivePipelineConfig(
        source=args.source,
        device=args.device,
        model=args.model or PLAYER_DETECTION_MODEL_PATH,
        max_width=args.max_width,
        jpeg_quality=args.jpeg_quality,
        process_every=args.process_every,
        conf_threshold=args.conf_threshold,
        disable_team_classification=disable_team,
        team_warmup_frames=args.team_warmup_frames,
        team_min_crops=args.team_min_crops,
        team_calibration_path=args.team_calibration_path,
        rtsp_transport=args.rtsp_transport,
        reconnect_delay=args.reconnect_delay,
        enable_pitch=args.enable_pitch,
        enable_undistortion=args.enable_undistortion,
        imgsz=args.imgsz,
    )
    run_live_server(config, host=args.host, port=args.port)


def _run_field_live(args: argparse.Namespace) -> None:
    from app.constants.paths import LEGACY_PLAYER_DETECTION_MODEL_PATH
    from live.field_pipeline import FieldLiveConfig
    from live.field_web_app import run_field_live_server
    from live.recorder import RecorderConfig, RecorderController

    config = FieldLiveConfig(
        source=args.source,
        camera_id=args.camera_id,
        device=args.device,
        model=args.model or LEGACY_PLAYER_DETECTION_MODEL_PATH,
        max_width=args.max_width,
        jpeg_quality=args.jpeg_quality,
        process_every=args.process_every,
        conf_threshold=args.conf_threshold,
        disable_team_classification=True,
        rtsp_transport=args.rtsp_transport,
        reconnect_delay=args.reconnect_delay,
        pitch_detection_interval=args.pitch_detection_interval,
        imgsz=args.imgsz,
        calibration_path=args.calibration_path,
        auto_load_calibration=not args.no_auto_load_calibration,
    )
    recorder = RecorderController(
        RecorderConfig(
            source=args.source,
            camera_id=args.camera_id,
            output_dir=args.output_dir,
            rtsp_transport=args.rtsp_transport,
            recording_backend=args.recording_backend,
        )
    )
    auto_dur = None
    if args.mode == "both":
        auto_dur = int(args.record_duration)
    run_field_live_server(
        config,
        host=args.host,
        port=args.port,
        recorder=recorder,
        auto_record_duration=auto_dur,
    )


def _run_record_live(args: argparse.Namespace) -> int:
    from live.record_cli import run_record_live

    forwarded: list[str] = [
        "--source",
        args.source,
        "--camera-id",
        args.camera_id,
        "--output-dir",
        args.output_dir,
        "--duration",
        str(args.duration),
        "--backend",
        args.backend,
        "--rtsp-transport",
        args.rtsp_transport,
        "--min-free-disk-gb",
        str(args.min_free_disk_gb),
    ]
    return run_record_live(forwarded)


def _run_record_preview(args: argparse.Namespace) -> None:
    from live.preview_pipeline import PreviewConfig
    from live.preview_web_app import run_record_preview_server

    config = PreviewConfig(
        source=args.source,
        camera_id=args.camera_id,
        output_dir=args.output_dir,
        max_width=args.max_width,
        jpeg_quality=args.jpeg_quality,
        rtsp_transport=args.rtsp_transport,
    )
    run_record_preview_server(config, host=args.host, port=args.port)


def _run_lite_detect(args: argparse.Namespace) -> None:
    from pathlib import Path

    from live.lite_pipeline import LiteDetectConfig
    from live.lite_web_app import run_lite_detect_server

    model = args.model or str(Path(__file__).resolve().parent / "assets" / "weights" / "yolo11n.pt")
    config = LiteDetectConfig(
        source=args.source,
        model=model,
        device=args.device if args.device != "auto" else "cpu",
        max_width=args.max_width,
        imgsz=args.imgsz,
        process_every=args.process_every,
        conf=args.conf,
        rtsp_transport=args.rtsp_transport,
        simple_track=not args.no_track,
    )
    run_lite_detect_server(config, host=args.host, port=args.port)


def _run_tracking(args: argparse.Namespace) -> None:
    from app.main import Mode, main as app_main

    app_main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=Mode.PLAYER_TRACKING,
        inference_backend=args.inference_backend,
    )


def _run_demo_vision(args: argparse.Namespace) -> int:
    from app.constants.paths import (
        BALL_DETECTION_MODEL_PATH,
        PITCH_DETECTION_MODEL_PATH,
    )
    from live.demo_vision import DemoVisionConfig, run_demo_vision

    cfg = DemoVisionConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        device=args.device,
        camera_id=args.camera_id,
        pitch_interval=args.pitch_interval,
        imgsz=args.imgsz,
        conf_threshold=args.conf_threshold,
        player_model=args.player_model,
        pitch_model=args.pitch_model or PITCH_DETECTION_MODEL_PATH,
        ball_model=args.ball_model or BALL_DETECTION_MODEL_PATH,
        start_seconds=args.start_seconds,
        end_seconds=args.end_seconds,
        max_frames=args.max_frames,
        stable_accept_count=args.stable_accept_count,
        max_stale_seconds=args.max_stale_seconds,
        stale_after_seconds=args.stale_after_seconds,
        smoke=args.smoke,
        overwrite=not args.no_overwrite,
    )
    summary = run_demo_vision(cfg)
    if cfg.smoke and not summary.get("smoke_ok"):
        return 2
    return 0


def _run_demo_tracking(args: argparse.Namespace) -> int:
    from live.tracking_only_demo import TrackingOnlyConfig, run_tracking_only

    cfg = TrackingOnlyConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        device=args.device,
        camera_id=args.camera_id,
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        track_activation_threshold=args.track_activation_threshold,
        lost_track_buffer=args.lost_track_buffer,
        minimum_matching_threshold=args.minimum_matching_threshold,
        minimum_consecutive_frames=args.minimum_consecutive_frames,
        trail_length=args.trail_length,
    )
    run_tracking_only(cfg)
    return 0


def _run_demo_offside(args: argparse.Namespace) -> int:
    if args.offside_command == "review":
        from live.offside_review_app import run_review_server

        run_review_server(
            input_path=args.input,
            discovery_dir=args.discovery_dir,
            host=args.host,
            port=args.port,
            device=args.device,
            config_path=args.config,
        )
        return 0

    from app.constants.paths import BALL_DETECTION_MODEL_PATH, PITCH_DETECTION_MODEL_PATH
    from live.offside_demo import DiscoverConfig, JudgeConfig, run_discover, run_judge

    if args.offside_command == "discover":
        summary = run_discover(
            DiscoverConfig(
                input_path=args.input,
                output_dir=args.output_dir,
                device=args.device,
                player_model=args.player_model,
                ball_model=args.ball_model or BALL_DETECTION_MODEL_PATH,
                pitch_model=args.pitch_model or PITCH_DETECTION_MODEL_PATH,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                pitch_interval=args.pitch_interval,
                thumb_interval_s=args.thumb_interval_s,
            )
        )
        assert summary.get("formal_verdict_generated") is False
        return 0
    if args.offside_command == "judge":
        run_judge(
            JudgeConfig(
                config_path=args.config,
                device=args.device,
                output_dir=args.output_dir,
            )
        )
        return 0
    raise SystemExit(f"Unknown offside command: {args.offside_command}")


def _run_pipeline(args: argparse.Namespace) -> None:
    from app.server.main import main as server_main

    argv = [sys.argv[0], "--device", args.device, "--host", args.host, "--port", str(args.port)]
    argv += ["--inference_backend", args.inference_backend]
    if args.video_source:
        argv += ["--video_source", args.video_source]
    sys.argv = argv
    server_main()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        raise SystemExit(0)

    if args.command == "live":
        _run_live(args)
    elif args.command == "field-live":
        _run_field_live(args)
    elif args.command == "record-live":
        code = _run_record_live(args)
        raise SystemExit(code)
    elif args.command == "record-preview":
        _run_record_preview(args)
    elif args.command == "lite-detect":
        _run_lite_detect(args)
    elif args.command == "tracking":
        _run_tracking(args)
    elif args.command == "demo-vision":
        code = _run_demo_vision(args)
        raise SystemExit(code)
    elif args.command == "demo-tracking":
        code = _run_demo_tracking(args)
        raise SystemExit(code)
    elif args.command == "demo-offside":
        code = _run_demo_offside(args)
        raise SystemExit(code)
    elif args.command == "demo-dashboard":
        from live.demo_dashboard_app import run_dashboard_server

        run_dashboard_server(config=args.config, host=args.host, port=args.port)
        raise SystemExit(0)
    elif args.command in {"pipeline", "server"}:
        _run_pipeline(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
