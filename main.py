import argparse
import os
import subprocess
import sys
from typing import Iterable, List


def _ensure_tracking_assets() -> None:
    required = [
        os.path.join("tracking", "data", "football-player-detection.pt"),
        os.path.join("tracking", "data", "football-ball-detection.pt"),
        os.path.join("tracking", "data", "football-pitch-detection.pt"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if not missing:
        return

    setup_script = os.path.join("tracking", "setup.sh")
    if not os.path.exists(setup_script):
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing tracking assets ({missing_list}) and setup script not found: {setup_script}"
        )

    print("[bootstrap] tracking models missing, running tracking/setup.sh ...")
    subprocess.run(["bash", setup_script], check=True)


def _resolve_device(device: str = "auto") -> str:
    normalized = (device or "auto").strip().lower()
    if normalized != "auto":
        return normalized

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


def _sync_players_with_projection(packet) -> None:
    """Sync projected field coordinates into packet.players."""
    if not packet.players or not packet.projection_tracklets:
        return
    projected_by_id = {
        int(t.track_id): t
        for t in packet.projection_tracklets
        if t.team != "BALL"
    }
    for player_id, player_state in packet.players.items():
        projected = projected_by_id.get(int(player_id))
        if projected is None:
            continue
        player_state.field_x = float(projected.map_x)
        player_state.field_y = float(projected.map_y)


def _run_tracking(args: argparse.Namespace) -> None:
    from tracking.main import main as run_tracking_main

    _ensure_tracking_assets()
    selected_device = _resolve_device(args.device)
    if args.device == "auto":
        print(f"[device] auto selected: {selected_device}")

    run_tracking_main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=selected_device,
        state_output_path=args.state_output_path,
        state_flush_interval=args.state_flush_interval,
        no_show=args.no_show,
    )


def _run_tracking_and_projection(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np
    from time import perf_counter

    import supervision as sv

    from core import AsyncPersistence, GameStateManager
    from projection.sn_projection_backend import create_projection_engine
    from projection.visualization import (
        build_projected_objects,
        draw_keypoints_on_field,
        draw_keypoints_on_frame,
        load_field_map,
        render_projection_frame,
    )
    from tracking.main import run_player_team_classification_packets

    _ensure_tracking_assets()
    selected_device = _resolve_device(args.device)
    if args.device == "auto":
        print(f"[device] auto selected: {selected_device}")

    field_img = cv2.imread(args.field)
    if field_img is None:
        field_img = load_field_map(args.field)
    map_h, map_w = field_img.shape[:2]

    calib_backend = getattr(args, "calib_backend", "nbjw")
    use_prev_homography = getattr(args, "use_prev_homography", True)
    runtime_dynamic = bool(args.dynamic or calib_backend in {"nbjw", "pnl"})
    if runtime_dynamic:
        print(
            f"[calibration] backend={calib_backend} dynamic=True "
            f"(interval={args.recalib_interval})"
        )
    elif args.calibration:
        print(f"[calibration] loading static homography from {args.calibration}")

    projection_engine = create_projection_engine(
        calib_backend=calib_backend,
        dynamic=runtime_dynamic,
        recalib_interval=args.recalib_interval,
        calibration_path=args.calibration,
        field_path=args.field,
        debug=args.debug,
        use_prev_homography=use_prev_homography,
    )

    video_info = None
    if args.projection_output_path or args.tracking_output_path or args.calibration_video:
        video_info = sv.VideoInfo.from_video_path(args.source_video_path)

    projection_writer = None
    if args.projection_output_path:
        fps = float(video_info.fps or 30)
        for fourcc_name in ("mp4v", "XVID", "MJPG"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            writer = cv2.VideoWriter(
                args.projection_output_path,
                fourcc,
                fps,
                (map_w, map_h),
            )
            if writer.isOpened():
                projection_writer = writer
                break
        if projection_writer is None or not projection_writer.isOpened():
            raise RuntimeError("Failed to create projection output video writer.")

    calib_video_writer = None
    if args.calibration_video:
        fps = float(video_info.fps or 30)
        orig_h = int(video_info.height)
        orig_w = int(video_info.width)
        composite_w = orig_w * 2
        composite_h = orig_h
        for fourcc_name in ("mp4v", "XVID", "MJPG"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            writer = cv2.VideoWriter(
                args.calibration_video,
                fourcc,
                fps,
                (composite_w, composite_h),
            )
            if writer.isOpened():
                calib_video_writer = writer
                break
        if calib_video_writer is None:
            print("Warning: Failed to create calibration video writer.", file=sys.stderr)

    frame_stream = run_player_team_classification_packets(
        source_video_path=args.source_video_path,
        device=selected_device,
    )

    game_state = None
    persistence = None
    if args.state_output_path:
        game_state = GameStateManager()
        persistence = AsyncPersistence(
            game_state=game_state,
            output_path=args.state_output_path,
            flush_interval=args.state_flush_interval,
        )
        persistence.start()

    def _process_packet(packet):
        project_started = perf_counter()
        h_adapter = projection_engine.update(packet.raw_frame)
        keypoints = projection_engine.get_keypoints()

        # Always draw keypoints on annotated frame when available
        if keypoints:
            packet.annotated_frame = draw_keypoints_on_frame(
                packet.annotated_frame,
                keypoints,
                show_labels=True,
            )

        packet.projection_tracklets = build_projected_objects(
            packet.tracked_objects,
            homography=h_adapter,
        )
        _sync_players_with_projection(packet)
        packet.metrics.project_ms = (perf_counter() - project_started) * 1000.0

        render_started = perf_counter()
        projection_frame = render_projection_frame(
            tracked_objects=packet.tracked_objects,
            field_img=field_img,
            homography=h_adapter,
        )
        if keypoints:
            projection_frame = draw_keypoints_on_field(
                projection_frame,
                keypoints,
                homography=h_adapter,
                show_labels=True,
            )
        packet.metrics.render_ms += (perf_counter() - render_started) * 1000.0
        packet.projection_frame = projection_frame

        if game_state is not None:
            persist_started = perf_counter()
            game_state.update_packet(packet)
            packet.metrics.persist_ms = (perf_counter() - persist_started) * 1000.0

        packet.metrics.total_ms = (
            packet.metrics.detect_ms
            + packet.metrics.track_ms
            + packet.metrics.classify_ms
            + packet.metrics.project_ms
            + packet.metrics.render_ms
            + packet.metrics.persist_ms
        )
        packet.metrics.fps = (
            float(1000.0 / packet.metrics.total_ms)
            if packet.metrics.total_ms > 0
            else 0.0
        )
        return projection_frame, keypoints

    try:
        tracking_sink = None
        if args.tracking_output_path:
            tracking_sink = sv.VideoSink(args.tracking_output_path, video_info)

        if tracking_sink:
            with tracking_sink:
                for packet in frame_stream:
                    projection_frame, keypoints = _process_packet(packet)
                    tracking_sink.write_frame(packet.annotated_frame)
                    if projection_writer:
                        projection_writer.write(projection_frame)
                    if calib_video_writer is not None and keypoints:
                        # Left: annotated frame with keypoint overlay
                        # Right: 2D field with template keypoint positions
                        calib_left = packet.annotated_frame
                        # Resize projection frame to match annotated frame height
                        target_h = calib_left.shape[0]
                        scale = target_h / projection_frame.shape[0]
                        new_w = int(projection_frame.shape[1] * scale)
                        calib_right = cv2.resize(projection_frame, (new_w, target_h))
                        calib_composite = np.hstack([calib_left, calib_right])
                        calib_video_writer.write(calib_composite)

                    if not args.no_show:
                        cv2.imshow("Tracking", packet.annotated_frame)
                        cv2.imshow("Projection 2D Map", projection_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
        else:
            # Headless mode - process frames without display
            for packet in frame_stream:
                projection_frame, keypoints = _process_packet(packet)
                if calib_video_writer is not None and keypoints:
                    calib_left = packet.annotated_frame
                    target_h = calib_left.shape[0]
                    scale = target_h / projection_frame.shape[0]
                    new_w = int(projection_frame.shape[1] * scale)
                    calib_right = cv2.resize(projection_frame, (new_w, target_h))
                    calib_composite = np.hstack([calib_left, calib_right])
                    calib_video_writer.write(calib_composite)
                if not args.no_show:
                    cv2.imshow("Tracking", packet.annotated_frame)
                    cv2.imshow("Projection 2D Map", projection_frame)
                    print(
                        f"FPS: {packet.metrics.fps:.1f} | "
                        f"Det: {packet.metrics.detect_ms:.0f}ms | "
                        f"Track: {packet.metrics.track_ms:.0f}ms | "
                        f"Class: {packet.metrics.classify_ms:.0f}ms | "
                        f"Project: {packet.metrics.project_ms:.0f}ms",
                        end="\r",
                    )
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        if projection_writer:
            projection_writer.release()
        if calib_video_writer is not None:
            calib_video_writer.release()
        if persistence is not None:
            persistence.stop()
            persistence.join()
        if not args.no_show:
            cv2.destroyAllWindows()


def _run_offside(args: argparse.Namespace) -> None:
    from offside.run_var_video import run_single_frame_pipeline

    if not os.path.exists(args.model):
        _ensure_tracking_assets()
    selected_device = _resolve_device(args.device)
    if args.device == "auto":
        print(f"[device] auto selected: {selected_device}")

    run_single_frame_pipeline(
        video_path=args.input,
        frame_index=args.frame_index,
        output_dir=args.output_dir,
        model_path=args.model,
        field_map_path=args.field,
        show_live=not args.no_show,
        state_output_path=args.state_output_path,
        state_flush_interval=args.state_flush_interval,
        device=selected_device,
    )


def _run_projection(args: argparse.Namespace) -> None:
    from projection.visualization import run_projection_video_pipeline

    if not os.path.exists(args.model):
        _ensure_tracking_assets()
    selected_device = _resolve_device(args.device)
    if args.device == "auto":
        print(f"[device] auto selected: {selected_device}")

    run_projection_video_pipeline(
        video_path=args.input,
        output_path=args.output,
        model_path=args.model,
        field_map_path=args.field,
        show_live=not args.no_show,
        device=selected_device,
        dynamic=getattr(args, "dynamic", False),
        recalib_interval=getattr(args, "recalib_interval", 10),
        calibration=getattr(args, "calibration", ""),
        calib_backend=getattr(args, "calib_backend", "nbjw"),
        use_prev_homography=getattr(args, "use_prev_homography", True),
        debug=getattr(args, "debug", False),
    )


def _run_modules(
    modules: Iterable[str],
    source_video_path: str,
    tracking_output_path: str,
    projection_output_path: str,
    offside_output_dir: str,
    offside_frame_index: int,
    device: str,
    model: str,
    field: str,
    no_show: bool,
    state_output_path: str,
    state_flush_interval: float,
    dynamic: bool = False,
    recalib_interval: int = 10,
    calibration: str = "",
    calib_backend: str = "nbjw",
    use_prev_homography: bool = True,
    debug: bool = False,
    calibration_video: str = "",
) -> None:
    ordered: List[str] = list(dict.fromkeys(modules))

    if ordered[:2] == ["tracking", "projection"]:
        shared_args = argparse.Namespace(
            source_video_path=source_video_path,
            tracking_output_path=tracking_output_path,
            projection_output_path=projection_output_path,
            calibration_video=calibration_video,
            device=device,
            field=field,
            no_show=no_show,
            state_output_path=state_output_path,
            state_flush_interval=state_flush_interval,
            dynamic=dynamic,
            recalib_interval=recalib_interval,
            calibration=calibration,
            calib_backend=calib_backend,
            use_prev_homography=use_prev_homography,
            debug=debug,
        )
        _run_tracking_and_projection(shared_args)
        ordered = ordered[2:]

    for module_name in ordered:
        if module_name == "tracking":
            tracking_args = argparse.Namespace(
                source_video_path=source_video_path,
                target_video_path=tracking_output_path,
                device=device,
                state_output_path=state_output_path,
                state_flush_interval=state_flush_interval,
                no_show=no_show,
            )
            _run_tracking(tracking_args)
            continue

        if module_name == "projection":
            projection_args = argparse.Namespace(
                input=source_video_path,
                output=projection_output_path,
                model=model,
                field=field,
                device=device,
                no_show=no_show,
                dynamic=dynamic,
                recalib_interval=recalib_interval,
                calibration=calibration,
                calib_backend=calib_backend,
                use_prev_homography=use_prev_homography,
                debug=debug,
            )
            _run_projection(projection_args)
            continue

        if module_name == "offside":
            offside_args = argparse.Namespace(
                input=source_video_path,
                frame_index=offside_frame_index,
                output_dir=offside_output_dir,
                model=model,
                field=field,
                device=device,
                no_show=no_show,
                state_output_path=state_output_path,
                state_flush_interval=state_flush_interval,
            )
            _run_offside(offside_args)
            continue

        raise ValueError(f"Unsupported module: {module_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sports Main unified CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tracking_parser = subparsers.add_parser("tracking", help="Run tracking module")
    tracking_parser.add_argument("--source_video_path", type=str, required=True)
    tracking_parser.add_argument("--target_video_path", type=str, required=False, default="")
    tracking_parser.add_argument("--device", type=str, default="auto")
    tracking_parser.add_argument("--state_output_path", type=str, default="")
    tracking_parser.add_argument("--state_flush_interval", type=float, default=0.5)
    tracking_parser.add_argument("--no_show", action="store_true", help="Do not display windows")
    tracking_parser.set_defaults(handler=_run_tracking)

    projection_parser = subparsers.add_parser("projection", help="Run projection module")
    projection_parser.add_argument("input", nargs="?", default="tracking/data/2e57b9_0.mp4")
    projection_parser.add_argument("-o", "--output", default="projection/projection_2d.mp4")
    projection_parser.add_argument("--model", default="tracking/data/football-player-detection.pt")
    projection_parser.add_argument("--field", default="field_map.png")
    projection_parser.add_argument("--device", type=str, default="auto")
    projection_parser.add_argument("--no_show", action="store_true")
    projection_parser.add_argument("--calibration", type=str, default="")
    projection_parser.add_argument("--dynamic", action="store_true")
    projection_parser.add_argument("--recalib_interval", type=int, default=10)
    projection_parser.add_argument(
        "--calib_backend",
        type=str,
        choices=["legacy", "nbjw", "pnl"],
        default="nbjw",
    )
    projection_parser.add_argument(
        "--no_use_prev_homography",
        "--no-use-prev-homography",
        action="store_false",
        dest="use_prev_homography",
    )
    projection_parser.set_defaults(use_prev_homography=True)
    projection_parser.add_argument("--debug", action="store_true")
    projection_parser.set_defaults(handler=_run_projection)

    offside_parser = subparsers.add_parser("offside", help="Run offside module")
    offside_parser.add_argument("input", nargs="?", default="offside/test.mp4")
    offside_parser.add_argument("--frame_index", type=int, required=True)
    offside_parser.add_argument("--output_dir", default="offside/offside_output")
    offside_parser.add_argument("--model", default="tracking/data/football-player-detection.pt")
    offside_parser.add_argument("--field", default="field_map.png")
    offside_parser.add_argument("--device", type=str, default="auto")
    offside_parser.add_argument("--no_show", action="store_true")
    offside_parser.add_argument("--state_output_path", default="")
    offside_parser.add_argument("--state_flush_interval", type=float, default=0.5)
    offside_parser.set_defaults(handler=_run_offside)

    modules_parser = subparsers.add_parser(
        "modules",
        help="Run one or more modules in sequence",
    )
    modules_parser.add_argument(
        "--modules",
        nargs="+",
        choices=["tracking", "projection", "offside"],
        required=True,
        help="Choose one or more modules",
    )
    modules_parser.add_argument("--source_video_path", type=str, required=True)
    modules_parser.add_argument("--tracking_output_path", type=str, default="")
    modules_parser.add_argument("--projection_output_path", type=str, default="")
    modules_parser.add_argument("--calibration_video", type=str, default="", help="Output path for calibration debug video (keypoints on original + field view, side-by-side)")
    modules_parser.add_argument(
        "--offside_output_dir",
        type=str,
        default="offside/modules-offside-output",
    )
    modules_parser.add_argument("--offside_frame_index", type=int, default=1)
    modules_parser.add_argument("--device", type=str, default="auto")
    modules_parser.add_argument("--model", type=str, default="tracking/data/football-player-detection.pt")
    modules_parser.add_argument("--field", type=str, default="field_map.png")
    modules_parser.add_argument("--no_show", action="store_true")
    modules_parser.add_argument("--state_output_path", type=str, default="")
    modules_parser.add_argument("--state_flush_interval", type=float, default=0.5)
    modules_parser.add_argument("--calibration", type=str, default="")
    modules_parser.add_argument("--dynamic", action="store_true")
    modules_parser.add_argument("--recalib_interval", type=int, default=10)
    modules_parser.add_argument(
        "--calib_backend",
        type=str,
        choices=["legacy", "nbjw", "pnl"],
        default="nbjw",
    )
    modules_parser.add_argument(
        "--no_use_prev_homography",
        "--no-use-prev-homography",
        action="store_false",
        dest="use_prev_homography",
    )
    modules_parser.set_defaults(use_prev_homography=True)
    modules_parser.add_argument("--debug", action="store_true")
    modules_parser.set_defaults(
        handler=lambda args: _run_modules(
            modules=args.modules,
            source_video_path=args.source_video_path,
            tracking_output_path=args.tracking_output_path,
            projection_output_path=args.projection_output_path,
            calibration_video=args.calibration_video,
            offside_output_dir=args.offside_output_dir,
            offside_frame_index=args.offside_frame_index,
            device=args.device,
            model=args.model,
            field=args.field,
            no_show=args.no_show,
            state_output_path=args.state_output_path,
            state_flush_interval=args.state_flush_interval,
            dynamic=args.dynamic,
            recalib_interval=args.recalib_interval,
            calibration=args.calibration,
            calib_backend=args.calib_backend,
            use_prev_homography=args.use_prev_homography,
            debug=args.debug,
        )
    )

    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline")
    pipeline_parser.add_argument("--source_video_path", type=str, required=True)
    pipeline_parser.add_argument("--tracking_output_path", type=str, default="")
    pipeline_parser.add_argument("--projection_output_path", type=str, default="")
    pipeline_parser.add_argument("--calibration_video", type=str, default="", help="Output path for calibration debug video (keypoints on original + field view, side-by-side)")
    pipeline_parser.add_argument(
        "--offside_output_dir",
        type=str,
        default="offside/pipeline-offside-output",
    )
    pipeline_parser.add_argument("--offside_frame_index", type=int, default=1)
    pipeline_parser.add_argument("--device", type=str, default="auto")
    pipeline_parser.add_argument("--model", type=str, default="tracking/data/football-player-detection.pt")
    pipeline_parser.add_argument("--field", type=str, default="field_map.png")
    pipeline_parser.add_argument("--no_show", action="store_true")
    pipeline_parser.add_argument("--state_output_path", type=str, default="")
    pipeline_parser.add_argument("--state_flush_interval", type=float, default=0.5)
    pipeline_parser.add_argument("--calibration", type=str, default="")
    pipeline_parser.add_argument("--dynamic", action="store_true")
    pipeline_parser.add_argument("--recalib_interval", type=int, default=10)
    pipeline_parser.add_argument(
        "--calib_backend",
        type=str,
        choices=["legacy", "nbjw", "pnl"],
        default="nbjw",
    )
    pipeline_parser.add_argument(
        "--no_use_prev_homography",
        "--no-use-prev-homography",
        action="store_false",
        dest="use_prev_homography",
    )
    pipeline_parser.set_defaults(use_prev_homography=True)
    pipeline_parser.add_argument("--debug", action="store_true")
    pipeline_parser.set_defaults(
        handler=lambda args: _run_modules(
            modules=["tracking", "projection", "offside"],
            source_video_path=args.source_video_path,
            tracking_output_path=args.tracking_output_path,
            projection_output_path=args.projection_output_path,
            calibration_video=args.calibration_video,
            offside_output_dir=args.offside_output_dir,
            offside_frame_index=args.offside_frame_index,
            device=args.device,
            model=args.model,
            field=args.field,
            no_show=args.no_show,
            state_output_path=args.state_output_path,
            state_flush_interval=args.state_flush_interval,
            dynamic=args.dynamic,
            recalib_interval=args.recalib_interval,
            calibration=args.calibration,
            calib_backend=args.calib_backend,
            use_prev_homography=args.use_prev_homography,
            debug=args.debug,
        )
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
