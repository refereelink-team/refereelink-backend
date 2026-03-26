import argparse
import os
import subprocess
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
        is_camera=args.is_camera,
    )




def _run_tracking_and_projection(args: argparse.Namespace) -> None:
    import cv2
    from time import perf_counter

    import supervision as sv

    from core import AsyncPersistence, GameStateManager
    from projection.dynamic_projector import create_dynamic_projector
    from projection.homography import load_calibration
    from projection.visualization import build_projected_objects, render_projection_frame
    from tracking.main import run_player_team_classification_packets

    _ensure_tracking_assets()
    selected_device = _resolve_device(args.device)
    if args.device == "auto":
        print(f"[device] auto selected: {selected_device}")

    # Determine if source is camera
    is_camera = args.is_camera or (args.source_video_path.isdigit() and int(args.source_video_path) >= 0)

    # Load field image
    field_img = cv2.imread(args.field)
    if field_img is None:
        from projection.visualization import load_field_map

        field_img = load_field_map(args.field)
    map_h, map_w = field_img.shape[:2]

    # Load calibration or create dynamic projector
    homography = None
    dynamic_projector = None

    if args.dynamic:
        print(f"[dynamic] enabling dynamic calibration (interval={args.recalib_interval})")
        dynamic_projector = create_dynamic_projector(
            recalib_interval=args.recalib_interval,
            field_path=args.field,
            debug=args.debug,
            min_inliers=4,
            min_inlier_ratio=0.35,
            max_reproj_err=20.0,
            max_consecutive_fails=10,
        )
    elif args.calibration:
        print(f"[calibration] loading from {args.calibration}")
        homography = load_calibration(args.calibration)

    # Setup video writer for projection output (skip for camera/realtime mode)
    projection_writer = None
    if args.projection_output_path and not is_camera:
        video_info = sv.VideoInfo.from_video_path(args.source_video_path)
        fps = float(video_info.fps or 30)
        for fourcc_name in ("mp4v", "XVID", "MJPG"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            writer = cv2.VideoWriter(args.projection_output_path, fourcc, fps, (map_w, map_h))
            if writer.isOpened():
                projection_writer = writer
                break
        if projection_writer is None or not projection_writer.isOpened():
            raise RuntimeError("无法创建 projection 输出视频，请检查路径与编码器。")
    else:
        video_info = None

    frame_stream = run_player_team_classification_packets(
        source_video_path=args.source_video_path,
        device=selected_device,
        is_camera=is_camera,
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

    try:
        tracking_sink = None
        if args.tracking_output_path and not is_camera:
            tracking_sink = sv.VideoSink(args.tracking_output_path, video_info)

        if tracking_sink:
            with tracking_sink:
                for packet in frame_stream:
                    project_started = perf_counter()

                    # Get homography: dynamic or static
                    if dynamic_projector is not None:
                        H_adapter = dynamic_projector.update(packet.raw_frame)

                        # 绘制关键点到 tracking 视图（调试用）
                        if args.debug:
                            from projection.visualization import draw_keypoints_on_frame, draw_keypoints_on_field
                            keypoints = dynamic_projector.keypoint_manager.keypoints
                            if keypoints:
                                debug_frame = draw_keypoints_on_frame(
                                    packet.annotated_frame, keypoints, show_labels=True
                                )
                                packet.annotated_frame = debug_frame
                    else:
                        H_adapter = homography

                    packet.projection_tracklets = build_projected_objects(
                        packet.tracked_objects, homography=H_adapter
                    )
                    packet.metrics.project_ms = (perf_counter() - project_started) * 1000.0

                    render_started = perf_counter()
                    projection_frame = render_projection_frame(
                        tracked_objects=packet.tracked_objects,
                        field_img=field_img,
                        homography=H_adapter,
                    )

                    # 绘制关键点到 projection 视图（调试用）
                    if args.debug and dynamic_projector is not None:
                        from projection.visualization import draw_keypoints_on_field
                        keypoints = dynamic_projector.keypoint_manager.keypoints
                        projection_frame = draw_keypoints_on_field(
                            projection_frame, keypoints, homography=H_adapter, show_labels=True
                        )

                    packet.metrics.render_ms += (perf_counter() - render_started) * 1000.0
                    packet.projection_frame = projection_frame

                    tracking_sink.write_frame(packet.annotated_frame)
                    if projection_writer:
                        projection_writer.write(projection_frame)

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
                    packet.metrics.fps = float(1000.0 / packet.metrics.total_ms) if packet.metrics.total_ms > 0 else 0.0

                    if not args.no_show:
                        cv2.imshow("Tracking", packet.annotated_frame)
                        cv2.imshow("Projection 2D Map", projection_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
        else:
            # Realtime/camera mode: display only, no file output
            for packet in frame_stream:
                project_started = perf_counter()

                # Get homography: dynamic or static
                if dynamic_projector is not None:
                    H_adapter = dynamic_projector.update(packet.raw_frame)

                    # 绘制关键点到 tracking 视图（调试用）
                    if args.debug:
                        from projection.visualization import draw_keypoints_on_frame, draw_keypoints_on_field
                        keypoints = dynamic_projector.keypoint_manager.keypoints
                        if keypoints:
                            debug_frame = draw_keypoints_on_frame(
                                packet.annotated_frame, keypoints, show_labels=True
                            )
                            packet.annotated_frame = debug_frame
                else:
                    H_adapter = homography

                packet.projection_tracklets = build_projected_objects(
                    packet.tracked_objects, homography=H_adapter
                )
                packet.metrics.project_ms = (perf_counter() - project_started) * 1000.0

                render_started = perf_counter()
                projection_frame = render_projection_frame(
                    tracked_objects=packet.tracked_objects,
                    field_img=field_img,
                    homography=H_adapter,
                )

                # 绘制关键点到 projection 视图（调试用）
                if args.debug and dynamic_projector is not None:
                    from projection.visualization import draw_keypoints_on_field
                    keypoints = dynamic_projector.keypoint_manager.keypoints
                    projection_frame = draw_keypoints_on_field(
                        projection_frame, keypoints, homography=H_adapter, show_labels=True
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
                packet.metrics.fps = float(1000.0 / packet.metrics.total_ms) if packet.metrics.total_ms > 0 else 0.0

                if not args.no_show:
                    cv2.imshow("Tracking", packet.annotated_frame)
                    cv2.imshow("Projection 2D Map", projection_frame)
                    # Print FPS for realtime monitoring
                    print(f"FPS: {packet.metrics.fps:.1f} | "
                          f"Det: {packet.metrics.detect_ms:.0f}ms | "
                          f"Track: {packet.metrics.track_ms:.0f}ms | "
                          f"Class: {packet.metrics.classify_ms:.0f}ms | "
                          f"Project: {packet.metrics.project_ms:.0f}ms",
                          end="\r")
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        if projection_writer:
            projection_writer.release()
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
    is_camera: bool = False,
    dynamic: bool = False,
    recalib_interval: int = 10,
    calibration: str = "",
    debug: bool = False,
) -> None:
    ordered: List[str] = list(dict.fromkeys(modules))

    if ordered[:2] == ["tracking", "projection"]:
        shared_args = argparse.Namespace(
            source_video_path=source_video_path,
            tracking_output_path=tracking_output_path,
            projection_output_path=projection_output_path,
            device=device,
            field=field,
            no_show=no_show,
            state_output_path=state_output_path,
            state_flush_interval=state_flush_interval,
            is_camera=is_camera,
            dynamic=dynamic,
            recalib_interval=recalib_interval,
            calibration=calibration,
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
    tracking_parser.add_argument("--is_camera", action="store_true", help="Treat source_video_path as camera index (0, 1, ...)")
    tracking_parser.add_argument("--no_show", action="store_true", help="Don't display video windows")
    tracking_parser.add_argument("--calibration", type=str, default="", help="Path to calibration JSON file")
    tracking_parser.set_defaults(handler=_run_tracking)

    projection_parser = subparsers.add_parser("projection", help="Run projection module")
    projection_parser.add_argument("input", nargs="?", default="tracking/data/2e57b9_0.mp4")
    projection_parser.add_argument(
        "-o", "--output", default="projection/projection_2d.mp4"
    )
    projection_parser.add_argument(
        "--model", default="tracking/data/football-player-detection.pt"
    )
    projection_parser.add_argument("--field", default="field_map.png")
    projection_parser.add_argument("--device", type=str, default="auto")
    projection_parser.add_argument("--no-show", action="store_true")
    projection_parser.set_defaults(handler=_run_projection)

    offside_parser = subparsers.add_parser("offside", help="Run offside module")
    offside_parser.add_argument("input", nargs="?", default="offside/test.mp4")
    offside_parser.add_argument("--frame_index", type=int, required=True)
    offside_parser.add_argument("--output_dir", default="offside/offside_output")
    offside_parser.add_argument(
        "--model", default="tracking/data/football-player-detection.pt"
    )
    offside_parser.add_argument("--field", default="field_map.png")
    offside_parser.add_argument("--device", type=str, default="auto")
    offside_parser.add_argument("--no-show", action="store_true")
    offside_parser.add_argument("--state_output_path", default="")
    offside_parser.add_argument("--state_flush_interval", type=float, default=0.5)
    offside_parser.set_defaults(handler=_run_offside)

    modules_parser = subparsers.add_parser(
        "modules", help="Run one or more modules in sequence"
    )
    modules_parser.add_argument(
        "--modules",
        nargs="+",
        choices=["tracking", "projection", "offside"],
        required=True,
        help="Choose one or more modules, e.g. --modules tracking projection offside",
    )
    modules_parser.add_argument("--source_video_path", type=str, required=True)
    modules_parser.add_argument(
        "--tracking_output_path",
        type=str,
        default="",
    )
    modules_parser.add_argument(
        "--projection_output_path", type=str, default=""
    )
    modules_parser.add_argument(
        "--offside_output_dir", type=str, default="offside/modules-offside-output"
    )
    modules_parser.add_argument(
        "--offside_frame_index", type=int, default=1
    )
    modules_parser.add_argument("--device", type=str, default="auto")
    modules_parser.add_argument(
        "--model", type=str, default="tracking/data/football-player-detection.pt"
    )
    modules_parser.add_argument("--field", type=str, default="field_map.png")
    modules_parser.add_argument("--no-show", action="store_true")
    modules_parser.add_argument("--state_output_path", type=str, default="")
    modules_parser.add_argument("--state_flush_interval", type=float, default=0.5)
    modules_parser.add_argument("--is_camera", action="store_true", help="Treat source_video_path as camera index")
    modules_parser.add_argument("--calibration", type=str, default="", help="Path to calibration JSON file")
    modules_parser.add_argument("--dynamic", action="store_true", help="Use dynamic calibration (auto-detect field lines)")
    modules_parser.add_argument("--recalib_interval", type=int, default=10, help="Recalibration interval for dynamic mode")
    modules_parser.add_argument("--debug", action="store_true", help="Enable debug output")
    modules_parser.set_defaults(
        handler=lambda args: _run_modules(
            modules=args.modules,
            source_video_path=args.source_video_path,
            tracking_output_path=args.tracking_output_path,
            projection_output_path=args.projection_output_path,
            offside_output_dir=args.offside_output_dir,
            offside_frame_index=args.offside_frame_index,
            device=args.device,
            model=args.model,
            field=args.field,
            no_show=args.no_show,
            state_output_path=args.state_output_path,
            state_flush_interval=args.state_flush_interval,
            is_camera=args.is_camera,
            dynamic=args.dynamic,
            recalib_interval=args.recalib_interval,
            calibration=args.calibration,
            debug=args.debug,
        )
    )

    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline")
    pipeline_parser.add_argument("--source_video_path", type=str, required=True)
    pipeline_parser.add_argument(
        "--tracking_output_path",
        type=str,
        default="",
    )
    pipeline_parser.add_argument(
        "--projection_output_path", type=str, default=""
    )
    pipeline_parser.add_argument(
        "--offside_output_dir", type=str, default="offside/pipeline-offside-output"
    )
    pipeline_parser.add_argument(
        "--offside_frame_index", type=int, default=1
    )
    pipeline_parser.add_argument("--device", type=str, default="auto")
    pipeline_parser.add_argument(
        "--model", type=str, default="tracking/data/football-player-detection.pt"
    )
    pipeline_parser.add_argument("--field", type=str, default="field_map.png")
    pipeline_parser.add_argument("--no-show", action="store_true")
    pipeline_parser.add_argument("--state_output_path", type=str, default="")
    pipeline_parser.add_argument("--state_flush_interval", type=float, default=0.5)
    pipeline_parser.add_argument("--is_camera", action="store_true", help="Treat source_video_path as camera index")
    pipeline_parser.add_argument("--calibration", type=str, default="", help="Path to calibration JSON file")
    pipeline_parser.add_argument("--dynamic", action="store_true", help="Use dynamic calibration (auto-detect field lines)")
    pipeline_parser.add_argument("--recalib_interval", type=int, default=10, help="Recalibration interval for dynamic mode")
    pipeline_parser.set_defaults(
        handler=lambda args: _run_modules(
            modules=["tracking", "projection", "offside"],
            source_video_path=args.source_video_path,
            tracking_output_path=args.tracking_output_path,
            projection_output_path=args.projection_output_path,
            offside_output_dir=args.offside_output_dir,
            offside_frame_index=args.offside_frame_index,
            device=args.device,
            model=args.model,
            field=args.field,
            no_show=args.no_show,
            state_output_path=args.state_output_path,
            state_flush_interval=args.state_flush_interval,
            is_camera=args.is_camera,
            dynamic=args.dynamic,
            recalib_interval=args.recalib_interval,
            calibration=args.calibration,
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
