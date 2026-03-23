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
    )




def _run_tracking_and_projection(args: argparse.Namespace) -> None:
    import cv2

    import supervision as sv

    from core import AsyncPersistence, GameStateManager
    from projection.visualization import build_projected_objects, render_projection_frame
    from tracking.main import run_player_team_classification_packets

    _ensure_tracking_assets()
    selected_device = _resolve_device(args.device)
    if args.device == "auto":
        print(f"[device] auto selected: {selected_device}")

    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    fps = float(video_info.fps or 30)
    field_img = cv2.imread(args.field)
    if field_img is None:
        from projection.visualization import load_field_map

        field_img = load_field_map(args.field)
    map_h, map_w = field_img.shape[:2]

    projection_writer = None
    for fourcc_name in ("mp4v", "XVID", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(args.projection_output_path, fourcc, fps, (map_w, map_h))
        if writer.isOpened():
            projection_writer = writer
            break
    if projection_writer is None or not projection_writer.isOpened():
        raise RuntimeError("无法创建 projection 输出视频，请检查路径与编码器。")

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

    try:
        with sv.VideoSink(args.tracking_output_path, video_info) as tracking_sink:
            for packet in frame_stream:
                tracking_sink.write_frame(packet.annotated_frame)
                packet.projection_tracklets = build_projected_objects(packet.tracked_objects)
                projection_frame = render_projection_frame(
                    tracked_objects=packet.tracked_objects,
                    field_img=field_img,
                )
                packet.projection_frame = projection_frame
                projection_writer.write(projection_frame)

                if game_state is not None:
                    game_state.update_packet(packet)

                if not args.no_show:
                    cv2.imshow("Tracking", packet.annotated_frame)
                    cv2.imshow("Projection 2D Map", projection_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
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
    tracking_parser.add_argument("--target_video_path", type=str, required=True)
    tracking_parser.add_argument("--device", type=str, default="auto")
    tracking_parser.add_argument("--state_output_path", type=str, default="")
    tracking_parser.add_argument("--state_flush_interval", type=float, default=0.5)
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
        default="tracking/data/modules-tracking-output.mp4",
    )
    modules_parser.add_argument(
        "--projection_output_path", type=str, default="projection/modules-projection-2d.mp4"
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
        )
    )

    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline")
    pipeline_parser.add_argument("--source_video_path", type=str, required=True)
    pipeline_parser.add_argument(
        "--tracking_output_path",
        type=str,
        default="tracking/data/pipeline-tracking-output.mp4",
    )
    pipeline_parser.add_argument(
        "--projection_output_path", type=str, default="projection/pipeline-projection-2d.mp4"
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
        )
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
