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
