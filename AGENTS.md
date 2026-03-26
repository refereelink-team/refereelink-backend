# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

This is a football (soccer) video analysis project with modules for:
- **tracking**: Player/goalkeeper/referee detection + team classification
- **projection**: 3D pixel to 2D field plane projection + 2D visualization
- **offside**: Offside judgment with single-frame visualization
- **core**: Shared data structures and state management

## Common Commands

```bash
# Setup environment
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt && pip install -e .

# Download required models
cd tracking && ./setup.sh

# Run tracking
python main.py tracking --source_video_path tracking/data/2e57b9_0.mp4 --target_video_path output.mp4

# Run tracking with camera input
python main.py tracking --source_video_path 0 --target_video_path output.mp4

# Run projection (2D field visualization)
python main.py projection tracking/data/2e57b9_0.mp4 -o projection/projection_2d.mp4 --field field_map.png

# Run offside detection
python main.py offside offside/test.mp4 --frame_index 120 --output_dir offside/output --field field_map.png

# Run full pipeline
python main.py pipeline --source_video_path tracking/data/2e57b9_0.mp4
```

## Camera Input Support

The `run_player_team_classification_packets` function in `tracking/main.py` supports both video files and camera inputs:
- Video file: Pass file path as usual
- Camera: Pass camera index (0, 1, etc.) - the function automatically detects numeric strings as camera indices

For camera input, a default team prototype model is used (no pre-collected crops required).

## Real-time Parallel Pipeline

For real-time display of both tracking and projection views simultaneously, use the parallel pipeline in `core/pipeline.py`:

```python
from core import create_parallel_pipeline

# Create and start pipeline
pipeline = create_parallel_pipeline(
    source="0",  # camera index or video path
    device="auto",
    is_camera=True,
)

# Get latest processed packet
packet = pipeline.get_latest_packet()
# packet.annotated_frame  # tracking view
# packet.projection_frame # projection view
```

The pipeline uses a thread-safe frame buffer (max 3 frames) to allow tracking and projection to run concurrently.

## Architecture

The system uses a unified `FramePacket` as the main data exchange object through the pipeline:

1. **tracking** produces `ObjectTrack` objects (pixel-space detections with team labels)
2. **projection** transforms pixels to 2D field coordinates via homography, outputs `ProjectedObject`
3. **offside** uses projection results to determine offside positions

### Key Data Structures

- `core/packet.py`: `FramePacket` - the main frame data carrier with fields like `tracked_objects`, `players`, `ball`, `projection_tracklets`, `metrics`
- `core/state.py`: `FrameState`, `PlayerState`, `BallState` - stable business state snapshots; `Team` enum (HOME, AWAY, REFEREE, UNKNOWN)

### Adding New Modules

Follow `MODULE_INTEGRATION.md`:
1. Read from `FramePacket` (use existing fields like `tracked_objects`)
2. Add results to packet's `events`, `debug_info`, or new optional fields
3. Use `GameStateManager.update_packet(packet)` for state persistence
4. Use `frame_state_from_packet(packet)` to convert packet to state

## Device Selection

Device auto-selection priority: `cuda` > `mps` > `cpu`. Pass `--device auto` (default) to auto-select.

## Code Structure

- `main.py`: Unified CLI entry point with subcommands (tracking, projection, offside, modules, pipeline)
- `core/`: Data structures, state management (`packet.py`, `state.py`, `persistence.py`, `store.py`, `pipeline.py`)
- `tracking/`: Detection/tracking using ultralytics (`main.py`, `backend.py`, `common/realtime_team.py`)
- `projection/`: Field projection (`homography.py`, `modeling.py`, `visualization.py`)
- `offside/`: Offside judgment (`judgement.py`, `run_var_video.py`)
