# Soccer Analysis

This repository is a soccer-only restructuring of the original `sports` project.
The code paths are organized by function instead of by reusable package plus example.

## Structure

```text
app/
  annotators/      pitch drawing helpers
  classification/  team classification
  config/          soccer pitch configuration
  constants/       class IDs, colors, asset paths
  geometry/        homography helpers
  modes/           runnable analysis modes
  tracking/        ball tracking helpers
  main.py          CLI entry point
assets/
  data/            sample videos
  weights/         model weights
tools/
  setup_assets.sh  downloads weights and sample videos
backup/
  legacy_repo/     preserved pre-migration structure for reference
notebooks/         model-training workflows
```

## Install

```bash
python3 -m pip install -e .
python3 -m pip install -e ".[tests]"
python3 -m pip install -r requirements.txt
./tools/setup_assets.sh #下载权重和示例视频，如果有了就不用再运行
```

## Run

```bash
python3 app/main.py \
  --source_video_path assets/data/2e57b9_0.mp4 \
  --target_video_path out.mp4 \
  --device cuda \
  --mode PLAYER_DETECTION
```

Available modes:

- `PITCH_DETECTION`
- `PLAYER_DETECTION`
- `BALL_DETECTION`
- `PLAYER_TRACKING`
- `TEAM_CLASSIFICATION`
- `RADAR`
- `RADAR_DASHBOARD`

`RADAR_DASHBOARD` uses a PySide6 + QML frontend with a larger tracking view on
the left, a smaller 2D pitch projection on the right, and a runtime log panel
across the bottom. The tracking view also overlays pitch keypoints. It still
writes the composed dashboard video to `--target_video_path`.

```bash
python3 app/main.py \
  --source_video_path assets/data/2e57b9_0.mp4 \
  --target_video_path out.mp4 \
  --device cpu \
  --mode RADAR_DASHBOARD
```

## Notes

- `backup/legacy_repo/` keeps the previous layout as migration reference.
- Large media and model weights are still kept outside version control.
