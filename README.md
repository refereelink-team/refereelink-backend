# Sports Main 项目说明（中文）

本仓库是一个以足球视频分析为主的视觉项目，当前结构已统一为：
- `core/`：稳定的数据结构与状态管理中台
- `projection/`：3D/像素到 2D 球场平面投影建模 + 2D 可视化视频输出
- `tracking/`：跟踪底座 + `PLAYER_TEAM_CLASSIFICATION` 主流程
- `offside/`：越位判定与单帧可视化（复用 `projection`）
- `main.py`：仓库统一 CLI 入口

## 1. 目录结构

```text
sports-main/
├─ requirements.txt
├─ setup.py
├─ README.md
├─ main.py
├─ core/
│  ├─ __init__.py
│  ├─ state.py
│  └─ README.md
├─ projection/
│  ├─ __init__.py
│  ├─ homography.py
│  ├─ modeling.py
│  ├─ visualization.py
│  └─ README.md
├─ tracking/
│  ├─ __init__.py
│  ├─ main.py
│  ├─ backend.py
│  ├─ README.md
│  ├─ setup.sh
│  ├─ requirements.txt
│  ├─ annotators/
│  ├─ common/
│  ├─ configs/
│  ├─ data/
│  └─ notebooks/
└─ offside/
   ├─ __init__.py
   ├─ run_var_video.py
   ├─ judgement.py
   ├─ offside_core_integration.py
   └─ README.md
```

## 2. 功能概览

- tracking：球员/门将/裁判识别 + 球队分类（`PLAYER_TEAM_CLASSIFICATION`），并提供统一跟踪底座
- projection：负责将跟踪结果映射为 2D 球场平面，并输出 2D 球场视频
- offside：基于 projection 结果执行越位判定，并输出单帧可视化结果
- 可选状态输出（CSV），由 `core` 提供统一数据结构与异步持久化

## 3. 环境安装（统一依赖）

推荐 Python `>=3.8`（建议 3.11）。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

说明：
- `requirements.txt` 是主目录统一依赖清单，可覆盖完整运行所需依赖（含 `tracking` 与 `offside`）。
- `pip install -e .` 用于本地可编辑安装，确保 `core` / `tracking` 在任意工作目录都可导入。

## 4. 准备模型与示例视频

```bash
cd tracking
./setup.sh
cd ..
```

资源会下载到 `tracking/data/`。

如果通过主入口执行 `tracking`、`modules` 或 `pipeline`，当检测到 `tracking/data/*.pt`
模型缺失时，会自动调用 `tracking/setup.sh` 下载（复用原脚本，不改变下载逻辑）。

## 5. 快速开始

建议优先使用统一入口在仓库根目录执行：

```bash
python main.py tracking \
  --source_video_path tracking/data/2e57b9_0.mp4 \
  --target_video_path tracking/data/2e57b9_0-player-tracking.mp4
```

运行 projection（导出 2D 球场视频）：

```bash
python main.py projection tracking/data/2e57b9_0.mp4 \
  -o projection/projection_2d.mp4 \
  --field field_map.png
```

运行 offside（关键帧单帧可视化）：

```bash
python main.py offside offside/test.mp4 \
  --frame_index 120 \
  --output_dir offside/offside_output \
  --model tracking/data/football-player-detection.pt \
  --field field_map.png
```

按模块组合运行：

```bash
python main.py modules \
  --modules tracking projection offside \
  --source_video_path tracking/data/2e57b9_0.mp4 \
  --tracking_output_path tracking/data/modules-tracking.mp4 \
  --projection_output_path projection/modules-projection-2d.mp4 \
  --offside_frame_index 120 \
  --offside_output_dir offside/modules-offside-output
```

运行整条 pipeline：

```bash
python main.py pipeline \
  --source_video_path tracking/data/2e57b9_0.mp4
```

### 摄像头实时输入（新增）

```bash
# 实时 tracking（摄像头）
python main.py tracking \
  --source_video_path 0 \
  --is_camera

# 实时 tracking + projection（双窗口显示）
python main.py modules \
  --modules tracking projection \
  --source_video_path 0 \
  --is_camera
```

启用状态输出（tracking 示例）：

```bash
python main.py tracking \
  --source_video_path tracking/data/2e57b9_0.mp4 \
  --target_video_path tracking/data/2e57b9_0-player-tracking.mp4 \
  --device cpu \
  --state_output_path tracking/data/2e57b9_0-player-tracking-state.csv \
  --state_flush_interval 0.5
```

## 6. 参数说明（统一入口）

- `--device`：默认 `auto`，按 `cuda > mps > cpu` 自动选择；也可手动指定
- `tracking` 子命令：`--source_video_path`、`--target_video_path`（固定运行 `PLAYER_TEAM_CLASSIFICATION`）、`--is_camera`（摄像头模式）
- `projection` 子命令：`input`、`--output`、`--field`
- `offside` 子命令：`input`、`--frame_index`、`--output_dir`、`--field`
- `modules`：`--modules tracking projection offside` 可顺序运行一个或多个模块，`--is_camera` 启用摄像头模式
- `pipeline`：固定执行 tracking + projection + offside
- 通用：`--state_output_path` / `--state_flush_interval` 启用 core 状态导出

## 7. 与 core 的关系

`tracking/main.py` 在运行时可选创建：
- `GameStateManager`
- `AsyncPersistence`

并按帧写入 `FrameState`。  
`core` 的详细接口见 `core/README.md`。

## 8. 扩展开发

新增模块接入规范见主目录文档：
- `MODULE_INTEGRATION.md`

## 9. 许可

- 本仓库代码遵循 MIT（见 `LICENSE`）
- `ultralytics` 遵循 AGPL-3.0，请按其许可条款使用
