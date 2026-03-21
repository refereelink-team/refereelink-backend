# Sports Main 项目说明（中文）

本仓库是一个以足球视频分析为主的视觉项目，现已统一为 **packet-first** 架构：

- `core/`：统一数据契约、状态快照与状态中台
- `tracking/`：产出 `FramePacket` 的检测/跟踪主流程
- `projection/`：消费 `FramePacket.tracked_objects`，补充投影结果与 2D 球场可视化
- `offside/`：消费投影结果，补充越位判定结果并写回 `GameStateManager`
- `main.py`：仓库统一 CLI 入口

## 1. 目录结构

```text
sports-main/
├─ requirements.txt
├─ setup.py
├─ README.md
├─ MODULE_INTEGRATION.md
├─ main.py
├─ core/
│  ├─ __init__.py
│  ├─ packet.py
│  ├─ state.py
│  ├─ store.py
│  ├─ persistence.py
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

## 2. 核心架构

### 2.1 统一主链路

```text
tracking -> FramePacket -> projection -> enriched FramePacket -> offside -> GameStateManager
                                                      |
                                                      v
                                           frame_state_from_packet
                                                      |
                                                      v
                                               AsyncPersistence
```

### 2.2 三层职责

- `core.packet.FramePacket`：实时逐帧交换对象，是主链路唯一事实来源。
- `core.state.FrameState`：由 `frame_state_from_packet(...)` 派生的稳定业务快照，用于历史查询与 CSV 输出。
- `core.store.GameStateManager`：统一消费 `update_packet(packet)`，同时缓存 packet 和 frame 快照。

## 3. 功能概览

- `tracking`：球员/门将/裁判识别 + 球队分类，输出 `FramePacket`
- `projection`：将 `tracked_objects` 投影为 `projection_tracklets`，并生成 `projection_frame`
- `offside`：基于投影结果执行越位判定，生成关键帧可视化与 `OffsideQuery`
- `core`：统一提供状态缓存、历史访问、回调与异步落盘

## 4. 环境安装（统一依赖）

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
- `pip install -e .` 用于本地可编辑安装，确保 `core` / `tracking` / `projection` / `offside` 在任意工作目录都可导入。

## 5. 准备模型与示例视频

```bash
cd tracking
./setup.sh
cd ..
```

资源会下载到 `tracking/data/`。

如果通过主入口执行 `tracking`、`modules` 或 `pipeline`，当检测到 `tracking/data/*.pt` 模型缺失时，会自动调用 `tracking/setup.sh` 下载。

## 6. 快速开始

建议优先使用统一入口在仓库根目录执行。

### 6.1 tracking

```bash
python main.py tracking \
  --source_video_path tracking/data/2e57b9_0.mp4 \
  --target_video_path tracking/data/2e57b9_0-player-tracking.mp4
```

### 6.2 projection（复用 packet 主链路）

```bash
python main.py projection tracking/data/2e57b9_0.mp4 \
  -o projection/projection_2d.mp4 \
  --field field_map.png
```

### 6.3 offside（关键帧单帧可视化）

```bash
python main.py offside offside/test.mp4 \
  --frame_index 120 \
  --output_dir offside/offside_output \
  --model tracking/data/football-player-detection.pt \
  --field field_map.png
```

### 6.4 按模块组合运行

```bash
python main.py modules \
  --modules tracking projection offside \
  --source_video_path tracking/data/2e57b9_0.mp4 \
  --tracking_output_path tracking/data/modules-tracking.mp4 \
  --projection_output_path projection/modules-projection-2d.mp4 \
  --offside_frame_index 120 \
  --offside_output_dir offside/modules-offside-output
```

### 6.5 运行整条 pipeline

```bash
python main.py pipeline \
  --source_video_path tracking/data/2e57b9_0.mp4
```

## 7. 状态输出与 `core`

启用状态输出（tracking 示例）：

```bash
python main.py tracking \
  --source_video_path tracking/data/2e57b9_0.mp4 \
  --target_video_path tracking/data/2e57b9_0-player-tracking.mp4 \
  --device cpu \
  --state_output_path tracking/data/2e57b9_0-player-tracking-state.csv \
  --state_flush_interval 0.5
```

运行时会：

- 创建 `GameStateManager`
- 可选创建 `AsyncPersistence`
- 由 `tracking` / `projection` / `offside` 通过 `update_packet(packet)` 写入统一状态
- 自动从 packet 派生 `FrameState` 用于持久化与历史查询

`core` 的详细接口见 `core/README.md`。

## 8. 模块扩展开发

新增模块接入规范见：

- `MODULE_INTEGRATION.md`

## 9. 许可

- 本仓库代码遵循 MIT（见 `LICENSE`）
- `ultralytics` 遵循 AGPL-3.0，请按其许可条款使用
