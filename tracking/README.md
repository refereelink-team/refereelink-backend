# tracking 模块 README

`tracking` 是项目的**检测与跟踪入口**：负责在像素坐标系输出统一 `FramePacket`，不负责投影和越位规则。

## 1. 模块结构

- `backend.py`：底层 YOLO + ByteTrack 能力，属于低层检测/跟踪原语。
- `main.py`：主流程，执行队伍分类并产出 `FramePacket`。
- `common/`：通用组件（队伍分类、可视化等）。
- `annotators/`、`configs/`：绘制与配置。
- `data/`：模型和样例视频资源。

## 2. 负责什么

- 目标检测与多目标跟踪。
- 为每个目标提供像素框、ID、置信度、粗粒度队伍标签。
- 产出统一 `FramePacket`：
  - `tracked_objects: list[core.ObjectTrack]`
  - `players: dict[int, core.PlayerState]`
  - `annotated_frame`

## 3. 引用了哪些模块（出向依赖）

- `tracking/main.py` 引用 `core`（`FramePacket`、`ObjectTrack`、`GameStateManager`、`AsyncPersistence` 等）用于统一 packet 输出与可选状态落盘。
- `tracking` 主流程不引用 `projection` 与 `offside`。

## 4. 被哪些模块引用（入向依赖）

- `projection/visualization.py`：消费 `run_player_team_classification_packets(...)` 产出的 `FramePacket.tracked_objects`。
- `main.py`：调用 `tracking.main` 作为统一 CLI 子命令与共享主链路入口。
- `offside`：在单帧分析场景下复用 `tracking.backend` 的低层检测/跟踪原语，再转入 packet-first 适配层。

## 5. 边界约束

- `tracking` 只做“像素空间感知”，不做：
  - 像素到球场坐标的映射（这是 `projection` 职责）
  - 越位/出界规则判定（这是 `offside` 职责）
- `tracking.backend.TrackedObject` 属于低层兼容类型；仓库级共享契约以 `core.ObjectTrack` / `core.FramePacket` 为准。
