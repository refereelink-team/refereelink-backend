# tracking 模块 README

`tracking` 是项目的**检测与跟踪底座**：在像素坐标系输出稳定 `tracklet`，不负责投影和越位规则。

## 1. 模块结构

- `backend.py`：底座能力（YOLO 检测 + ByteTrack 跟踪），输出 `TrackedObject`。
- `main.py`：球员/队伍识别主流程（现改为轻量颜色原型 + track 平滑），并按帧回填 `FrameMetrics` 性能指标，可选写入 `core` 状态。
- `common/`：通用组件（队伍分类、可视化等）。
- `annotators/`、`configs/`：绘制与配置。
- `data/`：模型和样例视频资源。

## 2. 负责什么

- 目标检测与多目标跟踪。
- 为每个目标提供像素框、ID、置信度、粗粒度队伍标签（如 `RED/BLUE/BALL`）。
- 给上游模块提供统一输入对象（`TrackedObject`）。

## 3. 引用了哪些模块（出向依赖）

- `tracking/main.py` 引用 `core`（`FramePacket`、`GameStateManager`、`AsyncPersistence`）用于逐帧中间结果输出与可选状态持久化。
- `tracking` 内部不引用 `projection` 与 `offside`。

## 4. 被哪些模块引用（入向依赖）

- `projection/modeling.py`：引用 `tracking.backend.TrackedObject` 作为投影输入类型。
- `projection/visualization.py`：引用 `tracking.backend` 生成 2D 球场投影视频。
- `offside/run_var_video.py`：引用 `tracking.backend` 做关键帧越位判定。
- 仓库根 `main.py`：调用 `tracking.main` 作为 CLI 子命令。

## 5. 边界约束

- `tracking` 只做“像素空间感知”，不做：
  - 像素到球场坐标的映射（这是 `projection` 职责）
  - 越位/出界规则判定（这是 `offside` 职责）
