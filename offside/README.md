# offside 模块 README

`offside` 是项目的**越位业务模块**：消费跟踪与投影结果，执行越位/出界判定并输出可视化结果。

## 1. 模块结构

- `run_var_video.py`：越位单帧可视化入口（检测跟踪 -> 投影 -> 判定 -> 输出关键帧结果）。
- `judgement.py`：越位线、攻守方向、出界/门线等规则判定逻辑。
- `offside_core_integration.py`：把本模块结果写入 `core`（`FrameState` / `OffsideQuery`）。

## 2. 负责什么

- 基于球场平面坐标执行越位相关判定。
- 生成关键帧判罚结果（`frame.jpg` / `map.jpg` / `json`）。
- 可选把关键帧状态与越位查询写入 `core` 状态中台。

## 3. 引用了哪些模块（出向依赖）

- `tracking.backend`：获取检测/跟踪能力和 `TrackedObject` 数据源。
- `projection.homography` + `projection.modeling`：获取投影能力和 `ProjectedTracklet`。
- `core`：可选状态写入（`GameStateManager`、`AsyncPersistence`、`FrameState`、`OffsideQuery`）。
- `offside` 内部模块：`judgement`、`offside_core_integration` 等。

## 4. 被哪些模块引用（入向依赖）

- 仓库根 `main.py`：`offside` 作为 CLI 子命令与 pipeline 组成部分。
- `offside` 包导出（`__init__.py`）供外部直接调用 `process_frame_with_core` 与球门方位配置接口。

## 5. 与 projection 的分工（重点）

- `offside` 不实现“像素->球场坐标”投影算法，统一使用 `projection` 提供的结果。
- `projection` 不实现越位规则，不输出判罚结论。

职责切分：

- `projection`：几何层（坐标映射、投影建模）
- `offside`：规则层（越位线、攻守方向、判罚与可视化）

这样可以保证：

- 标定/投影调整只改 `projection`；
- 规则/判罚策略调整只改 `offside`；
- 两者解耦，便于独立迭代与测试。

## 6. 用法（单帧越位可视化）

```bash
python main.py offside offside/test.mp4 \
  --frame_index 120 \
  --output_dir offside/offside_output \
  --field field_map.png
```
