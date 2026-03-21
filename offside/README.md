# offside 模块 README

`offside` 是项目的**越位业务模块**：消费跟踪与投影结果，执行越位/出界判定并输出可视化结果。

## 1. 模块结构

- `run_var_video.py`：越位单帧可视化入口（检测跟踪 -> 投影 -> 判定 -> 输出关键帧结果）。
- `judgement.py`：越位线、攻守方向、出界/门线等规则判定逻辑。
- `offside_core_integration.py`：将投影结果封装为 `FramePacket` 并写入 `core.GameStateManager`。

## 2. 负责什么

- 基于球场平面坐标执行越位相关判定。
- 生成关键帧判罚结果（`frame.jpg` / `map.jpg` / `json`）。
- 可选把关键帧 packet 与 `OffsideQuery` 写入 `core` 状态中台。

## 3. 引用了哪些模块（出向依赖）

- `tracking.backend`：在单帧分析模式下获取底层检测/跟踪能力。
- `projection.homography` + `projection.modeling`：获取投影能力与 `ProjectedObject`。
- `core`：通过 `GameStateManager.update_packet(...)` 写入统一状态，并附加 `OffsideQuery`。
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

## 6. packet-first 状态接入

`offside_core_integration.py` 当前流程为：

1. 接收投影后的对象列表；
2. 构造 `FramePacket(tracked_objects + projection_tracklets)`；
3. 调用 `GameStateManager.update_packet(packet)`；
4. 按需追加 `OffsideQuery`。

因此越位模块已不再以 `FrameState -> update_frame(...)` 作为主链路。

## 7. 用法（单帧越位可视化）

```bash
python main.py offside offside/test.mp4 \
  --frame_index 120 \
  --output_dir offside/offside_output \
  --field field_map.png
```
