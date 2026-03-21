# projection 模块 README

`projection` 是项目的**坐标投影模块**：把统一 packet 主链路中的像素空间结果映射到 2D 球场平面坐标。

## 1. 模块结构

- `homography.py`：`HomographyAdapter` 与默认单应矩阵配置（`DEFAULT_SRC_PTS` / `DEFAULT_DST_PTS`）。
- `modeling.py`：将 `core.ObjectTrack` 投影为 `core.ProjectedObject`。
- `visualization.py`：把投影结果渲染为 2D 球场视频或单帧画面。
- `__init__.py`：统一导出。

## 2. 负责什么

- 管理像素坐标 -> 球场平面坐标（map 坐标）的映射逻辑。
- 将 `ObjectTrack` 转换为 `ProjectedObject`（保留原像素信息 + 补充 `map_x/map_y`）。
- 负责 2D 球场可视化视频导出（仅投影层，不含越位业务判定）。
- 提供可复用、与业务规则无关的几何建模能力。

## 3. 引用了哪些模块（出向依赖）

- `modeling.py` 引用 `core.ObjectTrack` / `core.ProjectedObject` 作为统一输入输出契约。
- `visualization.py` 在 standalone CLI 场景下复用 `tracking.main.run_player_team_classification_packets(...)`，保证独立运行也走 packet 主链路。
- 不引用 `offside`。

## 4. 被哪些模块引用（入向依赖）

- `main.py`：`projection` 子命令调用 `run_projection_video_pipeline`。
- `main.py` 的 tracking+projection 联跑路径：在 `FramePacket` 上补充 `projection_tracklets` / `projection_frame`。
- `offside/run_var_video.py`：调用投影接口把跟踪结果映射到球场平面后再做判罚。
- `offside/offside_core_integration.py`：消费投影对象并封装为 packet 写回状态中台。

## 5. 与 offside 的分工（重点）

- `projection` 只解决**“坐标变换与几何建模”**：
  - 输入：像素空间 `ObjectTrack`
  - 输出：球场空间 `ProjectedObject`
  - 不做任何“是否越位/是否出界”结论

- `offside` 只解决**“规则判定与业务流程”**：
  - 消费 `ProjectedObject`
  - 计算越位线、进攻方向、出界/门线结果
  - 进行可视化和关键帧判罚输出

一句话：`projection` 负责“把点投对”，`offside` 负责“按规则判对”。

## 6. 用法（2D 投影视频）

```bash
python main.py projection tracking/data/2e57b9_0.mp4 \
  -o projection/projection_2d.mp4 \
  --field field_map.png
```
