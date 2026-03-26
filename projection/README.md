# projection 模块 README

`projection` 是项目的**坐标投影模块**：把 `tracking` 的像素空间结果映射到 2D 球场平面坐标。

## 1. 模块结构

- `homography.py`：`HomographyAdapter` 与默认单应矩阵配置（`DEFAULT_SRC_PTS` / `DEFAULT_DST_PTS`）。
- `modeling.py`：`ProjectedTracklet` 数据结构与 `project_tracked_objects` 投影函数。
- `visualization.py`：`run_projection_video_pipeline`，输出投影后的 2D 球场视频。
- `__init__.py`：统一导出。

## 2. 负责什么

- 管理像素坐标 -> 球场平面坐标（map 坐标）的映射逻辑。
- 将 `TrackedObject` 转换为 `ProjectedTracklet`（保留原像素信息 + 补充 `map_x/map_y`）。
- 负责 2D 球场可视化视频导出（仅投影层，不含越位判罚）。
- 提供可复用、与业务规则无关的几何建模能力。

## 3. 引用了哪些模块（出向依赖）

- `modeling.py` 引用 `tracking.backend.TrackedObject`（输入类型定义）。
- `modeling.py` 引用 `projection.homography.HomographyAdapter`。
- 不引用 `offside` 与 `core`。

## 4. 被哪些模块引用（入向依赖）

- `main.py`：`projection` 子命令调用 `run_projection_video_pipeline`。
- `offside/run_var_video.py`：调用投影接口把 tracklet 映射到球场平面后再做判罚。
- `offside/offside_core_integration.py`：使用 `ProjectedTracklet` 类型。

## 5. 与 offside 的分工（重点）

- `projection` 只解决**”坐标变换与几何建模”**：
  - 输入：像素空间 `TrackedObject`
  - 输出：球场空间 `ProjectedTracklet`
  - 不做任何”是否越位/是否出界”结论

- `offside` 只解决**”规则判定与业务流程”**：
  - 消费 `ProjectedTracklet`
  - 计算越位线、进攻方向、出界/门线结果
  - 进行可视化和关键帧判罚输出

一句话：`projection` 负责”把点投对”，`offside` 负责”按规则判对”。

## 6. 用法（2D 投影视频）

```bash
python main.py projection tracking/data/2e57b9_0.mp4 \
  -o projection/projection_2d.mp4 \
  --field field_map.png

# 实时模式（显示窗口）
python main.py modules --modules tracking projection \
  --source_video_path 0 \
  --is_camera \
  --no-show
```
