# core 模块 README

`core` 现在被拆分为三层：

1. **中间结果包层**：`packet.py`，定义 `FramePacket`、`ObjectTrack`、`ProjectedObject`、`FrameMetrics`。
2. **状态快照层**：`state.py`，定义稳定业务语义对象 `Team`、`PlayerState`、`BallState`、`FrameState`、`FoulEvent`、`OffsideQuery`，并提供 `frame_state_from_packet` 适配函数。
3. **状态管理与持久化层**：`store.py` 提供 `GameStateManager`，`persistence.py` 提供 `AsyncPersistence`。
4. **并行处理层**：`pipeline.py` 提供 `FrameBuffer`、`ParallelPipeline`、`create_parallel_pipeline`，支持实时并行处理。

## 1. 模块结构

- `packet.py`：实时主链路的统一逐帧交换对象。
- `state.py`：稳定业务状态快照与事件对象。
- `store.py`：线程安全状态仓库，同时支持 `FramePacket` 与 `FrameState`。
- `persistence.py`：异步持久化线程。
- `pipeline.py`：并行处理管道，支持实时显示 tracking + projection 双视图。
- `__init__.py`：统一导出对外 API。

## 2. 设计目标

- 主处理链路优先传递 `FramePacket`；
- 持久化、查询、历史回放优先消费 `FrameState`；
- 通过 `frame_state_from_packet` 把实时逐帧中间结果抽取为稳定状态快照；
- `GameStateManager` 同时缓存 packet 与 frame，方便前端实时消费和赛后分析。
- `ParallelPipeline` 支持 tracking 和 projection 并行处理，实现实时双视图显示。

## 3. 推荐数据流

```text
tracking -> FramePacket -> projection/offside/frontend -> FramePacket
                                     |
                                     v
                         frame_state_from_packet
                                     |
                                     v
                    GameStateManager / AsyncPersistence
```

## 4. 实时并行处理（新增）

```python
from core import create_parallel_pipeline

# 创建并行管道（支持摄像头和视频）
pipeline = create_parallel_pipeline(
    source="0",  # 摄像头索引或视频路径
    device="auto",
    is_camera=True,
)

# 获取最新处理后的帧
packet = pipeline.get_latest_packet()
# packet.annotated_frame  # 跟踪视图
# packet.projection_frame # 投影视图
```

## 5. 边界约束

- `packet.py` 是实时主链路契约，应保持轻量、可扩展。
- `state.py` 是稳定业务语义层，应尽量避免频繁破坏式变更。
- `store.py` / `persistence.py` 是消费层，不应该反向主导上游模块的逐帧主循环。
