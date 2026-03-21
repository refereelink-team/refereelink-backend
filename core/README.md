# core 模块 README

`core` 是仓库统一的数据契约与状态中台，现已固定为三层结构：

1. **中间结果包层**：`packet.py`，定义 `FramePacket`、`ObjectTrack`、`ProjectedObject`、`FrameMetrics`。
2. **状态快照层**：`state.py`，定义稳定业务语义对象 `Team`、`PlayerState`、`BallState`、`FrameState`、`FoulEvent`、`OffsideQuery`，并提供 `frame_state_from_packet(...)`。
3. **状态管理与持久化层**：`store.py` 提供 `GameStateManager`，`persistence.py` 提供 `AsyncPersistence`。

## 1. 模块结构

- `packet.py`：实时主链路的统一逐帧交换对象。
- `state.py`：稳定业务状态快照与事件对象。
- `store.py`：线程安全状态仓库，统一消费 `update_packet(packet)`。
- `persistence.py`：异步持久化线程。
- `__init__.py`：统一导出对外 API。

## 2. 设计目标

- 主处理链路只传递 `FramePacket`；
- 持久化、查询、历史回放消费 `FrameState`；
- 通过 `frame_state_from_packet` 把实时逐帧中间结果抽取为稳定状态快照；
- `GameStateManager` 同时缓存 packet 与 frame，方便实时消费和赛后分析。

## 3. 推荐数据流

```text
tracking -> FramePacket -> projection/offside/frontend -> enriched FramePacket
                                     |
                                     v
                         frame_state_from_packet
                                     |
                                     v
                    GameStateManager / AsyncPersistence
```

## 4. 核心接口

### 4.1 `FramePacket`

逐帧主链路统一对象，常用字段包括：

- `tracked_objects`
- `players`
- `ball`
- `projection_tracklets`
- `projection_frame`
- `events`
- `metrics`
- `debug_info`

### 4.2 `frame_state_from_packet(packet)`

把 packet 派生为稳定 `FrameState`，供：

- `AsyncPersistence` 落盘
- `GameStateManager.get_recent_frames(...)`
- 历史查询 / 回放 / 统计

### 4.3 `GameStateManager`

推荐用法：

```python
state = GameStateManager()
state.update_packet(packet)
state.on_packet(callback)
recent_packets = state.get_recent_packets(32)
recent_frames = state.get_recent_frames(32)
```

## 5. 边界约束

- `packet.py` 是实时主链路契约，应保持轻量、可扩展。
- `state.py` 是稳定业务语义层，应尽量避免频繁破坏式变更。
- `store.py` / `persistence.py` 是消费层，不应反向主导上游模块的逐帧主循环。
- 新模块不再以手写 `FrameState` 作为实时主链路输入，应统一先构造 `FramePacket`。
