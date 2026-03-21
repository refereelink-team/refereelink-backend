# 新增模块接入指南（中文）

本文档用于指导你把新的分析模块快速接入当前 **packet-first** 架构。

## 1. 接入原则

- 保持 `core.packet.FramePacket` 作为主链路统一逐帧交换对象
- 保持 `core.state.FrameState` 作为由 packet 派生的稳定业务状态快照
- `GameStateManager` / `AsyncPersistence` 仅作为消费层
- 新模块优先遵循“读取一个 packet，补充一个 packet，再写回一个 packet”

## 2. 推荐目录

新增模块建议按功能新增子目录（例如 `/yyy/xxx.py`）。

## 3. 最小接入步骤

1. 读取 `FramePacket`
   读取已有字段（如 `tracked_objects`、`players`、`projection_tracklets`、`projection_frame`、`events`、`debug_info`）。

2. 在模块内补充 `FramePacket`
   将你的分析结果补充到 `events`、`debug_info`、`metrics` 或新增的可选字段中。

3. 将补充后的 packet 写回状态中台
   统一调用 `GameStateManager.update_packet(packet)`。

4. 如需稳定状态落盘或历史查询
   复用 `frame_state_from_packet(packet)` 的派生结果，不再直接手写 `FrameState` 主链路。

5. 如需异步落盘
   复用 `AsyncPersistence`，或新增独立后台线程。

## 4. 代码模板（示意）

```python
import time

from core import FramePacket, GameStateManager

state = GameStateManager()

def process_one_frame(frame_id: int, frame, tracked_objects):
    packet = FramePacket(
        frame_id=frame_id,
        timestamp=time.time(),
        source_id="NEW_MODULE",
        raw_frame=frame,
        tracked_objects=tracked_objects,
    )
    packet.debug_info["producer"] = "new-module"
    state.update_packet(packet)
```

## 5. 回调式扩展（推荐）

```python
def on_new_packet(packet):
    # 做二次分析、告警、统计等
    pass

state.on_packet(on_new_packet)
```

## 6. 数据结构扩展策略

- 实时字段优先加到 `FramePacket`
- 稳定业务字段优先由 `frame_state_from_packet` 映射到 `FrameState` / 事件结构
- 变更后同步更新：
  - `core/packet.py`
  - `core/state.py`
  - `core/__init__.py`
  - 文档（`core/README.md`）

## 7. 接入完成检查清单

- 新模块可独立运行并稳定补充 `FramePacket`
- 主流程无明显性能回退
- `FrameState` / 事件结构字段完整
- CSV 输出可用且字段语义正确
- 异常不会导致主循环崩溃
