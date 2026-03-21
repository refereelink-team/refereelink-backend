# core 模块 README

`core` 是项目的**状态中台**：定义统一数据结构，并提供线程安全状态管理与异步落盘能力。

## 1. 模块结构

- `state.py`：核心数据结构与管理器实现（`Team`、`PlayerState`、`FrameState`、`OffsideQuery`、`GameStateManager`、`AsyncPersistence`）。
- `__init__.py`：统一导出对外 API。

## 2. 负责什么

- 定义跨模块共享的标准状态契约（球员、球、帧、事件）。
- 提供统一状态读写入口（`GameStateManager`）。
- 提供异步持久化（`AsyncPersistence`，CSV 输出）。

## 3. 引用了哪些模块（出向依赖）

- 标准库（`threading`、`dataclasses`、`csv` 等）。
- **不依赖** `tracking` / `projection` / `offside`（保持中立底座）。

## 4. 被哪些模块引用（入向依赖）

- `tracking/main.py`：写入 `FrameState`，可选启用 `AsyncPersistence`。
- `offside/offside_core_integration.py`：把越位流程产物映射到 `FrameState` / `OffsideQuery`。
- `offside/run_var_video.py`：在关键帧流程中可选创建 `GameStateManager` 与 `AsyncPersistence`。

## 5. 边界约束

- `core` 只做“状态定义 + 状态管理 + 持久化”，不做检测、投影、越位业务判定。
- 上层模块通过 `core` 交换数据，避免相互直接共享可变结构。
