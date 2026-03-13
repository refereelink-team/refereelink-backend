from __future__ import annotations

import csv
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Tuple,
)

logger = logging.getLogger(__name__)


class Team(Enum):
    """队伍枚举。

    使用统一的队伍标识，便于后续扩展（如多摄像头、多比赛等场景）。
    """

    HOME = "HOME"
    AWAY = "AWAY"
    REFEREE = "REFEREE"
    UNKNOWN = "UNKNOWN"


@dataclass
class PlayerState:
    """单个球员在某一帧上的完整状态信息。

    说明：
    - player_id 通常来自跟踪模块的 tracker_id；
    - 像素坐标 pixel_x / pixel_y 来自原始视频坐标系；
    - 场地图坐标 field_x / field_y 由 3D→2D 投影模块写入，单位为米；
    - speed、jersey_number 等字段可以由后续模块逐步补充。
    """

    player_id: int
    team: Team
    pixel_x: float
    pixel_y: float
    field_x: float = 0.0
    field_y: float = 0.0
    speed: float = 0.0
    confidence: float = 0.0
    jersey_number: Optional[int] = None


@dataclass
class BallState:
    """球在某一帧的状态。

    设计与 PlayerState 类似，但不需要 player_id / team 等字段。
    """

    pixel_x: float
    pixel_y: float
    field_x: float = 0.0
    field_y: float = 0.0
    speed: float = 0.0
    confidence: float = 0.0


@dataclass
class FrameState:
    """单帧的完整状态快照。

    含义：
    - frame_id：全局递增帧号（建议由摄像头处理主循环统一维护）；
    - timestamp：time.time() 的绝对时间，用于跨进程/模块同步；
    - video_ts：视频内时间戳（可选），单位为秒；
    - players：当前帧所有球员的状态，键为 player_id；
    - ball：当前帧的球状态（可选）；
    - source：摄像头来源标识，例如 "wide_angle"。
    """

    frame_id: int
    timestamp: float
    video_ts: Optional[float] = None
    players: Dict[int, PlayerState] = field(default_factory=dict)
    ball: Optional[BallState] = None
    source: str = "default"


@dataclass
class FoulEvent:
    """犯规事件描述。

    说明：
    - event_id：全局递增的事件编号（由调用方或上层逻辑保证）；
    - frame_id：犯规发生的大致帧号，便于回放与精确分析；
    - timestamp：事件生成时间（通常也是 time.time()）；
    - foul_type：犯规大类（身体接触、战术犯规、手球等）；
    - severity：严重程度，建议 "suspected" / "confirmed"；
    - location_field：球场平面坐标 (x, y)，单位米；
    - involved_players：涉及的球员 ID 列表；
    - camera_source：事件依据的摄像头来源。
    - offending_player_field：主要犯规球员在球场平面坐标中的位置 (x, y)，单位米；
     
    """

    event_id: int
    frame_id: int
    timestamp: float
    foul_type: str
    severity: str
    location_field: Tuple[float, float]
    involved_players: List[int]
    camera_source: str
    offending_player_field: Optional[Tuple[float, float]] = None


@dataclass
class OffsideQuery:
    """越位判定查询请求。

    说明：
    - query_id：查询编号（用于日志与 UI 标识）；
    - key_frame_id：裁判选择的关键帧号；
    - timestamp：请求创建时间；
    - result：越位与否（True/False），None 表示尚未给出结论；
    - details：额外说明（例如判罚理由、规则版本等）；
    - offending_player_field：被判定疑似/实际越位的球员在球场平面坐标中的位置 (x, y)，单位米，
      方便在前端高亮该球员或在分析时快速定位；
    - offside_line_field_x：越位线在球场坐标系中的“横坐标”（x 轴，单位米），
      一般取防守方最后一名非门将球员或倒数第二名防守队员的 field_x；
    - offside_line_pixel_x：越位线在原始视频中的“横坐标”（像素 x 轴），
      方便在图像坐标系中直接画出竖直的越位线。
    """

    query_id: int
    key_frame_id: int
    timestamp: float
    result: Optional[bool] = None
    details: str = ""
    offending_player_field: Optional[Tuple[float, float]] = None
    offside_line_field_x: Optional[float] = None
    offside_line_pixel_x: Optional[float] = None


class GameStateManager:
    """全局比赛状态管理器。

    该类在内存中保存最近一段时间的帧数据和事件，并提供线程安全的读写接口。

    设计要点：
    - 所有写入 / 读取接口一律通过内部的 RLock 保证线程安全；
    - 写入与回调分离：在持有锁期间只做状态更新，不执行耗时逻辑；
    - 上层所有模块（跟踪、投影、犯规识别、越位判定、前端 UI 等）
      都通过 GameStateManager 读写数据，避免直接共享裸字典。
    """

    def __init__(self, max_history: int = 900, max_events: int = 500) -> None:
        """
        :param max_history: 帧历史缓存长度，默认 900（约 30fps * 30s）
        :param max_events: 犯规事件与越位查询的最大缓存数量，默认 500
        """
        # 可重入锁，确保在复杂回调场景中也不会死锁
        self._lock = threading.RLock()

        # 帧数据
        self._max_history: int = max_history
        self._current_frame: Optional[FrameState] = None
        self._frame_history: Deque[FrameState] = deque(maxlen=max_history)
        self._frame_index: Dict[int, FrameState] = {}

        # 事件数据（有上限，防止长时间运行内存无限增长）
        self._foul_events: Deque[FoulEvent] = deque(maxlen=max_events)
        self._offside_queries: Deque[OffsideQuery] = deque(maxlen=max_events)

        # 回调列表
        self._frame_callbacks: List[Callable[[FrameState], None]] = []
        self._foul_callbacks: List[Callable[[FoulEvent], None]] = []

        # 统计信息
        self._total_frames: int = 0
        self._start_time: Optional[float] = None

    def update_frame(self, frame: FrameState) -> None:
        """写入最新一帧数据（线程安全）。

        操作步骤：
        1. 加锁，更新当前帧与历史队列；
        2. 维护 frame_id → FrameState 的索引，并清理太旧的索引；
        3. 更新统计信息；
        4. 复制回调列表，释放锁；
        5. 在锁外依次调用所有帧回调（防止长时间占锁）。
        """ 
        # 先在锁内更新状态
        with self._lock:
            if self._start_time is None:
                # 首帧到来时记录起始时间
                self._start_time = time.time()

            self._current_frame = frame
            self._frame_history.append(frame)
            self._frame_index[frame.frame_id] = frame
            self._total_frames += 1

            # 清理过旧的索引，保持索引规模与历史缓存大体一致
            if self._frame_history:
                # 历史队列中最早一帧的 frame_id，之前的都可以安全删除
                earliest_id = self._frame_history[0].frame_id
                obsolete_ids = [
                    fid for fid in self._frame_index.keys() if fid < earliest_id
                ]
                for fid in obsolete_ids:
                    self._frame_index.pop(fid, None)

            # 拷贝回调列表，避免锁外遍历时被修改
            callbacks = list(self._frame_callbacks)

        # 在锁外执行回调，避免阻塞写入线程
        for cb in callbacks:
            try:
                cb(frame)
            except Exception:
                logger.exception("帧回调执行时发生异常")

    def add_foul_event(self, event: FoulEvent) -> None:
        """新增犯规事件，并触发对应的回调（线程安全）。"""
        with self._lock:
            self._foul_events.append(event)
            callbacks = list(self._foul_callbacks)

        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                logger.exception("犯规事件回调执行时发生异常")

    def add_offside_query(self, query: OffsideQuery) -> None:
        """记录一次越位查询请求（线程安全）。"""
        with self._lock:
            self._offside_queries.append(query)

 


    def get_current_frame(self) -> Optional[FrameState]:
        """获取当前最新帧。

        注意：返回的是内部对象的引用，如需在其他线程长期持有并修改，
        建议在调用方自行复制，以免影响底层状态。
        """
        with self._lock:
            return self._current_frame

    def get_recent_frames(self, n: int) -> List[FrameState]:
        """获取最近 n 帧（按时间顺序从旧到新）。"""
        if n <= 0:
            return []

        with self._lock:
            # deque 支持负索引切片，转换为 list 后再截取
            frames = list(self._frame_history)[-n:]
        return frames

    def get_frame_by_id(self, frame_id: int) -> Optional[FrameState]:
        """按 frame_id 精确获取某一帧，用于越位判定等场景。"""
        with self._lock:
            return self._frame_index.get(frame_id)

    def get_frames_range(self, start_id: int, end_id: int) -> List[FrameState]:
        """获取指定帧号区间内的所有帧（包含边界）。"""
        if end_id < start_id:
            return []

        with self._lock:
            # 这里按 frame_id 排序，保证返回结果按时间顺序
            ids = sorted(self._frame_index.keys())
            selected = [
                self._frame_index[fid]
                for fid in ids
                if start_id <= fid <= end_id
            ]
        return selected

    def get_player_trajectory(
        self, player_id: int, n_frames: int
    ) -> List[Tuple[float, float]]:
        """获取某球员最近 n 帧的球场坐标轨迹。

        返回：
            List[(field_x, field_y)]，按时间顺序从旧到新。
        """
        if n_frames <= 0:
            return []

        with self._lock:
            # 从最近的帧向前遍历，提高命中效率
            trajectory_rev: List[Tuple[float, float]] = []
            count = 0
            for frame in reversed(self._frame_history):
                player = frame.players.get(player_id)
                if player is not None:
                    trajectory_rev.append((player.field_x, player.field_y))
                    count += 1
                    if count >= n_frames:
                        break

        # 当前是从新到旧收集，需要反转为从旧到新
        trajectory_rev.reverse()
        return trajectory_rev

    def get_foul_events(self, last_n: int) -> List[FoulEvent]:
        """获取最近的犯规事件列表。

        :param last_n: 返回的最大事件数；<=0 时返回空列表。
        """
        if last_n <= 0:
            return []

        with self._lock:
            events = list(self._foul_events)[-last_n:]
        return events

    def get_offside_queries(self) -> List[OffsideQuery]:
        """获取所有越位查询记录。"""
        with self._lock:
            return list(self._offside_queries)

    def get_stats(self) -> Dict[str, float]:
        """获取运行统计信息。

        返回字段包括：
        - total_frames：总共接收到的帧数；
        - buffered_frames：当前缓冲中的帧数（历史队列长度）；
        - foul_events：记录的犯规事件数量；
        - runtime_sec：运行时间（秒）；
        - estimated_fps：根据 total_frames / runtime_sec 估算的帧率。
        """
        with self._lock:
            total_frames = self._total_frames
            buffered_frames = len(self._frame_history)
            foul_events = len(self._foul_events)
            start_time = self._start_time

        now = time.time()
        if start_time is None:
            runtime_sec = 0.0
        else:
            runtime_sec = max(0.0, now - start_time)

        estimated_fps = (
            float(total_frames) / runtime_sec if runtime_sec > 0 else 0.0
        )

        return {
            "total_frames": float(total_frames),
            "buffered_frames": float(buffered_frames),
            "foul_events": float(foul_events),
            "runtime_sec": runtime_sec,
            "estimated_fps": estimated_fps,
        }

    # =====================
    # 观察者注册接口
    # =====================

    def on_frame(self, callback: Callable[[FrameState], None]) -> None:
        """注册帧更新回调。

        每次 update_frame 被调用时，都会依次触发这些回调。
        """
        with self._lock:
            self._frame_callbacks.append(callback)

    def on_foul(self, callback: Callable[[FoulEvent], None]) -> None:
        """注册犯规事件回调。"""
        with self._lock:
            self._foul_callbacks.append(callback)


class AsyncPersistence(threading.Thread):
    """异步持久化线程。

    功能：
    - 周期性地从 GameStateManager 中拉取新帧；
    - 将帧展开为“行”，追加写入到输出文件（csv 等）；
    - 专注于日志记录与赛后分析，不参与模块间数据通信。

    使用方式示例：
        state = GameStateManager(max_history=900)
        logger = AsyncPersistence(state, "tracking_log.csv", flush_interval=0.5)
        logger.start()
        ...
        logger.stop()
        logger.join()
    """

    def __init__(
        self,
        game_state: GameStateManager,
        output_path: str,
        flush_interval: float = 0.5,
    ) -> None:
        super().__init__(daemon=True)
        self._game_state = game_state
        self._output_path = output_path
        self._flush_interval = flush_interval

        # 记录上次写入的最大 frame_id，只处理更大的新帧
        self._last_frame_id: int = -1

        # 线程停止信号
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """请求停止后台线程。"""
        self._stop_event.set()

    def run(self) -> None:
        """主循环：周期性刷新数据到持久化介质。"""
        # 在循环外缓存一份 header，避免反复创建
        header = [
            "frame_id",
            "timestamp",
            "video_ts",
            "source",
            "player_id",
            "team",
            "pixel_x",
            "pixel_y",
            "field_x",
            "field_y",
            "speed",
            "confidence",
            "jersey_number",
        ]

        while not self._stop_event.is_set():
            try:
                # 一次性最多拉取最近 1000 帧，然后只保留 frame_id 更大的部分
                frames = self._game_state.get_recent_frames(1000)
                new_frames = [
                    f for f in frames if f.frame_id > self._last_frame_id
                ]

                if new_frames:
                    # 按 frame_id 排序，保证时序正确
                    new_frames.sort(key=lambda f: f.frame_id)

                    # 判断文件是否存在，用于决定是否写入表头
                    file_exists = os.path.exists(self._output_path)

                    # 以追加模式写入 csv
                    with open(
                        self._output_path,
                        mode="a",
                        newline="",
                        encoding="utf-8",
                    ) as f:
                        writer = csv.writer(f)

                        # 如果是新文件，先写入表头
                        if not file_exists:
                            writer.writerow(header)

                        # 展开每一帧的玩家信息为多行
                        for frame in new_frames:
                            for player in frame.players.values():
                                writer.writerow(
                                    [
                                        frame.frame_id,
                                        frame.timestamp,
                                        frame.video_ts
                                        if frame.video_ts is not None
                                        else "",
                                        frame.source,
                                        player.player_id,
                                        player.team.value,
                                        player.pixel_x,
                                        player.pixel_y,
                                        player.field_x,
                                        player.field_y,
                                        player.speed,
                                        player.confidence,
                                        player.jersey_number
                                        if player.jersey_number is not None
                                        else "",
                                    ]
                                )

                        # 更新最新写入的 frame_id
                        self._last_frame_id = new_frames[-1].frame_id

            except Exception:
                logger.exception("AsyncPersistence 持久化时发生异常")

            # 休眠一段时间，再进行下一轮刷新
            self._stop_event.wait(self._flush_interval)

