from __future__ import annotations

import threading

from app.state.events import EventBus
from app.state.models import (
    GameEvent,
    FrameState,
    MetricsSnapshot,
    PipelineConfig,
    SourceStatus,
)
from app.state.store import StateStore


def test_state_store_latest_frame_state():
    s = StateStore()
    assert s.latest_frame_state is None
    fs = FrameState(frame_id=1)
    s.latest_frame_state = fs
    assert s.latest_frame_state == fs


def test_state_store_config_update():
    s = StateStore()
    cfg = s.update_config({"device": "cuda", "mode": "realtime"})
    assert cfg.device == "cuda"
    assert cfg.mode == "realtime"
    assert s.config.device == "cuda"


def test_state_store_events_bounded():
    s = StateStore()
    for i in range(600):
        s.add_event(GameEvent(
            event_type="foul", confidence=0.5, severity="likely",
            timestamp=float(i), frame_id=i,
        ))
    assert len(s.events) == 500


def test_state_store_metrics():
    s = StateStore()
    m = MetricsSnapshot(processing_fps=22.5)
    s.metrics = m
    assert s.metrics.processing_fps == 22.5


def test_state_store_source_status():
    s = StateStore()
    assert s.source_status == SourceStatus.DISCONNECTED
    s.source_status = SourceStatus.CONNECTED
    assert s.source_status == SourceStatus.CONNECTED


def test_state_store_log_lines_bounded():
    s = StateStore()
    for i in range(300):
        s.append_log(f"line {i}")
    assert len(s.log_lines) == 200


def test_state_store_thread_safety():
    s = StateStore()

    def writer():
        for i in range(1000):
            s.latest_frame_state = FrameState(frame_id=i)
            s.append_log(f"log {i}")
            s.add_event(GameEvent(
                event_type="foul", confidence=0.5, severity="likely",
                timestamp=0.0, frame_id=i,
            ))

    threads = [threading.Thread(target=writer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert s.latest_frame_state is not None
    assert len(s.log_lines) == 200
    assert len(s.events) == 500


def test_event_bus_subscribe_and_emit():
    bus = EventBus()
    received = []
    bus.subscribe("test", lambda **kw: received.append(kw))
    bus.emit("test", x=1)
    bus.emit("test", x=2)
    assert received == [{"x": 1}, {"x": 2}]


def test_event_bus_multiple_subscribers():
    bus = EventBus()
    a, b = [], []
    bus.subscribe("e", lambda **kw: a.append(kw))
    bus.subscribe("e", lambda **kw: b.append(kw))
    bus.emit("e", n=1)
    assert a == b == [{"n": 1}]


def test_event_bus_unsubscribe():
    bus = EventBus()
    received = []

    def cb(**kw):
        received.append(kw)

    bus.subscribe("e", cb)
    bus.emit("e", x=1)
    bus.unsubscribe("e", cb)
    bus.emit("e", x=2)
    assert received == [{"x": 1}]


def test_event_bus_subscriber_exception_isolated():
    bus = EventBus()
    a, b = [], []

    def bad_cb(**kw):
        raise RuntimeError("boom")

    bus.subscribe("e", bad_cb)
    bus.subscribe("e", lambda **kw: b.append(kw))
    bus.emit("e", n=1)
    assert b == [{"n": 1}]


def test_state_store_snapshot():
    s = StateStore()
    s.latest_frame_state = FrameState(frame_id=5)
    s.append_log("hello")
    snap = s.snapshot()
    assert snap["frame_state"]["frame_id"] == 5
    assert "hello" in snap["log_lines"]
