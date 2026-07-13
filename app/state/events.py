from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Callable

Subscriber = Callable[..., None]


class EventBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[Subscriber]] = defaultdict(list)

    def subscribe(self, event_type: str, callback: Subscriber) -> None:
        with self._lock:
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Subscriber) -> None:
        with self._lock:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)

    def emit(self, event_type: str, **kwargs: Any) -> None:
        with self._lock:
            callbacks = list(self._subscribers.get(event_type, []))
        for callback in callbacks:
            try:
                callback(**kwargs)
            except Exception:
                pass

    def clear(self) -> None:
        with self._lock:
            self._subscribers.clear()
