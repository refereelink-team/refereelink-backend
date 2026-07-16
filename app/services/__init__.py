"""Service helpers.

Imports are lazy so the state store can use the frame encoder without creating
the publisher -> store import cycle during module initialization.
"""

__all__ = ["WebSocketPublisher"]


def __getattr__(name: str):
    if name == "WebSocketPublisher":
        from app.services.publisher import WebSocketPublisher

        return WebSocketPublisher
    raise AttributeError(name)
