from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Event, Lock, Thread
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import supervision as sv


QML_PATH = Path(__file__).with_name('radar_dashboard.qml')
LOG_BUFFER_SIZE = 14
TRACKING_PANEL_RATIO = 0.68
LOG_PANEL_RATIO = 0.28


try:
    from PySide6.QtCore import QObject, Property, QTimer, Qt, QUrl, Signal, Slot
    from PySide6.QtGui import QImage
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuick import QQuickImageProvider
    from PySide6.QtWidgets import QApplication

    HAVE_PYSIDE6 = True
    PYSIDE6_IMPORT_ERROR = ''
except ImportError as error:  # pragma: no cover - exercised only without Qt installed
    HAVE_PYSIDE6 = False
    PYSIDE6_IMPORT_ERROR = str(error)


@dataclass
class RadarDashboardFrame:
    frame_index: int
    tracked_frame: np.ndarray
    radar_frame: np.ndarray
    log_text: str


def _decorate_panel(panel: np.ndarray, title: str) -> np.ndarray:
    decorated = panel.copy()
    card_color = (32, 36, 42)
    border_color = (64, 72, 84)

    cv2.rectangle(decorated, (0, 0), (decorated.shape[1], decorated.shape[0]), border_color, 2)
    cv2.rectangle(decorated, (0, 0), (decorated.shape[1], 52), card_color, -1)
    cv2.putText(
        decorated,
        title,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return decorated


def frame_to_qimage(frame: np.ndarray) -> 'QImage':
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape
    return QImage(
        image.data,
        width,
        height,
        channels * width,
        QImage.Format_RGB888,
    ).copy()


def resize_with_letterbox(
    image: np.ndarray,
    size_wh: Tuple[int, int],
    background_color: Tuple[int, int, int] = (14, 18, 24),
) -> np.ndarray:
    target_width, target_height = size_wh
    canvas = np.full((target_height, target_width, 3), background_color, dtype=np.uint8)
    if target_width <= 0 or target_height <= 0:
        return canvas

    image_height, image_width = image.shape[:2]
    scale = min(target_width / image_width, target_height / image_height)
    resized_width = max(1, int(round(image_width * scale)))
    resized_height = max(1, int(round(image_height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height))

    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized
    return canvas


def render_log_panel(
    log_lines: Sequence[str],
    size_wh: Tuple[int, int],
) -> np.ndarray:
    width, height = size_wh
    panel = np.full((height, width, 3), (16, 20, 26), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width, height), (58, 66, 78), 2)
    cv2.rectangle(panel, (0, 0), (width, 54), (28, 32, 38), -1)
    cv2.putText(
        panel,
        'Runtime Log',
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        'live pipeline events',
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (160, 168, 180),
        1,
        cv2.LINE_AA,
    )

    max_lines = max(1, (height - 86) // 26)
    visible_lines = list(log_lines)[-max_lines:]
    for index, line in enumerate(visible_lines):
        y = 88 + index * 26
        cv2.putText(
            panel,
            line[:128],
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 224, 230),
            1,
            cv2.LINE_AA,
        )
    return panel


def compose_dashboard_frame(
    tracked_frame: np.ndarray,
    radar_frame: np.ndarray,
    log_lines: Sequence[str],
) -> np.ndarray:
    frame_height, frame_width = tracked_frame.shape[:2]
    log_height = min(frame_height - 1, max(180, int(frame_height * LOG_PANEL_RATIO)))
    top_height = frame_height - log_height
    left_width = int(round(frame_width * TRACKING_PANEL_RATIO))
    right_width = frame_width - left_width

    tracking_panel = resize_with_letterbox(tracked_frame, (left_width, top_height))
    radar_panel = resize_with_letterbox(radar_frame, (right_width, top_height))
    log_panel = render_log_panel(log_lines, (frame_width, log_height))

    tracking_panel = _decorate_panel(tracking_panel, 'Tracking View')
    radar_panel = _decorate_panel(radar_panel, '2D Projection')

    return np.vstack([np.hstack([tracking_panel, radar_panel]), log_panel])


class RadarDashboardWorker(Thread):
    def __init__(
        self,
        source_video_path: str,
        target_video_path: str,
        device: str,
        event_queue: Queue,
    ) -> None:
        super().__init__(daemon=True)
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.device = device
        self.event_queue = event_queue
        self.stop_event = Event()
        self.log_lines: List[str] = []

    def stop(self) -> None:
        self.stop_event.set()

    def append_log(self, message: str) -> None:
        self.log_lines.append(message)
        self.log_lines = self.log_lines[-LOG_BUFFER_SIZE:]
        self.event_queue.put(('log', '\n'.join(self.log_lines)))

    def run(self) -> None:
        try:
            from app.modes.radar import iter_radar_analysis

            video_info = sv.VideoInfo.from_video_path(self.source_video_path)
            with sv.VideoSink(self.target_video_path, video_info) as sink:
                for update in iter_radar_analysis(
                    source_video_path=self.source_video_path,
                    device=self.device,
                    log_callback=self.append_log,
                ):
                    if self.stop_event.is_set():
                        break

                    dashboard_frame = compose_dashboard_frame(
                        tracked_frame=update.tracked_frame,
                        radar_frame=update.radar_frame,
                        log_lines=self.log_lines,
                    )
                    sink.write_frame(dashboard_frame)
                    self.event_queue.put(
                        (
                            'frame',
                            RadarDashboardFrame(
                                frame_index=update.frame_index,
                                tracked_frame=update.tracked_frame,
                                radar_frame=update.radar_frame,
                                log_text='\n'.join(self.log_lines),
                            ),
                        )
                    )
        except Exception as error:  # pragma: no cover - requires runtime dependencies
            self.append_log(f'error: {error}')
            self.event_queue.put(('error', str(error)))
        finally:
            self.event_queue.put(('finished', None))


if HAVE_PYSIDE6:
    class RadarFrameProvider(QQuickImageProvider):
        def __init__(self) -> None:
            super().__init__(QQuickImageProvider.Image)
            self.lock = Lock()
            self.placeholder = QImage(640, 360, QImage.Format_RGB888)
            self.placeholder.fill(Qt.black)
            self.images = {
                'tracking': self.placeholder,
                'radar': self.placeholder,
            }

        def update_frame(self, name: str, frame: np.ndarray) -> None:
            image = frame_to_qimage(frame)
            with self.lock:
                self.images[name] = image

        def requestImage(self, image_id, size, requested_size):  # type: ignore[override]
            frame_name = image_id.split('?', 1)[0]
            with self.lock:
                image = self.images.get(frame_name, self.placeholder)

            if size is not None:
                size.setWidth(image.width())
                size.setHeight(image.height())

            if requested_size.width() > 0 and requested_size.height() > 0:
                return image.scaled(
                    requested_size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )

            return image


    class RadarDashboardController(QObject):
        trackingRevisionChanged = Signal()
        radarRevisionChanged = Signal()
        logTextChanged = Signal()
        statusTextChanged = Signal()
        finishedChanged = Signal()

        def __init__(
            self,
            source_video_path: str,
            target_video_path: str,
            device: str,
            frame_provider: 'RadarFrameProvider',
        ) -> None:
            super().__init__()
            self.frame_provider = frame_provider
            self.event_queue: Queue = Queue()
            self.worker = RadarDashboardWorker(
                source_video_path=source_video_path,
                target_video_path=target_video_path,
                device=device,
                event_queue=self.event_queue,
            )
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.drain_events)

            self._tracking_revision = 0
            self._radar_revision = 0
            self._log_text = ''
            self._status_text = 'Starting radar dashboard...'
            self._finished = False

        @Property(int, notify=trackingRevisionChanged)
        def trackingRevision(self) -> int:
            return self._tracking_revision

        @Property(int, notify=radarRevisionChanged)
        def radarRevision(self) -> int:
            return self._radar_revision

        @Property(str, notify=logTextChanged)
        def logText(self) -> str:
            return self._log_text

        @Property(str, notify=statusTextChanged)
        def statusText(self) -> str:
            return self._status_text

        @Property(bool, notify=finishedChanged)
        def finished(self) -> bool:
            return self._finished

        @Slot()
        def start(self) -> None:
            if not self.worker.is_alive():
                self.timer.start(30)
                self.worker.start()

        @Slot()
        def shutdown(self) -> None:
            self.timer.stop()
            self.worker.stop()
            self.worker.join(timeout=1.0)

        def set_log_text(self, value: str) -> None:
            if value != self._log_text:
                self._log_text = value
                self.logTextChanged.emit()

        def set_status_text(self, value: str) -> None:
            if value != self._status_text:
                self._status_text = value
                self.statusTextChanged.emit()

        def set_finished(self, value: bool) -> None:
            if value != self._finished:
                self._finished = value
                self.finishedChanged.emit()

        @Slot()
        def drain_events(self) -> None:
            while True:
                try:
                    event_type, payload = self.event_queue.get_nowait()
                except Empty:
                    break

                if event_type == 'log':
                    self.set_log_text(payload)
                elif event_type == 'frame':
                    self.frame_provider.update_frame('tracking', payload.tracked_frame)
                    self.frame_provider.update_frame('radar', payload.radar_frame)
                    self._tracking_revision += 1
                    self._radar_revision += 1
                    self.trackingRevisionChanged.emit()
                    self.radarRevisionChanged.emit()
                    self.set_log_text(payload.log_text)
                    self.set_status_text(f'Frame {payload.frame_index}')
                elif event_type == 'error':
                    self.set_status_text(f'Error: {payload}')
                elif event_type == 'finished':
                    self.timer.stop()
                    self.set_finished(True)
                    self.set_status_text('Processing finished')


def run_radar_dashboard(
    source_video_path: str,
    target_video_path: str,
    device: str,
) -> None:
    if not HAVE_PYSIDE6:
        raise RuntimeError(
            'PySide6 is required for RADAR_DASHBOARD mode. '
            f'Current interpreter: {sys.executable}. '
            f'Import error: {PYSIDE6_IMPORT_ERROR}'
        )

    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    frame_provider = RadarFrameProvider()
    controller = RadarDashboardController(
        source_video_path=source_video_path,
        target_video_path=target_video_path,
        device=device,
        frame_provider=frame_provider,
    )

    engine.addImageProvider('radarFrames', frame_provider)
    engine.rootContext().setContextProperty('dashboard', controller)
    engine.load(QUrl.fromLocalFile(str(QML_PATH.resolve())))
    if not engine.rootObjects():
        raise RuntimeError(f'Failed to load QML UI: {QML_PATH}')

    app.aboutToQuit.connect(controller.shutdown)
    controller.start()
    app.exec()
