from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import numpy as np

from app.field_ingest.frames import DecodedVideoFrame, utc_timestamp

logger = logging.getLogger(__name__)

_SHOWINFO_RE = re.compile(
    r"showinfo.*?n:\s*(?P<n>\d+).*?pts:\s*(?P<pts>-?\d+).*?pts_time:\s*(?P<pts_time>-?[0-9.]+)"
)


def probe_srt_support(ffmpeg_bin: str) -> tuple[bool, str]:
    """Return whether the configured FFmpeg exposes the SRT protocol."""
    try:
        completed = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-protocols"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    output = f"{completed.stdout}\n{completed.stderr}".lower()
    if completed.returncode != 0:
        return False, output.strip()[-500:]
    supported = bool(re.search(r"(?m)^\s*srt\s*$", output))
    return supported, "" if supported else "ffmpeg does not advertise the srt protocol"


@dataclass
class ReceiverMetrics:
    state: str = "starting"
    decoded_frame_count: int = 0
    raw_frame_count: int = 0
    first_pts90k: int | None = None
    last_pts90k: int | None = None
    last_error: str | None = None
    exit_code: int | None = None
    log_tail: list[str] = field(default_factory=list)

    def append_log(self, line: str) -> None:
        self.log_tail.append(line[-500:])
        del self.log_tail[:-40]


class SRTReceiver:
    """Own one FFmpeg listener process and keep only bounded diagnostics."""

    def __init__(
        self,
        *,
        ffmpeg_bin: str,
        host: str,
        port: int,
        session_id: str,
        stream_epoch: int,
        stream_token: str,
        output_path: Path,
        latency_ms: int,
        width: int = 1280,
        height: int = 720,
        on_frame: Callable[[int], None] | None = None,
        on_decoded_frame: Callable[[DecodedVideoFrame], None] | None = None,
        on_end: Callable[[], None] | None = None,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self.ffmpeg_bin = ffmpeg_bin
        self.host = host
        self.port = port
        self.session_id = session_id
        self.stream_epoch = stream_epoch
        self.stream_token = stream_token
        self.output_path = output_path
        self.latency_ms = latency_ms
        self.width = width
        self.height = height
        self.on_frame = on_frame
        self.on_decoded_frame = on_decoded_frame
        self.on_end = on_end
        self._popen = popen
        self._process: subprocess.Popen | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stdout_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._pts_condition = threading.Condition(self._lock)
        self._pts_by_index: dict[int, int] = {}
        self.metrics = ReceiverMetrics()

    @property
    def command(self) -> list[str]:
        stream_id = f"refereelink/{self.session_id}/{self.stream_epoch}/{self.stream_token}"
        url = (
            f"srt://{self.host}:{self.port}"
            f"?mode=listener&transtype=live&latency={self.latency_ms}"
            # FFmpeg's SRT URL parser does not expose payloadsize on all
            # libsrt builds.  The iOS sender uses MPEG-TS and the negotiated
            # SRT payload size, so leaving this optional socket setting out
            # keeps the receiver compatible with the installed FFmpeg while
            # retaining the supported low-latency options below.
            f"&tlpktdrop=1"
            f"&streamid={quote(stream_id, safe='')}"
            f"&passphrase={quote(self.stream_token, safe='')}"
            f"&pbkeylen=16"
        )
        # First output preserves the original MPEG-TS/PTS.  The second output
        # decodes frames into a bounded rawvideo pipe for the inference bridge.
        return [
            self.ffmpeg_bin,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "info",
            "-i",
            url,
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-f",
            "mpegts",
            str(self.output_path),
            "-map",
            "0:v:0",
            "-vf",
            "showinfo",
            "-s",
            f"{self.width}x{self.height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._process is not None:
                return
            try:
                self._process = self._popen(
                    self.command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                    start_new_session=True,
                )
            except OSError as exc:
                self.metrics.state = "failed"
                self.metrics.last_error = str(exc)
                raise
            self.metrics.state = "listening"
            self._stderr_thread = threading.Thread(
                target=self._read_output,
                name=f"srt-receiver-stderr-{self.stream_epoch}",
                daemon=True,
            )
            self._stdout_thread = threading.Thread(
                target=self._read_frames,
                name=f"srt-receiver-frames-{self.stream_epoch}",
                daemon=True,
            )
            self._stderr_thread.start()
            self._stdout_thread.start()

    def _read_output(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        try:
            for raw_line in process.stderr:
                if isinstance(raw_line, bytes):
                    line = raw_line.decode("utf-8", errors="replace").strip()
                else:
                    line = raw_line.strip()
                if not line:
                    continue
                match = _SHOWINFO_RE.search(line)
                with self._lock:
                    self.metrics.append_log(line)
                    if match:
                        pts = int(match.group("pts"))
                        self.metrics.decoded_frame_count += 1
                        if self.metrics.first_pts90k is None:
                            self.metrics.first_pts90k = pts
                        self.metrics.last_pts90k = pts
                        self._pts_by_index[int(match.group("n"))] = pts
                        self._pts_condition.notify_all()
                        if self.on_frame is not None:
                            self.on_frame(pts)
        finally:
            return_code = process.poll()
            with self._lock:
                self.metrics.exit_code = return_code
                if self.metrics.state not in {"stopping", "released"}:
                    self.metrics.state = "ended" if return_code == 0 else "failed"
                    if return_code not in (None, 0):
                        self.metrics.last_error = f"ffmpeg exited with code {return_code}"
                self._pts_condition.notify_all()
            if self._stdout_thread is not None and self._stdout_thread is not threading.current_thread():
                self._stdout_thread.join(timeout=1)
            if self.on_end is not None:
                self.on_end()

    def _read_frames(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        frame_size = self.width * self.height * 3
        decode_index = 0
        try:
            while True:
                payload = _read_exact(process.stdout, frame_size)
                if not payload:
                    break
                image = np.frombuffer(payload, dtype=np.uint8).reshape(
                    (self.height, self.width, 3)
                ).copy()
                pts = self._wait_for_pts(decode_index)
                with self._lock:
                    self.metrics.raw_frame_count += 1
                if self.on_decoded_frame is not None:
                    self.on_decoded_frame(
                        DecodedVideoFrame(
                            image=image,
                            transport_pts90k=pts,
                            decode_index=decode_index,
                            received_at=utc_timestamp(),
                        )
                    )
                decode_index += 1
        except (EOFError, OSError, ValueError) as exc:
            with self._lock:
                self.metrics.last_error = f"rawvideo reader failed: {exc}"
                self.metrics.state = "failed"
        finally:
            with self._lock:
                self._pts_condition.notify_all()

    def _wait_for_pts(self, decode_index: int) -> int | None:
        end = time.monotonic() + 0.25
        with self._pts_condition:
            while decode_index not in self._pts_by_index and time.monotonic() < end:
                self._pts_condition.wait(timeout=max(0.0, end - time.monotonic()))
            return self._pts_by_index.pop(decode_index, None)

    def stop(self) -> None:
        with self._lock:
            process = self._process
            if process is None:
                self.metrics.state = "released"
                return
            self.metrics.state = "stopping"
        try:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        process.kill()
                    process.wait(timeout=3)
        finally:
            for stream in (process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
            for thread in (self._stderr_thread, self._stdout_thread):
                if thread is not None and thread is not threading.current_thread():
                    thread.join(timeout=1)
            with self._lock:
                self.metrics.exit_code = process.returncode
                self.metrics.state = "released"

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "state": self.metrics.state,
                "decoded_frame_count": self.metrics.decoded_frame_count,
                "raw_frame_count": self.metrics.raw_frame_count,
                "first_pts90k": self.metrics.first_pts90k,
                "last_pts90k": self.metrics.last_pts90k,
                "last_error": self.metrics.last_error,
                "exit_code": self.metrics.exit_code,
                "log_tail": list(self.metrics.log_tail),
            }


def _read_exact(stream: object, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)  # type: ignore[attr-defined]
        if not chunk:
            if not chunks:
                return b""
            raise EOFError(f"rawvideo pipe ended with {remaining} bytes pending")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
