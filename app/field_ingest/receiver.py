from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import quote

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
        on_frame: Callable[[int], None] | None = None,
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
        self.on_frame = on_frame
        self._popen = popen
        self._process: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._lock = threading.RLock()
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
        )
        # First output preserves the original MPEG-TS/PTS.  The second output
        # decodes frames through showinfo so a successful session proves that
        # the backend can receive and decode, not merely accept UDP packets.
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
            "-f",
            "null",
            os.devnull,
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
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
            except OSError as exc:
                self.metrics.state = "failed"
                self.metrics.last_error = str(exc)
                raise
            self.metrics.state = "listening"
            self._reader_thread = threading.Thread(
                target=self._read_output,
                name=f"srt-receiver-{self.stream_epoch}",
                daemon=True,
            )
            self._reader_thread.start()

    def _read_output(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        try:
            for raw_line in process.stderr:
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
            with self._lock:
                self.metrics.exit_code = process.returncode
                self.metrics.state = "released"

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "state": self.metrics.state,
                "decoded_frame_count": self.metrics.decoded_frame_count,
                "first_pts90k": self.metrics.first_pts90k,
                "last_pts90k": self.metrics.last_pts90k,
                "last_error": self.metrics.last_error,
                "exit_code": self.metrics.exit_code,
                "log_tail": list(self.metrics.log_tail),
            }
