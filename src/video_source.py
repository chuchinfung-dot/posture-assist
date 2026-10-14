"""Camera capture via PyAV (no OpenCV). Returns RGB frames."""

from __future__ import annotations

import errno
import sys
import time
from typing import Optional, Tuple

import av
import numpy as np


class VideoSource:
    """
    Open the laptop camera using FFmpeg backends:
      macOS: avfoundation | Linux: v4l2 | Windows: dshow

    Frames are RGB uint8 arrays (H, W, 3).
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self._container: Optional[av.container.InputContainer] = None
        self._fmt: Optional[str] = None
        self._dev: Optional[str] = None
        self._fail_streak = 0

    def open(self) -> None:
        device = str(self.source)
        errors: list[str] = []
        for fmt, dev in self._candidates(device):
            try:
                self._open_device(fmt, dev)
                self._fmt, self._dev = fmt, dev
                self._fail_streak = 0
                return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{fmt}:{dev} → {exc}")
                self.release()

        raise RuntimeError(
            "Cannot open camera (PyAV, no OpenCV).\n"
            + "\n".join(errors)
            + "\nCheck System Settings → Privacy & Security → Camera."
        )

    def _open_device(self, fmt: str, dev: str) -> None:
        options = {
            "framerate": str(self.fps),
            "video_size": f"{self.width}x{self.height}",
        }
        self._container = av.open(dev, format=fmt, options=options)
        # Live devices: prefer low latency
        try:
            self._container.streams.video[0].thread_type = "AUTO"
        except Exception:
            pass

    @staticmethod
    def _candidates(device: str) -> list[Tuple[str, str]]:
        if sys.platform == "darwin":
            return [
                ("avfoundation", device),
                ("avfoundation", f"{device}:none"),
            ]
        if sys.platform.startswith("linux"):
            path = device if device.startswith("/dev/") else f"/dev/video{device}"
            return [("v4l2", path)]
        if sys.platform == "win32":
            return [("dshow", f"video={device}")]
        return [("avfoundation", device)]

    def _is_temp_unavailable(self, exc: BaseException) -> bool:
        if isinstance(exc, BlockingIOError):
            return True
        if isinstance(exc, OSError) and getattr(exc, "errno", None) in (
            errno.EAGAIN,
            errno.EWOULDBLOCK,
            35,
        ):
            return True
        msg = str(exc).lower()
        return "resource temporarily unavailable" in msg or "eagain" in msg

    def _reopen(self) -> None:
        fmt, dev = self._fmt, self._dev
        self.release()
        if fmt and dev:
            time.sleep(0.05)
            self._open_device(fmt, dev)
            self._fmt, self._dev = fmt, dev

    def read(self, wait_s: float = 0.8) -> Optional[np.ndarray]:
        """
        Read next RGB frame.
        Retries temporary EAGAIN; reopens device after repeated failures.
        """
        if self._container is None:
            self.open()
        assert self._container is not None

        deadline = time.perf_counter() + wait_s
        while time.perf_counter() < deadline:
            try:
                for packet in self._container.demux(video=0):
                    if packet.dts is None and packet.size == 0:
                        continue
                    for frame in packet.decode():
                        self._fail_streak = 0
                        return frame.to_ndarray(format="rgb24")
                    # After one demux pass without a frame, continue waiting
                    if time.perf_counter() >= deadline:
                        break
            except StopIteration:
                break
            except Exception as exc:  # noqa: BLE001
                if self._is_temp_unavailable(exc):
                    time.sleep(0.01)
                    continue
                self._fail_streak += 1
                if self._fail_streak >= 3:
                    try:
                        self._reopen()
                        self._fail_streak = 0
                    except Exception:
                        return None
                time.sleep(0.02)
                continue

        self._fail_streak += 1
        if self._fail_streak >= 15:
            try:
                self._reopen()
                self._fail_streak = 0
            except Exception:
                pass
        return None

    def release(self) -> None:
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
        self._container = None

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
