"""Camera / video capture helpers."""

from __future__ import annotations

from typing import Optional

import cv2


class VideoSource:
    """Open the laptop camera (default index 0) or an optional video file."""

    def __init__(self, source: int | str = 0, width: int = 1280, height: int = 720) -> None:
        self.source = source
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.source)
        if isinstance(self.source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open video source {self.source!r}. "
                "Check camera permission in System Settings → Privacy → Camera."
            )

    def read(self):
        if self._cap is None:
            self.open()
        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
