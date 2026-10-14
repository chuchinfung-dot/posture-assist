"""MediaPipe Face Landmarker — RGB in, no OpenCV."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
)


@dataclass
class HeadPose:
    pitch: float
    yaw: float
    roll: float
    ok: bool = True


def _rotation_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    r20 = float(np.clip(-R[2, 0], -1.0, 1.0))
    pitch = float(np.degrees(np.arcsin(r20)))
    yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    roll = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
    return pitch, yaw, roll


class FaceEstimator:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        min_detection_confidence: float = 0.4,
        min_tracking_confidence: float = 0.4,
    ) -> None:
        path = Path(model_path) if model_path else DEFAULT_MODEL
        if not path.exists():
            raise FileNotFoundError(f"Face model not found: {path}")

        options = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._frame_ts_ms = 0
        self._t0 = time.perf_counter()

    def process(self, frame_rgb: np.ndarray) -> Optional[HeadPose]:
        if not frame_rgb.flags.writeable:
            frame_rgb = np.ascontiguousarray(frame_rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        ts = int((time.perf_counter() - self._t0) * 1000)
        if ts <= self._frame_ts_ms:
            ts = self._frame_ts_ms + 1
        self._frame_ts_ms = ts
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        if not result.facial_transformation_matrixes:
            return None

        M = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)
        if M.size == 16:
            M = M.reshape(4, 4)
        pitch, yaw, roll = _rotation_to_euler(M[:3, :3])
        return HeadPose(pitch=pitch, yaw=yaw, roll=roll, ok=True)

    def close(self) -> None:
        self._landmarker.close()
