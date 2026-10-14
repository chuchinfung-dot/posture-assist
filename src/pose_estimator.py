"""MediaPipe Pose Landmarker (Tasks API) — landmarks + skeleton overlay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

# Landmark indices (same as classic MediaPipe Pose)
NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_HIP = 23
RIGHT_HIP = 24

DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"
)


@dataclass
class Landmark:
    x: float  # normalized [0, 1]
    y: float
    z: float
    visibility: float

    def px(self, w: int, h: int) -> Tuple[int, int]:
        return int(self.x * w), int(self.y * h)


class PoseEstimator:
    """Estimate body pose with MediaPipe Tasks PoseLandmarker."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        num_poses: int = 1,
    ) -> None:
        path = Path(model_path) if model_path else DEFAULT_MODEL
        if not path.exists():
            raise FileNotFoundError(
                f"Pose model not found: {path}. "
                "Download pose_landmarker_lite.task into models/."
            )

        options = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._frame_ts_ms = 0
        self._connections = list(mp_vision.PoseLandmarksConnections.POSE_LANDMARKS)

    def process(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[Dict[str, Landmark]], Optional[object]]:
        """Return named landmark dict and raw pose landmarks list (or None)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts_ms += 33  # ~30 FPS clock for VIDEO mode
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        if not result.pose_landmarks:
            return None, None

        lm_list = result.pose_landmarks[0]
        named = {
            "nose": _to_lm(lm_list[NOSE]),
            "left_eye": _to_lm(lm_list[LEFT_EYE]),
            "right_eye": _to_lm(lm_list[RIGHT_EYE]),
            "left_ear": _to_lm(lm_list[LEFT_EAR]),
            "right_ear": _to_lm(lm_list[RIGHT_EAR]),
            "left_shoulder": _to_lm(lm_list[LEFT_SHOULDER]),
            "right_shoulder": _to_lm(lm_list[RIGHT_SHOULDER]),
            "left_elbow": _to_lm(lm_list[LEFT_ELBOW]),
            "right_elbow": _to_lm(lm_list[RIGHT_ELBOW]),
            "left_wrist": _to_lm(lm_list[LEFT_WRIST]),
            "right_wrist": _to_lm(lm_list[RIGHT_WRIST]),
            "left_index": _to_lm(lm_list[LEFT_INDEX]),
            "right_index": _to_lm(lm_list[RIGHT_INDEX]),
            "left_hip": _to_lm(lm_list[LEFT_HIP]),
            "right_hip": _to_lm(lm_list[RIGHT_HIP]),
        }
        return named, lm_list

    def draw(self, frame_bgr: np.ndarray, pose_landmarks) -> np.ndarray:
        out = frame_bgr.copy()
        if pose_landmarks is None:
            return out
        mp_vision.drawing_utils.draw_landmarks(
            out,
            pose_landmarks,
            self._connections,
            landmark_drawing_spec=mp_vision.drawing_utils.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=2
            ),
            connection_drawing_spec=mp_vision.drawing_utils.DrawingSpec(
                color=(0, 200, 255), thickness=2
            ),
        )
        return out

    def close(self) -> None:
        self._landmarker.close()


def _to_lm(p) -> Landmark:
    vis = getattr(p, "visibility", None)
    if vis is None:
        vis = getattr(p, "presence", 1.0)
    return Landmark(x=p.x, y=p.y, z=p.z, visibility=float(vis if vis is not None else 1.0))
