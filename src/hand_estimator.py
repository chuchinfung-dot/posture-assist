"""MediaPipe Hand Landmarker — accurate fingertips for chin-rest detection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

from .pose_estimator import Landmark

DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
)

# Key fingertip / palm indices
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
INDEX_MCP = 5
MIDDLE_MCP = 9


@dataclass
class HandTips:
    wrist: Landmark
    tips: List[Landmark]  # thumb..pinky tips
    palm: List[Landmark]  # MCP joints near palm


class HandEstimator:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        num_hands: int = 2,
        min_detection_confidence: float = 0.4,
        min_tracking_confidence: float = 0.4,
    ) -> None:
        path = Path(model_path) if model_path else DEFAULT_MODEL
        if not path.exists():
            raise FileNotFoundError(f"Hand model not found: {path}")

        options = mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._frame_ts_ms = 0
        self._t0 = time.perf_counter()
        self._connections = list(mp_vision.HandLandmarksConnections.HAND_CONNECTIONS)

    def process(self, frame_bgr: np.ndarray) -> Tuple[List[HandTips], list]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.perf_counter() - self._t0) * 1000)
        if ts <= self._frame_ts_ms:
            ts = self._frame_ts_ms + 1
        self._frame_ts_ms = ts
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        hands: List[HandTips] = []
        raw = result.hand_landmarks or []
        for lm_list in raw:
            hands.append(
                HandTips(
                    wrist=_to_lm(lm_list[WRIST]),
                    tips=[
                        _to_lm(lm_list[THUMB_TIP]),
                        _to_lm(lm_list[INDEX_TIP]),
                        _to_lm(lm_list[MIDDLE_TIP]),
                        _to_lm(lm_list[RING_TIP]),
                        _to_lm(lm_list[PINKY_TIP]),
                    ],
                    palm=[
                        _to_lm(lm_list[INDEX_MCP]),
                        _to_lm(lm_list[MIDDLE_MCP]),
                    ],
                )
            )
        return hands, raw

    def draw(self, frame_bgr: np.ndarray, hand_landmarks_list: list) -> np.ndarray:
        out = frame_bgr.copy()
        for hand_lms in hand_landmarks_list:
            mp_vision.drawing_utils.draw_landmarks(
                out,
                hand_lms,
                self._connections,
                landmark_drawing_spec=mp_vision.drawing_utils.DrawingSpec(
                    color=(255, 100, 0), thickness=2, circle_radius=3
                ),
                connection_drawing_spec=mp_vision.drawing_utils.DrawingSpec(
                    color=(0, 140, 255), thickness=2
                ),
            )
        return out

    def close(self) -> None:
        self._landmarker.close()


def _to_lm(p) -> Landmark:
    return Landmark(x=p.x, y=p.y, z=getattr(p, "z", 0.0), visibility=1.0)
