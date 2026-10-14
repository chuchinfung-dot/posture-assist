"""Relative screen-distance estimate via apparent body/face size + calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .pose_estimator import Landmark


@dataclass
class DistanceState:
    calibrated: bool
    apparent_size: float
    baseline_size: float
    ratio: float  # apparent / baseline; >1 means closer
    too_close: bool


def apparent_size(lms: Dict[str, Landmark]) -> float:
    """
    Apparent size proxy: prefer inter-ear distance, fall back to shoulder width.
    Larger value ⇒ subject is closer to the camera / screen.
    """
    le, re = lms["left_ear"], lms["right_ear"]
    ear_w = abs(le.x - re.x)
    if ear_w > 0.02 and le.visibility > 0.3 and re.visibility > 0.3:
        return float(ear_w)

    ls, rs = lms["left_shoulder"], lms["right_shoulder"]
    return float(abs(ls.x - rs.x))


def evaluate_distance(
    lms: Dict[str, Landmark],
    baseline_size: Optional[float],
    *,
    close_ratio: float = 1.25,
) -> DistanceState:
    """
    After calibration, if apparent_size / baseline > close_ratio → too close.
    close_ratio 1.25 means ~25% larger on screen than at calibration distance.
    """
    size = apparent_size(lms)
    if baseline_size is None or baseline_size <= 1e-6:
        return DistanceState(
            calibrated=False,
            apparent_size=size,
            baseline_size=0.0,
            ratio=1.0,
            too_close=False,
        )
    ratio = size / baseline_size
    return DistanceState(
        calibrated=True,
        apparent_size=size,
        baseline_size=baseline_size,
        ratio=float(ratio),
        too_close=ratio >= close_ratio,
    )


class CalibrationStore:
    """Persist baseline apparent size under data/calibration.json."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> Optional[float]:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            value = float(data.get("baseline_size", 0))
            return value if value > 0 else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def save(self, baseline_size: float) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"baseline_size": baseline_size}, indent=2),
            encoding="utf-8",
        )
