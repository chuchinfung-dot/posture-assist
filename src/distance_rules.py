"""Relative screen-distance estimate via apparent body/face size + calibration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

from .pose_estimator import Landmark
from .posture_rules import PostureCalibration


@dataclass
class DistanceState:
    calibrated: bool
    apparent_size: float
    baseline_size: float
    ratio: float  # apparent / baseline; >1 means closer
    too_close: bool


@dataclass
class FullCalibration:
    baseline_size: float = 0.0
    posture: Optional[PostureCalibration] = None


def apparent_size(lms: Dict[str, Landmark]) -> float:
    """Prefer inter-ear distance; fall back to shoulder width."""
    le, re = lms["left_ear"], lms["right_ear"]
    ear_w = abs(le.x - re.x)
    if ear_w > 0.02:
        return float(ear_w)
    ls, rs = lms["left_shoulder"], lms["right_shoulder"]
    return float(abs(ls.x - rs.x))


def evaluate_distance(
    lms: Dict[str, Landmark],
    baseline_size: Optional[float],
    *,
    close_ratio: float = 1.35,
) -> DistanceState:
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
    """Persist distance + posture baselines under data/calibration.json."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> FullCalibration:
        if not self.path.exists():
            return FullCalibration()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            size = float(data.get("baseline_size", 0) or 0)
            posture = None
            raw = data.get("posture")
            if isinstance(raw, dict) and raw.get("face_size"):
                posture = PostureCalibration(
                    face_size=float(raw.get("face_size", 0)),
                    head_shoulder_ratio=float(raw.get("head_shoulder_ratio", 0)),
                    neck_drop=float(raw.get("neck_drop", 0)),
                    roll_deg=float(raw.get("roll_deg", 0)),
                    pitch_deg=float(raw.get("pitch_deg", 0)),
                    lateral_deg=float(raw.get("lateral_deg", 0)),
                    pitch_proxy=float(raw.get("pitch_proxy", 0)),
                    nose_shoulder_z=float(raw.get("nose_shoulder_z", 0)),
                )
            return FullCalibration(
                baseline_size=size if size > 0 else 0.0,
                posture=posture,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return FullCalibration()

    def save(self, calib: FullCalibration) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"baseline_size": calib.baseline_size}
        if calib.posture is not None:
            payload["posture"] = asdict(calib.posture)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
