"""Geometric rules for chin-rest, forward head, and head tilt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .pose_estimator import Landmark


@dataclass
class PostureFlags:
    chin_rest: bool
    forward_head: bool
    head_tilt: bool
    # Debug metrics (useful in sidebar)
    chin_dist_norm: float
    forward_offset: float
    tilt_ratio: float


def _visible(lm: Landmark, thr: float = 0.4) -> bool:
    return lm.visibility >= thr


def _dist(a: Landmark, b: Landmark) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def _shoulder_width(lms: Dict[str, Landmark]) -> float:
    return max(_dist(lms["left_shoulder"], lms["right_shoulder"]), 1e-6)


def evaluate_posture(
    lms: Dict[str, Landmark],
    *,
    chin_rest_threshold: float = 0.35,
    forward_head_threshold: float = 0.12,
    head_tilt_threshold: float = 0.12,
) -> PostureFlags:
    """
    All thresholds are relative to shoulder width (normalized image coords).

    chin_rest: wrist/index near face (nose / cheek region)
    forward_head: nose closer to camera than shoulders (MediaPipe z), or
                  nose projects far below ear–shoulder baseline (front cam proxy)
    head_tilt: ear height difference vs shoulder width
    """
    sw = _shoulder_width(lms)
    nose = lms["nose"]
    face_refs = [nose, lms["left_ear"], lms["right_ear"]]

    hand_points = [
        lms["left_wrist"],
        lms["right_wrist"],
        lms["left_index"],
        lms["right_index"],
    ]

    min_hand_face = 1e9
    for hand in hand_points:
        if not _visible(hand):
            continue
        for face in face_refs:
            if not _visible(face, 0.3):
                continue
            min_hand_face = min(min_hand_face, _dist(hand, face) / sw)

    chin_dist_norm = float(min_hand_face if min_hand_face < 1e8 else 1.0)
    chin_rest = chin_dist_norm < chin_rest_threshold

    # Forward head: nose z more negative than shoulder midpoint z → closer to camera
    ls, rs = lms["left_shoulder"], lms["right_shoulder"]
    shoulder_z = (ls.z + rs.z) / 2.0
    # In MediaPipe, smaller (more negative) z ≈ closer to the camera
    forward_offset = float(shoulder_z - nose.z)
    # Also use vertical: if nose is much lower than ear midline relative to shoulders
    ear_y = (lms["left_ear"].y + lms["right_ear"].y) / 2.0
    shoulder_y = (ls.y + rs.y) / 2.0
    # When leaning forward toward screen, head often drops; combine with z
    drop = max(0.0, (nose.y - ear_y) / sw)
    forward_score = forward_offset + 0.5 * drop
    forward_head = forward_score > forward_head_threshold and _visible(nose)

    # Head tilt: |left_ear.y - right_ear.y| / shoulder_width
    tilt_ratio = abs(lms["left_ear"].y - lms["right_ear"].y) / sw
    head_tilt = tilt_ratio > head_tilt_threshold and (
        _visible(lms["left_ear"], 0.3) and _visible(lms["right_ear"], 0.3)
    )

    return PostureFlags(
        chin_rest=chin_rest,
        forward_head=forward_head,
        head_tilt=head_tilt,
        chin_dist_norm=chin_dist_norm,
        forward_offset=forward_score,
        tilt_ratio=float(tilt_ratio),
    )
