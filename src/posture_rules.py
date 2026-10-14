"""Geometric rules for chin-rest, forward head, and head tilt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .face_estimator import HeadPose
from .hand_estimator import HandTips
from .pose_estimator import Landmark


@dataclass
class PostureCalibration:
    """Good-posture baselines captured with the `c` key."""

    face_size: float = 0.0
    head_shoulder_ratio: float = 0.0
    neck_drop: float = 0.0
    # Neck / head orientation baselines (degrees or unitless)
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    lateral_deg: float = 0.0
    pitch_proxy: float = 0.0
    nose_shoulder_z: float = 0.0


@dataclass
class PostureFlags:
    chin_rest: bool
    forward_head: bool
    head_tilt: bool
    chin_dist_norm: float
    forward_offset: float
    tilt_ratio: float
    chin_score: float = 0.0
    # Extra debug for neck
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    neck_score: float = 0.0


def _dist(a: Landmark, b: Landmark) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def _shoulder_width(lms: Dict[str, Landmark]) -> float:
    return max(_dist(lms["left_shoulder"], lms["right_shoulder"]), 1e-6)


def _face_width(lms: Dict[str, Landmark]) -> float:
    return max(abs(lms["left_ear"].x - lms["right_ear"].x), 1e-6)


def _face_anchor_points(lms: Dict[str, Landmark]) -> List[Tuple[float, float]]:
    nose = lms["nose"]
    le, re = lms["left_ear"], lms["right_ear"]
    ley, rey = lms["left_eye"], lms["right_eye"]
    fw = _face_width(lms)
    eye_y = (ley.y + rey.y) / 2.0
    face_h = max(abs(nose.y - eye_y) * 2.8, fw * 0.55)

    chin = (nose.x, nose.y + face_h * 0.55)
    mouth = (nose.x, nose.y + face_h * 0.28)
    left_cheek = ((nose.x + le.x) / 2.0, (nose.y + le.y) / 2.0 + face_h * 0.12)
    right_cheek = ((nose.x + re.x) / 2.0, (nose.y + re.y) / 2.0 + face_h * 0.12)
    left_jaw = (le.x * 0.65 + nose.x * 0.35, nose.y + face_h * 0.4)
    right_jaw = (re.x * 0.65 + nose.x * 0.35, nose.y + face_h * 0.4)
    return [chin, mouth, left_cheek, right_cheek, left_jaw, right_jaw]


def _min_hand_to_face(
    anchors: Sequence[Tuple[float, float]],
    points: Sequence[Landmark],
    face_w: float,
) -> float:
    best = 1e9
    for p in points:
        for ax, ay in anchors:
            best = min(best, float(np.hypot(p.x - ax, p.y - ay)) / face_w)
    return float(best if best < 1e8 else 1.0)


def _elbow_bent(shoulder: Landmark, elbow: Landmark, wrist: Landmark) -> bool:
    v1 = np.array([shoulder.x - elbow.x, shoulder.y - elbow.y])
    v2 = np.array([wrist.x - elbow.x, wrist.y - elbow.y])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return False
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cos_a)))
    return angle < 100.0


def _pose_neck_geometry(lms: Dict[str, Landmark]) -> dict:
    """Pose-only neck features (works even without Face Landmarker)."""
    fw = _face_width(lms)
    sw = _shoulder_width(lms)
    le, re = lms["left_ear"], lms["right_ear"]
    ls, rs = lms["left_shoulder"], lms["right_shoulder"]
    nose = lms["nose"]
    ley, rey = lms["left_eye"], lms["right_eye"]

    ear_mid_x = (le.x + re.x) / 2.0
    ear_mid_y = (le.y + re.y) / 2.0
    ear_mid_z = (le.z + re.z) / 2.0
    sh_mid_x = (ls.x + rs.x) / 2.0
    sh_mid_y = (ls.y + rs.y) / 2.0
    sh_mid_z = (ls.z + rs.z) / 2.0

    # Roll from ear line + eye line (degrees from horizontal)
    ear_roll = float(np.degrees(np.arctan2(le.y - re.y, le.x - re.x)))
    eye_roll = float(np.degrees(np.arctan2(ley.y - rey.y, ley.x - rey.x)))
    roll_deg = 0.6 * ear_roll + 0.4 * eye_roll

    # Lateral lean of neck column (0 = vertical)
    lateral_deg = float(
        np.degrees(np.arctan2(ear_mid_x - sh_mid_x, max(sh_mid_y - ear_mid_y, 1e-6)))
    )

    # Forward / tucked-chin proxies
    neck_drop = (ear_mid_y - (sh_mid_y - sw * 0.55)) / sw
    head_shoulder_ratio = fw / sw
    # Nose closer to camera than shoulders → more negative nose.z relative to shoulders
    nose_shoulder_z = float(sh_mid_z - nose.z)
    ear_forward_z = float(sh_mid_z - ear_mid_z)
    # Combined pitch proxy (unitless): anterior translation + head drop
    pitch_proxy = ear_forward_z * 1.4 + max(0.0, neck_drop) * 0.9 + max(
        0.0, nose_shoulder_z - ear_forward_z
    ) * 0.5

    return {
        "fw": fw,
        "sw": sw,
        "roll_deg": roll_deg,
        "lateral_deg": lateral_deg,
        "neck_drop": float(neck_drop),
        "head_shoulder_ratio": float(head_shoulder_ratio),
        "nose_shoulder_z": nose_shoulder_z,
        "pitch_proxy": float(pitch_proxy),
        "face_size": fw,
    }


def evaluate_posture(
    lms: Dict[str, Landmark],
    hands: Optional[List[HandTips]] = None,
    head_pose: Optional[HeadPose] = None,
    *,
    calibration: Optional[PostureCalibration] = None,
    chin_rest_threshold: float = 0.55,
    forward_head_threshold: float = 0.14,
    head_tilt_threshold: float = 12.0,  # degrees when using roll
    forward_pitch_deg: float = 12.0,  # pitch delta vs calibration
) -> PostureFlags:
    """
    Neck accuracy stack:
      1) Face Landmarker pitch/roll (best) when available
      2) Pose ear/eye roll + neck lateral angle
      3) Calibrated deltas vs `c` baseline (strongly preferred)
    """
    hands = hands or []
    geom = _pose_neck_geometry(lms)
    fw = geom["fw"]
    anchors = _face_anchor_points(lms)

    # --- Chin rest (unchanged logic) ---
    hand_points: List[Landmark] = []
    for hand in hands:
        hand_points.extend(hand.tips)
        hand_points.extend(hand.palm)
        hand_points.append(hand.wrist)

    if hand_points:
        chin_dist = _min_hand_to_face(anchors, hand_points, fw)
    else:
        pose_hands = [
            lms["left_wrist"],
            lms["right_wrist"],
            lms["left_index"],
            lms["right_index"],
        ]
        chin_dist = _min_hand_to_face(anchors, pose_hands, fw)

    left_bent = _elbow_bent(lms["left_shoulder"], lms["left_elbow"], lms["left_wrist"])
    right_bent = _elbow_bent(
        lms["right_shoulder"], lms["right_elbow"], lms["right_wrist"]
    )
    bent_boost = 0.12 if (left_bent or right_bent) else 0.0
    chin_score = max(0.0, (chin_rest_threshold + bent_boost) - chin_dist)
    chin_rest = chin_dist < (chin_rest_threshold + bent_boost)

    # --- Merge face pose if present ---
    if head_pose is not None and head_pose.ok:
        roll_deg = float(head_pose.roll)
        pitch_deg = float(head_pose.pitch)
        face_weight = 1.0
    else:
        roll_deg = float(geom["roll_deg"])
        pitch_deg = 0.0  # unknown without face matrix
        face_weight = 0.0

    lateral_deg = float(geom["lateral_deg"])

    # --- Head tilt ---
    if calibration is not None and (
        calibration.face_size > 0 or abs(calibration.roll_deg) > 1e-6
    ):
        roll_delta = abs(roll_deg - calibration.roll_deg)
        lateral_delta = abs(lateral_deg - calibration.lateral_deg)
    else:
        roll_delta = abs(roll_deg)
        lateral_delta = abs(lateral_deg)

    # Tilt if roll OR neck lean exceeds degree threshold
    tilt_metric = max(roll_delta, lateral_delta * 0.9)
    # Also keep mild ear asymmetry as backup (normalized → rough degrees * 40)
    ear_asym = abs(lms["left_ear"].y - lms["right_ear"].y) / fw * 40.0
    tilt_metric = max(tilt_metric, ear_asym * 0.5)
    head_tilt = tilt_metric > head_tilt_threshold

    # --- Forward head ---
    if calibration is not None and calibration.face_size > 0:
        ratio_delta = (
            geom["head_shoulder_ratio"]
            / max(calibration.head_shoulder_ratio, 1e-6)
            - 1.0
        )
        drop_delta = geom["neck_drop"] - calibration.neck_drop
        z_delta = geom["nose_shoulder_z"] - calibration.nose_shoulder_z
        proxy_delta = geom["pitch_proxy"] - calibration.pitch_proxy

        pitch_delta = 0.0
        if face_weight > 0:
            # Chin-down / forward nod relative to calibration
            pitch_delta = pitch_deg - calibration.pitch_deg

        # Weighted score: face pitch (strong) + geometry
        forward_score = (
            max(0.0, pitch_delta) / max(forward_pitch_deg, 1e-3) * 0.55
            + max(0.0, proxy_delta) * 1.1
            + max(0.0, ratio_delta) * 1.0
            + max(0.0, drop_delta) * 0.7
            + max(0.0, z_delta) * 0.8
        )
        # Dual gate: score threshold OR clear pitch delta
        forward_head = forward_score > forward_head_threshold or (
            face_weight > 0 and pitch_delta > forward_pitch_deg
        )
        neck_score = float(forward_score)
    else:
        # No calibration: conservative — need strong evidence
        z_forward = max(0.0, geom["nose_shoulder_z"])
        forward_score = z_forward * 0.5 + max(0.0, geom["neck_drop"]) * 0.4
        if face_weight > 0:
            forward_score += max(0.0, pitch_deg) / 25.0
        forward_head = forward_score > (forward_head_threshold + 0.15)
        neck_score = float(forward_score)

    return PostureFlags(
        chin_rest=chin_rest,
        forward_head=forward_head,
        head_tilt=head_tilt,
        chin_dist_norm=float(chin_dist),
        forward_offset=float(forward_score),
        tilt_ratio=float(tilt_metric),
        chin_score=float(chin_score),
        pitch_deg=float(pitch_deg),
        roll_deg=float(roll_deg),
        neck_score=float(neck_score),
    )


def capture_posture_calibration(
    lms: Dict[str, Landmark],
    head_pose: Optional[HeadPose] = None,
) -> PostureCalibration:
    geom = _pose_neck_geometry(lms)
    roll = float(head_pose.roll) if head_pose and head_pose.ok else geom["roll_deg"]
    pitch = float(head_pose.pitch) if head_pose and head_pose.ok else 0.0
    return PostureCalibration(
        face_size=geom["face_size"],
        head_shoulder_ratio=geom["head_shoulder_ratio"],
        neck_drop=geom["neck_drop"],
        roll_deg=roll,
        pitch_deg=pitch,
        lateral_deg=geom["lateral_deg"],
        pitch_proxy=geom["pitch_proxy"],
        nose_shoulder_z=geom["nose_shoulder_z"],
    )
