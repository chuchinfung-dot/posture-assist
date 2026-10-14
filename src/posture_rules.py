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


def _face_band(lms: Dict[str, Landmark]) -> dict:
    """Lower-face geometry used for chin-rest (excludes forehead / nape)."""
    nose = lms["nose"]
    le, re = lms["left_ear"], lms["right_ear"]
    ley, rey = lms["left_eye"], lms["right_eye"]
    fw = _face_width(lms)
    eye_y = (ley.y + rey.y) / 2.0
    face_h = max(abs(nose.y - eye_y) * 2.8, fw * 0.55)
    ear_y = (le.y + re.y) / 2.0
    return {
        "fw": fw,
        "nose": nose,
        "eye_y": eye_y,
        "ear_y": ear_y,
        "face_h": face_h,
        # Only lower face / cheek — not near ears or crown
        "anchors": [
            (nose.x, nose.y + face_h * 0.55),  # chin
            (nose.x, nose.y + face_h * 0.30),  # mouth
            (nose.x - fw * 0.22, nose.y + face_h * 0.22),  # left cheek (front)
            (nose.x + fw * 0.22, nose.y + face_h * 0.22),  # right cheek
            (nose.x - fw * 0.18, nose.y + face_h * 0.42),  # left jaw
            (nose.x + fw * 0.18, nose.y + face_h * 0.42),  # right jaw
        ],
        # Exclusion magnets: crown / nape / behind-ear
        "nape_anchors": [
            (nose.x, eye_y - face_h * 0.55),  # crown
            (le.x, ear_y - face_h * 0.15),
            (re.x, ear_y - face_h * 0.15),
            (le.x, ear_y + face_h * 0.05),
            (re.x, ear_y + face_h * 0.05),
            ((le.x + re.x) / 2.0, ear_y - face_h * 0.25),  # back of head (top)
        ],
    }


def _min_dist_norm(
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


def _hand_behind_neck(
    lms: Dict[str, Landmark],
    hand: HandTips,
    band: dict,
) -> bool:
    """
    True when hand is likely behind the head / on the nape
    (common false positive for chin-rest).
    """
    fw = band["fw"]
    eye_y = band["eye_y"]
    ear_y = band["ear_y"]
    face_h = band["face_h"]
    nose = band["nose"]

    pts = list(hand.tips) + list(hand.palm) + [hand.wrist]
    # How close to nape/crown vs lower face
    d_nape = _min_dist_norm(band["nape_anchors"], pts, fw)
    d_chin = _min_dist_norm(band["anchors"], pts, fw)

    ys = [p.y for p in pts]
    xs = [p.x for p in pts]
    top_y = min(ys)
    mid_y = float(np.median(ys))
    mid_x = float(np.median(xs))

    # Hand mass above the eyes / near crown
    above_eyes = top_y < eye_y - face_h * 0.05
    near_ear_height = abs(mid_y - ear_y) < face_h * 0.35
    outside_cheek = abs(mid_x - nose.x) > fw * 0.45

    # Raised elbow typical of "hand behind head"
    left = mid_x < nose.x
    shoulder = lms["left_shoulder"] if left else lms["right_shoulder"]
    elbow = lms["left_elbow"] if left else lms["right_elbow"]
    elbow_raised = elbow.y < shoulder.y - fw * 0.08

    # Wrist above ear line
    wrist_high = hand.wrist.y < ear_y + face_h * 0.05

    if d_nape + 0.08 < d_chin:
        return True
    if above_eyes and (near_ear_height or outside_cheek):
        return True
    if elbow_raised and wrist_high and d_nape < 0.85:
        return True
    if wrist_high and outside_cheek and d_chin > 0.35:
        return True
    return False


def _points_in_chin_zone(
    points: Sequence[Landmark], band: dict
) -> List[Landmark]:
    """Keep only hand points that sit in the lower-face / cheek band."""
    eye_y = band["eye_y"]
    face_h = band["face_h"]
    nose = band["nose"]
    fw = band["fw"]
    y_top = eye_y + face_h * 0.02  # allow slightly below eyes only
    y_bot = nose.y + face_h * 0.85
    kept: List[Landmark] = []
    for p in points:
        if p.y < y_top or p.y > y_bot:
            continue
        if abs(p.x - nose.x) > fw * 0.70:
            continue
        kept.append(p)
    return kept


def _hand_span_norm(hand: HandTips, face_w: float) -> float:
    xs = [p.x for p in hand.tips]
    ys = [p.y for p in hand.tips]
    return float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)) / max(face_w, 1e-6))


def _hand_is_face_false_positive(hand: HandTips, band: dict) -> bool:
    """
    Close-up faces are often misread as a 'hand' glued onto the cheek.
    Real hands usually have a larger tip span and a wrist offset from the face center.
    """
    fw = band["fw"]
    nose = band["nose"]
    span = _hand_span_norm(hand, fw)
    if span < 0.18:
        return True
    # Wrist almost at nose/mouth → likely face landmarks mistaken as hand
    if _dist(hand.wrist, nose) / fw < 0.25:
        return True
    # All tips clustered on face center
    tips_near = sum(
        1
        for t in hand.tips
        if abs(t.x - nose.x) < fw * 0.35 and abs(t.y - nose.y) < band["face_h"] * 0.45
    )
    if tips_near >= 4 and span < 0.35:
        return True
    return False


def _face_very_close(lms: Dict[str, Landmark], band: dict) -> bool:
    """True when the face fills much of the frame (leaning into the camera)."""
    fw = band["fw"]
    sw = _shoulder_width(lms)
    # Absolute face size or face dominating shoulders
    return fw >= 0.30 or (sw > 1e-6 and fw / sw >= 0.85)


def _evaluate_chin_rest(
    lms: Dict[str, Landmark],
    hands: List[HandTips],
    *,
    chin_rest_threshold: float,
    suppress_when_close: bool = True,
) -> Tuple[bool, float, float]:
    """
    Returns (chin_rest, chin_dist_norm, chin_score).
    Rejects hand-behind-neck / crown poses and close-up face false positives.
    """
    band = _face_band(lms)
    close = _face_very_close(lms, band)

    # Drop nape poses and close-up face→hand misdetections
    valid_hands = [
        h
        for h in hands
        if not _hand_behind_neck(lms, h, band)
        and not _hand_is_face_false_positive(h, band)
    ]

    hand_points: List[Landmark] = []
    used_hand_landmarks = False
    for hand in valid_hands:
        used_hand_landmarks = True
        hand_points.extend(hand.tips)
        hand_points.extend(hand.palm)
        hand_points.append(hand.wrist)

    hand_points = _points_in_chin_zone(hand_points, band)

    # When close to camera: never use Pose wrist fallback (very noisy / false chin-rest)
    if not hand_points and not close:
        pose_candidates = [
            lms["left_wrist"],
            lms["right_wrist"],
            lms["left_index"],
            lms["right_index"],
        ]
        for side in ("left", "right"):
            sh = lms[f"{side}_shoulder"]
            el = lms[f"{side}_elbow"]
            wr = lms[f"{side}_wrist"]
            if el.y < sh.y - band["fw"] * 0.08 and wr.y < band["ear_y"] + band["face_h"] * 0.1:
                pose_candidates = [
                    p
                    for p in pose_candidates
                    if p is not wr and p is not lms[f"{side}_index"]
                ]
        hand_points = _points_in_chin_zone(pose_candidates, band)

    if not hand_points:
        return False, 1.0, 0.0

    # Close-up without a real Hand Landmarker hand → not chin-rest
    if close and not used_hand_landmarks:
        return False, 1.0, 0.0

    chin_dist = _min_dist_norm(band["anchors"], hand_points, band["fw"])
    d_nape = _min_dist_norm(band["nape_anchors"], hand_points, band["fw"])
    if d_nape + 0.12 < chin_dist:
        return False, 1.0, 0.0

    left_bent = _elbow_bent(lms["left_shoulder"], lms["left_elbow"], lms["left_wrist"])
    right_bent = _elbow_bent(
        lms["right_shoulder"], lms["right_elbow"], lms["right_wrist"]
    )
    bent_boost = 0.0
    if left_bent and any(
        abs(p.x - lms["left_wrist"].x) < 0.15 for p in hand_points
    ):
        bent_boost = 0.08
    if right_bent and any(
        abs(p.x - lms["right_wrist"].x) < 0.15 for p in hand_points
    ):
        bent_boost = max(bent_boost, 0.08)

    # Close to camera: require bent elbow (classic chin-rest), stricter distance
    if close:
        if not (left_bent or right_bent):
            return False, 1.0, 0.0
        thr = chin_rest_threshold * 0.65
        if suppress_when_close and not used_hand_landmarks:
            return False, 1.0, 0.0
    else:
        thr = chin_rest_threshold + bent_boost

    chin_rest = chin_dist < thr
    chin_score = max(0.0, thr - chin_dist)
    return chin_rest, float(chin_dist), float(chin_score)


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

    # --- Chin rest (rejects hand-behind-neck / crown) ---
    chin_rest, chin_dist, chin_score = _evaluate_chin_rest(
        lms, hands, chin_rest_threshold=chin_rest_threshold
    )

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
