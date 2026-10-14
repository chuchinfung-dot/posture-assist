"""Landmark drawing helpers on RGB numpy arrays (no OpenCV)."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


def _line(
    draw: ImageDraw.ImageDraw,
    a: Tuple[int, int],
    b: Tuple[int, int],
    color: Tuple[int, int, int],
    width: int = 2,
) -> None:
    draw.line([a, b], fill=color, width=width)


def _dot(
    draw: ImageDraw.ImageDraw,
    p: Tuple[int, int],
    color: Tuple[int, int, int],
    r: int = 3,
) -> None:
    x, y = p
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def draw_pose_landmarks(
    rgb: np.ndarray,
    landmarks,
    connections: Sequence,
    *,
    point_color: Tuple[int, int, int] = (0, 220, 80),
    line_color: Tuple[int, int, int] = (0, 180, 255),
) -> np.ndarray:
    if landmarks is None:
        return rgb
    h, w = rgb.shape[:2]
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    pts = []
    for lm in landmarks:
        pts.append((int(lm.x * w), int(lm.y * h)))

    for conn in connections:
        start = getattr(conn, "start", None)
        end = getattr(conn, "end", None)
        if start is None and isinstance(conn, (tuple, list)) and len(conn) == 2:
            start, end = conn[0], conn[1]
        if start is None or end is None:
            continue
        if 0 <= start < len(pts) and 0 <= end < len(pts):
            _line(draw, pts[start], pts[end], line_color, 2)

    for p in pts:
        _dot(draw, p, point_color, 3)

    return np.asarray(img)


def draw_hand_landmarks(
    rgb: np.ndarray,
    hands: Iterable,
    connections: Sequence,
    *,
    point_color: Tuple[int, int, int] = (255, 120, 0),
    line_color: Tuple[int, int, int] = (255, 160, 40),
) -> np.ndarray:
    out = rgb
    for hand in hands:
        out = draw_pose_landmarks(
            out, hand, connections, point_color=point_color, line_color=line_color
        )
    return out
