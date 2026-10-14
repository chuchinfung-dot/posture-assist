#!/usr/bin/env python3
"""
坐姿健康提醒 — 本地 OpenCV 窗口版

快捷键：
  c     校准（坐直后按；距离 + 颈部基准 pitch/roll）
  s     开关骨架/手部点
  m     开关提示音
  d     开关实时调试数值
  q/ESC 退出
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alert import (
    ISSUE_LABELS,
    ISSUE_MESSAGES,
    AlertLogger,
    IssueTimer,
    play_beep,
)
from src.distance_rules import (
    CalibrationStore,
    FullCalibration,
    apparent_size,
    evaluate_distance,
)
from src.face_estimator import FaceEstimator
from src.hand_estimator import HandEstimator
from src.pose_estimator import PoseEstimator
from src.posture_rules import capture_posture_calibration, evaluate_posture
from src.smoother import EmaSmoother
from src.video_source import VideoSource

DATA_DIR = ROOT / "data"
CALIB_PATH = DATA_DIR / "calibration.json"
LOG_PATH = DATA_DIR / "posture_log.csv"

_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size, index=0)
            except OSError:
                continue
    return ImageFont.load_default()


def put_overlay(
    frame: np.ndarray,
    lines: list[str],
    *,
    mode: str,
    help_line: str,
) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img, "RGBA")
    font = _load_font(26)
    small = _load_font(18)

    colors = {
        "ok": (40, 160, 70, 150),
        "watch": (220, 160, 20, 160),
        "alert": (220, 40, 40, 175),
    }
    bar_h = 36 + 30 * max(len(lines), 1)
    draw.rectangle([(0, 0), (img.width, bar_h)], fill=colors.get(mode, colors["ok"]))

    y = 6
    for line in lines:
        draw.text((14, y), line, font=font, fill=(255, 255, 255, 255))
        y += 30

    draw.rectangle([(0, img.height - 34), (img.width, img.height)], fill=(0, 0, 0, 150))
    draw.text((10, img.height - 28), help_line, font=small, fill=(230, 230, 230, 255))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="坐姿健康提醒（OpenCV）")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--dwell", type=float, default=3.0, help="持续秒数后告警")
    p.add_argument("--chin", type=float, default=0.55, help="托腮距离阈值")
    p.add_argument("--forward", type=float, default=0.14, help="颈前倾综合分数阈值")
    p.add_argument(
        "--tilt",
        type=float,
        default=12.0,
        help="头侧倾角度阈值（度，相对校准）",
    )
    p.add_argument(
        "--pitch",
        type=float,
        default=12.0,
        help="低头/前倾 pitch 角度阈值（度，相对校准）",
    )
    p.add_argument("--close-ratio", type=float, default=1.35, help="过近比例")
    p.add_argument("--no-skeleton", action="store_true")
    p.add_argument("--no-sound", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    show_skeleton = not args.no_skeleton
    sound_on = not args.no_sound
    show_debug = True

    store = CalibrationStore(CALIB_PATH)
    full_calib = store.load()
    baseline = full_calib.baseline_size if full_calib.baseline_size > 0 else None
    posture_calib = full_calib.posture

    timer = IssueTimer()
    logger = AlertLogger(LOG_PATH)
    smooth = EmaSmoother(alpha=0.35)

    print("=" * 56)
    print("坐姿健康提醒（颈部加强：Face pitch/roll + Pose 颈角）")
    print("1) 坐直正对镜头  2) 按 c 校准  3) 再测伸颈/歪头")
    print("快捷键: c校准  s骨架  m声音  d调试  q退出")
    if baseline and posture_calib:
        print(
            f"已载入校准 distance={baseline:.4f} "
            f"pitch0={getattr(posture_calib, 'pitch_deg', 0):.1f} "
            f"roll0={getattr(posture_calib, 'roll_deg', 0):.1f}"
        )
    else:
        print("尚未校准：请先按 c（颈部准确度依赖校准）")
    print("=" * 56)

    pose = PoseEstimator()
    hands_est = HandEstimator()
    face_est = FaceEstimator()
    cap = VideoSource(args.camera)
    try:
        cap.open()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        pose.close()
        hands_est.close()
        face_est.close()
        sys.exit(1)

    window = "Posture Assist"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = cap.read()
            if frame is None:
                print("无法读取摄像头画面", file=sys.stderr)
                break

            frame = cv2.flip(frame, 1)
            landmarks, lm_raw = pose.process(frame)
            hand_tips, hand_raw = hands_est.process(frame)
            head_pose = face_est.process(frame)

            watch_lines: list[str] = []
            alert_lines: list[str] = []
            debug = ""
            mode = "ok"

            if landmarks is not None:
                posture = evaluate_posture(
                    landmarks,
                    hand_tips,
                    head_pose,
                    calibration=posture_calib,
                    chin_rest_threshold=args.chin,
                    forward_head_threshold=args.forward,
                    head_tilt_threshold=args.tilt,
                    forward_pitch_deg=args.pitch,
                )
                dist = evaluate_distance(
                    landmarks, baseline, close_ratio=args.close_ratio
                )

                chin_s = smooth.update("chin", posture.chin_dist_norm)
                fwd_s = smooth.update("fwd", posture.forward_offset)
                tilt_s = smooth.update("tilt", posture.tilt_ratio)
                pitch_s = smooth.update("pitch", posture.pitch_deg)
                roll_s = smooth.update("roll", posture.roll_deg)
                ratio_s = smooth.update("ratio", dist.ratio)

                chin_on = chin_s < args.chin or posture.chin_rest
                # Prefer rule engine for neck (already uses smoothed-ish face+pose);
                # reinforce with smoothed degree thresholds vs calibration.
                if posture_calib is not None:
                    tilt_on = tilt_s > args.tilt or posture.head_tilt
                    # forward: keep model flag, or large pitch delta after smooth
                    pitch0 = getattr(posture_calib, "pitch_deg", 0.0)
                    fwd_on = posture.forward_head or (
                        (pitch_s - pitch0) > args.pitch
                    )
                else:
                    tilt_on = posture.head_tilt
                    fwd_on = posture.forward_head

                active = {
                    "chin_rest": chin_on,
                    "forward_head": fwd_on,
                    "head_tilt": tilt_on,
                    "too_close": dist.too_close,
                }

                newly = timer.update(active, dwell_seconds=args.dwell)
                for key in newly:
                    duration = timer.elapsed(key)
                    logger.log(key, duration)
                    play_beep(sound_on)
                    print(
                        f"[告警] {ISSUE_LABELS[key]} · {duration:.0f}s — {ISSUE_MESSAGES[key]}"
                    )

                for key, on in active.items():
                    if not on:
                        continue
                    elapsed = timer.elapsed(key)
                    label = ISSUE_LABELS[key]
                    if key in timer.alerted:
                        alert_lines.append(ISSUE_MESSAGES[key])
                    else:
                        watch_lines.append(
                            f"监测中 {label} {elapsed:.1f}/{args.dwell:.0f}s"
                        )

                face_ok = "Y" if head_pose is not None else "N"
                if show_debug:
                    debug = (
                        f"face={face_ok} pitch={pitch_s:.1f}° roll={roll_s:.1f}° "
                        f"tiltΔ={tilt_s:.1f} fwd={fwd_s:.2f} "
                        f"chin={chin_s:.2f} dist×={ratio_s:.2f} hands={len(hand_tips)}"
                    )
            else:
                timer.update(
                    {
                        "chin_rest": False,
                        "forward_head": False,
                        "head_tilt": False,
                        "too_close": False,
                    },
                    dwell_seconds=args.dwell,
                )
                watch_lines.append("未检测到上半身，请靠近镜头")

            if show_skeleton:
                frame = pose.draw(frame, lm_raw)
                frame = hands_est.draw(frame, hand_raw)

            if alert_lines:
                mode = "alert"
                lines = alert_lines
            elif watch_lines:
                mode = "watch"
                lines = watch_lines
            else:
                mode = "ok"
                lines = ["姿态正常"]

            if debug:
                lines = lines + [debug]

            totals = " ".join(
                f"{ISSUE_LABELS[k]}:{timer.today_seconds.get(k, 0):.0f}s"
                for k in ISSUE_LABELS
            )
            help_line = f"[c]校准 [s]骨架 [m]声 [d]调试 [q]退出  {totals}"
            frame = put_overlay(frame, lines, mode=mode, help_line=help_line)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                show_skeleton = not show_skeleton
            if key == ord("m"):
                sound_on = not sound_on
            if key == ord("d"):
                show_debug = not show_debug
            if key == ord("c"):
                if landmarks is None:
                    print("未检测到人体，请入镜后再校准")
                else:
                    size = apparent_size(landmarks)
                    # Prefer a fresh face sample for calibration
                    hp = head_pose
                    if hp is None:
                        # try a few more frames
                        for _ in range(5):
                            fr = cap.read()
                            if fr is None:
                                break
                            fr = cv2.flip(fr, 1)
                            hp = face_est.process(fr)
                            if hp is not None:
                                break
                    posture_calib = capture_posture_calibration(landmarks, hp)
                    baseline = size
                    full_calib = FullCalibration(
                        baseline_size=size, posture=posture_calib
                    )
                    store.save(full_calib)
                    smooth.reset()
                    face_note = (
                        f"pitch0={posture_calib.pitch_deg:.1f}° "
                        f"roll0={posture_calib.roll_deg:.1f}°"
                        if hp
                        else "（未拿到 Face，仅用 Pose 基准）"
                    )
                    print(f"已校准 ✓ distance={size:.4f} {face_note}")
    finally:
        cap.release()
        pose.close()
        hands_est.close()
        face_est.close()
        cv2.destroyAllWindows()
        print(f"日志：{LOG_PATH}")


if __name__ == "__main__":
    main()
