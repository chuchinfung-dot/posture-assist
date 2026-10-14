#!/usr/bin/env python3
"""
坐姿健康提醒 — PyAV 摄像头 + Pygame 窗口（不使用 OpenCV）

侦测：MediaPipe Pose / Hand / Face
采集：PyAV (avfoundation)
显示：Pygame

快捷键：
  c     校准
  s     开关骨架
  m     开关提示音
  d     开关调试
  q/ESC 退出
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pygame
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
    rgb: np.ndarray,
    lines: list[str],
    *,
    mode: str,
    help_line: str,
) -> np.ndarray:
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
    return np.asarray(img)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="坐姿健康提醒（无 OpenCV）")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--dwell", type=float, default=3.0)
    p.add_argument("--chin", type=float, default=0.55)
    p.add_argument("--forward", type=float, default=0.14)
    p.add_argument("--tilt", type=float, default=12.0)
    p.add_argument("--pitch", type=float, default=12.0)
    p.add_argument("--close-ratio", type=float, default=1.35)
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

    def log(msg: str) -> None:
        print(msg, flush=True)

    log("=" * 56)
    log("坐姿健康提醒（PyAV + Pygame，无 OpenCV）")
    log("侦测仍用 MediaPipe Pose / Hand / Face")
    log("1) 坐直  2) 按 c 校准  3) 再测试")
    log("快捷键: c校准  s骨架  m声音  d调试  q退出")
    log("=" * 56)

    pose = PoseEstimator()
    hands_est = HandEstimator()
    face_est = FaceEstimator()
    cap = VideoSource(args.camera)

    try:
        cap.open()
    except RuntimeError as exc:
        log(str(exc))
        pose.close()
        hands_est.close()
        face_est.close()
        sys.exit(1)

    pygame.init()
    pygame.display.set_caption("Posture Assist (no OpenCV)")
    clock = pygame.time.Clock()
    # Create window immediately so it does not look like the app "flashed and quit"
    screen = pygame.display.set_mode((960, 540))
    screen.fill((30, 30, 30))
    font = pygame.font.SysFont(None, 36)
    screen.blit(font.render("Starting camera...", True, (240, 240, 240)), (40, 240))
    pygame.display.flip()

    running = True
    miss_frames = 0
    try:
        while running:
            try:
                frame = cap.read()
            except Exception as exc:  # noqa: BLE001
                log(f"摄像头读取异常（将重试）: {exc}")
                miss_frames += 1
                pygame.event.pump()
                clock.tick(30)
                continue

            if frame is None:
                miss_frames += 1
                if miss_frames % 30 == 1:
                    log("等待摄像头帧…（若一直如此，检查 Camera 权限）")
                    screen.fill((30, 30, 30))
                    screen.blit(
                        font.render("Waiting for camera...", True, (240, 240, 240)),
                        (40, 240),
                    )
                    pygame.display.flip()
                if miss_frames > 300:
                    log("长时间读不到画面，退出")
                    break
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN
                        and event.key in (pygame.K_q, pygame.K_ESCAPE)
                    ):
                        running = False
                clock.tick(30)
                continue
            miss_frames = 0

            # Mirror for selfie view
            frame = np.ascontiguousarray(frame[:, ::-1, :])

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

                chin_on = posture.chin_rest and chin_s < (args.chin + 0.12)
                if posture_calib is not None:
                    tilt_on = tilt_s > args.tilt or posture.head_tilt
                    pitch0 = getattr(posture_calib, "pitch_deg", 0.0)
                    fwd_on = posture.forward_head or ((pitch_s - pitch0) > args.pitch)
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
                    log(
                        f"[告警] {ISSUE_LABELS[key]} · {duration:.0f}s — {ISSUE_MESSAGES[key]}"
                    )

                for key, on in active.items():
                    if not on:
                        continue
                    elapsed = timer.elapsed(key)
                    if key in timer.alerted:
                        alert_lines.append(ISSUE_MESSAGES[key])
                    else:
                        watch_lines.append(
                            f"监测中 {ISSUE_LABELS[key]} {elapsed:.1f}/{args.dwell:.0f}s"
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
                mode, lines = "alert", alert_lines
            elif watch_lines:
                mode, lines = "watch", watch_lines
            else:
                mode, lines = "ok", ["姿态正常"]
            if debug:
                lines = lines + [debug]

            totals = " ".join(
                f"{ISSUE_LABELS[k]}:{timer.today_seconds.get(k, 0):.0f}s"
                for k in ISSUE_LABELS
            )
            help_line = f"[c]校准 [s]骨架 [m]声 [d]调试 [q]退出  {totals}"
            frame = put_overlay(frame, lines, mode=mode, help_line=help_line)

            h, w = frame.shape[:2]
            if screen is None or screen.get_size() != (w, h):
                screen = pygame.display.set_mode((w, h))

            surf = pygame.image.frombuffer(frame.tobytes(), (w, h), "RGB")
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_s:
                        show_skeleton = not show_skeleton
                    elif event.key == pygame.K_m:
                        sound_on = not sound_on
                    elif event.key == pygame.K_d:
                        show_debug = not show_debug
                    elif event.key == pygame.K_c:
                        if landmarks is None:
                            log("未检测到人体，请入镜后再校准")
                        else:
                            size = apparent_size(landmarks)
                            hp = head_pose
                            if hp is None:
                                for _ in range(5):
                                    fr = cap.read()
                                    if fr is None:
                                        break
                                    fr = np.ascontiguousarray(fr[:, ::-1, :])
                                    hp = face_est.process(fr)
                                    if hp is not None:
                                        break
                            posture_calib = capture_posture_calibration(landmarks, hp)
                            baseline = size
                            store.save(
                                FullCalibration(
                                    baseline_size=size, posture=posture_calib
                                )
                            )
                            smooth.reset()
                            face_note = (
                                f"pitch0={posture_calib.pitch_deg:.1f}° "
                                f"roll0={posture_calib.roll_deg:.1f}°"
                                if hp
                                else "（仅 Pose 基准）"
                            )
                            log(f"已校准 ✓ distance={size:.4f} {face_note}")
    except Exception as exc:  # noqa: BLE001
        import traceback

        log(f"程序异常: {exc}")
        traceback.print_exc()
        raise
    finally:
        cap.release()
        pose.close()
        hands_est.close()
        face_est.close()
        pygame.quit()
        log(f"日志：{LOG_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        try:
            input("出错了。按 Enter 关闭…")
        except EOFError:
            pass
        sys.exit(1)
