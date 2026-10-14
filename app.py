"""
Posture Assist — Streamlit app
Laptop camera → MediaPipe Pose → chin-rest / neck / screen-distance alerts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

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
from src.distance_rules import CalibrationStore, evaluate_distance
from src.pose_estimator import PoseEstimator
from src.posture_rules import evaluate_posture
from src.video_source import VideoSource

DATA_DIR = ROOT / "data"
CALIB_PATH = DATA_DIR / "calibration.json"
LOG_PATH = DATA_DIR / "posture_log.csv"


def _init_state() -> None:
    if "timer" not in st.session_state:
        st.session_state.timer = IssueTimer()
    if "logger" not in st.session_state:
        st.session_state.logger = AlertLogger(LOG_PATH)
    if "calib" not in st.session_state:
        store = CalibrationStore(CALIB_PATH)
        st.session_state.calib_store = store
        st.session_state.baseline_size = store.load()
    if "running" not in st.session_state:
        st.session_state.running = False
    if "last_alerts" not in st.session_state:
        st.session_state.last_alerts = []
    if "pose" not in st.session_state:
        st.session_state.pose = None
    if "cap" not in st.session_state:
        st.session_state.cap = None


def _overlay_status(frame: np.ndarray, messages: list[str], ok: bool) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    bar_h = 36 + 28 * max(len(messages), 1)
    overlay = out.copy()
    color = (40, 160, 70) if ok else (40, 40, 220)
    cv2.rectangle(overlay, (0, 0), (w, bar_h), color, -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)
    if ok:
        cv2.putText(
            out,
            "Posture OK",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        y = 28
        for msg in messages:
            cv2.putText(
                out,
                msg,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 28
    return out


def main() -> None:
    st.set_page_config(page_title="坐姿健康提醒", page_icon="🪑", layout="wide")
    _init_state()

    st.title("坐姿健康提醒")
    st.caption(
        "托腮 · 颈前倾 / 头侧倾 · 萤幕过近｜辅助提醒原型，非医疗诊断。数据仅保存在本机。"
    )

    with st.sidebar:
        st.header("控制")
        camera_index = st.number_input("摄像头编号", min_value=0, max_value=4, value=0, step=1)
        show_skeleton = st.checkbox("显示骨架", value=True)
        sound_on = st.checkbox("告警提示音", value=True)

        st.header("灵敏度")
        dwell = st.slider("持续多少秒才提醒", 3, 30, 10)
        chin_thr = st.slider("托腮灵敏度（越小越严）", 0.15, 0.60, 0.35, 0.01)
        forward_thr = st.slider("颈前倾阈值", 0.05, 0.40, 0.12, 0.01)
        tilt_thr = st.slider("头侧倾阈值", 0.05, 0.40, 0.12, 0.01)
        close_ratio = st.slider("过近比例（相对校准）", 1.05, 2.00, 1.25, 0.01)

        st.header("萤幕距离校准")
        st.write(
            "坐在舒适距离、头肩入镜后，点击校准。"
            "之后若身体在画面中明显变大（凑近），会提醒拉远。"
        )
        calib_btn = st.button("校准当前距离", use_container_width=True)
        if st.session_state.baseline_size:
            st.success(f"已校准 · baseline={st.session_state.baseline_size:.4f}")
        else:
            st.warning("尚未校准 — 过近侦测暂不生效")

        col_a, col_b = st.columns(2)
        with col_a:
            start = st.button("开始", type="primary", use_container_width=True)
        with col_b:
            stop = st.button("停止", use_container_width=True)

        if start:
            st.session_state.running = True
        if stop:
            st.session_state.running = False
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            if st.session_state.pose is not None:
                st.session_state.pose.close()
                st.session_state.pose = None

    left, right = st.columns([2, 1])
    frame_slot = left.empty()
    status_slot = right.empty()
    metrics_slot = right.empty()
    log_slot = right.empty()

    if not st.session_state.running:
        frame_slot.info("点击左侧「开始」，允许浏览器/系统使用摄像头后即可实时监测。")
        with status_slot.container():
            st.subheader("今日累计不良时间（秒）")
            for key, label in ISSUE_LABELS.items():
                secs = st.session_state.timer.today_seconds.get(key, 0.0)
                st.metric(label, f"{secs:.0f}")
        return

    # Lazy-init camera + pose
    if st.session_state.pose is None:
        st.session_state.pose = PoseEstimator(model_complexity=1)
    if st.session_state.cap is None:
        try:
            cap = VideoSource(int(camera_index))
            cap.open()
            st.session_state.cap = cap
        except RuntimeError as exc:
            st.session_state.running = False
            st.error(str(exc))
            return

    cap: VideoSource = st.session_state.cap
    pose: PoseEstimator = st.session_state.pose
    timer: IssueTimer = st.session_state.timer
    logger: AlertLogger = st.session_state.logger

    # Streamlit reruns; loop a batch of frames per run then request rerun
    frame = cap.read()
    if frame is None:
        st.warning("无法读取摄像头画面。")
        st.session_state.running = False
        return

    # Mirror for natural feedback (selfie view)
    frame = cv2.flip(frame, 1)
    landmarks, results = pose.process(frame)

    pending_calib_size = None
    active_msgs: list[str] = []
    flags_debug = {}

    if landmarks is not None:
        posture = evaluate_posture(
            landmarks,
            chin_rest_threshold=chin_thr,
            forward_head_threshold=forward_thr,
            head_tilt_threshold=tilt_thr,
        )
        dist = evaluate_distance(
            landmarks,
            st.session_state.baseline_size,
            close_ratio=close_ratio,
        )
        pending_calib_size = dist.apparent_size

        if calib_btn and pending_calib_size > 0:
            st.session_state.baseline_size = pending_calib_size
            st.session_state.calib_store.save(pending_calib_size)
            st.toast("距离已校准", icon="✅")

        active = {
            "chin_rest": posture.chin_rest,
            "forward_head": posture.forward_head,
            "head_tilt": posture.head_tilt,
            "too_close": dist.too_close,
        }
        newly = timer.update(active, dwell_seconds=float(dwell))
        for key in newly:
            duration = timer.elapsed(key)
            logger.log(key, duration)
            play_beep(sound_on)
            st.session_state.last_alerts.insert(
                0, f"{ISSUE_LABELS[key]} · {duration:.0f}s"
            )
            st.session_state.last_alerts = st.session_state.last_alerts[:8]

        for key, on in active.items():
            if on and key in timer.alerted:
                active_msgs.append(ISSUE_MESSAGES[key])

        flags_debug = {
            "chin_dist": posture.chin_dist_norm,
            "forward": posture.forward_offset,
            "tilt": posture.tilt_ratio,
            "dist_ratio": dist.ratio,
            "apparent": dist.apparent_size,
            "calibrated": dist.calibrated,
        }
    else:
        timer.update(
            {
                "chin_rest": False,
                "forward_head": False,
                "head_tilt": False,
                "too_close": False,
            },
            dwell_seconds=float(dwell),
        )

    if show_skeleton:
        frame = pose.draw(frame, results)
    frame = _overlay_status(frame, active_msgs, ok=len(active_msgs) == 0)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_slot.image(rgb, channels="RGB", use_container_width=True)

    with status_slot.container():
        if active_msgs:
            for m in active_msgs:
                st.error(m)
        else:
            st.success("姿态正常（或尚未达到提醒时长）")

        if flags_debug:
            st.caption(
                f"chin={flags_debug['chin_dist']:.2f} · "
                f"fwd={flags_debug['forward']:.2f} · "
                f"tilt={flags_debug['tilt']:.2f} · "
                f"dist×={flags_debug['dist_ratio']:.2f}"
            )

    with metrics_slot.container():
        st.subheader("今日累计（秒）")
        cols = st.columns(2)
        for i, (key, label) in enumerate(ISSUE_LABELS.items()):
            secs = timer.today_seconds.get(key, 0.0)
            cols[i % 2].metric(label, f"{secs:.0f}")

    with log_slot.container():
        st.subheader("最近告警")
        if st.session_state.last_alerts:
            for line in st.session_state.last_alerts:
                st.write(f"- {line}")
        else:
            st.caption("暂无（不良姿势需持续达到设定秒数）")
        st.caption(f"日志文件：`{LOG_PATH}`")

    # Keep streaming
    st.rerun()


if __name__ == "__main__":
    main()
