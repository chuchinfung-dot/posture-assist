# 坐姿健康提醒（Posture Assist）

用 **laptop 摄像头** 实时监测坐姿，侦测并提醒：

1. **托腮** — 手靠近脸颊 / 下巴过久  
2. **颈部异常** — 颈前倾（头前伸）、头侧倾  
3. **萤幕过近** — 相对校准距离明显凑近  

技术栈：Python · OpenCV · MediaPipe Pose · Streamlit（几何规则 + 持续时间阈值，无需自训模型）。

> **免责声明**：本项目是学习 / 作品集用的 **辅助提醒原型**，**不是医疗器械或医疗诊断**。存在误报与漏报，不能替代专业建议或真人照护。萤幕距离为单目相对估计，不提供精确厘米测距。

## 环境要求

- macOS / Windows / Linux
- Python **3.10 或 3.11**（推荐；系统若是 3.14，请用 conda 建 3.11 环境）
- Laptop 内置摄像头；头、肩、手尽量入镜

## 安装

```bash
cd posture-assist

# 需要 Python 3.10/3.11（系统若是 3.14，请用 conda 3.11 建 venv）
python3.11 -m venv .venv   # 或：/opt/miniconda3/envs/EEE4463_4490_py311/bin/python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash scripts/download_model.sh   # 下载 MediaPipe Pose Landmarker（~5.5MB）
```

> 本仓库使用 **MediaPipe Tasks API**（`PoseLandmarker`），模型文件：`models/pose_landmarker_lite.task`。

## 运行（PyAV 摄像头 + Pygame 窗口，**不使用 OpenCV**）

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -u main.py
```

或双击 `run.command`（macOS 系统终端）。

| 层 | 技术 |
|----|------|
| 采集 | **PyAV**（macOS `avfoundation`） |
| 显示 | **Pygame** |
| 侦测 | **MediaPipe** Pose / Hand / Face |
| 文字叠加 | Pillow |

应用代码已不再调用 OpenCV。环境里可能仍有 `opencv-contrib`（MediaPipe 依赖），但摄像头/窗口/画骨架都唔经 OpenCV。

**第一次请先坐直，按 `c` 校准**，再测托腮 / 凑近 / 颈部。

| 键 | 作用 |
|----|------|
| `c` | 校准距离 + 颈部基准 |
| `s` | 开关骨架/手部点 |
| `m` | 开关提示音 |
| `d` | 开关调试数值 |
| `q` / `Esc` | 退出 |

```bash
python -u main.py --dwell 3 --tilt 10 --pitch 10
```

告警：`data/posture_log.csv`；校准：`data/calibration.json`。

> `app.py`（Streamlit）仍可能引用旧 OpenCV 路径，日常请用 `main.py`。

## 使用提示

| 项目 | 建议 |
|------|------|
| 机位 | 镜头大致平视上半身，光线充足 |
| 托腮 | 手放脸旁多停几秒 |
| 颈前倾 | 头往萤幕方向伸 |
| 侧倾 | 头明显歪向一侧 |
| 距离 | 先校准，再明显凑近 |

侧边栏可调：持续秒数、各项灵敏度、是否提示音、是否画骨架。

## 项目结构

```
posture-assist/
  app.py                  # Streamlit 入口
  requirements.txt
  README.md
  src/
    pose_estimator.py     # MediaPipe Pose
    posture_rules.py      # 托腮、颈前倾、侧倾
    distance_rules.py     # 相对萤幕距离（校准）
    alert.py              # 计时、提示音、CSV 日志
    video_source.py       # 摄像头
  data/
    posture_log.csv       # 运行后生成
    calibration.json      # 校准后生成
```

## 隐私

- 视频帧在本地处理，默认不上传云端  
- 日志仅含时间与问题类型，不含录像  

## 后续可扩展

- MediaPipe Face Mesh 更稳的双眼距离  
- 圆肩、长时间低头  
- 可选桌面通知 / Telegram 提醒（仍建议本地优先）
