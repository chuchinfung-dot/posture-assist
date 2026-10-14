#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/models"

download() {
  local url="$1" out="$2"
  if [[ -f "$out" ]]; then
    echo "Exists: $out"
    return
  fi
  echo "Downloading → $out"
  curl -L -o "$out" "$url"
}

download \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" \
  "$ROOT/models/pose_landmarker_lite.task"
download \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" \
  "$ROOT/models/hand_landmarker.task"
download \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
  "$ROOT/models/face_landmarker.task"
ls -lh "$ROOT/models"
