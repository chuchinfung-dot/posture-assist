#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/models"
URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
OUT="$ROOT/models/pose_landmarker_lite.task"
if [[ -f "$OUT" ]]; then
  echo "Model already exists: $OUT"
  exit 0
fi
echo "Downloading Pose Landmarker lite → $OUT"
curl -L -o "$OUT" "$URL"
ls -lh "$OUT"
