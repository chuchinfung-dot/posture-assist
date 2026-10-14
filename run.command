#!/bin/bash
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

if [[ ! -d .venv ]]; then
  echo "Missing .venv — see README."
  read -r -p "Press Enter to close..."
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
echo "Starting Posture Assist (PyAV + Pygame, no OpenCV)..."
echo "Allow Camera if macOS asks."
echo ""
python -u main.py
echo ""
read -r -p "Exited. Press Enter to close..."
