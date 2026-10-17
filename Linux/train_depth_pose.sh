#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
LAUNCHER="$ROOT/runtime_python_launcher.sh"
[[ -x "$LAUNCHER" ]] || { echo "[ERROR] Runtime launcher missing: $LAUNCHER"; exit 2; }
PY="$LAUNCHER"
echo "============================================================"
echo "Kinect Depth AutoLearn - trainer"
echo "============================================================"
echo "Python launcher: $PY"
if [[ $# -gt 0 && "${1:-}" != --* ]]; then echo "Dataset: $1"; else echo "Dataset: latest valid session"; fi
echo "Checkpoints: every 100 batches + safe pause checkpoint"
echo "Resume: automatic from output/last.pt"
echo "Console: progress / elapsed time / ETA every 15 seconds"
echo "============================================================"
exec "$PY" trainer/train_depth_pose.py "$@" --resume auto --checkpoint-batches 100 --console-seconds 15
