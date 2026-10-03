#!/usr/bin/env bash
set -euo pipefail
R="$(cd "$(dirname "$0")" && pwd)"; cd "$R"
PTR="$R/runtime_python.txt"
[[ -f "$PTR" ]] || PTR="${XDG_DATA_HOME:-$HOME/.local/share}/kinect-depth-autolearn/runtime_python.txt"
[[ -f "$PTR" ]] || { echo "[ERROR] Runtime not found. Run ./install_autolearn.sh"; exit 2; }
PY="$(head -n1 "$PTR" | tr -d '\r\n')"
"$PY" trainer/training_control.py status "$@"
