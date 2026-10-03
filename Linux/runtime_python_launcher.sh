#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}/kinect-depth-autolearn"
PTR="$ROOT/runtime_python.txt"
[[ -f "$PTR" ]] || PTR="$DATA_HOME/runtime_python.txt"
[[ -f "$PTR" ]] || { echo "[ERROR] Runtime not found. Run ./install_autolearn.sh" >&2; exit 2; }
PY="$(head -n1 "$PTR" | tr -d '\r\n')"
[[ -n "$PY" && -x "$PY" ]] || { echo "[ERROR] Configured Python is missing or not executable: $PY" >&2; exit 3; }
exec "$PY" "$@"
