#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND="${1:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}/kinect-depth-autolearn"
CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}/kinect-depth-autolearn"
RUNTIME_HOME="$DATA_HOME/runtime"
ASSET_HOME="$DATA_HOME/assets"
mkdir -p "$RUNTIME_HOME" "$ASSET_HOME" "$CACHE_HOME" "$ROOT/output" "$ROOT/app/KinectDepthAutoLearn/code" "$ROOT/app/KinectDepthAutoLearn/data"
command -v "$PYTHON_BIN" >/dev/null || { echo "[ERROR] python3 not found"; exit 2; }
if [[ "$BACKEND" == auto ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then BACKEND=cuda
  elif command -v rocminfo >/dev/null 2>&1 || command -v rocm-smi >/dev/null 2>&1; then BACKEND=rocm
  elif [[ -e /dev/kfd ]] && command -v lspci >/dev/null 2>&1 && lspci 2>/dev/null | grep -Eqi 'AMD|ATI|Radeon'; then BACKEND=rocm
  else BACKEND=cpu; fi
fi
BACKEND="${BACKEND,,}"
case "$BACKEND" in
  cuda) NAME=CUDA; INDEX="${PYTORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}" ;;
  rocm) NAME=ROCM; INDEX="${PYTORCH_ROCM_INDEX:-https://download.pytorch.org/whl/rocm6.3}" ;;
  cpu) NAME=CPU; INDEX="https://download.pytorch.org/whl/cpu" ;;
  *) echo "Usage: $0 [auto|cpu|cuda|rocm]"; exit 2 ;;
esac
VENV="$RUNTIME_HOME/venv-${BACKEND}"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Creating persistent Python environment: $VENV"
  "$PYTHON_BIN" -m venv "$VENV"
fi
PY="$VENV/bin/python"; PIP="$VENV/bin/pip"

JAVA_PY="$PY"

export PIP_CACHE_DIR="$CACHE_HOME/pip"
"$PY" -m pip install -U pip wheel setuptools
"$PIP" install torch --index-url "$INDEX"
"$PIP" install numpy pillow onnx onnxruntime onnxscript
fetch(){ local url="$1" dst="$2" min="$3"; if [[ -f "$dst" ]] && [[ $(stat -c%s "$dst") -ge "$min" ]]; then echo "Using cached $(basename "$dst")"; return; fi; local tmp="$dst.part"; rm -f "$tmp"; curl --fail --location --retry 3 --connect-timeout 30 --max-time 600 --proto "=https" --tlsv1.2 -o "$tmp" "$url"; [[ $(stat -c%s "$tmp") -ge "$min" ]] || { echo "[ERROR] invalid download: $url"; rm -f "$tmp"; exit 3; }; mv "$tmp" "$dst"; }
ORT_VERSION="${ORT_VERSION:-1.22.0}"
ORT_JAR="$ASSET_HOME/onnxruntime-${ORT_VERSION}.jar"
fetch "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime/${ORT_VERSION}/onnxruntime-${ORT_VERSION}.jar" "$ORT_JAR" 1000000
fetch "https://huggingface.co/Xenova/movenet-singlepose-lightning/resolve/ed0f314bb7356fd1dbf1e4f52c2d40791bf6534f/onnx/model.onnx?download=true" "$ASSET_HOME/movenet.onnx" 100000
fetch "https://huggingface.co/tasmulaev/rtmpose-m-distill/resolve/6fe5cf88e76efbef02648f341d75b10b8eb4acfa/onnx/rtmpose-m-distill-256x256.onnx?download=true" "$ASSET_HOME/rtmpose-hand.onnx" 100000
rm -f "$ROOT/app/KinectDepthAutoLearn/code/onnxruntime-"*.jar 2>/dev/null || true
cp -f "$ORT_JAR" "$ROOT/app/KinectDepthAutoLearn/code/"
cp -f "$ASSET_HOME/movenet.onnx" "$ROOT/app/KinectDepthAutoLearn/data/movenet.onnx"
cp -f "$ASSET_HOME/rtmpose-hand.onnx" "$ROOT/app/KinectDepthAutoLearn/data/rtmpose-hand.onnx"
printf '%s\n' "$JAVA_PY" > "$DATA_HOME/runtime_python.txt"
printf '%s\n' "$JAVA_PY" > "$ROOT/runtime_python.txt"
GPU="$($PY - <<'PY'
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
PY
)"
PYVER="$($PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
# Serialize through Python to correctly escape device names and filesystem paths.
RUNTIME_NAME="$NAME" RUNTIME_GPU="$GPU" RUNTIME_PYTHON="$JAVA_PY" \
RUNTIME_PYVER="$PYVER" RUNTIME_DATA_HOME="$DATA_HOME" \
"$PY" - "$DATA_HOME/runtime_status.json" <<'PYJSON'
import json
import os
import sys
from pathlib import Path

status = {
    "version": "1",
    "platform": "linux",
    "backend": os.environ["RUNTIME_NAME"],
    "selection": os.environ["RUNTIME_NAME"],
    "gpu": os.environ["RUNTIME_GPU"],
    "python": os.environ["RUNTIME_PYTHON"],
    "python_version": os.environ["RUNTIME_PYVER"],
    "persistent_root": os.environ["RUNTIME_DATA_HOME"],
}
destination = Path(sys.argv[1])
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_text(json.dumps(status, ensure_ascii=True) + "\n", encoding="utf-8")
temporary.replace(destination)
PYJSON
cp -f "$DATA_HOME/runtime_status.json" "$ROOT/runtime_status.json"
"$PY" "$ROOT/trainer/diagnose_runtime.py"

HELPER_HOME="$DATA_HOME/helper"
BIN_HOME="$DATA_HOME/bin"
mkdir -p "$HELPER_HOME" "$BIN_HOME"
cp -f "$ROOT/helper/autolearn_helper.py" "$HELPER_HOME/autolearn_helper.py"
chmod 755 "$HELPER_HOME/autolearn_helper.py"
if [[ -f "$ROOT/bridges/bin/remold_bridge" ]]; then
  cp -f "$ROOT/bridges/bin/remold_bridge" "$BIN_HOME/remold_bridge"
  chmod 755 "$BIN_HOME/remold_bridge"
fi
if [[ -f "$ROOT/bridges/bin/openkinect_bridge" ]]; then
  cp -f "$ROOT/bridges/bin/openkinect_bridge" "$BIN_HOME/openkinect_bridge"
  chmod 755 "$BIN_HOME/openkinect_bridge"
fi

SYSTEMD_OK=0
if command -v systemctl >/dev/null 2>&1; then
  UNIT_DIR="$HOME/.config/systemd/user"
  UNIT_FILE="$UNIT_DIR/kinect-depth-autolearn-helper.service"
  mkdir -p "$UNIT_DIR"
  SYS_PY="$(command -v python3)"
  cat > "$UNIT_FILE" <<UNIT
[Unit]
Description=Kinect Depth AutoLearn local helper
After=default.target

[Service]
Type=simple
Environment="XDG_DATA_HOME=${XDG_DATA_HOME:-$HOME/.local/share}"
ExecStart="$SYS_PY" "$HELPER_HOME/autolearn_helper.py"
Restart=always
RestartSec=1

[Install]
WantedBy=default.target
UNIT
  if systemctl --user daemon-reload >/dev/null 2>&1 \
     && systemctl --user enable kinect-depth-autolearn-helper.service >/dev/null 2>&1 \
     && systemctl --user restart kinect-depth-autolearn-helper.service >/dev/null 2>&1; then
    SYSTEMD_OK=1
  fi
fi
if [[ "$SYSTEMD_OK" != 1 ]]; then
  if [[ ! -f "$DATA_HOME/helper_status.json" ]] || ! "$PYTHON_BIN" - "$DATA_HOME/helper_status.json" <<'PYHELP' >/dev/null 2>&1
import json,sys,time
try:
 d=json.load(open(sys.argv[1],encoding='utf-8')); raise SystemExit(0 if time.time()*1000-d.get('heartbeat_ms',0)<8000 else 1)
except Exception: raise SystemExit(1)
PYHELP
  then
    nohup "$(command -v python3)" "$HELPER_HOME/autolearn_helper.py" >> "$DATA_HOME/helper-launch.log" 2>&1 &
  fi
fi
sleep 1

echo ""
echo "[OK] Runtime ready: $NAME / $GPU"
echo "Persistent runtime: $DATA_HOME"
echo "Persistent download cache: $CACHE_HOME"
echo "Re-extracting/deleting the project will not remove these dependencies."
echo "Processing renderer: JAVA2D (no JOGL/JogAmp native dependency)."
echo "Local helper: installed and started (bridges + training launcher)."
echo "If you rebuild a bridge later, rerun this installer to refresh the persistent helper copy."
