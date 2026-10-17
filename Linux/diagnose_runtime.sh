#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "Kinect Depth AutoLearn — Linux diagnostics"
echo "Kernel: $(uname -srmo 2>/dev/null || true)"
echo "Python: $(python3 --version 2>&1 || true)"
echo "Processing: $(command -v processing-java 2>/dev/null || echo not-in-PATH)"
echo ""
echo "USB Kinect devices:"
lsusb 2>/dev/null | grep -i -E 'Kinect|Microsoft|Xbox' || echo "  none found / lsusb unavailable"
echo ""
RBASE="${REMOLD_RUNDIR:-/run}/kinect360-remold"
[[ "${REMOLD_RUNDIR:-}" == *kinect360-remold ]] && RBASE="${REMOLD_RUNDIR}"
echo "Remold runtime: $RBASE"
[[ -S "$RBASE/sdk.sock" ]] && echo "  sdk.sock: OK" || echo "  sdk.sock: missing"
[[ -S "$RBASE/control.sock" ]] && echo "  control.sock: OK" || echo "  control.sock: missing"
if command -v kinect360-remoldctl >/dev/null 2>&1; then
  echo "  kinect360-remoldctl: $(command -v kinect360-remoldctl)"
  kinect360-remoldctl status 2>&1 | sed 's/^/  status: /'
else
  echo "  kinect360-remoldctl: not in PATH"
fi
echo ""
[[ -x "$ROOT/bridges/bin/remold_bridge" ]] && echo "Remold bridge: built" || echo "Remold bridge: not built (run ./bridges/build_bridges.sh)"
[[ -x "$ROOT/bridges/bin/openkinect_bridge" ]] && echo "OpenKinect bridge: built" || echo "OpenKinect bridge: not built/optional"
if ldconfig -p 2>/dev/null | grep -q libfreenect; then echo "libfreenect: available"; else echo "libfreenect: not detected (only needed for source 2)"; fi
echo ""
echo "Persistent Linux helper:"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}/kinect-depth-autolearn"
STATUS="$DATA_HOME/helper_status.json"
if [[ -f "$STATUS" ]]; then
  python3 - "$STATUS" <<'PYHELP'
import json,sys,time
p=sys.argv[1]
try:
 d=json.load(open(p,encoding='utf-8')); age=(time.time()*1000-d.get('heartbeat_ms',0))/1000
 print(f"  heartbeat age: {age:.1f}s")
 print(f"  remold bridge: {'running' if d.get('remold_bridge') else 'not running'}")
 print(f"  openkinect bridge: {'running' if d.get('openkinect_bridge') else 'not running'}")
 print(f"  training: {'running' if d.get('training') else 'idle'}")
except Exception as e: print('  invalid helper status:',e)
PYHELP
else
  echo "  helper_status.json: missing (run ./install_autolearn.sh auto or cuda)"
fi
if command -v systemctl >/dev/null 2>&1; then
  systemctl --user --no-pager --full status kinect-depth-autolearn-helper.service 2>/dev/null | sed -n '1,8p' || true
fi
echo ""
echo "Training runtime:"
PTR="$ROOT/runtime_python.txt"
[[ -f "$PTR" ]] || PTR="$DATA_HOME/runtime_python.txt"
if [[ -f "$PTR" ]]; then
  PY="$(head -n1 "$PTR" | tr -d '\r\n')"
  echo "  configured python: $PY"
  if [[ -x "$PY" ]] && "$PY" -c 'import torch,sys; print("  effective python: "+sys.executable); print("  torch: "+torch.__version__); print("  cuda available: "+str(torch.cuda.is_available())); print("  gpu: "+(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))' 2>/dev/null; then
    :
  else
    echo "  runtime check: FAILED"
  fi
else
  echo "  runtime_python.txt: missing (run ./install_autolearn.sh)"
fi
