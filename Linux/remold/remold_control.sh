#!/usr/bin/env bash
set -euo pipefail
CTL="${REMOLD_CTL:-kinect360-remoldctl}"
if ! command -v "$CTL" >/dev/null 2>&1; then
  echo "kinect360-remoldctl was not found in PATH." >&2
  echo "Install/start the Kinect Remold Xbox 360 Linux runtime first, or set REMOLD_CTL=/path/to/kinect360-remoldctl." >&2
  exit 3
fi
exec "$CTL" "$@"
