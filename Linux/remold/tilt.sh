#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
DEG="${1:-0}"
case "$DEG" in -[0-9]*|[0-9]*) ;; *) echo "usage: $0 DEG" >&2; exit 2;; esac
exec "$DIR/remold_control.sh" tilt "$DEG"
