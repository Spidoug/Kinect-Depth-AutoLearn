#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-green}"
case "$MODE" in off|green|red|blink-green) ;; *) echo "usage: $0 off|green|red|blink-green" >&2; exit 2;; esac
exec "$DIR/remold_control.sh" led "$MODE"
