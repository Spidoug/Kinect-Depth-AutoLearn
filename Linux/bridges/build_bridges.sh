#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build bin
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc 2>/dev/null || echo 2)"
echo "Built bridges:"
[[ -x bin/remold_bridge ]] && echo "  Remold:     $(pwd)/bin/remold_bridge"
[[ -x bin/openkinect_bridge ]] && echo "  OpenKinect: $(pwd)/bin/openkinect_bridge" || echo "  OpenKinect: skipped (libfreenect development files not found)"
