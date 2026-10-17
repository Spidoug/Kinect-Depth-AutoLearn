#!/usr/bin/env bash
set -euo pipefail
if ! command -v apt-get >/dev/null 2>&1; then
  echo "This helper targets Debian/Ubuntu/Mint. Install equivalents manually: build-essential cmake libusb-1.0-dev python3-venv curl wmctrl. libfreenect-dev is optional when using Remold." >&2
  exit 2
fi
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config libusb-1.0-0-dev python3 python3-venv python3-pip curl git wmctrl
if apt-cache show libfreenect-dev >/dev/null 2>&1; then
  sudo apt-get install -y libfreenect-dev || true
fi
echo "Core dependencies installed."
echo "For Remold source 1: install/start Kinect Remold Xbox 360 Linux runtime, then run ./remold/status.sh"
echo "Build AutoLearn bridges with: ./bridges/build_bridges.sh"
echo "Install training runtime with: ./install_autolearn.sh auto"
