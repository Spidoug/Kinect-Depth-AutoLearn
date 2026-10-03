#!/usr/bin/env bash
set -euo pipefail
WORK="${HOME}/.cache/kinect-depth-autolearn/libfreenect"
sudo apt-get update
sudo apt-get install -y git cmake build-essential libusb-1.0-0-dev
rm -rf "$WORK"; mkdir -p "$(dirname "$WORK")"
git clone --depth 1 https://github.com/OpenKinect/libfreenect.git "$WORK"
cmake -S "$WORK" -B "$WORK/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_PYTHON=OFF
cmake --build "$WORK/build" -j"$(nproc 2>/dev/null || echo 2)"
sudo cmake --install "$WORK/build"
if [[ -f "$WORK/platform/linux/udev/51-kinect.rules" ]]; then
  sudo cp "$WORK/platform/linux/udev/51-kinect.rules" /etc/udev/rules.d/51-kinect.rules
  sudo udevadm control --reload-rules
  sudo udevadm trigger
fi
sudo ldconfig
echo "Latest libfreenect installed. Unplug/replug the Kinect, then run ./bridges/build_bridges.sh"
