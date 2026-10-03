# Kinect Depth AutoLearn — Linux

This Linux distribution preserves the dataset/training format and tracking improvements of the Windows distribution, and supports two Kinect Xbox 360 acquisition paths: **Kinect Remold (default)** and **OpenKinect/libfreenect (fallback)**.

## Linux architecture

- Default Kinect Xbox 360 source: installed Kinect Remold Linux runtime.
- Remold discovery/control: public Remold SDK socket at `$REMOLD_RUNDIR/kinect360-remold/sdk.sock` (`/run/kinect360-remold/sdk.sock` by default).
- Remold camera stream: per-device Unix socket returned by the SDK.
- Remold bridge: `bridges/bin/remold_bridge`, localhost TCP port `38174` toward Processing.
- OpenKinect fallback: `bridges/bin/openkinect_bridge`, localhost TCP port `38173`.
- Processing UI/tracking: `app/KinectDepthAutoLearn`.
- Training: Python/PyTorch with CPU, NVIDIA CUDA, or AMD ROCm.
- Dataset/checkpoint format: compatible with the Windows distribution.
- Pause/resume, automatic session recovery, periodic checkpoints, ETA, closed-hand tracking retention and hand ROI improvements are preserved.

When Remold is selected, AutoLearn does **not** claim the Kinect USB interfaces itself. Remold remains the hardware owner and AutoLearn communicates through Remold's SDK/camera endpoints.

## Quick start

Make scripts executable and build the bridges:

```bash
chmod +x *.sh bridges/*.sh remold/*.sh
./install_linux_dependencies.sh
./bridges/build_bridges.sh
./install_autolearn.sh auto
```

If you use Remold, install/start your Kinect Remold Xbox 360 Linux runtime first. Verify it with:

```bash
./remold/status.sh
```

Then open in Processing 4:

```text
app/KinectDepthAutoLearn/KinectDepthAutoLearn.pde
```

The application starts on source `1` (Remold).

If Remold is not installed or you want direct OpenKinect access, press `2`. If your distro does not provide libfreenect development files:

```bash
./build_libfreenect_latest.sh
./bridges/build_bridges.sh
```

## Remold communication

The Remold bridge performs this sequence automatically:

1. Connect to `sdk.sock`.
2. Send `LIST` and select the first Kinect whose state is `Ready`.
3. Read the per-device camera socket endpoint from the SDK row.
4. Subscribe to RGB + Depth through scanner ABI v1.
5. Read the 68-byte negotiation reply, including factory depth calibration.
6. Decode Bayer GRBG8 RGB and packed 11-bit depth.
7. Convert depth to millimetres using the Remold-provided calibration.
8. Publish BGRA32 + DepthMm16 to Processing over loopback only.

More detail and control helpers are in `remold/README.md`.

Useful commands:

```bash
./remold/status.sh
./remold/led.sh green
./remold/led.sh red
./remold/led.sh blink-green
./remold/led.sh off
./remold/tilt.sh 5
./remold/tilt.sh 0
```

## GPU selection

Automatic:

```bash
./install_autolearn.sh auto
```

Explicit:

```bash
./install_autolearn.sh cpu
./install_autolearn.sh cuda
./install_autolearn.sh rocm
```

For CUDA/ROCm, the driver/runtime must already support the GPU.

## Training from terminal

Newest dataset automatically:

```bash
./train_depth_pose.sh
```

Specific dataset:

```bash
./train_depth_pose.sh "dataset/session_20261001_200920"
```

Controls:

```bash
./pause_training.sh
./resume_training.sh
./training_status.sh
```

The trainer saves complete periodic checkpoints. A normal pause saves safely, and a restart uses `output/last.pt` automatically.

### Linux local helper architecture

On some Linux desktop/Processing installations, the JVM can read and write files and open localhost sockets but is not allowed to execute child processes. In that situation Java reports `error=13, Permission denied` even for `/bin/bash` or `/usr/bin/python3`; CUDA has not started yet.

The current Linux architecture avoids that path completely. `./install_autolearn.sh` installs a persistent per-user helper under `~/.local/share/kinect-depth-autolearn/` and starts it with a user `systemd` service when available (with a background fallback otherwise). Processing never calls `ProcessBuilder` for training or bridges. It only writes a training request file and communicates with the already-running localhost bridge.

The helper:

- keeps the Remold bridge available on `127.0.0.1:38174`;
- keeps the optional OpenKinect bridge available on `127.0.0.1:38173` when that binary is installed;
- launches the configured persistent CUDA/ROCm/CPU Python for training;
- writes `helper_status.json` heartbeats so the UI can distinguish an offline helper from a Kinect that is still becoming ready.

After replacing or re-extracting the project, rerun the installer once so the persistent helper receives the current bridge/helper files:

```bash
./install_autolearn.sh cuda
```

Use `auto` instead of `cuda` if automatic backend selection is preferred.


## Final model export

After training completes (or early stopping selects the best validation checkpoint), the trainer exports:

```text
output/SynSkeleton.onnx
output/SynHand.onnx
output/SynSkeleton_layout.json
output/dataset_split.json
output/quality_report.json
```

`SynSkeleton.onnx` is the full-frame depth-only student. When enough hand supervision exists, `SynHand.onnx` is trained automatically and the Processing runtime uses it to refine each hand from a high-resolution depth crop. `SynSkeleton_layout.json`, `dataset_split.json`, and `quality_report.json` document layout, leakage-safe temporal splitting, and validation metrics.

## Dataset migration Windows ↔ Linux

Copy the whole session folder, for example:

```text
dataset/session_20261001_200920/
  depth/
  labels.csv
  output/
```

The PNG/CSV format is portable. PyTorch checkpoints are portable at the model/state-dict level; optimizer state is moved to the active device when resumed.

## Application keys

- `A`: start/pause collection
- `X`: stop collection
- `T`: start/resume training
- `P`: pause/resume training
- `R`: create a new dataset session
- `V`: teacher/student view
- `L`: reload latest student model
- `1`: Kinect Xbox 360 via Kinect Remold (default)
- `2`: Kinect Xbox 360 via OpenKinect/libfreenect fallback
- `3`: reserved Kinect v2 source

## Diagnostics

```bash
./diagnose_runtime.sh
./training_status.sh
./remold/status.sh
```

Manual bridge tests:

```bash
./bridges/bin/remold_bridge 38174
./bridges/bin/openkinect_bridge 38173
```

Each bridge waits for the Processing client on localhost.

---

## Linux robustness improvements

### Remold bridge connection-refused loop

The Remold bridge now binds/listens on `127.0.0.1:38174` immediately and accepts Processing before waiting for the Remold device to reach `Ready`. This removes the previous startup window where Processing repeatedly reported `waiting for bridge: connection refused` even though the bridge process was alive and merely waiting for Remold.

The bridge keeps the Processing connection while it waits for Remold, reconnects to the camera endpoint after a Remold/device interruption, and reopens the localhost listener when Processing itself disconnects.

### Training `error=13` / `Permission denied`

Training I/O now retries `progress.json`, `control.json`, `last.pt`, and `best.pt` writes and handles filesystems that allow data writes but reject an atomic rename/replace operation. Telemetry-file failures are non-fatal, so a locked progress file cannot terminate an otherwise healthy training run.

UI-launched training is delegated to the persistent Linux helper, so the confined Processing JVM does not execute Python, `/bin/bash`, or bridge binaries at all. The helper launches the configured persistent CUDA/ROCm/CPU environment outside the JVM confinement. `runtime_python_launcher.sh` remains available for explicit terminal use.

### Closed/partially closed hands

Hand inference runs at a 60 ms cadence and uses the proven Body3D forearm-derived RGB crop and original RTMPose confidence mapping for both hands. Left and right hands use the same detector and thresholds. A post-detection geometry filter rejects impossible finger chains and repairs only isolated landmarks while a real hand is detected in the current frame. If a complete hand is not detected, it is not synthesized, retained, extrapolated, or rendered.

## SynSkeleton v2 quality pipeline

The capture target is 12,000 diverse accepted body samples, with at least 3,500 useful samples for each hand before automatic training starts. The recorder rejects near-duplicate frames using landmark motion, visibility changes, and coarse depth-geometry change rather than simply saving every 100 ms.

The trainer uses 30-second temporal validation groups with a 1.5-second guard by default, synchronized geometric augmentation, hand-rich sample balancing, 160×120 full-frame heatmaps, and local depth-conditioned Z prediction. A second shared hand model trains on 128×128 crops from the original 640×480 depth frames and exports `SynHand.onnx`.
