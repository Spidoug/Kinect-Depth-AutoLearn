# Kinect Depth AutoLearn

Kinect Depth AutoLearn is a cross-platform Processing + ONNX/PyTorch system for acquiring synchronized Kinect RGB-D data, building structured pose datasets, training a depth-only student model, and running that model back inside the application.

The project uses RGB-based teacher models during data collection and trains a student that consumes metric depth only. The shared representation contains 58 landmarks: 16 body/head landmarks plus 21 landmarks for each hand, together with skeletal segments, endpoints, metric depth, confidence, and structural losses.

## Interface

The Processing frontend uses a panel-based layout designed to keep the live previews visually dominant and diagnostics secondary. RGB Teacher and Metric/Student Depth occupy the main workspace. Dataset progress is grouped below the previews, while Session, Training, Runtime, and Controls live in a dedicated side rail.

The former bottom shortcut strip and duplicated telemetry were removed. Primary commands are now both clickable and available through their keyboard shortcuts. Sensor-source selection is grouped inside the Controls panel instead of being repeated across the screen.

## Project status and scope

The Windows and Linux distributions share the same dataset layout, trainer architecture, model layout, pause/resume checkpoint format, hand/body representation, and depth-only ONNX export. Platform directories contain only the capture/runtime details specific to each operating system.

Current project features include:

- Kinect Xbox 360 and Kinect v2 acquisition paths where supported by the platform.
- Kinect Remold integration.
- Microsoft Kinect SDK 1.8 and SDK 2.0 capture on Windows.
- OpenKinect/libfreenect capture.
- RGB teacher inference for body and hands during dataset creation.
- Metric-depth PNG datasets with CSV labels.
- 58-landmark body + hands model without eye landmarks.
- Five finger chains per hand and structural segment supervision.
- Depth-only PyTorch student training and ONNX export.
- CPU and supported GPU backends.
- Automatic latest-session recovery.
- Periodic full checkpoints containing model, optimizer, scheduler, epoch, batch, validation state, ETA state, and resume position.
- Safe pause/resume from the Processing UI or platform scripts.
- Resilient progress/checkpoint I/O with retry and fallback for transient Windows/Linux permission or rename failures.
- Improved closed-hand and flexed-finger retention.
- Resizable panel-based Processing interface with clickable controls and keyboard shortcuts.

## Directory layout

```text
KinectDepthAutoLearn/
  README.md                   complete project overview
  PROJECT_VERSION
  PROJECT_METADATA.json
  docs/images/                documentation screenshots

  Windows/                    Windows-specific runtime and capture implementation
    README.md
    app/KinectDepthAutoLearn/
    bridges/
    trainer/
    install_autolearn.bat
    train_depth_pose.bat
    pause_training.bat
    resume_training.bat
    training_status.bat
    diagnose_runtime.bat

  Linux/                      Linux-specific runtime and capture implementation
    README.md
    app/KinectDepthAutoLearn/
    bridges/
    remold/
    trainer/
    install_autolearn.sh
    train_depth_pose.sh
    pause_training.sh
    resume_training.sh
    training_status.sh
    diagnose_runtime.sh
```

## Shared application model

The application receives RGB + metric depth from the selected sensor backend. RGB is used only during supervised data acquisition. MoveNet provides body landmarks and RTMPose-Hand provides 21 landmarks per hand. Those 2D teacher landmarks are fused with Kinect metric depth to build a structured 3D/depth training target.

The student network is trained from depth alone. The final exported model is:

```text
SynSkeleton.onnx
```

The export name is fixed and case-sensitive: `SynSkeleton.onnx`. The companion `SynSkeleton_layout.json` documents the landmark order and structural outputs. `SynSkeleton.onnx` consumes metric depth and outputs keypoints, structural segments, and selected endpoints.

## Hand tracking

The hand pipeline uses the proven Body3D RTMPose crop and confidence mapping as its primary RGB detector. Both hands are evaluated with the same geometry and acceptance criteria. Stability is added after inference rather than by making the detector increasingly permissive:

- the original forearm-derived hand crop and RTMPose SimCC logit mapping are used for primary detection;
- both hands are evaluated on the same capture cycle with identical thresholds;
- a light motion-adaptive landmark smoother reduces jitter without delaying deliberate finger motion;
- a hand is rendered only when a coherent hand is detected in the current frame; if the complete hand is absent, it is not synthesized, retained, extrapolated, or rendered;
- isolated landmark outliers are constrained to the current finger chain and may be repaired from the previous palm-relative geometry only while the hand itself remains detected;
- the 3D renderer can display a useful partial hand with four valid landmarks instead of hiding the entire hand;
- hand inference runs at a 60 ms cadence.

Depth resolution remains the physical limitation: individual fingers can be ambiguous at long range, so good hand datasets should include near, medium, open, closed, flexed, rotated, and partially occluded examples.

## Dataset format

A session normally contains:

```text
dataset/
  session_YYYYMMDD_HHMMSS/
    depth/
      0000000.png
      0000001.png
      ...
    labels.csv
    source.txt
    output/
      progress.json
      control.json
      last.pt
      best.pt
      train.log
      SynSkeleton.onnx
      SynHand.onnx
      SynSkeleton_layout.json
      dataset_split.json
      quality_report.json
```

Hands remain optional for an individual body frame, but automatic full-quality training now requires 12,000 accepted diverse body samples plus at least 3,500 useful samples for each hand. Near-duplicate frames are rejected using landmark motion, visibility changes, and a coarse depth-geometry signature; long static holds are sampled only occasionally.

## Training and recovery

SynSkeleton v2 trains the full-frame student for up to 90 epochs with early stopping, then trains the optional high-resolution `SynHand.onnx` crop refiner for up to 45 epochs when sufficient hand data is present. The full-frame model uses 160×120 heatmaps, a dedicated hand head, and local metric-depth sampling for Z. The hand refiner uses 128×128 depth crops and 64×64 hand heatmaps. `last.pt`/`best.pt` and `hand_last.pt`/`hand_best.pt` preserve both phases independently.

Progress and checkpoint writes are now hardened against transient filesystem failures. `progress.json` and `control.json` use retry logic and a non-atomic fallback on filesystems that permit writes but temporarily reject rename/replace operations. Progress telemetry is non-fatal: a momentarily locked status file no longer terminates a long training run. Checkpoint writes also retry and use a direct-save fallback.

Before training begins, the trainer performs a write/replace probe in the session `output` directory. This makes permission, ownership, FUSE/network filesystem, antivirus, and rename-semantics problems visible before substantial compute is spent.

### SynSkeleton v2 data-quality safeguards

Training/validation is no longer split by random neighboring frames. Validation uses contiguous temporal groups with a guard interval, preventing frames a few hundred milliseconds apart from appearing on opposite sides of the split. `dataset_split.json` records the exact policy and split fingerprint.

Geometric augmentation is synchronized with labels: horizontal flip swaps left/right semantics, rotation/scale/translation moves depth and landmarks together, metric-range perturbation updates Z labels consistently, and Kinect-style holes/noise are injected only in the input. Hand-rich samples are automatically rebalanced during full-frame training so body-only frames cannot dominate finger learning.

After export, ONNX Runtime performs a self-test on the generated model before training is reported as complete. `quality_report.json` records body/hand XY and Z validation errors, hand-refiner metrics, and dataset split information.

## Windows

Read `Windows/README.md` for Windows-only setup and capture details.

Supported capture paths:

1. Kinect Remold / Kinect Xbox 360
2. Microsoft Kinect SDK 1.8 / Kinect Xbox 360
3. OpenKinect/libfreenect / Kinect Xbox 360
4. Microsoft Kinect SDK 2.0 / Kinect v2

Typical setup:

```text
install_autolearn.bat
bridges\build_bridges.bat
```

Then open `Windows/app/KinectDepthAutoLearn/KinectDepthAutoLearn.pde` in Processing 4.

Training:

```text
train_depth_pose.bat
pause_training.bat
resume_training.bat
training_status.bat
```

Windows backends can include CPU, NVIDIA CUDA, and AMD DirectML depending on the installed runtime and hardware. The managed AMD/DirectML environment is created with Python 3.10.

## Linux

Read `Linux/README.md` for Linux-only setup and capture details.

Supported capture paths:

1. Kinect Remold / Kinect Xbox 360
2. OpenKinect/libfreenect / Kinect Xbox 360
3. Kinect v2 slot reserved for a future Linux bridge

Typical setup:

```bash
chmod +x *.sh bridges/*.sh remold/*.sh
./install_linux_dependencies.sh
./bridges/build_bridges.sh
./install_autolearn.sh auto
```

The Linux installer also installs/starts a persistent Linux helper. The Processing UI uses files and localhost sockets only; bridge and training subprocesses are launched by that helper, preventing JVM `error=13 / Permission denied` failures on confined desktop installations.

Then open `Linux/app/KinectDepthAutoLearn/KinectDepthAutoLearn.pde` in Processing 4.

The Remold bridge now opens its localhost TCP listener immediately, before waiting for the Remold runtime/device. Processing can therefore connect to the bridge while the Kinect is still becoming Ready instead of repeatedly reporting `connection refused`. If the Processing client disconnects, the bridge reopens the listener and accepts a new client without requiring a manual restart.

On Linux, the Processing UI does not launch Python, `/bin/bash`, or bridge executables. A persistent per-user helper starts the selected CUDA/ROCm/CPU Python runtime and maintains the bridge processes outside the Processing JVM. This avoids Java/Processing `error=13, Permission denied` failures that can occur before Python or CUDA starts. Manual shell helpers remain available for terminal use.

Linux backends can include CPU, NVIDIA CUDA, and AMD ROCm depending on the installed driver/runtime.

## Common application controls

- `A` — start/pause automatic dataset capture
- `X` — stop/pause capture
- `T` — start training or resume from the latest checkpoint
- `P` — pause/resume active training safely
- `R` — create a new dataset session
- `V` — toggle Teacher / Student view
- `L` — reload the newest trained student model

Source-selection keys are platform-specific and documented in the corresponding platform README.

## Persistent runtime

Windows:

```text
%LOCALAPPDATA%\KinectDepthAutoLearn
```

Linux:

```text
${XDG_DATA_HOME:-~/.local/share}/kinect-depth-autolearn
${XDG_CACHE_HOME:-~/.cache}/kinect-depth-autolearn
```

The installers reuse downloaded models, Python environments, and package caches where possible, so replacing the project folder does not normally require rebuilding the full runtime.

## Diagnostics

Windows:

```text
diagnose_runtime.bat
training_status.bat
```

Linux:

```bash
./diagnose_runtime.sh
./training_status.sh
./remold/status.sh
```

The primary project-level training log is:

```text
output/latest_training.log
```

Each session also has its own `output/train.log`.

## Recommended data collection

Prefer diversity over repeated near-identical frames. Include front/side body poses, arms at different heights, hands close to the sensor, fists, half-closed hands, spread fingers, curled fingers, wrist rotation, and both hands at different distances from the torso. A preliminary model should be trained early, tested in Student view, and followed by targeted captures of poses where the student is weak.

## Platform documentation

- `Windows/README.md` — Windows installation, native SDK bridges, DirectML/CUDA, Windows capture paths, and Windows diagnostics.
- `Linux/README.md` — Linux runtime, Remold SDK/socket integration, libfreenect fallback, CUDA/ROCm, Linux launcher behavior, and Linux diagnostics.

### Adaptive tracking stability
Body, hands and fingers use motion-adaptive smoothing. Very small stationary fluctuations are damped; real movements reduce smoothing automatically so gestures remain responsive.

