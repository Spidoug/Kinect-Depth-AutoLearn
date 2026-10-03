# Kinect Remold communication — Linux

Kinect Depth AutoLearn uses the installed Kinect Remold Xbox 360 runtime directly.

Camera/data path:

1. `bridges/bin/remold_bridge` connects to `$REMOLD_RUNDIR/kinect360-remold/sdk.sock` (default `/run/kinect360-remold/sdk.sock`).
2. It sends the public SDK `LIST` command and selects the first `Ready` Kinect 360.
3. From the SDK row it obtains that device's camera Unix-socket endpoint.
4. It subscribes to `RGB | Depth` using Remold scanner ABI v1.
5. Bayer GRBG8 is demosaiced to BGRA32.
6. packed 11-bit depth is converted to metric millimetres using the factory calibration returned by Remold.
7. Frames are sent only over localhost to Processing on TCP port `38174`.

This means AutoLearn does not take direct USB ownership when source 1/Remold is selected; the Remold runtime remains the hardware owner.

Source keys in Processing:

- `1` — Remold / Kinect 360 (default)
- `2` — OpenKinect/libfreenect fallback
- `3` — reserved Kinect v2 source

Control helpers use the Remold-provided `kinect360-remoldctl` command:

```bash
./remold/status.sh
./remold/led.sh green
./remold/led.sh red
./remold/led.sh off
./remold/tilt.sh 5
```

If the Remold runtime is installed outside PATH, set:

```bash
export REMOLD_CTL=/path/to/kinect360-remoldctl
```

If the runtime directory differs from `/run`, export the parent runtime directory used by Remold:

```bash
export REMOLD_RUNDIR=/your/runtime/root
```
