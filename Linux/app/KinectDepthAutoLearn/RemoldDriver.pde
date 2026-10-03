class DepthFrame {
  final int width=640, height=480;
  final int[] raw=new int[width*height];
  final float[] mm=new float[width*height];
  long sequence, tickMs;
  int flags;
}

class RgbFrame {
  final int width=640, height=480;
  final int[] argb=new int[width*height];
  long sequence, tickMs;
  int flags;
}

class DepthCalibration {
  boolean valid;
  double constShift, emitterDistance, referenceDistance, referencePixelSize;

  float rawToMm(int raw) {
    if (raw<=0 || raw>=2047) return 0;
    if (!valid) {
      double m=0.1236*Math.tan(raw/2842.5+1.1863);
      return (float)(m*1000.0);
    }
    final double parameterCoefficient=4.0, shiftScale=10.0, s2dConstOffset=0.375;
    double fixedRefX=((raw-parameterCoefficient*constShift)/parameterCoefficient)-s2dConstOffset;
    double metric=fixedRefX*referencePixelSize;
    double denom=emitterDistance-metric;
    if (Math.abs(denom)<1e-9) return 0;
    double z=shiftScale*((metric*referenceDistance/denom)+referenceDistance);
    if (!Double.isFinite(z)||z<300||z>10000) return 0;
    return (float)z;
  }
}

// Linux Remold transport is implemented by bridges/bin/remold_bridge.
// The bridge speaks the native Kinect Remold SDK/camera Unix-socket ABI and publishes
// the same localhost RGB+Depth protocol consumed by TcpBridgeRgbdSource.
