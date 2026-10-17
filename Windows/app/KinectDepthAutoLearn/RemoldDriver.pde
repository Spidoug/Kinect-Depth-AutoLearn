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

class RemoldRgbdSource implements Runnable, RgbdSource {
  static final int REQ_MAGIC=0x43534D52, FRAME_MAGIC=0x46534D52;
  static final int STREAM_RGB=0x1, STREAM_DEPTH=0x4;
  static final int PIXEL_BAYER_GRBG8=4, PIXEL_DEPTH11=6;
  static final int RGB_BYTES=640*480, DEPTH_BYTES=640*480*11/8;

  volatile boolean running, connected;
  volatile String message="searching for driver...";
  volatile DepthFrame depthPublished;
  volatile RgbFrame rgbPublished;
  DepthCalibration calibration=new DepthCalibration();
  Thread thread;
  RandomAccessFile pipe;

  void start(){running=true;thread=new Thread(this,"RemoldRgbdReader");thread.setDaemon(true);thread.start();}
  void stop(){running=false;closePipe();}
  DepthFrame latestDepth(){return depthPublished;}
  RgbFrame latestRgb(){return rgbPublished;}
  String status(){return "REMOLD: "+message;}
  String sourceName(){return "REMOLD";}

  public void run(){
    while(running){
      try{
        String endpoint=findScannerPipe();
        if(endpoint==null){connected=false;message="Kinect Remold not found in devices.tsv";sleepQuiet(700);continue;}
        pipe=new RandomAccessFile(endpoint,"rw");
        sendSubscribe(pipe);
        int accepted=readReply(pipe);
        if((accepted&(STREAM_RGB|STREAM_DEPTH))!=(STREAM_RGB|STREAM_DEPTH)) throw new IOException("driver did not accept RGB+Depth: mask="+accepted);
        connected=true;
        message=calibration.valid?"connected • RGB 640x480 + Depth 640x480 • depth calibration OK":"connected • RGB+Depth • fallback depth calibration";
        streamFrames(pipe);
      }catch(Throwable ex){connected=false;message="reconnecting: "+shortError(ex);}finally{closePipe();}
      sleepQuiet(450);
    }
  }

  String findScannerPipe() throws Exception {
    String pd=System.getenv("ProgramData");
    if(pd==null||pd.trim().isEmpty()) return null;
    File f=new File(new File(new File(new File(pd,"Kinect Remold"),"Runtime"),"kinect-xbox-360-remold"),"devices.tsv");
    if(!f.exists()) return null;
    BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(f),"UTF-8"));
    try{
      String line;
      while((line=br.readLine())!=null){
        if(line.startsWith("#")||line.trim().isEmpty())continue;
        String[] c=line.split("\\t",-1);
        if(c.length>4&&"Ready".equalsIgnoreCase(c[2])&&c[4].startsWith("\\\\.\\pipe\\Kinect360RemoldScanner-")) return c[4];
      }
    }finally{br.close();}
    return null;
  }

  void sendSubscribe(RandomAccessFile p)throws IOException{
    ByteBuffer b=ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
    b.putInt(REQ_MAGIC);b.putInt(1);b.putInt(1);b.putInt(STREAM_RGB|STREAM_DEPTH);p.write(b.array());
  }

  int readReply(RandomAccessFile p)throws IOException{
    byte[] r=new byte[68];p.readFully(r);ByteBuffer b=ByteBuffer.wrap(r).order(ByteOrder.LITTLE_ENDIAN);
    int magic=b.getInt(),version=b.getInt(),result=b.getInt(),mask=b.getInt();
    b.getInt();b.getInt();b.getInt();b.getInt();
    calibration.valid=b.getInt()!=0;
    calibration.constShift=b.getDouble();calibration.emitterDistance=b.getDouble();calibration.referenceDistance=b.getDouble();calibration.referencePixelSize=b.getDouble();
    if(magic!=REQ_MAGIC||version!=1||result!=0)throw new IOException("invalid handshake result="+result);
    return mask;
  }

  void streamFrames(RandomAccessFile p)throws IOException{
    byte[] hdr=new byte[76];byte[] payload=new byte[DEPTH_BYTES];
    while(running){
      p.readFully(hdr);ByteBuffer b=ByteBuffer.wrap(hdr).order(ByteOrder.LITTLE_ENDIAN);
      int magic=b.getInt(),version=b.getInt(),mode=b.getInt(),w=b.getInt(),h=b.getInt(),pixel=b.getInt(),bytes=b.getInt(),flags=b.getInt();
      long frameNo=b.getLong(),tick=b.getLong(); b.position(b.position()+28);
      if(magic!=FRAME_MAGIC||version!=1||bytes<0||bytes>2_000_000)throw new IOException("invalid frame header");
      if(payload.length!=bytes)payload=new byte[bytes];p.readFully(payload);

      if(mode==2&&pixel==PIXEL_DEPTH11&&w==640&&h==480&&bytes==DEPTH_BYTES){
        DepthFrame f=new DepthFrame();f.sequence=frameNo;f.tickMs=tick;f.flags=flags;unpack11(payload,f.raw);
        for(int i=0;i<f.raw.length;i++)f.mm[i]=calibration.rawToMm(f.raw[i]);depthPublished=f;
      } else if(mode==0&&pixel==PIXEL_BAYER_GRBG8&&w==640&&h==480&&bytes==RGB_BYTES){
        RgbFrame f=new RgbFrame();f.sequence=frameNo;f.tickMs=tick;f.flags=flags;demosaicGrbg(payload,f.argb,640,480);rgbPublished=f;
      }
    }
  }

  void demosaicGrbg(byte[] src,int[] dst,int w,int h){
    for(int y=0;y<h;y++)for(int x=0;x<w;x++){
      int c=sample(src,w,h,x,y),r,g,bl;
      boolean ye=(y&1)==0, xe=(x&1)==0;
      if(ye&&xe){g=c;r=avg2(sample(src,w,h,x-1,y),sample(src,w,h,x+1,y));bl=avg2(sample(src,w,h,x,y-1),sample(src,w,h,x,y+1));}
      else if(ye){r=c;g=avg4(sample(src,w,h,x-1,y),sample(src,w,h,x+1,y),sample(src,w,h,x,y-1),sample(src,w,h,x,y+1));bl=avg4(sample(src,w,h,x-1,y-1),sample(src,w,h,x+1,y-1),sample(src,w,h,x-1,y+1),sample(src,w,h,x+1,y+1));}
      else if(xe){bl=c;g=avg4(sample(src,w,h,x-1,y),sample(src,w,h,x+1,y),sample(src,w,h,x,y-1),sample(src,w,h,x,y+1));r=avg4(sample(src,w,h,x-1,y-1),sample(src,w,h,x+1,y-1),sample(src,w,h,x-1,y+1),sample(src,w,h,x+1,y+1));}
      else {g=c;r=avg2(sample(src,w,h,x,y-1),sample(src,w,h,x,y+1));bl=avg2(sample(src,w,h,x-1,y),sample(src,w,h,x+1,y));}
      dst[y*w+x]=0xFF000000|(r<<16)|(g<<8)|bl;
    }
  }
  int sample(byte[] a,int w,int h,int x,int y){x=max(0,min(w-1,x));y=max(0,min(h-1,y));return a[y*w+x]&255;}
  int avg2(int a,int b){return(a+b)>>1;}int avg4(int a,int b,int c,int d){return(a+b+c+d)>>2;}

  void unpack11(byte[] src,int[] dst){
    int si=0,di=0;while(si+10<src.length&&di+7<dst.length){
      int b0=src[si]&255,b1=src[si+1]&255,b2=src[si+2]&255,b3=src[si+3]&255,b4=src[si+4]&255,b5=src[si+5]&255,b6=src[si+6]&255,b7=src[si+7]&255,b8=src[si+8]&255,b9=src[si+9]&255,b10=src[si+10]&255;
      dst[di]=((b0<<3)|(b1>>5))&0x7FF;dst[di+1]=(((b1&0x1F)<<6)|(b2>>2))&0x7FF;dst[di+2]=(((b2&3)<<9)|(b3<<1)|(b4>>7))&0x7FF;dst[di+3]=(((b4&0x7F)<<4)|(b5>>4))&0x7FF;dst[di+4]=(((b5&0x0F)<<7)|(b6>>1))&0x7FF;dst[di+5]=(((b6&1)<<10)|(b7<<2)|(b8>>6))&0x7FF;dst[di+6]=(((b8&0x3F)<<5)|(b9>>3))&0x7FF;dst[di+7]=(((b9&7)<<8)|b10)&0x7FF;si+=11;di+=8;
    }
  }
  void closePipe(){try{if(pipe!=null)pipe.close();}catch(Throwable ignored){}pipe=null;}
  String shortError(Throwable e){String s=e.getMessage();if(s==null||s.isEmpty())s=e.getClass().getSimpleName();return s.length()>100?s.substring(0,100):s;}
  void sleepQuiet(long ms){try{Thread.sleep(ms);}catch(InterruptedException ignored){}}
}
