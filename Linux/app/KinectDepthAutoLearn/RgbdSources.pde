import java.net.*;

interface RgbdSource {
  void start();
  void stop();
  DepthFrame latestDepth();
  RgbFrame latestRgb();
  String status();
  String sourceName();
}

class RgbdSourceHub implements RgbdSource {
  RgbdSource active;
  int sourceId=1;
  volatile String lastSwitchMessage="";

  RgbdSourceHub(){ active=createSource(1); }

  synchronized boolean switchTo(int id){
    if(id<1||id>3)return false;
    if(id==sourceId && active!=null)return true;
    if(active!=null)active.stop();
    sourceId=id;
    active=createSource(id);
    active.start();
    lastSwitchMessage="source changed to "+sourceName();
    return true;
  }

  RgbdSource createSource(int id){
    if(id==1)return new TcpBridgeRgbdSource("REMOLD / KINECT 360",38174,"remold_bridge");
    if(id==2)return new TcpBridgeRgbdSource("OPENKINECT / LIBFREENECT",38173,"openkinect_bridge");
    return new DisabledRgbdSource("KINECT V2", "Linux Kinect v2 source is reserved for a future libfreenect2/Remold bridge");
  }

  void start(){ if(active!=null)active.start(); }
  void stop(){ if(active!=null)active.stop(); }
  DepthFrame latestDepth(){ return active!=null?active.latestDepth():null; }
  RgbFrame latestRgb(){ return active!=null?active.latestRgb():null; }
  String status(){ return active!=null?active.status():"no source"; }
  String sourceName(){ return active!=null?active.sourceName():"NONE"; }
}

class DisabledRgbdSource implements RgbdSource {
  final String label,reason;
  DisabledRgbdSource(String l,String r){label=l;reason=r;}
  void start(){} void stop(){}
  DepthFrame latestDepth(){return null;} RgbFrame latestRgb(){return null;}
  String status(){return label+": "+reason;} String sourceName(){return label;}
}

class TcpBridgeRgbdSource implements Runnable, RgbdSource {
  static final int MAGIC=0x4241444B;
  static final int STREAM_RGB=1, STREAM_DEPTH=2;
  static final int FORMAT_BGRA32=1, FORMAT_DEPTH_MM16=2;
  final String label,exeName; final int port;
  volatile boolean running=false,connected=false;
  volatile String message="starting local bridge...";
  volatile DepthFrame depthPublished;
  volatile RgbFrame rgbPublished;
  Thread thread;
  Socket socket;
  DataInputStream input;

  TcpBridgeRgbdSource(String label,int port,String exeName){this.label=label;this.port=port;this.exeName=exeName;}
  void start(){ if(running)return; running=true; message="waiting for local Linux helper..."; thread=new Thread(this,"RgbdBridge-"+label);thread.setDaemon(true);thread.start(); }
  void stop(){ running=false; closeSocket(); }
  DepthFrame latestDepth(){return depthPublished;}
  RgbFrame latestRgb(){return rgbPublished;}
  String status(){return label+": "+message;}
  String sourceName(){return label;}

  boolean helperOnline(){
    try{
      File f=persistentRuntimeFile("helper_status.json");
      if(!f.isFile())return false;
      JSONObject j=loadJSONObject(f.getAbsolutePath());
      if(j==null)return false;
      long hb=(long)j.getDouble("heartbeat_ms",0);
      return hb>0 && Math.abs(System.currentTimeMillis()-hb)<8000;
    }catch(Throwable ignored){return false;}
  }

  public void run(){
    while(running){
      try{
        socket=new Socket(); socket.setTcpNoDelay(true); socket.connect(new InetSocketAddress("127.0.0.1",port),5000);
        input=new DataInputStream(new BufferedInputStream(socket.getInputStream(),1024*1024)); connected=true; message="bridge connected • waiting for Kinect/Remold frames...";
        stream();
      }catch(Throwable e){
        connected=false;
        if(helperOnline()) message="bridge initializing • waiting for Kinect/Remold...";
        else message="Linux helper offline • run ./install_autolearn.sh auto";
      }
      finally{closeSocket();}
      sleepBridge(500);
    }
  }

  void stream()throws IOException{
    byte[] hdr=new byte[48]; byte[] payload=new byte[640*480*4];
    while(running){
      input.readFully(hdr);
      ByteBuffer b=ByteBuffer.wrap(hdr).order(ByteOrder.LITTLE_ENDIAN);
      int magic=b.getInt(),version=b.getInt(),stream=b.getInt(),w=b.getInt(),h=b.getInt(),format=b.getInt(),bytes=b.getInt(),flags=b.getInt();
      long seq=b.getLong(),tick=b.getLong();
      if(magic!=MAGIC||version!=1||w!=640||h!=480||bytes<0||bytes>640*480*4)throw new IOException("invalid frame");
      if(payload.length<bytes)payload=new byte[bytes]; input.readFully(payload,0,bytes);
      if(stream==STREAM_RGB&&format==FORMAT_BGRA32&&bytes==640*480*4){
        RgbFrame f=new RgbFrame();f.sequence=seq;f.tickMs=tick;f.flags=flags;
        for(int i=0,p=0;i<f.argb.length;i++,p+=4){int bl=payload[p]&255,g=payload[p+1]&255,r=payload[p+2]&255;f.argb[i]=0xFF000000|(r<<16)|(g<<8)|bl;}
        rgbPublished=f; message="connected • aligned RGB+Depth 640x480";
      }else if(stream==STREAM_DEPTH&&format==FORMAT_DEPTH_MM16&&bytes==640*480*2){
        DepthFrame f=new DepthFrame();f.sequence=seq;f.tickMs=tick;f.flags=flags;
        ByteBuffer d=ByteBuffer.wrap(payload,0,bytes).order(ByteOrder.LITTLE_ENDIAN);
        for(int i=0;i<f.mm.length;i++){int mm=d.getShort()&0xFFFF;f.raw[i]=mm;f.mm[i]=(mm>=300&&mm<=10000)?mm:0;}
        depthPublished=f; message="connected • aligned RGB+Depth 640x480";
      }
    }
  }

  void closeSocket(){try{if(input!=null)input.close();}catch(Throwable ignored){}try{if(socket!=null)socket.close();}catch(Throwable ignored){}input=null;socket=null;}
  void sleepBridge(long ms){try{Thread.sleep(ms);}catch(InterruptedException ignored){}}
  String shortErrorBridge(Throwable e){String s=e.getMessage();if(s==null||s.isEmpty())s=e.getClass().getSimpleName();return s.length()>110?s.substring(0,110):s;}
}
