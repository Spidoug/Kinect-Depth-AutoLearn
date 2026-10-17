class PoseLandmark {
  float x,y,score;
  PoseLandmark(float X,float Y,float S){x=X;y=Y;score=S;}
}

class PoseFrame {
  long sourceSequence, receivedMs;
  PoseLandmark[] lm=new PoseLandmark[17];
  float score;
  boolean metricDepthInput=false;
}

class JavaPoseEngine implements Runnable {
  volatile boolean running=false;
  volatile PoseFrame published;
  volatile String message="waiting ONNX...";
  Thread thread;
  RgbdSource source;
  OrtEnvironment env;
  OrtSession session;
  String inputName;
  int inputSize=192;
  long lastSourceSeq=-1;
  long lastInferenceTick=0;
  final int inferenceIntervalMs=100;
  float inferenceMs=0;
  volatile boolean metricDepthMode=false;
  volatile float metricCenterMm=2200;
  PoseFrame temporalPose=null;

  void start(RgbdSource s){source=s;running=true;thread=new Thread(this,"MoveNetJavaPose");thread.setDaemon(true);thread.start();}
  void stop(){running=false;try{if(session!=null)session.close();}catch(Throwable ignored){}try{if(env!=null)env.close();}catch(Throwable ignored){}}
  PoseFrame latest(){return published;}
  boolean isFresh(){PoseFrame p=published;return p!=null&&millis()-p.receivedMs<900;}
  boolean usesMetricDepth(){return metricDepthMode;}
  void toggleInputMode(){metricDepthMode=!metricDepthMode;lastSourceSeq=-1;published=null;message=metricDepthMode?"switching to metric depth...":"switching to RGB...";}
  String inputModeName(){return metricDepthMode?"METRIC DEPTH":"RGB";}
  String status(){return "TEACHER RGB: "+message;}

  public void run(){
    try{
      File model=sketchFile("data/movenet.onnx");
      if(!model.exists()){message="model missing — run install_autolearn.bat";return;}
      env=OrtEnvironment.getEnvironment();
      OrtSession.SessionOptions opt=new OrtSession.SessionOptions();
      opt.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
      session=env.createSession(model.getAbsolutePath(),opt);
      inputName=session.getInputNames().iterator().next();
      NodeInfo ni=session.getInputInfo().get(inputName);
      if(ni!=null&&ni.getInfo() instanceof TensorInfo){long[] sh=((TensorInfo)ni.getInfo()).getShape();if(sh.length==4&&sh[1]>0)inputSize=(int)sh[1];}
      message="MoveNet active • "+inputSize+"x"+inputSize;
      while(running){
        long wait=inferenceIntervalMs-(millis()-lastInferenceTick);
        if(wait>0){sleepMs(Math.min(wait,10L));continue;}
        lastInferenceTick=millis();
        boolean metric=metricDepthMode;
        long t0=System.nanoTime();
        PoseFrame p=null;
        if(metric){
          DepthFrame d=source.latestDepth();
          if(d==null||d.sequence==lastSourceSeq){sleepMs(3L);continue;}
          lastSourceSeq=d.sequence;
          p=inferMetricDepth(d);
        }else{
          RgbFrame r=source.latestRgb();
          if(r==null||r.sequence==lastSourceSeq){sleepMs(3L);continue;}
          lastSourceSeq=r.sequence;
          p=infer(r);
        }
        inferenceMs=(System.nanoTime()-t0)/1_000_000.0f;
        if(p!=null){
          if(metric) p=stabilizePose(p); else temporalPose=null;
          published=p;
          message=String.format(Locale.US,"MoveNet • %.1f ms • conf %.0f%%",inferenceMs,p.score*100.0f);
        }
      }
    }catch(Throwable e){message="ONNX error: "+shortError(e);}
  }

  PoseFrame infer(RgbFrame r)throws Exception{
    long[] shape={1,inputSize,inputSize,3};
    IntBuffer ib=IntBuffer.allocate(inputSize*inputSize*3);
    for(int y=0;y<inputSize;y++){
      int sy=min(r.height-1,(int)((y+0.5f)*r.height/inputSize));
      for(int x=0;x<inputSize;x++){
        int sx=min(r.width-1,(int)((x+0.5f)*r.width/inputSize));
        int c=r.argb[sy*r.width+sx];
        ib.put((c>>16)&255);ib.put((c>>8)&255);ib.put(c&255);
      }
    }
    ib.rewind();
    try(OnnxTensor input=OnnxTensor.createTensor(env,ib,shape)){
      HashMap<String,OnnxTensor> inputs=new HashMap<String,OnnxTensor>();inputs.put(inputName,input);
      try(OrtSession.Result result=session.run(inputs)){
        Object v=result.get(0).getValue();
        float[][][][] out=(float[][][][])v;
        if(out.length<1||out[0].length<1||out[0][0].length<17)return null;
        PoseFrame p=new PoseFrame();p.sourceSequence=r.sequence;p.receivedMs=millis();p.metricDepthInput=false;
        float sum=0;int n=0;
        for(int i=0;i<17;i++){
          float yy=out[0][0][i][0],xx=out[0][0][i][1],sc=out[0][0][i][2];
          p.lm[i]=new PoseLandmark(constrain(xx,0,1),constrain(yy,0,1),constrain(sc,0,1));
          if(i==5||i==6||i==7||i==8||i==9||i==10||i>=11){sum+=sc;n++;}
        }
        p.score=n>0?sum/n:0;return p;
      }
    }
  }
  PoseFrame inferMetricDepth(DepthFrame d)throws Exception{
    metricCenterMm=estimateMetricCenter(d);
    final float nearMm=450.0f, farMm=5000.0f;
    long[] shape={1,inputSize,inputSize,3};
    IntBuffer ib=IntBuffer.allocate(inputSize*inputSize*3);
    for(int y=0;y<inputSize;y++){
      int sy=min(d.height-1,max(0,(int)((y+0.5f)*d.height/inputSize)));
      for(int x=0;x<inputSize;x++){
        int sx=min(d.width-1,max(0,(int)((x+0.5f)*d.width/inputSize)));
        int[] c=metricChannels(d,sx,sy,metricCenterMm,nearMm,farMm);
        ib.put(c[0]);ib.put(c[1]);ib.put(c[2]);
      }
    }
    ib.rewind();
    try(OnnxTensor input=OnnxTensor.createTensor(env,ib,shape)){
      HashMap<String,OnnxTensor> inputs=new HashMap<String,OnnxTensor>();inputs.put(inputName,input);
      try(OrtSession.Result result=session.run(inputs)){
        Object v=result.get(0).getValue();
        float[][][][] out=(float[][][][])v;
        if(out.length<1||out[0].length<1||out[0][0].length<17)return null;
        PoseFrame p=new PoseFrame();p.sourceSequence=d.sequence;p.receivedMs=millis();p.metricDepthInput=true;
        float sum=0;int n=0;
        for(int i=0;i<17;i++){
          float yy=out[0][0][i][0],xx=out[0][0][i][1],sc=out[0][0][i][2];
          p.lm[i]=new PoseLandmark(constrain(xx,0,1),constrain(yy,0,1),constrain(sc,0,1));
          if(i==5||i==6||i==7||i==8||i==9||i==10||i>=11){sum+=sc;n++;}
        }
        p.score=n>0?sum/n:0;return p;
      }
    }
  }

  PoseFrame stabilizePose(PoseFrame cur){
    PoseFrame prev=temporalPose;
    if(prev==null||prev.metricDepthInput!=cur.metricDepthInput){temporalPose=cur;return cur;}
    for(int i=0;i<17;i++){
      PoseLandmark a=cur.lm[i],b=prev.lm[i];if(a==null||b==null)continue;
      float dx=a.x-b.x,dy=a.y-b.y,d=sqrt(dx*dx+dy*dy);
      boolean arm=(i>=5&&i<=10);
      float alpha=arm?map(constrain(a.score,.10f,.85f),.10f,.85f,.18f,.58f):map(constrain(a.score,.10f,.85f),.10f,.85f,.28f,.66f);
      if(d>.12f)alpha=max(alpha,.62f); else if(d<.015f)alpha*=.72f;
      a.x=lerp(b.x,a.x,constrain(alpha,.12f,.78f));
      a.y=lerp(b.y,a.y,constrain(alpha,.12f,.78f));
      a.score=max(a.score,b.score*.72f);
    }
    temporalPose=cur;return cur;
  }

  float estimateMetricCenter(DepthFrame d){
    int[] hist=new int[46];
    int x0=(int)(d.width*.18f),x1=(int)(d.width*.82f),y0=(int)(d.height*.06f),y1=(int)(d.height*.96f);
    for(int y=y0;y<y1;y+=3)for(int x=x0;x<x1;x+=3){
      float mm=d.mm[y*d.width+x];
      if(mm>=500&&mm<5000){int b=constrain((int)(mm/100.0f),5,45);hist[b]++;}
    }
    int best=0;float bestScore=0;
    for(int b=5;b<hist.length;b++){
      float score=hist[b]*(1.0f+constrain((3200-b*100)/5000.0f,0,.42f));
      if(score>bestScore){bestScore=score;best=b;}
    }
    float center=bestScore>12?best*100+50:metricCenterMm;
    return constrain(lerp(metricCenterMm,center,.28f),650,4500);
  }

  int[] metricChannels(DepthFrame d,int x,int y,float center,float nearMm,float farMm){
    float z0=d.mm[y*d.width+x];
    if(z0<nearMm||z0>farMm)return new int[]{0,0,0};
    float norm=constrain((z0-nearMm)/(farMm-nearMm),0,1);
    int distanceTone=constrain(round(255.0f*(1.0f-norm)),0,255);
    float zl=metricSample(d,x-2,y,z0),zr=metricSample(d,x+2,y,z0),zu=metricSample(d,x,y-2,z0),zd=metricSample(d,x,y+2,z0);
    float grad=abs(zl-zr)+abs(zu-zd);
    int edgeTone=constrain(round(grad*.42f),0,255);
    float dz=abs(z0-center);
    int bodyTone=constrain(round(255.0f*(1.0f-dz/900.0f)),0,255);
    bodyTone=constrain(round(bodyTone*.82f+distanceTone*.18f),0,255);
    return new int[]{distanceTone,edgeTone,bodyTone};
  }

  float metricSample(DepthFrame d,int x,int y,float fallback){
    x=constrain(x,0,d.width-1);y=constrain(y,0,d.height-1);float v=d.mm[y*d.width+x];
    return v>=400&&v<=6000?v:fallback;
  }

  void sleepMs(long ms){try{Thread.sleep(ms);}catch(InterruptedException ignored){Thread.currentThread().interrupt();}}
  String shortError(Throwable e){String s=e.getMessage();if(s==null||s.isEmpty())s=e.getClass().getSimpleName();return s.length()>120?s.substring(0,120):s;}
}
class HandLandmark2D {
  float x,y,score;
  HandLandmark2D(float X,float Y,float S){x=X;y=Y;score=S;}
}

class HandPose2D {
  boolean left;
  boolean metricDepthInput=false;
  HandLandmark2D[] lm=new HandLandmark2D[21];
  float score;
  long receivedMs, observedMs, sourceSequence;
  float cropX0,cropY0,cropX1,cropY1;
}

class HandPoseFrame {
  HandPose2D left,right;
  long receivedMs;
}

class SimCCPoint {
  float coord,confidence,peak;
  SimCCPoint(float c,float q,float p){coord=c;confidence=q;peak=p;}
}

class HandPoseEngine implements Runnable {
  volatile boolean running=false;
  volatile HandPoseFrame published;
  volatile String message="optional hand model missing";
  Thread thread;
  RgbdSource source;
  JavaPoseEngine bodyPose;
  OrtEnvironment env;
  OrtSession session;
  String inputName;
  int inputSize=256;
  long lastSourceSeq=-1;
  long lastInferenceTick=0;
  final int inferenceIntervalMs=60;
  float inferenceMs=0;
  boolean inferLeftNext=true;
  HandPose2D lastLeft,lastRight;

  void start(RgbdSource s,JavaPoseEngine bp){source=s;bodyPose=bp;running=true;thread=new Thread(this,"RTMPoseHandJava");thread.setDaemon(true);thread.start();}
  void stop(){running=false;try{if(session!=null)session.close();}catch(Throwable ignored){}}
  HandPoseFrame latest(){return published;}
  boolean isFresh(){HandPoseFrame h=published;return h!=null&&millis()-h.receivedMs<900;}
  String status(){return "JAVA HANDS: "+message;}

  public void run(){
    try{
      File model=sketchFile("data/rtmpose-hand.onnx");
      if(!model.exists()){message="optional model disabled — run install_autolearn.bat";return;}
      env=OrtEnvironment.getEnvironment();
      OrtSession.SessionOptions opt=new OrtSession.SessionOptions();
      opt.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
      session=env.createSession(model.getAbsolutePath(),opt);
      inputName=session.getInputNames().iterator().next();
      message="RTMPose Hand active • 21 points/hand • temporally stabilized";
      while(running){
        long wait=inferenceIntervalMs-(millis()-lastInferenceTick);
        if(wait>0){sleepMs(Math.min(wait,10L));continue;}
        lastInferenceTick=millis();
        PoseFrame p=bodyPose.latest();
        if(p==null||millis()-p.receivedMs>650){
          if(published!=null&&millis()-published.receivedMs>300){published=null;lastLeft=null;lastRight=null;}
          sleepMs(4L);continue;
        }
        boolean metric=p.metricDepthInput;
        long seq=-1;RgbFrame r=null;DepthFrame d=null;
        if(metric){d=source.latestDepth();if(d==null){sleepMs(4L);continue;}seq=d.sequence;}
        else{r=source.latestRgb();if(r==null){sleepMs(4L);continue;}seq=r.sequence;}
        if(seq==lastSourceSeq){sleepMs(4L);continue;}lastSourceSeq=seq;
        long t0=System.nanoTime();
        HandPoseFrame hf=new HandPoseFrame();
        if(metric){
          HandPose2D candL=inferOneMetric(d,p,true);
          HandPose2D candR=inferOneMetric(d,p,false);
          lastLeft=stabilizeHand(candL,lastLeft);
          lastRight=stabilizeHand(candR,lastRight);
        }else{
          HandPose2D candL=inferOne(r,p,true);
          HandPose2D candR=inferOne(r,p,false);
          lastLeft=filterRgbHand(candL,lastLeft,p,true,r.width,r.height);
          lastRight=filterRgbHand(candR,lastRight,p,false,r.width,r.height);
        }
        hf.left=lastLeft; hf.right=lastRight; hf.receivedMs=millis();
        inferenceMs=(System.nanoTime()-t0)/1_000_000.0f;
        published=hf;
        int n=(hf.left!=null?1:0)+(hf.right!=null?1:0);
        message=String.format(Locale.US,"RTMPose Hand [%s] • %d hand(s) • %.1f ms",metric?"DEPTH":"RGB",n,inferenceMs);
      }
    }catch(Throwable e){message="hand ONNX error: "+shortError(e);}
  }

  HandPose2D inferOne(RgbFrame r,PoseFrame p,boolean left)throws Exception{
    int wi=left?9:10, ei=left?7:8;
    PoseLandmark wrist=p.lm[wi], elbow=p.lm[ei];
    if(wrist==null||elbow==null||wrist.score<.20f||elbow.score<.18f)return null;
    float wx=wrist.x*r.width, wy=wrist.y*r.height, ex=elbow.x*r.width, ey=elbow.y*r.height;
    float fore=dist(wx,wy,ex,ey); if(fore<12)return null;
    float dx=wx-ex,dy=wy-ey,m=max(1,sqrt(dx*dx+dy*dy));dx/=m;dy/=m;
    float cx=wx+dx*fore*.20f, cy=wy+dy*fore*.20f;
    float side=constrain(fore*1.55f,72,190);
    float x0=cx-side*.5f,y0=cy-side*.5f,x1=x0+side,y1=y0+side;

    FloatBuffer fb=FloatBuffer.allocate(1*3*inputSize*inputSize);
    float[] mean={123.675f,116.28f,103.53f},std={58.395f,57.12f,57.375f};
    for(int c=0;c<3;c++)for(int y=0;y<inputSize;y++)for(int x=0;x<inputSize;x++){
      float sx=x0+(x+.5f)*side/inputSize, sy=y0+(y+.5f)*side/inputSize;
      int rgb=sampleArgb(r,sx,sy); int val=c==0?((rgb>>16)&255):c==1?((rgb>>8)&255):(rgb&255);
      fb.put((val-mean[c])/std[c]);
    }
    fb.rewind(); long[] shape={1,3,inputSize,inputSize};
    try(OnnxTensor in=OnnxTensor.createTensor(env,fb,shape)){
      HashMap<String,OnnxTensor> inputs=new HashMap<String,OnnxTensor>();inputs.put(inputName,in);
      try(OrtSession.Result result=session.run(inputs)){
        if(result.size()<2)return null;
        float[][][] ox=(float[][][])result.get(0).getValue();
        float[][][] oy=(float[][][])result.get(1).getValue();
        if(ox.length<1||oy.length<1||ox[0].length<21||oy[0].length<21)return null;
        HandPose2D out=new HandPose2D();out.left=left;out.metricDepthInput=false;out.receivedMs=millis();out.observedMs=out.receivedMs;out.sourceSequence=r.sequence;
        out.cropX0=x0/r.width;out.cropY0=y0/r.height;out.cropX1=x1/r.width;out.cropY1=y1/r.height;
        float sum=0;int good=0;
        for(int k=0;k<21;k++){
          int ix=argmax(ox[0][k]), iy=argmax(oy[0][k]);
          float cxp=ix/2.0f,cyp=iy/2.0f;
          float gx=(x0+cxp*side/inputSize)/r.width, gy=(y0+cyp*side/inputSize)/r.height;
          float sc=min(maxVal(ox[0][k]),maxVal(oy[0][k]));
          float conf=1.0f/(1.0f+exp(-sc));
          out.lm[k]=new HandLandmark2D(constrain(gx,0,1),constrain(gy,0,1),conf);
          if(conf>.45f){sum+=conf;good++;}
        }
        out.score=good>0?sum/good:0;
        return good>=8?out:null;
      }
    }
  }

  HandPose2D filterRgbHand(HandPose2D cur,HandPose2D prev,PoseFrame body,boolean left,int frameW,int frameH){
    if(cur==null||!rgbHandRenderSafe(cur,body,left,frameW,frameH))return null;

    enforceHandGeometry(cur,prev,body,left,frameW,frameH);
    if(!rgbHandRenderSafe(cur,body,left,frameW,frameH))return null;
    if(prev==null||prev.left!=cur.left||prev.metricDepthInput)return cur;

    for(int i=0;i<21;i++){
      HandLandmark2D a=cur.lm[i],b=prev.lm[i];if(a==null||b==null)continue;
      float dx=a.x-b.x,dy=a.y-b.y,d=sqrt(dx*dx+dy*dy);
      float alpha=d<.0025f?.24f:d<.008f?.46f:.82f;
      a.x=lerp(b.x,a.x,alpha);a.y=lerp(b.y,a.y,alpha);
      a.score=max(a.score,b.score*.68f);
    }
    return cur;
  }

  void enforceHandGeometry(HandPose2D cur,HandPose2D prev,PoseFrame body,boolean left,int frameW,int frameH){
    if(cur==null||cur.lm[0]==null)return;
    float side=max(1,max((cur.cropX1-cur.cropX0)*frameW,(cur.cropY1-cur.cropY0)*frameH));
    HandLandmark2D root=cur.lm[0];
    HandLandmark2D prevRoot=prev!=null?prev.lm[0]:null;
    int[][] chains={{0,1,2,3,4},{0,5,6,7,8},{0,9,10,11,12},{0,13,14,15,16},{0,17,18,19,20}};

    for(int[] chain:chains){
      for(int n=1;n<chain.length;n++){
        int pid=chain[n-1],id=chain[n];
        HandLandmark2D parent=cur.lm[pid],q=cur.lm[id];
        if(q==null)continue;

        if(parent==null||parent.score<.18f){cur.lm[id]=null;continue;}

        float dx=(q.x-parent.x)*frameW,dy=(q.y-parent.y)*frameH,len=sqrt(dx*dx+dy*dy);
        float maxLen=(n==1?side*.40f:side*.30f);
        boolean impossible=len<side*.006f||len>maxLen;

        boolean temporalJump=false;
        HandLandmark2D old=prev!=null?prev.lm[id]:null;
        if(old!=null&&prevRoot!=null){
          float rx=(q.x-root.x)-(old.x-prevRoot.x);
          float ry=(q.y-root.y)-(old.y-prevRoot.y);
          temporalJump=sqrt(rx*rx+ry*ry)*max(frameW,frameH)>side*.24f;
        }

        if(impossible||temporalJump){
          if(old!=null&&prevRoot!=null&&old.score>=.18f){
            q.x=constrain(root.x+(old.x-prevRoot.x),0,1);
            q.y=constrain(root.y+(old.y-prevRoot.y),0,1);
            q.score=min(q.score,old.score*.72f);
          }else cur.lm[id]=null;
        }
      }
    }
  }

  boolean rgbHandRenderSafe(HandPose2D h,PoseFrame body,boolean left,int frameW,int frameH){
    if(h==null||body==null||h.lm[0]==null)return false;
    int wi=left?9:10,ei=left?7:8;
    PoseLandmark wrist=body.lm[wi],elbow=body.lm[ei];
    if(wrist==null||elbow==null||wrist.score<.20f||elbow.score<.18f)return false;
    float cropW=max(1,(h.cropX1-h.cropX0)*frameW),cropH=max(1,(h.cropY1-h.cropY0)*frameH);
    float side=max(cropW,cropH);
    HandLandmark2D root=h.lm[0];
    float rdx=(root.x-wrist.x)*frameW,rdy=(root.y-wrist.y)*frameH;
    if(sqrt(rdx*rdx+rdy*rdy)>side*.58f)return false;

    int good=0,palmGood=0,saneSeg=0;
    float minx=1e9f,miny=1e9f,maxx=-1e9f,maxy=-1e9f,cx=0,cy=0;
    int[] palm={0,5,9,13,17};
    for(int i=0;i<21;i++){
      HandLandmark2D q=h.lm[i];if(q==null||q.score<.45f)continue;
      float x=q.x*frameW,y=q.y*frameH;
      minx=min(minx,x);maxx=max(maxx,x);miny=min(miny,y);maxy=max(maxy,y);cx+=x;cy+=y;good++;
    }
    for(int id:palm){HandLandmark2D q=h.lm[id];if(q!=null&&q.score>=.45f)palmGood++;}
    if(good<8||palmGood<2)return false;
    cx/=good;cy/=good;
    float cdx=cx-wrist.x*frameW,cdy=cy-wrist.y*frameH;
    if(sqrt(cdx*cdx+cdy*cdy)>side*.72f)return false;
    float span=max(maxx-minx,maxy-miny);
    if(span<side*.045f||span>side*1.02f)return false;

    int[][] chains={{0,1,2,3,4},{0,5,6,7,8},{0,9,10,11,12},{0,13,14,15,16},{0,17,18,19,20}};
    for(int[] chain:chains)for(int j=0;j<chain.length-1;j++){
      HandLandmark2D a=h.lm[chain[j]],b=h.lm[chain[j+1]];
      if(a==null||b==null||a.score<.40f||b.score<.40f)continue;
      float dx=(a.x-b.x)*frameW,dy=(a.y-b.y)*frameH,len=sqrt(dx*dx+dy*dy);
      if(len>=side*.008f&&len<=side*.46f)saneSeg++;
    }
    return saneSeg>=3;
  }


  HandPose2D copyHand(HandPose2D src){
    if(src==null)return null;
    HandPose2D h=new HandPose2D();h.left=src.left;h.metricDepthInput=src.metricDepthInput;
    h.score=src.score;h.receivedMs=src.receivedMs;h.observedMs=src.observedMs;h.sourceSequence=src.sourceSequence;
    h.cropX0=src.cropX0;h.cropY0=src.cropY0;h.cropX1=src.cropX1;h.cropY1=src.cropY1;
    for(int i=0;i<21;i++){HandLandmark2D q=src.lm[i];if(q!=null)h.lm[i]=new HandLandmark2D(q.x,q.y,q.score);}
    return h;
  }

  HandPose2D inferOneMetric(DepthFrame d,PoseFrame p,boolean left)throws Exception{
    HandPose2D a=inferOneMetricPass(d,p,left,false);
    if(handNeedsRetry(a)){
      HandPose2D b=inferOneMetricPass(d,p,left,true);
      return chooseBetterHand(a,b);
    }
    return a;
  }

  HandPose2D inferOneMetricPass(DepthFrame d,PoseFrame p,boolean left,boolean mirrorInput)throws Exception{
    int wi=left?9:10, ei=left?7:8;
    PoseLandmark wrist=p.lm[wi], elbow=p.lm[ei];
    if(wrist==null||elbow==null||wrist.score<.07f||elbow.score<.07f)return null;
    float wx=wrist.x*d.width,wy=wrist.y*d.height,ex=elbow.x*d.width,ey=elbow.y*d.height;
    float fore=dist(wx,wy,ex,ey);if(fore<10)return null;
    float dx=wx-ex,dy=wy-ey,m=max(1,sqrt(dx*dx+dy*dy));dx/=m;dy/=m;
    float cx=wx+dx*fore*.13f,cy=wy+dy*fore*.13f;
    float side=constrain(fore*2.12f,84,250);
    cx=constrain(cx,side*.5f,d.width-side*.5f);
    cy=constrain(cy,side*.5f,d.height-side*.5f);
    float localCenter=localDepthMedian(d,round(wx),round(wy),12);
    if(localCenter<=0)localCenter=bodyPose.metricCenterMm;
    float x0=cx-side*.5f,y0=cy-side*.5f,x1=x0+side,y1=y0+side;

    FloatBuffer fb=FloatBuffer.allocate(1*3*inputSize*inputSize);
    float[] mean={123.675f,116.28f,103.53f},std={58.395f,57.12f,57.375f};
    for(int c=0;c<3;c++)for(int y=0;y<inputSize;y++)for(int x=0;x<inputSize;x++){
      float ix=mirrorInput?(inputSize-1-x):x;
      int sx=constrain(round(x0+(ix+.5f)*side/inputSize),0,d.width-1);
      int sy=constrain(round(y0+(y+.5f)*side/inputSize),0,d.height-1);
      int[] ch=handMetricChannels(d,sx,sy,localCenter);
      fb.put((ch[c]-mean[c])/std[c]);
    }
    fb.rewind();long[] shape={1,3,inputSize,inputSize};
    try(OnnxTensor in=OnnxTensor.createTensor(env,fb,shape)){
      HashMap<String,OnnxTensor> inputs=new HashMap<String,OnnxTensor>();inputs.put(inputName,in);
      try(OrtSession.Result result=session.run(inputs)){
        if(result.size()<2)return null;
        float[][][] ox=(float[][][])result.get(0).getValue();float[][][] oy=(float[][][])result.get(1).getValue();
        if(ox.length<1||oy.length<1||ox[0].length<21||oy[0].length<21)return null;
        HandPose2D out=new HandPose2D();out.left=left;out.metricDepthInput=true;out.receivedMs=millis();out.observedMs=out.receivedMs;out.sourceSequence=d.sequence;
        out.cropX0=x0/d.width;out.cropY0=y0/d.height;out.cropX1=x1/d.width;out.cropY1=y1/d.height;
        float sum=0;int good=0;
        for(int k=0;k<21;k++){
          SimCCPoint sxp=decodeSimCC(ox[0][k]), syp=decodeSimCC(oy[0][k]);
          float cxp=sxp.coord/2.0f,cyp=syp.coord/2.0f;
          if(mirrorInput)cxp=(inputSize-1)-cxp;
          float gx=(x0+cxp*side/inputSize)/d.width,gy=(y0+cyp*side/inputSize)/d.height;
          float conf=sqrt(max(0,sxp.confidence*syp.confidence));
          out.lm[k]=new HandLandmark2D(constrain(gx,0,1),constrain(gy,0,1),conf);
          if(conf>.070f){sum+=conf;good++;}
        }
        out.score=good>0?sum/good:0;
        return (good>=3&&validateHand(out,wrist.x,wrist.y,d.width,d.height,side))?out:null;
      }
    }
  }

  boolean handNeedsRetry(HandPose2D h){
    if(h==null)return true;
    int good=0,palm=0;float sum=0;int[] palmIds={0,5,9,13,17};
    for(int i=0;i<21;i++){HandLandmark2D q=h.lm[i];if(q!=null&&q.score>=.075f){good++;sum+=q.score;}}
    for(int id:palmIds){HandLandmark2D q=h.lm[id];if(q!=null&&q.score>=.070f)palm++;}
    float avg=good>0?sum/good:0;
    return good<9||palm<3||avg<.16f;
  }

  float handEvidence(HandPose2D h){
    if(h==null)return -1000;
    int good=0,palm=0;float sum=0;int[] palmIds={0,5,9,13,17};
    for(int i=0;i<21;i++){HandLandmark2D q=h.lm[i];if(q!=null&&q.score>=.065f){good++;sum+=q.score;}}
    for(int id:palmIds){HandLandmark2D q=h.lm[id];if(q!=null&&q.score>=.065f)palm++;}
    float avg=good>0?sum/good:0;
    return good*1.0f+palm*1.7f+avg*10.0f;
  }

  HandPose2D chooseBetterHand(HandPose2D a,HandPose2D b){
    if(a==null)return b;if(b==null)return a;
    return handEvidence(b)>handEvidence(a)+.20f?b:a;
  }


  SimCCPoint decodeSimCC(float[] logits){
    if(logits==null||logits.length==0)return new SimCCPoint(0,0,0);
    int im=0;float mx=logits[0];
    for(int i=1;i<logits.length;i++)if(logits[i]>mx){mx=logits[i];im=i;}
    double sum=0; for(int i=0;i<logits.length;i++)sum+=Math.exp((double)logits[i]-mx);
    float peak=(float)(1.0/Math.max(1e-12,sum));
    int radius=5,a=max(0,im-radius),b=min(logits.length-1,im+radius);
    double local=0,weighted=0;
    for(int i=a;i<=b;i++){double w=Math.exp((double)logits[i]-mx);local+=w;weighted+=w*i;}
    float coord=local>0?(float)(weighted/local):im;
    float uniform=1.0f/max(1,logits.length);
    float peakRatio=peak/max(1e-6f,uniform);
    float localMass=(float)(local/Math.max(1e-12,sum));
    float quality=constrain((peakRatio-1.10f)/4.25f,0,1);
    quality=constrain(quality*.72f+constrain((localMass-.08f)/.42f,0,1)*.28f,0,1);
    return new SimCCPoint(coord,quality,peak);
  }


  boolean validateHand(HandPose2D h,float wristX,float wristY,int frameW,int frameH,float cropSide){
    if(h==null||h.lm[0]==null)return false;
    HandLandmark2D root=h.lm[0];
    float rootDx=(root.x-wristX)*frameW,rootDy=(root.y-wristY)*frameH;
    if(sqrt(rootDx*rootDx+rootDy*rootDy)>cropSide*.72f)return false;

    float minx=1e9f,miny=1e9f,maxx=-1e9f,maxy=-1e9f;int visible=0,palmVisible=0;
    int[] palm={0,5,9,13,17};
    for(int i=0;i<21;i++){
      HandLandmark2D q=h.lm[i];if(q==null||q.score<.09f)continue;
      float x=q.x*frameW,y=q.y*frameH;minx=min(minx,x);maxx=max(maxx,x);miny=min(miny,y);maxy=max(maxy,y);visible++;
    }
    for(int id:palm){HandLandmark2D q=h.lm[id];if(q!=null&&q.score>.075f)palmVisible++;}
    if(visible<3||palmVisible<2)return false;
    float span=max(maxx-minx,maxy-miny);
    if(span<cropSide*.018f||span>cropSide*1.25f)return false;

    int[][] chains={{0,1,2,3,4},{0,5,6,7,8},{0,9,10,11,12},{0,13,14,15,16},{0,17,18,19,20}};
    int sane=0,totalSeg=0;
    for(int[] chain:chains){
      int validSeg=0;
      for(int j=0;j<chain.length-1;j++){
        HandLandmark2D a=h.lm[chain[j]],b=h.lm[chain[j+1]];
        if(a==null||b==null||a.score<.065f||b.score<.065f)continue;
        float dx=(a.x-b.x)*frameW,dy=(a.y-b.y)*frameH,len=sqrt(dx*dx+dy*dy);
        if(len>=.45f&&len<=cropSide*.55f){validSeg++;totalSeg++;}
      }
      if(validSeg>=1)sane++;
    }
    return sane>=1 || (palmVisible>=3&&totalSeg>=1);
  }

  HandPose2D stabilizeHand(HandPose2D cur,HandPose2D prev){
    if(cur==null)return null;
    if(prev==null||prev.left!=cur.left||prev.metricDepthInput!=cur.metricDepthInput)return cur;
    for(int i=0;i<21;i++){
      HandLandmark2D a=cur.lm[i],b=prev.lm[i];if(a==null||b==null)continue;
      if(a.score<.10f&&b.score>.14f&&millis()-prev.receivedMs<1100){a.x=b.x;a.y=b.y;a.score=b.score*.72f;}
      float dx=a.x-b.x,dy=a.y-b.y,d=sqrt(dx*dx+dy*dy);
      float alpha=map(constrain(a.score,.35f,.85f),.35f,.85f,.20f,.60f);
      if(d<.0025f)alpha=.07f;
      else if(d<.0070f)alpha=lerp(.10f,.24f,(d-.0025f)/.0045f);
      else if(d>.030f)alpha=max(alpha,.58f);
      if(d>.065f)alpha=max(alpha,.68f);
      a.x=lerp(b.x,a.x,constrain(alpha,.07f,.76f));a.y=lerp(b.y,a.y,constrain(alpha,.07f,.76f));
      a.score=max(a.score,b.score*.78f);
    }
    return cur;
  }

  float localDepthMedian(DepthFrame d,int u,int v,int r){
    ArrayList<Float> a=new ArrayList<Float>();
    for(int y=max(0,v-r);y<=min(d.height-1,v+r);y+=2)for(int x=max(0,u-r);x<=min(d.width-1,u+r);x+=2){float z=d.mm[y*d.width+x];if(z>=450&&z<=5000)a.add(z);}
    if(a.size()<3)return 0;Collections.sort(a);return a.get(a.size()/2);
  }

  int[] handMetricChannels(DepthFrame d,int x,int y,float center){
    float z=bodyPose.metricSample(d,x,y,center);
    if(z<450||z>5000)return new int[]{0,0,0};
    float rel=constrain((z-(center-360.0f))/720.0f,0,1);
    int localTone=constrain(round(255.0f*(1.0f-rel)),0,255);
    float zl=bodyPose.metricSample(d,x-1,y,z),zr=bodyPose.metricSample(d,x+1,y,z),zu=bodyPose.metricSample(d,x,y-1,z),zd=bodyPose.metricSample(d,x,y+1,z);
    float grad=abs(zl-zr)+abs(zu-zd);
    int edge=constrain(round(grad*.90f),0,255);
    float dz=abs(z-center);
    int palm=constrain(round(255.0f*(1.0f-dz/300.0f)),0,255);
    palm=constrain(round(palm*.78f+edge*.22f),0,255);
    return new int[]{localTone,edge,palm};
  }

  int sampleArgb(RgbFrame r,float xf,float yf){int x=constrain(round(xf),0,r.width-1),y=constrain(round(yf),0,r.height-1);return r.argb[y*r.width+x];}
  int argmax(float[] a){int bi=0;float bv=-Float.MAX_VALUE;for(int i=0;i<a.length;i++)if(a[i]>bv){bv=a[i];bi=i;}return bi;}
  float maxVal(float[] a){float v=-Float.MAX_VALUE;for(float x:a)if(x>v)v=x;return v;}
  void sleepMs(long ms){try{Thread.sleep(ms);}catch(InterruptedException ignored){Thread.currentThread().interrupt();}}
  String shortError(Throwable e){String s=e.getMessage();if(s==null||s.isEmpty())s=e.getClass().getSimpleName();return s.length()>120?s.substring(0,120):s;}
}
