class StudentDepthFrame {
  long sourceSequence, receivedMs;
  float[][] kp=new float[58][4]; // x,y,z_mm,conf
  float[][] segments; // [segment][x1,y1,z1,x2,y2,z2,conf,length_mm]
  float[][] endpoints; // structural endpoint subset
  float score;
}

class StudentDepthEngine implements Runnable {
  volatile boolean running=false;
  volatile boolean enabled=false;
  volatile StudentDepthFrame published;
  volatile String message="model not loaded yet";
  Thread thread;
  RgbdSource source;
  OrtEnvironment env;
  OrtSession session,handSession;
  String inputName,handInputName;
  File modelFile,handModelFile;
  long lastSeq=-1;
  long lastInferenceTick=0;
  final int inferenceIntervalMs=100;
  float inferenceMs=0;
  int preprocessVersion=1;

  void start(RgbdSource s){ source=s; running=true; thread=new Thread(this,"DepthStudentONNX"); thread.setDaemon(true); thread.start(); }
  void stop(){ running=false; closeSession(); }
  StudentDepthFrame latest(){ return published; }
  String status(){ return "STUDENT DEPTH: "+message; }

  void setEnabled(boolean v){ enabled=v; if(v && session==null) reloadLatest(); }
  void toggle(){ setEnabled(!enabled); }

  synchronized void reloadLatest(){
    closeSession(); published=null; lastSeq=-1;
    try{
      modelFile=findLatestModel();
      if(modelFile==null){ message="SynSkeleton.onnx not found — train a model first"; return; }
      env=OrtEnvironment.getEnvironment();
      OrtSession.SessionOptions opt=new OrtSession.SessionOptions();
      opt.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
      session=env.createSession(modelFile.getAbsolutePath(),opt);
      inputName=session.getInputNames().iterator().next();
      loadModelMetadata();
      loadHandRefiner(opt);
      message="loaded v"+preprocessVersion+(handSession!=null?" + HQ hands":"")+": "+modelFile.getParentFile().getParentFile().getName();
    }catch(Throwable e){ message="error loading ONNX: "+shortErrorStudent(e); closeSession(); }
  }

  File findLatestModel(){
    File root=projectFile("dataset");
    if(!root.exists())return null;
    File[] sessions=root.listFiles(); if(sessions==null)return null;
    File best=null; long bestTime=-1;
    for(File s:sessions){
      if(!s.isDirectory())continue;
      File f=new File(new File(s,"output"),"SynSkeleton.onnx");
      if(f.exists()&&f.lastModified()>bestTime){best=f;bestTime=f.lastModified();}
    }
    // Also accepts a model copied manually to data/SynSkeleton.onnx.
    File manual=sketchFile("data/SynSkeleton.onnx");
    if(manual.exists()&&manual.lastModified()>bestTime)best=manual;
    return best;
  }

  void loadModelMetadata(){
    preprocessVersion=1;
    try{
      File meta=new File(modelFile.getParentFile(),"SynSkeleton_layout.json");
      if(meta.isFile()){JSONObject j=loadJSONObject(meta.getAbsolutePath());if(j!=null&&j.hasKey("preprocess_version"))preprocessVersion=j.getInt("preprocess_version");}
    }catch(Throwable ignored){preprocessVersion=1;}
  }

  void loadHandRefiner(OrtSession.SessionOptions opt){
    handSession=null;handInputName=null;handModelFile=null;
    try{
      File f=new File(modelFile.getParentFile(),"SynHand.onnx");
      if(!f.isFile()&&modelFile.equals(sketchFile("data/SynSkeleton.onnx")))f=sketchFile("data/SynHand.onnx");
      if(!f.isFile())return;
      handSession=env.createSession(f.getAbsolutePath(),opt);handInputName=handSession.getInputNames().iterator().next();handModelFile=f;
    }catch(Throwable e){try{if(handSession!=null)handSession.close();}catch(Throwable ignored){}handSession=null;handInputName=null;handModelFile=null;}
  }

  public void run(){
    while(running){
      try{
        if(!enabled){ Thread.sleep(20); continue; }
        if(session==null){ Thread.sleep(100); continue; }
        long wait=inferenceIntervalMs-(millis()-lastInferenceTick);
        if(wait>0){ Thread.sleep(Math.min(wait,10)); continue; }
        lastInferenceTick=millis();
        DepthFrame d=source.latestDepth();
        if(d==null||d.sequence==lastSeq){ Thread.sleep(3); continue; }
        lastSeq=d.sequence;
        long t0=System.nanoTime();
        StudentDepthFrame f=infer(d);
        inferenceMs=(System.nanoTime()-t0)/1_000_000.0f;
        if(f!=null){ published=f; message=String.format(Locale.US,"%.1f ms • conf %.0f%% • %s",inferenceMs,f.score*100.0f,modelFile.getName()); }
      }catch(Throwable e){ message="inference error: "+shortErrorStudent(e); try{Thread.sleep(200);}catch(Exception ignored){} }
    }
  }

  StudentDepthFrame infer(DepthFrame d)throws Exception{
    final int W=320,H=240;
    FloatBuffer fb=FloatBuffer.allocate(W*H);
    for(int y=0;y<H;y++)for(int x=0;x<W;x++){
      float mm=robustDepthForStudent(d,x,y,W,H);
      float v;
      if(mm>350&&mm<6000)v=constrain((mm-2200.0f)/1800.0f,-1.0f,preprocessVersion>=2?1.8f:1.5f);
      else v=preprocessVersion>=2?-2.0f:0.0f;
      fb.put(v);
    }
    fb.rewind(); long[] shape={1,1,H,W};
    try(OnnxTensor in=OnnxTensor.createTensor(env,fb,shape)){
      HashMap<String,OnnxTensor> inputs=new HashMap<String,OnnxTensor>(); inputs.put(inputName,in);
      try(OrtSession.Result result=session.run(inputs)){
        Object o=result.get(0).getValue();
        float[][][] out=(float[][][])o;
        if(out.length<1||out[0].length<58)return null;
        StudentDepthFrame f=new StudentDepthFrame(); f.sourceSequence=d.sequence; f.receivedMs=millis();
        float sum=0; int n=0;
        for(int i=0;i<58;i++){
          f.kp[i][0]=constrain(out[0][i][0],0,1);
          f.kp[i][1]=constrain(out[0][i][1],0,1);
          f.kp[i][2]=constrain(out[0][i][2],0,6000);
          f.kp[i][3]=constrain(out[0][i][3],0,1);
          if(f.kp[i][3]>.05f){sum+=f.kp[i][3];n++;}
        }
        try{
          if(result.size()>1){ Object so=result.get(1).getValue(); float[][][] seg=(float[][][])so; if(seg.length>0)f.segments=seg[0]; }
          if(result.size()>2){ Object eo=result.get(2).getValue(); float[][][] ep=(float[][][])eo; if(ep.length>0)f.endpoints=ep[0]; }
        }catch(Throwable ignored){}
        f.score=n>0?sum/n:0;
        if(handSession!=null)refineHandsHQ(d,f);
        recomputeStudentScore(f);
        stabilizeStudentFrame(f,published);
        // Render segments from stabilized keypoints so brief finger confidence dips do not
        // make the hand skeleton disappear while the raw model catches up.
        f.segments=null;
        return f;
      }
    }
  }

  void refineHandsHQ(DepthFrame d,StudentDepthFrame f)throws Exception{
    refineOneHandHQ(d,f,true);refineOneHandHQ(d,f,false);
  }

  void refineOneHandHQ(DepthFrame d,StudentDepthFrame f,boolean left)throws Exception{
    int elbow=left?6:7,wrist=left?8:9,base=left?16:37;float[] e=f.kp[elbow],w=f.kp[wrist];
    if(e[3]<.14f||w[3]<.16f)return;
    float ex=e[0]*(d.width-1),ey=e[1]*(d.height-1),wx=w[0]*(d.width-1),wy=w[1]*(d.height-1);float dx=wx-ex,dy=wy-ey,fore=sqrt(dx*dx+dy*dy);if(fore<8)return;dx/=fore;dy/=fore;
    float cx=wx+dx*fore*.18f,cy=wy+dy*fore*.18f,scale=d.width/640.0f,side=constrain(fore*1.60f,72.0f*scale,190.0f*scale),x0=cx-side*.5f,y0=cy-side*.5f;
    final int S=128;FloatBuffer fb=FloatBuffer.allocate(S*S);
    for(int yy=0;yy<S;yy++)for(int xx=0;xx<S;xx++){int u=round(x0+(xx+.5f)*side/S),v=round(y0+(yy+.5f)*side/S);float mm=0;if(u>=0&&u<d.width&&v>=0&&v<d.height)mm=d.mm[v*d.width+u];float q=(mm>350&&mm<6000)?constrain((mm-2200.0f)/1800.0f,-1.0f,1.8f):-2.0f;fb.put(q);}
    fb.rewind();long[] shape={1,1,S,S};
    try(OnnxTensor in=OnnxTensor.createTensor(env,fb,shape)){HashMap<String,OnnxTensor> inputs=new HashMap<String,OnnxTensor>();inputs.put(handInputName,in);try(OrtSession.Result result=handSession.run(inputs)){Object o=result.get(0).getValue();float[][][] out=(float[][][])o;if(out.length<1||out[0].length<21)return;int good=0;float[][] cand=new float[21][4];for(int i=0;i<21;i++){float lx=constrain(out[0][i][0],0,1),ly=constrain(out[0][i][1],0,1),hz=constrain(out[0][i][2],0,6000),hc=constrain(out[0][i][3],0,1);cand[i][0]=constrain((x0+lx*side)/max(1,d.width-1),0,1);cand[i][1]=constrain((y0+ly*side)/max(1,d.height-1),0,1);cand[i][2]=hz;cand[i][3]=hc;if(hc>.12f)good++;}if(good<4)return;for(int i=0;i<21;i++){float[] old=f.kp[base+i],q=cand[i];if(q[3]<.08f&&old[3]>.10f)continue;if(i==0){q[0]=lerp(w[0],q[0],.35f);q[1]=lerp(w[1],q[1],.35f);q[2]=lerp(w[2],q[2],.35f);}float blend=old[3]>.10f?constrain(map(q[3],.08f,.80f,.58f,.92f),.55f,.94f):1.0f;old[0]=lerp(old[0],q[0],blend);old[1]=lerp(old[1],q[1],blend);old[2]=lerp(old[2],q[2],blend);old[3]=max(q[3],old[3]*.55f);}}}
  }

  void recomputeStudentScore(StudentDepthFrame f){float sum=0;int n=0;for(int i=0;i<58;i++)if(f.kp[i][3]>.05f){sum+=f.kp[i][3];n++;}f.score=n>0?sum/n:0;}

  float robustDepthForStudent(DepthFrame d,int ox,int oy,int outW,int outH){
    int x0=constrain((int)floor(ox*d.width/(float)outW),0,d.width-1),x1=constrain((int)floor((ox+1)*d.width/(float)outW)-1,0,d.width-1);
    int y0=constrain((int)floor(oy*d.height/(float)outH),0,d.height-1),y1=constrain((int)floor((oy+1)*d.height/(float)outH)-1,0,d.height-1);
    if(x1<x0)x1=x0;if(y1<y0)y1=y0;
    float a=d.mm[y0*d.width+x0],b=d.mm[y0*d.width+x1],c=d.mm[y1*d.width+x0],e=d.mm[y1*d.width+x1];
    float sum=0,mn=Float.MAX_VALUE,mx=-Float.MAX_VALUE;int n=0;
    if(a>350&&a<6000){sum+=a;mn=min(mn,a);mx=max(mx,a);n++;}
    if(b>350&&b<6000){sum+=b;mn=min(mn,b);mx=max(mx,b);n++;}
    if(c>350&&c<6000){sum+=c;mn=min(mn,c);mx=max(mx,c);n++;}
    if(e>350&&e<6000){sum+=e;mn=min(mn,e);mx=max(mx,e);n++;}
    if(n==0)return 0;if(n<=2)return sum/n;return n==3?sum-mn-mx:(sum-mn-mx)*.5f;
  }

  void stabilizeStudentFrame(StudentDepthFrame cur,StudentDepthFrame prev){
    if(cur==null||prev==null||millis()-prev.receivedMs>650)return;
    for(int i=0;i<58;i++){
      float[] a=cur.kp[i],b=prev.kp[i]; boolean hand=i>=16;
      if(hand && a[3]<.10f && b[3]>.16f){
        a[0]=b[0];a[1]=b[1];a[2]=b[2];a[3]=b[3]*.82f;continue;
      }
      if(a[3]<.06f||b[3]<.06f)continue;
      float dx=a[0]-b[0],dy=a[1]-b[1],dist2=sqrt(dx*dx+dy*dy);
      float alpha=hand?constrain(map(a[3],.12f,.75f,.28f,.68f),.24f,.72f):constrain(map(a[3],.18f,.80f,.34f,.72f),.30f,.76f);
      if(dist2>.075f)alpha=max(alpha,.70f);
      a[0]=lerp(b[0],a[0],alpha);a[1]=lerp(b[1],a[1],alpha);a[2]=lerp(b[2],a[2],min(.78f,alpha+.10f));
      if(hand)a[3]=max(a[3],b[3]*.72f);
    }
  }

  void drawOverlay(int ox,int oy,int w,int h){
    StudentDepthFrame f=published; if(f==null||millis()-f.receivedMs>800)return;
    if(f.segments!=null){
      strokeWeight(max(2,ui.s(3)));
      for(int i=0;i<f.segments.length;i++){
        float[] q=f.segments[i]; if(q.length<8||q[6]<.14f)continue;
        if(i<17) stroke(ui.success);
        else if(i<37) stroke(ui.handSegmentColor((i-17)%20));
        else if(i<57) stroke(ui.handSegmentColor((i-37)%20));
        else stroke(ui.textPrimary);
        line(ox+q[0]*w,oy+q[1]*h,ox+q[3]*w,oy+q[4]*h);
      }
    }else{
      int[][] body={{0,4},{0,5},{1,0},{2,0},{3,0},{4,5},{4,6},{6,8},{5,7},{7,9},{4,10},{5,11},{10,11},{10,12},{12,14},{11,13},{13,15}};
      strokeWeight(max(2,ui.s(3))); stroke(ui.success); for(int[] e:body)drawStudentEdge(f,e[0],e[1],ox,oy,w,h,.20f);
      int[][] fingers={{0,1,2,3,4},{0,5,6,7,8},{0,9,10,11,12},{0,13,14,15,16},{0,17,18,19,20}};
      strokeWeight(max(2,ui.s(2)));
      for(int fi=0;fi<5;fi++){
        stroke(ui.fingerColor(fi)); int[] ids=fingers[fi];
        for(int j=0;j<ids.length-1;j++){
          drawStudentEdge(f,16+ids[j],16+ids[j+1],ox,oy,w,h,.15f);
          drawStudentEdge(f,37+ids[j],37+ids[j+1],ox,oy,w,h,.15f);
        }
      }
      stroke(ui.textPrimary); drawStudentEdge(f,8,16,ox,oy,w,h,.15f); drawStudentEdge(f,9,37,ox,oy,w,h,.15f);
    }
    noStroke();
    for(int i=0;i<58;i++){
      float[] q=f.kp[i]; if(q[3]<(i<16?.18f:.12f))continue;
      if(i<16) fill(ui.success); else fill(ui.handPointColor((i-16)%21));
      float r=i<16?ui.s(9):ui.s(6); if(isStudentEndpoint(i))r+=ui.s(3);
      ellipse(ox+q[0]*w,oy+q[1]*h,r,r);
    }
  }

  boolean isStudentEndpoint(int i){return i==0||i==1||i==2||i==3||i==8||i==9||i==14||i==15||i==20||i==24||i==28||i==32||i==36||i==41||i==45||i==49||i==53||i==57;}

  void drawStudentEdge(StudentDepthFrame f,int ia,int ib,int ox,int oy,int w,int h,float minc){
    float[] a=f.kp[ia],b=f.kp[ib]; if(a[3]<minc||b[3]<minc)return;
    line(ox+a[0]*w,oy+a[1]*h,ox+b[0]*w,oy+b[1]*h);
  }

  synchronized void closeSession(){
    try{if(session!=null)session.close();}catch(Throwable ignored){} session=null;
    try{if(handSession!=null)handSession.close();}catch(Throwable ignored){} handSession=null;handInputName=null;handModelFile=null;
    // OrtEnvironment is a singleton; keep it alive so the teacher is not affected.
  }
  String shortErrorStudent(Throwable e){String s=e.getMessage();if(s==null||s.isEmpty())s=e.getClass().getSimpleName();return s.length()>140?s.substring(0,140):s;}
}
