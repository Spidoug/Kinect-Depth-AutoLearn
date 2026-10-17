class DatasetRecorder {
  final int BODY_KP=16, HAND_KP=21, KP=58;
  boolean collecting=false, training=false, trainingPaused=false;
  Process trainProcess=null;
  long lastProgressPoll=0;
  float trainProgress=0,trainLoss=0,trainVal=0,trainBest=0;
  int trainEpoch=0,trainEpochs=0,trainBatch=0,trainBatches=0;
  String trainState="idle",trainDevice="",trainGpu="",trainEta="--:--:--",trainElapsed="00:00:00";
  long lastSavedMs=0,lastDepthSeq=-1;
  int accepted=0,rejected=0,target=12000,nextSampleIndex=0;
  int targetHandEach=3500; // minimum useful supervision per left/right hand before auto-training
  int lastBodyGood=0,lastHandGood=0,acceptedWithHands=0,acceptedBodyOnly=0,acceptedLeftHand=0,acceptedRightHand=0;
  int noveltyRejected=0;
  int minIntervalMs=80; // fast polling; novelty gate below decides whether a frame is worth storing
  int maxDuplicateHoldMs=2200;
  final int SIG_W=20,SIG_H=15;
  float[] lastDepthSignature=new float[SIG_W*SIG_H];
  boolean haveDepthSignature=false;
  File sessionDir,depthDir,manifestFile;
  PrintWriter manifest;
  float[][] lastAccepted=new float[KP][4];
  String message="press A to start";
  String captureSource="REMOLD";

  DatasetRecorder(){ recoverLatestOrNew(); }

  void recoverLatestOrNew(){
    try{
      File root=projectFile("dataset"); root.mkdirs();
      File[] dirs=root.listFiles(new FileFilter(){public boolean accept(File f){return f.isDirectory()&&f.getName().startsWith("session_")&&new File(f,"labels.csv").isFile();}});
      if(dirs==null||dirs.length==0){newSession();return;}
      Arrays.sort(dirs,new Comparator<File>(){public int compare(File a,File b){return b.getName().compareTo(a.getName());}});
      for(File d:dirs){ if(loadSession(d)){message="recovered session • "+accepted+" samples • "+d.getName();return;} }
    }catch(Exception ignored){}
    newSession();
  }

  boolean loadSession(File dir){
    closeManifest();
    try{
      File mf=new File(dir,"labels.csv"),dd=new File(dir,"depth"); if(!mf.isFile()||!dd.isDirectory())return false;
      int count=0,maxIdx=-1,hands=0,bodyOnly=0,leftHands=0,rightHands=0; String last=null;
      BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(mf),"UTF-8"));
      String line=br.readLine();
      while((line=br.readLine())!=null){
        if(line.trim().length()==0)continue; String[] c=line.split(",",-1); if(c.length<3+KP*4)continue;
        File df=new File(dir,c[0].replace("/",File.separator)); if(!df.isFile())continue;
        count++; last=line; String fn=df.getName(); int dot=fn.lastIndexOf('.');
        try{int n=Integer.parseInt(dot>0?fn.substring(0,dot):fn);maxIdx=max(maxIdx,n);}catch(Exception ignored){}
        int hg=0,hgl=0,hgr=0; for(int i=BODY_KP;i<KP;i++){try{if(Float.parseFloat(c[3+i*4+3])>.14f){hg++;if(i<BODY_KP+HAND_KP)hgl++;else hgr++;}}catch(Exception ignored){}}
        if(hgl>=4)leftHands++; if(hgr>=4)rightHands++;
        if(hgl>=4||hgr>=4)hands++; else bodyOnly++;
      }
      br.close();
      sessionDir=dir; depthDir=dd; manifestFile=mf; accepted=count; nextSampleIndex=maxIdx+1; rejected=0; noveltyRejected=0; acceptedWithHands=hands; acceptedBodyOnly=bodyOnly; acceptedLeftHand=leftHands; acceptedRightHand=rightHands; haveDepthSignature=false; collecting=false; training=false; trainingPaused=false;
      if(last!=null){
        try{String[] c=last.split(",",-1);for(int i=0;i<KP;i++){lastAccepted[i][0]=Float.parseFloat(c[3+i*4]);lastAccepted[i][1]=Float.parseFloat(c[3+i*4+1]);lastAccepted[i][2]=Float.parseFloat(c[3+i*4+2]);lastAccepted[i][3]=Float.parseFloat(c[3+i*4+3]);} lastBodyGood=0;lastHandGood=0;for(int i=0;i<BODY_KP;i++)if(lastAccepted[i][3]>.18f)lastBodyGood++;for(int i=BODY_KP;i<KP;i++)if(lastAccepted[i][3]>.14f)lastHandGood++;}catch(Exception ignored){}
      }
      File src=new File(dir,"source.txt"); if(src.isFile()){try{captureSource=readFirstLine(src);}catch(Exception ignored){}}
      manifest=new PrintWriter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream(manifestFile,true),"UTF-8")));
      loadExistingTrainingProgress();
      return true;
    }catch(Exception e){message="session recovery failed: "+e.getMessage();return false;}
  }

  void loadExistingTrainingProgress(){
    try{
      File f=new File(new File(sessionDir,"output"),"progress.json"); if(!f.isFile())return; JSONObject j=loadJSONObject(f.getAbsolutePath()); if(j==null)return;
      trainState=j.getString("state",trainState); trainProgress=(float)j.getFloat("progress",trainProgress); trainLoss=(float)j.getFloat("loss",trainLoss); trainVal=(float)j.getFloat("val",trainVal); trainBest=(float)j.getFloat("best",trainBest); trainEpoch=j.getInt("epoch",trainEpoch); trainEpochs=j.getInt("epochs",trainEpochs); trainBatch=j.getInt("batch",trainBatch); trainBatches=j.getInt("batches",trainBatches); trainEta=j.getString("eta",trainEta); trainElapsed=j.getString("elapsed",trainElapsed); trainingPaused="paused".equals(trainState)||"pausing".equals(trainState);
    }catch(Exception ignored){}
  }

  void setCaptureSource(String name){captureSource=(name==null||name.length()==0)?"UNKNOWN":name;}

  void newSession(){
    closeManifest();
    String stamp=new java.text.SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
    sessionDir=new File(projectFile("dataset"),"session_"+stamp);
    depthDir=new File(sessionDir,"depth"); depthDir.mkdirs();
    new File(sessionDir,"output").mkdirs();
    projectFile("output").mkdirs();
    manifestFile=new File(sessionDir,"labels.csv");
    try{
      manifest=new PrintWriter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream(manifestFile),"UTF-8")));
      manifest.print("file,sequence,tick_ms");
      for(int i=0;i<KP;i++)manifest.print(",x"+i+",y"+i+",z"+i+",s"+i);
      manifest.println(); manifest.flush();
      accepted=0;nextSampleIndex=0;rejected=0;noveltyRejected=0;acceptedWithHands=0;acceptedBodyOnly=0;acceptedLeftHand=0;acceptedRightHand=0;lastBodyGood=0;lastHandGood=0;haveDepthSignature=false;collecting=false;training=false;trainingPaused=false;trainProgress=0;trainState="idle";message="new session • source "+captureSource+" • "+sessionDir.getName();
      try{PrintWriter meta=new PrintWriter(new OutputStreamWriter(new FileOutputStream(new File(sessionDir,"source.txt")),"UTF-8"));meta.println(captureSource);meta.close();}catch(Exception ignored){}
    }catch(Exception e){ message="error creating dataset: "+e.getMessage(); }
  }

  void toggle(){ collecting=!collecting; message=collecting?"collecting automatically...":"capture paused"; }
  void stopCollecting(){ collecting=false; message="session closed with "+accepted+" samples"; }

  void update(RgbFrame r,DepthFrame d,PoseFrame p,HandPoseFrame h,BodyTracker3D tr){
    if(!collecting||training||r==null||d==null||p==null)return;
    if(d.sequence==lastDepthSeq||millis()-lastSavedMs<minIntervalMs)return;
    if(abs((long)r.tickMs-(long)d.tickMs)>90){ rejected++; return; }
    if(millis()-p.receivedMs>500||p.score<.28f||!tr.tracked||tr.goodJointCount()<10){ rejected++; return; }
    float[][] kp=buildLabels(tr,p);
    int bodyGood=0,handGood=0,leftGood=0,rightGood=0;
    for(int i=0;i<BODY_KP;i++)if(kp[i][3]>.18f)bodyGood++;
    for(int i=BODY_KP;i<KP;i++)if(kp[i][3]>.14f){handGood++;if(i<BODY_KP+HAND_KP)leftGood++;else rightGood++;}
    lastBodyGood=bodyGood; lastHandGood=handGood;
    if(bodyGood<10){ rejected++; return; }
    boolean hasLeftHand=leftGood>=4,hasRightHand=rightGood>=4,hasUsefulHand=hasLeftHand||hasRightHand;
    if(!isNovelSample(kp,d,hasLeftHand,hasRightHand)){rejected++;noveltyRejected++;return;}
    try{
      String name=String.format(Locale.US,"%07d.png",nextSampleIndex);
      saveDepthPng(d,new File(depthDir,name));
      manifest.print("depth/"+name+","+d.sequence+","+d.tickMs);
      for(int i=0;i<KP;i++) manifest.print(String.format(Locale.US,",%.7f,%.7f,%.2f,%.4f",kp[i][0],kp[i][1],kp[i][2],kp[i][3]));
      manifest.println(); if((accepted%20)==0)manifest.flush();
      lastAccepted=kp; accepted++; nextSampleIndex++; if(hasUsefulHand)acceptedWithHands++; else acceptedBodyOnly++; if(hasLeftHand)acceptedLeftHand++; if(hasRightHand)acceptedRightHand++; rememberDepthSignature(d); lastSavedMs=millis(); lastDepthSeq=d.sequence;
      int needL=max(0,targetHandEach-acceptedLeftHand),needR=max(0,targetHandEach-acceptedRightHand);
      message="collecting diverse poses • body "+bodyGood+"/16 • hands "+leftGood+"L/"+rightGood+"R • hand quota L "+acceptedLeftHand+" R "+acceptedRightHand;
      if(accepted>=target && acceptedLeftHand>=targetHandEach && acceptedRightHand>=targetHandEach){ collecting=false; startTraining(); }
      else if(accepted>=target && (needL>0||needR>0))message="body target reached • keep showing varied hands • need L "+needL+" / R "+needR;
    }catch(Exception e){ collecting=false; message="save error: "+e.getMessage(); }
  }

  float[][] buildLabels(BodyTracker3D tr,PoseFrame pose){
    float[][] out=new float[KP][4];
    putProjected(out,0,tr.j.get("head"));
    putProjected(out,1,tr.projectPoseFace(pose,0,12));
    putProjected(out,2,tr.projectPoseFace(pose,3,12));
    putProjected(out,3,tr.projectPoseFace(pose,4,12));
    String[] body={"shoulderL","shoulderR","elbowL","elbowR","wristL","wristR","hipL","hipR","kneeL","kneeR","ankleL","ankleR"};
    for(int i=0;i<body.length;i++)putProjected(out,4+i,tr.j.get(body[i]));
    if(tr.handL!=null&&tr.handL.valid)for(int i=0;i<HAND_KP;i++)putProjected(out,BODY_KP+i,tr.handL.k[i]);
    if(tr.handR!=null&&tr.handR.valid)for(int i=0;i<HAND_KP;i++)putProjected(out,BODY_KP+HAND_KP+i,tr.handR.k[i]);
    return out;
  }

  void putProjected(float[][] out,int idx,Joint3D q){
    if(q==null||!q.valid||q.confidence<.10f||q.p.z<400||q.p.z>6000)return;
    float u=q.p.x*594.2f/q.p.z+339.3f, v=q.p.y*591.0f/q.p.z+242.7f;
    if(u<0||u>=640||v<0||v>=480)return;
    out[idx][0]=u/639.0f; out[idx][1]=v/479.0f; out[idx][2]=q.p.z; out[idx][3]=constrain(q.confidence,0,1);
  }

  void saveDepthPng(DepthFrame d,File file)throws Exception{
    BufferedImage img=new BufferedImage(640,480,BufferedImage.TYPE_USHORT_GRAY);
    short[] px=((DataBufferUShort)img.getRaster().getDataBuffer()).getData();
    for(int i=0;i<px.length;i++){ int z=(d.mm[i]>=350&&d.mm[i]<=10000)?round(d.mm[i]):0; px[i]=(short)(z&0xFFFF); }
    ImageIO.write(img,"png",file);
  }

  boolean isNovelSample(float[][] kp,DepthFrame d,boolean hasLeftHand,boolean hasRightHand){
    if(accepted==0||lastSavedMs==0)return true;
    float bodyMove=0,bodyMax=0,bodyZ=0;int bodyN=0,visibilityChanges=0;
    float handMove=0,handMax=0,handZ=0;int handN=0;
    for(int i=0;i<KP;i++){
      float[] a=kp[i],b=lastAccepted[i];boolean va=a[3]>(i<BODY_KP?.18f:.14f),vb=b[3]>(i<BODY_KP?.18f:.14f);
      if(va!=vb)visibilityChanges++; if(!va||!vb)continue;
      float dx=a[0]-b[0],dy=a[1]-b[1],mv=sqrt(dx*dx+dy*dy),dz=abs(a[2]-b[2]);
      if(i<BODY_KP){bodyMove+=mv;bodyMax=max(bodyMax,mv);bodyZ+=dz;bodyN++;}
      else {handMove+=mv;handMax=max(handMax,mv);handZ+=dz;handN++;}
    }
    bodyMove/=max(1,bodyN);bodyZ/=max(1,bodyN);handMove/=max(1,handN);handZ/=max(1,handN);
    boolean poseNovel=bodyMove>.0065f||bodyMax>.018f||bodyZ>22.0f||visibilityChanges>=3;
    boolean handNovel=(hasLeftHand||hasRightHand)&&(handMove>.0075f||handMax>.024f||handZ>16.0f);
    boolean geometryNovel=depthGeometryNovel(d);
    boolean keepAlive=millis()-lastSavedMs>=maxDuplicateHoldMs;
    return poseNovel||handNovel||geometryNovel||keepAlive;
  }

  boolean depthGeometryNovel(DepthFrame d){
    if(!haveDepthSignature)return true;int common=0,changed=0;float changedSum=0;
    for(int gy=0;gy<SIG_H;gy++)for(int gx=0;gx<SIG_W;gx++){
      int idx=gy*SIG_W+gx;float cur=depthSignatureValue(d,gx,gy),prev=lastDepthSignature[idx];
      if(cur<400||prev<400)continue;common++;float dif=abs(cur-prev);if(dif>25){changed++;changedSum+=dif;}
    }
    return common>35&&changed>=10&&changed/(float)common>.055f&&changedSum/max(1,changed)>34.0f;
  }

  void rememberDepthSignature(DepthFrame d){
    for(int gy=0;gy<SIG_H;gy++)for(int gx=0;gx<SIG_W;gx++)lastDepthSignature[gy*SIG_W+gx]=depthSignatureValue(d,gx,gy);
    haveDepthSignature=true;
  }

  float depthSignatureValue(DepthFrame d,int gx,int gy){
    int x=constrain(round((gx+.5f)*d.width/SIG_W),0,d.width-1),y=constrain(round((gy+.5f)*d.height/SIG_H),0,d.height-1);float v=d.mm[y*d.width+x];
    return (v>=400&&v<=6000)?v:0;
  }

  void startTraining(){
    if(training)return;
    collecting=false; closeManifest();
    if(accepted<250){ message="recommended minimum: 250 samples before training"; reopenManifest(); return; }
    try{
      File sessionOutput=new File(sessionDir,"output");
      File rootOutput=projectFile("output");
      if(!sessionOutput.exists()&&!sessionOutput.mkdirs())throw new IOException("cannot create session output folder: "+sessionOutput.getAbsolutePath());
      if(!rootOutput.exists()&&!rootOutput.mkdirs())throw new IOException("cannot create project output folder: "+rootOutput.getAbsolutePath());

      File pointer=runtimeConfigFile("runtime_python.txt");
      if(!pointer.isFile())throw new IOException("runtime_python.txt is missing. Run ./install_autolearn.sh cuda (or auto) first.");
      String py=readFirstLine(pointer);
      if(py.length()==0)throw new IOException("runtime_python.txt is empty.");
      File trainer=projectFile("trainer/train_depth_pose.py");
      if(!trainer.isFile())throw new IOException("trainer/train_depth_pose.py is missing.");

      if(!linuxHelperOnline()){
        throw new IOException("Linux helper is offline. Run ./install_autolearn.sh cuda (or auto) once from a terminal, then reopen the app.");
      }

      File sessionLog=new File(sessionOutput,"train.log");
      File mirrorLog=new File(rootOutput,"latest_training.log");
      File progress=new File(sessionOutput,"progress.json");
      writeTrainingControl(false);
      writeTextFile(mirrorLog,"Training request\r\nSession: "+sessionDir.getAbsolutePath()+"\r\nPython configured: "+py+"\r\nLaunch mode: persistent Linux helper (Processing does not spawn executables)\r\n\r\n");
      writeTextFile(progress,"{\"state\":\"queued\",\"progress\":0.0,\"epoch\":0,\"epochs\":90,\"batch\":0,\"batches\":0,\"loss\":0.0,\"val\":0.0,\"best\":0.0,\"eta\":\"--:--:--\",\"elapsed\":\"00:00:00\",\"device\":\"\",\"gpu\":\"\"}");

      JSONObject req=new JSONObject();
      req.setString("request_id",sessionDir.getName()+"_"+System.currentTimeMillis());
      req.setString("dataset",sessionDir.getAbsolutePath());
      req.setString("trainer",trainer.getAbsolutePath());
      req.setString("python",py);
      req.setString("session_log",sessionLog.getAbsolutePath());
      req.setString("mirror_log",mirrorLog.getAbsolutePath());
      req.setInt("progress_ms",100);
      req.setInt("checkpoint_batches",10);
      req.setInt("console_seconds",15);
      File request=persistentRuntimeFile("training_request.json");
      File tmp=persistentRuntimeFile("training_request.tmp");
      saveJSONObject(req,tmp.getAbsolutePath());
      if(request.exists()&&!request.delete())throw new IOException("cannot replace training request file");
      if(!tmp.renameTo(request)){
        saveJSONObject(req,request.getAbsolutePath());
        tmp.delete();
      }

      training=true; trainingPaused=false; trainState="queued"; trainProgress=0;
      message=(new File(sessionOutput,"last.pt").isFile()?"training queued • checkpoint will resume":"training queued")+" • Linux helper • "+(runtimeInfo!=null?runtimeInfo.backend:"runtime");
    }catch(Exception e){
      training=false;
      message="failed to start training: "+e.getMessage()+" • see output/latest_training.log";
      try{
        File rootOutput=projectFile("output"); rootOutput.mkdirs();
        File errorLog=new File(rootOutput,"latest_training.log");
        if(errorLog.isFile())appendTextFile(errorLog,"\r\nTRAINING LAUNCH ERROR\r\n"+e.toString()+"\r\n");
        else writeTextFile(errorLog,"TRAINING LAUNCH ERROR\r\n"+e.toString()+"\r\n");
      }catch(Exception ignored){}
      reopenManifest();
    }
  }

  boolean linuxHelperOnline(){
    try{
      File f=persistentRuntimeFile("helper_status.json");
      if(!f.isFile())return false;
      JSONObject j=loadJSONObject(f.getAbsolutePath());
      if(j==null)return false;
      long heartbeat=(long)j.getDouble("heartbeat_ms",0);
      return heartbeat>0 && Math.abs(System.currentTimeMillis()-heartbeat)<8000;
    }catch(Exception e){return false;}
  }

  void appendTextFile(File f,String text){
    try{PrintWriter w=new PrintWriter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream(f,true),"UTF-8")));w.print(text);w.close();}catch(Exception ignored){}
  }

  void writeTrainingControl(boolean paused)throws Exception{
    File out=new File(sessionDir,"output");out.mkdirs(); File f=new File(out,"control.json"),tmp=new File(out,"control.tmp");
    writeTextFile(tmp,"{\"paused\":"+(paused?"true":"false")+",\"updated\":"+(System.currentTimeMillis()/1000.0)+"}");
    if(f.exists())f.delete(); if(!tmp.renameTo(f)){writeTextFile(f,"{\"paused\":"+(paused?"true":"false")+"}");tmp.delete();}
  }

  void toggleTrainingPause(){
    if(!training){
      File ck=new File(new File(sessionDir,"output"),"last.pt"); if(ck.isFile()){message="checkpoint found • press T to resume training";} else message="training is not running"; return;
    }
    try{trainingPaused=!trainingPaused;writeTrainingControl(trainingPaused);trainState=trainingPaused?"pausing":"resuming";message=trainingPaused?"pause requested • saving checkpoint":"resume requested";}catch(Exception e){message="training control error: "+e.getMessage();}
  }

  String readFirstLine(File f)throws Exception{
    BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(f),"UTF-8"));
    String line=br.readLine(); br.close();
    if(line==null)return "";
    return line.replace("\uFEFF","").trim();
  }

  void writeTextFile(File f,String text)throws Exception{
    File parent=f.getParentFile(); if(parent!=null)parent.mkdirs();
    PrintWriter w=new PrintWriter(new OutputStreamWriter(new FileOutputStream(f),"UTF-8"));
    w.print(text); w.close();
  }


  void pollTrainingProgress(){
    if(!training)return;
    if(millis()-lastProgressPoll<100)return;
    lastProgressPoll=millis();
    File f=new File(new File(sessionDir,"output"),"progress.json");
    if(f.exists()){
      try{
        JSONObject j=loadJSONObject(f.getAbsolutePath());
        if(j!=null){
          trainState=j.getString("state",trainState); trainDevice=j.getString("device",trainDevice); trainGpu=j.getString("gpu",trainGpu);
          trainProgress=(float)j.getFloat("progress",trainProgress); trainLoss=(float)j.getFloat("loss",trainLoss); trainVal=(float)j.getFloat("val",trainVal); trainBest=(float)j.getFloat("best",trainBest);
          trainEpoch=j.getInt("epoch",trainEpoch); trainEpochs=j.getInt("epochs",trainEpochs); trainBatch=j.getInt("batch",trainBatch); trainBatches=j.getInt("batches",trainBatches);
          trainEta=j.getString("eta",trainEta); trainElapsed=j.getString("elapsed",trainElapsed); trainingPaused="paused".equals(trainState)||"pausing".equals(trainState);
          if("error".equals(trainState)){ String er=j.getString("error",""); if(er!=null&&er.length()>0) message="training error: "+er; }
        }
      }catch(Exception ignored){}
    }
    if(training && "done".equals(trainState)){
      training=false; trainProcess=null; trainProgress=1;
      message="training complete • ONNX ready"; if(student!=null)student.reloadLatest();
      reopenManifest();
    } else if(training && "error".equals(trainState)){
      training=false; trainProcess=null;
      File pf=new File(new File(sessionDir,"output"),"progress.json");
      String extra="";
      try{ JSONObject ej=loadJSONObject(pf.getAbsolutePath()); if(ej!=null) extra=ej.getString("error",""); }catch(Exception ignored){}
      message="training ended with error"+(extra.length()>0?" • "+extra:" • see project output/latest_training.log");
      reopenManifest();
    }
  }

  void drawTrainingBar(int x,int y,int w,int h){
    if(!training && trainProgress<=0)return;
    noStroke(); fill(38); rect(x,y,w,h,5);
    float p=constrain(trainProgress,0,1); fill(training?color(70,190,255):color(80,230,150)); rect(x,y,w*p,h,5);
    fill(245); textSize(14);
    String st=training?trainState:"done";
    text(String.format(Locale.US,"TRAIN %s  %.1f%%  epoch %d/%d  batch %d/%d",st,p*100,trainEpoch,trainEpochs,trainBatch,trainBatches),x,y-6);
    String hw=(trainGpu!=null&&trainGpu.length()>0)?trainGpu:trainDevice;
    text(String.format(Locale.US,"%s   loss %.5f   val %.5f   best %.5f   ETA %s",hw,trainLoss,trainVal,trainBest,trainEta),x,y+h+18);
  }

  void reopenManifest(){
    try{ manifest=new PrintWriter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream(manifestFile,true),"UTF-8"))); }catch(Exception ignored){}
  }
  void closeManifest(){ if(manifest!=null){manifest.flush();manifest.close();manifest=null;} }
  void close(){closeManifest();}

  void drawAcceptedOverlay(int ox,int oy,int w,int h){
    if(lastAccepted==null)return;
    int[][] bodyEdges={{0,4},{0,5},{1,0},{2,0},{3,0},{4,5},{4,6},{6,8},{5,7},{7,9},{4,10},{5,11},{10,11},{10,12},{12,14},{11,13},{13,15}};
    stroke(ui.success); strokeWeight(max(2,ui.s(3)));
    for(int[] e:bodyEdges)drawAcceptedEdge(e[0],e[1],ox,oy,w,h,.16f);
    int[][] fingers={{0,1,2,3,4},{0,5,6,7,8},{0,9,10,11,12},{0,13,14,15,16},{0,17,18,19,20}};
    strokeWeight(max(2,ui.s(2)));
    for(int fi=0;fi<5;fi++){
      stroke(ui.fingerColor(fi)); int[] ids=fingers[fi];
      for(int j=0;j<ids.length-1;j++){
        drawAcceptedEdge(BODY_KP+ids[j],BODY_KP+ids[j+1],ox,oy,w,h,.13f);
        drawAcceptedEdge(BODY_KP+HAND_KP+ids[j],BODY_KP+HAND_KP+ids[j+1],ox,oy,w,h,.13f);
      }
    }
    stroke(ui.textPrimary); drawAcceptedEdge(8,BODY_KP,ox,oy,w,h,.13f); drawAcceptedEdge(9,BODY_KP+HAND_KP,ox,oy,w,h,.13f);
    noStroke();
    for(int i=0;i<KP;i++){
      float[] q=lastAccepted[i]; if(q[3]<.13f)continue;
      if(i<BODY_KP)fill(ui.success); else fill(ui.handPointColor((i-BODY_KP)%HAND_KP));
      float rr=(i<BODY_KP?ui.s(9):ui.s(6)); if(isEndpoint(i))rr+=ui.s(2);
      ellipse(ox+q[0]*w,oy+q[1]*h,rr,rr);
    }
  }

  void drawAcceptedEdge(int a,int b,int ox,int oy,int w,int h,float minc){float[] p=lastAccepted[a],q=lastAccepted[b];if(p[3]<minc||q[3]<minc)return;line(ox+p[0]*w,oy+p[1]*h,ox+q[0]*w,oy+q[1]*h);}
  boolean isEndpoint(int i){return i==0||i==1||i==2||i==3||i==8||i==9||i==14||i==15||i==20||i==24||i==28||i==32||i==36||i==41||i==45||i==49||i==53||i==57;}

  void drawHud(int x,int y){
    fill(collecting?color(80,255,150):color(220)); textSize(15);
    text("AutoLearn: "+message,x,y);
    fill(200); textSize(13);
    text("accepted "+accepted+" / "+target+"   rejected "+rejected+" (duplicates "+noveltyRejected+")   hands L "+acceptedLeftHand+"/"+targetHandEach+" R "+acceptedRightHand+"/"+targetHandEach,x,y+24);
    text("dataset: "+sessionDir.getAbsolutePath(),x,y+46,1180,40);
    drawTrainingBar(x,y+94,1180,18);
  }
}
