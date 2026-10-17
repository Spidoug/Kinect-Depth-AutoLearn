class RuntimeInfo {
  String backend="NOT INSTALLED";
  String selection="";
  String gpu="";
  String pythonVersion="";
  boolean ready=false;
  long lastLoad=0;

  void reload(){
    ready=false; backend="NOT INSTALLED"; selection=""; gpu=""; pythonVersion="";
    try{
      File f=runtimeConfigFile("runtime_status.json");
      if(!f.exists())return;
      JSONObject j=loadJSONObject(f.getAbsolutePath());
      if(j==null)return;
      backend=j.getString("backend","UNKNOWN");
      selection=j.getString("selection",backend);
      gpu=j.getString("gpu","");
      pythonVersion=j.getString("python_version","");
      ready=true; lastLoad=millis();
    }catch(Exception ignored){}
  }
}

class DashboardUI {
  final int bg=color(9,12,17);
  final int panel=color(17,22,29);
  final int panelSoft=color(21,27,36);
  final int panelRaised=color(25,32,42);
  final int border=color(42,52,65);
  final int divider=color(35,44,56);
  final int textPrimary=color(238,243,249);
  final int textMuted=color(151,166,184);
  final int textDim=color(105,120,139);
  final int accent=color(71,192,244);
  final int success=color(73,218,151);
  final int warning=color(245,184,74);
  final int danger=color(244,91,104);
  final int magenta=color(225,103,198);

  float uiScale=1.0f;
  boolean compact=false;
  boolean windowTooSmall=false;
  final int minimumWidth=960, minimumHeight=640;
  int margin,gap,topH;
  int contentY,contentBottom,contentH;
  int workspaceX,workspaceW,sidebarX,sidebarW;
  int cardY,cardW,cardH,leftX,rightX,imagePad,imageY,imageW,imageH;
  int bottomY,bottomH;
  int sessionY,sessionH,trainingY,trainingH,systemY,systemH,controlsY,controlsH;

  final int MAX_BUTTONS=16;
  int[] buttonX=new int[MAX_BUTTONS], buttonY=new int[MAX_BUTTONS], buttonW=new int[MAX_BUTTONS], buttonH=new int[MAX_BUTTONS];
  char[] buttonCommand=new char[MAX_BUTTONS];
  int buttonCount=0;

  int s(float v){ return max(1,round(v*uiScale)); }
  float fs(float base,float minimum){ return max(minimum,base*uiScale); }
  void font(float base,float minimum){ textSize(fs(base,minimum)); }

  void layout(){
    windowTooSmall=width<minimumWidth || height<minimumHeight;
    uiScale=constrain(min(width/1440.0f,height/810.0f),0.78f,1.24f);
    compact=width<1120 || height<680;
    margin=s(compact?14:16);
    gap=s(compact?10:12);
    topH=s(compact?62:70);
    contentY=topH+margin;
    contentBottom=height-margin;
    contentH=max(0,contentBottom-contentY);

    sidebarW=round(width*(compact?0.30f:0.27f));
    sidebarW=max(s(300),min(sidebarW,s(430)));
    workspaceX=margin;
    workspaceW=max(0,width-margin*2-gap-sidebarW);
    sidebarX=workspaceX+workspaceW+gap;

    cardY=contentY;
    cardW=max(0,(workspaceW-gap)/2);
    leftX=workspaceX;
    rightX=workspaceX+cardW+gap;

    int desiredPreview=round(cardW*0.82f);
    int maxPreview=round(contentH*0.62f);
    cardH=max(0,min(desiredPreview,maxPreview));
    bottomY=cardY+cardH+gap;
    bottomH=max(0,contentBottom-bottomY);

    int headerSpace=s(58), pad=s(12);
    int availW=max(s(100),cardW-pad*2);
    int availH=max(s(80),cardH-headerSpace-pad);
    imageW=min(availW,round(availH*4.0f/3.0f));
    imageH=round(imageW*3.0f/4.0f);
    imagePad=(cardW-imageW)/2;
    imageY=cardY+headerSpace+max(0,(availH-imageH)/2);

    sessionH=s(compact?100:104);
    trainingH=s(compact?224:236);
    systemH=s(compact?100:104);
    sessionY=contentY;
    trainingY=sessionY+sessionH+gap;
    systemY=trainingY+trainingH+gap;
    controlsY=systemY+systemH+gap;
    controlsH=max(0,contentBottom-controlsY);
  }

  void begin(){
    layout();
    buttonCount=0;
    background(bg);
  }

  void smallWindowNotice(){
    background(bg);
    fill(textPrimary); textAlign(CENTER,CENTER);
    font(22,18); text("Window is too small",width/2,height/2-s(26));
    fill(textMuted); font(15,13);
    text("Resize to at least 960 × 640 to display the dashboard.",width/2,height/2+s(12));
    textAlign(LEFT,BASELINE);
  }

  int fingerColor(int finger){
    if(finger==0)return color(255,166,76);
    if(finger==1)return color(83,210,255);
    if(finger==2)return color(96,231,150);
    if(finger==3)return color(204,135,255);
    return color(255,105,154);
  }
  int handPointColor(int localIndex){
    if(localIndex<=0)return textPrimary;
    if(localIndex<=4)return fingerColor(0);
    if(localIndex<=8)return fingerColor(1);
    if(localIndex<=12)return fingerColor(2);
    if(localIndex<=16)return fingerColor(3);
    return fingerColor(4);
  }
  int handSegmentColor(int localSegment){
    if(localSegment<4)return fingerColor(0);
    if(localSegment<8)return fingerColor(1);
    if(localSegment<12)return fingerColor(2);
    if(localSegment<16)return fingerColor(3);
    return fingerColor(4);
  }

  void header(RuntimeInfo rt, boolean studentView, RgbdSourceHub sensor){
    noStroke(); fill(13,17,23); rect(0,0,width,topH);
    fill(textPrimary); font(compact?24:27,21); text("Kinect Depth AutoLearn",margin,s(compact?36:40));
    if(!compact){
      fill(textDim); font(12,11); text("Depth-only pose learning",margin,s(58));
    }

    String mode=studentView?"Student view":"Teacher view";
    String runtime=rt.ready?rt.backend:"Runtime off";
    String source=sensor.sourceName();
    float modeW=pillWidth(mode), runW=pillWidth(runtime), srcW=pillWidth(source);
    float reserved=margin+textWidth("Kinect Depth AutoLearn")+s(36);
    float x=width-margin-modeW;
    pill(round(x),s(18),mode,studentView?accent:textMuted);
    x-=runW+s(8);
    if(x>reserved)pill(round(x),s(18),runtime,rt.ready?success:warning);
    x-=srcW+s(8);
    if(!compact && x>reserved)pill(round(x),s(18),source,accent);
  }

  void card(int x,String title,String subtitle,boolean active,int tagColor){
    panelBox(x,cardY,cardW,cardH);
    fill(textPrimary); font(18,16); text(fitText(cleanCardTitle(title),cardW-s(115)),x+s(16),cardY+s(27));
    fill(textMuted); font(11,11); text(fitText(cleanCardSubtitle(subtitle),cardW-s(34)),x+s(16),cardY+s(47));
    String state=active?"Live":"Idle";
    statePillRight(x+cardW-s(14),cardY+s(14),state,active?tagColor:textDim);
    noStroke(); fill(6,8,11); rect(x+imagePad,imageY,imageW,imageH,s(7));
  }

  String cleanCardTitle(String title){
    if(title.startsWith("RGB"))return "Teacher RGB";
    if(title.startsWith("METRIC DEPTH"))return title.indexOf("STUDENT")>=0?"Student Depth":"Metric depth";
    return title;
  }
  String cleanCardSubtitle(String subtitle){
    if(subtitle.indexOf("MoveNet")>=0)return "MoveNet + RTMPose • pseudo-labels";
    if(subtitle.indexOf("SynSkeleton")>=0)return "SynSkeleton + HQ hand refinement";
    if(subtitle.indexOf("structured")>=0)return "Metric labels • 58 landmarks";
    return subtitle;
  }

  void footer(RuntimeInfo rt, DatasetRecorder rec, StudentDepthEngine st, boolean studentView, RgbdSourceHub sensor){
    drawDatasetPanel(rec);
    drawSessionPanel(rec,sensor);
    drawTrainingPanel(rt,rec);
    drawSystemPanel(rt,st,studentView);
    drawControlsPanel(rec,studentView);
  }

  void drawDatasetPanel(DatasetRecorder rec){
    panelBox(workspaceX,bottomY,workspaceW,bottomH);
    int x=workspaceX+s(18), y=bottomY+s(29);
    fill(textPrimary); font(17,15); text("Dataset",x,y);
    String state=rec.collecting?"Capturing":"Ready";
    statePillRight(workspaceX+workspaceW-s(16),bottomY+s(15),state,rec.collecting?success:textMuted);

    float bodyProg=rec.target>0?rec.accepted/(float)rec.target:0;
    float leftProg=rec.targetHandEach>0?rec.acceptedLeftHand/(float)rec.targetHandEach:1;
    float rightProg=rec.targetHandEach>0?rec.acceptedRightHand/(float)rec.targetHandEach:1;
    int rightStatsW=min(s(260),round(workspaceW*0.31f));
    int progW=max(s(240),workspaceW-rightStatsW-s(58));
    int py=bottomY+s(61);
    progressRow(x,py,progW,"Body diversity",rec.accepted+" / "+rec.target,bodyProg,success);
    progressRow(x,py+s(48),progW,"Left hand",rec.acceptedLeftHand+" / "+rec.targetHandEach,leftProg,magenta);
    progressRow(x,py+s(96),progW,"Right hand",rec.acceptedRightHand+" / "+rec.targetHandEach,rightProg,accent);

    int sx=workspaceX+workspaceW-rightStatsW-s(18);
    float ar=(rec.accepted+rec.rejected)>0 ? rec.accepted/(float)(rec.accepted+rec.rejected) : 0;
    smallStat(sx,py,"Accepted",str(rec.accepted),success,rightStatsW/2-s(6));
    smallStat(sx+rightStatsW/2,py,"Rejected",str(rec.rejected),textMuted,rightStatsW/2-s(6));
    smallStat(sx,py+s(58),"Body visible",rec.lastBodyGood+" / 16",accent,rightStatsW/2-s(6));
    smallStat(sx+rightStatsW/2,py+s(58),"Hand points",rec.lastHandGood+" / 42",magenta,rightStatsW/2-s(6));
    if(bottomH>=s(225))smallStat(sx,py+s(116),"Acceptance",nf(ar*100,0,1)+"%",ar>.35?success:warning,rightStatsW-s(6));

    if(bottomH>s(245)){
      fill(textMuted); font(12,11); text(fitText(rec.message,workspaceW-s(36)),x,bottomY+bottomH-s(18));
    }
  }

  void drawSessionPanel(DatasetRecorder rec,RgbdSourceHub sensor){
    panelBox(sidebarX,sessionY,sidebarW,sessionH);
    int x=sidebarX+s(15), y=sessionY+s(26);
    fill(textPrimary); font(16,14); text("Session",x,y);
    int sc=rec.training?accent:(rec.collecting?success:textMuted);
    String state=rec.training?"Training":(rec.collecting?"Capturing":"Ready");
    statePillRight(sidebarX+sidebarW-s(13),sessionY+s(13),state,sc);
    fill(textDim); font(11,10); text("Source",x,sessionY+s(49));
    fill(textPrimary); font(11,10); text(fitText(sensor.sourceName(),sidebarW-s(30)),x,sessionY+s(65));
    if(sessionH>s(88)){
      fill(textDim); font(10,10); text("Session path",x,sessionY+s(81));
      fill(textMuted); font(10,9); String p=rec.sessionDir!=null?rec.sessionDir.getAbsolutePath():"—";
      text(fitText(p,sidebarW-s(30)),x,sessionY+s(97));
    }
  }

  void drawTrainingPanel(RuntimeInfo rt,DatasetRecorder rec){
    panelBox(sidebarX,trainingY,sidebarW,trainingH);
    int x=sidebarX+s(15), inner=sidebarW-s(30);
    fill(textPrimary); font(16,14); text("Training",x,trainingY+s(27));
    String state=rec.training?rec.trainState:(rec.trainProgress>=1?"Done":"Waiting");
    statePillRight(sidebarX+sidebarW-s(13),trainingY+s(13),state,rec.training?accent:(rec.trainProgress>=1?success:textMuted));

    int py=trainingY+s(55);
    fill(textMuted); font(10,10); text("Overall progress",x,py);
    textAlign(RIGHT,BASELINE); fill(textPrimary); font(11,10);
    text(nf(rec.trainProgress*100,0,1)+"%",x+inner,py);
    textAlign(LEFT,BASELINE);
    progress(x,py+s(9),inner,s(9),rec.trainProgress,rec.training?accent:success);

    int rowY=py+s(34);
    int half=(inner-s(8))/2;
    trainingInfoRow(x,rowY,half,"Epoch",rec.trainEpoch+" / "+rec.trainEpochs);
    trainingInfoRow(x+half+s(8),rowY,half,"Batch",rec.trainBatch+" / "+rec.trainBatches);

    int metricsY=rowY+s(37);
    int metricGap=s(6);
    int metricW=(inner-metricGap*2)/3;
    trainingMetricBox(x,metricsY,metricW,"Loss",nf(rec.trainLoss,0,5));
    trainingMetricBox(x+metricW+metricGap,metricsY,metricW,"Validation",nf(rec.trainVal,0,5));
    trainingMetricBox(x+(metricW+metricGap)*2,metricsY,metricW,"Best",nf(rec.trainBest,0,5));

    int timeY=trainingY+trainingH-s(43);
    stroke(divider); strokeWeight(1); line(x,timeY-s(10),x+inner,timeY-s(10));
    int timeHalf=(inner-s(12))/2;
    trainingTimeBlock(x,timeY,timeHalf,"Time remaining",cleanTime(rec.trainEta));
    trainingTimeBlock(x+timeHalf+s(12),timeY,timeHalf,"Elapsed",cleanTime(rec.trainElapsed));
  }

  void drawSystemPanel(RuntimeInfo rt,StudentDepthEngine st,boolean studentView){
    panelBox(sidebarX,systemY,sidebarW,systemH);
    int x=sidebarX+s(15), inner=sidebarW-s(30);
    fill(textPrimary); font(16,14); text("Runtime",x,systemY+s(27));
    String status=rt.ready?"Ready":"Offline";
    statePillRight(sidebarX+sidebarW-s(13),systemY+s(13),status,rt.ready?success:warning);

    int y=systemY+s(50);
    keyValue(x,y,inner,"Backend",rt.ready?rt.backend:"Not installed");
    y+=s(20);
    String gpu=rt.gpu!=null&&rt.gpu.length()>0?rt.gpu:"—";
    keyValue(x,y,inner,"Device",gpu);
    y+=s(20);
    if(systemH>s(92)){
      String model=st.session!=null?(st.handSession!=null?"SynSkeleton + SynHand":"SynSkeleton"):("No model loaded");
      keyValue(x,y,inner,"Model",model);
    }
  }

  void drawControlsPanel(DatasetRecorder rec,boolean studentView){
    panelBox(sidebarX,controlsY,sidebarW,controlsH);
    int x=sidebarX+s(15), inner=sidebarW-s(30);
    fill(textPrimary); font(16,14); text("Controls",x,controlsY+s(27));
    fill(textDim); font(10,10); textAlign(RIGHT,BASELINE); text("Click or key",sidebarX+sidebarW-s(15),controlsY+s(27)); textAlign(LEFT,BASELINE);

    int colGap=s(7), bw=(inner-colGap)/2, bh=s(32);
    int y=controlsY+s(42);
    commandButton(x,y,bw,bh,'A',rec.collecting?"Stop capture":"Capture",rec.collecting?success:textPrimary);
    commandButton(x+bw+colGap,y,bw,bh,'T',rec.training?"Training active":"Train / Resume",rec.training?accent:textPrimary);
    y+=bh+s(6);
    commandButton(x,y,bw,bh,'V',studentView?"Teacher view":"Student view",studentView?accent:textPrimary);
    commandButton(x+bw+colGap,y,bw,bh,'L',"Reload model",textPrimary);
    y+=bh+s(6);
    commandButton(x,y,bw,bh,'R',"New session",textPrimary);
    commandButton(x+bw+colGap,y,bw,bh,'P',rec.trainingPaused?"Resume train":"Pause train",rec.trainingPaused?warning:textPrimary);
    y+=bh+s(6);
    commandButton(x,y,bw,bh,'X',"Pause capture",textPrimary);

    int buttonsBottom=y+bh;
    if(controlsY+controlsH-buttonsBottom >= s(61)){
      int sy=controlsY+controlsH-s(27);
      stroke(divider); strokeWeight(1); line(x,sy-s(20),x+inner,sy-s(20));
      fill(textDim); font(10,10); text("Input source",x,sy-s(7));
      int sourceCount=System.getProperty("os.name","").toLowerCase().indexOf("win")>=0?4:3;
      int sg=s(6), sw=(inner-sg*(sourceCount-1))/sourceCount;
      for(int i=0;i<sourceCount;i++)sourceButton(x+(sw+sg)*i,sy,sw,s(24),(char)('1'+i));
    }
  }

  void panelBox(int x,int y,int w,int h){
    stroke(border); strokeWeight(1); fill(panel); rect(x,y,w,h,s(10));
  }

  void progressRow(int x,int y,int w,String label,String value,float p,int c){
    fill(textMuted); font(11,10); text(label,x,y);
    textAlign(RIGHT,BASELINE); fill(textPrimary); text(value,x+w,y); textAlign(LEFT,BASELINE);
    progress(x,y+s(10),w,s(10),p,c);
  }

  void progress(float x,float y,float w,float h,float value,int c){
    noStroke(); fill(38,46,57); rect(x,y,w,h,h/2);
    float p=constrain(value,0,1);
    if(p>0){ fill(c); rect(x,y,max(h,w*p),h,h/2); }
  }

  void smallStat(int x,int y,String label,String value,int c,float maxW){
    fill(textDim); font(10,10); text(fitText(label,maxW),x,y);
    fill(c); font(17,14); text(fitText(value,maxW),x,y+s(22));
  }

  void trainingMetric(int x,int y,String label,String value,int w){
    fill(textDim); font(10,10); text(label,x,y);
    fill(textPrimary); font(13,11); text(fitText(value,w-s(6)),x,y+s(18));
  }

  void trainingInfoRow(int x,int y,int w,String label,String value){
    fill(textDim); font(10,10); text(label,x,y);
    textAlign(RIGHT,BASELINE); fill(textPrimary); font(11,10);
    text(fitText(value,w-s(42)),x+w,y);
    textAlign(LEFT,BASELINE);
  }

  void trainingMetricBox(int x,int y,int w,String label,String value){
    noStroke(); fill(panelSoft); rect(x,y,w,s(40),s(6));
    fill(textDim); font(11,10); text(fitText(label,w-s(16)),x+s(8),y+s(14));
    fill(textPrimary); font(12,10); text(fitText(value,w-s(16)),x+s(8),y+s(31));
  }

  void trainingTimeBlock(int x,int y,int w,String label,String value){
    fill(textDim); font(10,10); text(fitText(label,w),x,y);
    fill(textPrimary); font(12,10); text(fitText(value,w),x,y+s(17));
  }

  String cleanTime(String value){
    if(value==null||value.trim().length()==0)return "—";
    String v=value.trim();
    if(v.equals("--:--:--")||v.equals("00:00:00")||v.equalsIgnoreCase("unknown"))return v;
    return v;
  }

  void keyValue(int x,int y,int w,String key,String value){
    fill(textDim); font(10,10); text(key,x,y);
    textAlign(RIGHT,BASELINE); fill(textPrimary); font(11,10); text(fitText(value,w-s(90)),x+w,y); textAlign(LEFT,BASELINE);
  }

  float pillWidth(String label){ font(11,10); return textWidth(label)+s(24); }
  void pill(int x,int y,String label,int c){
    font(11,10); float tw=textWidth(label)+s(24);
    noStroke(); fill(red(c),green(c),blue(c),24); rect(x,y,tw,s(30),s(15));
    fill(c); text(label,x+s(12),y+s(20));
  }

  void statePillRight(int right,int y,String label,int c){
    font(9,9); int w=round(textWidth(label))+s(18), h=s(24); int x=right-w;
    noStroke(); fill(red(c),green(c),blue(c),25); rect(x,y,w,h,s(12));
    fill(c); text(label,x+s(9),y+s(16));
  }

  void commandButton(int x,int y,int w,int h,char cmd,String label,int c){
    boolean hover=mouseX>=x&&mouseX<=x+w&&mouseY>=y&&mouseY<=y+h;
    noStroke(); fill(hover?panelRaised:panelSoft); rect(x,y,w,h,s(6));
    fill(c); rect(x+s(6),y+s(5),s(20),h-s(10),s(5));
    fill(bg); font(10,10); textAlign(CENTER,CENTER); text(str(cmd),x+s(16),y+h/2);
    textAlign(LEFT,CENTER); fill(textPrimary); font(12,11); text(fitText(label,w-s(38)),x+s(33),y+h/2); textAlign(LEFT,BASELINE);
    rememberButton(x,y,w,h,cmd);
  }

  void sourceButton(int x,int y,int w,int h,char cmd){
    boolean hover=mouseX>=x&&mouseX<=x+w&&mouseY>=y&&mouseY<=y+h;
    noStroke(); fill(hover?panelRaised:panelSoft); rect(x,y,w,h,s(5));
    fill(textPrimary); font(10,10); textAlign(CENTER,CENTER); text(fitText(sourceButtonLabel(cmd),w-s(8)),x+w/2,y+h/2); textAlign(LEFT,BASELINE);
    rememberButton(x,y,w,h,cmd);
  }

  String sourceButtonLabel(char cmd){
    boolean win=System.getProperty("os.name","").toLowerCase().indexOf("win")>=0;
    if(win){
      if(cmd=='1')return "1  Remold";
      if(cmd=='2')return "2  SDK 1.8";
      if(cmd=='3')return "3  OpenK";
      return "4  Kinect v2";
    }
    if(cmd=='1')return "1  Remold";
    if(cmd=='2')return "2  OpenK";
    return "3  Kinect v2";
  }

  void rememberButton(int x,int y,int w,int h,char cmd){
    if(buttonCount>=MAX_BUTTONS)return;
    buttonX[buttonCount]=x; buttonY[buttonCount]=y; buttonW[buttonCount]=w; buttonH[buttonCount]=h; buttonCommand[buttonCount]=cmd; buttonCount++;
  }

  char commandAt(int mx,int my){
    for(int i=buttonCount-1;i>=0;i--){
      if(mx>=buttonX[i]&&mx<=buttonX[i]+buttonW[i]&&my>=buttonY[i]&&my<=buttonY[i]+buttonH[i])return buttonCommand[i];
    }
    return 0;
  }

  String fitText(String value,float maxWidth){
    if(value==null)return "";
    if(maxWidth<=0)return "";
    if(textWidth(value)<=maxWidth)return value;
    String ell="…";
    if(textWidth(ell)>maxWidth)return "";
    int lo=0,hi=value.length();
    while(lo<hi){
      int mid=(lo+hi+1)/2;
      String t=value.substring(0,mid)+ell;
      if(textWidth(t)<=maxWidth)lo=mid;else hi=mid-1;
    }
    return value.substring(0,lo)+ell;
  }

  String shortPath(String value,int maxChars){
    if(value==null)return "";
    if(value.length()<=maxChars)return value;
    int keep=max(1,maxChars-3); return "..."+value.substring(value.length()-keep);
  }
}
