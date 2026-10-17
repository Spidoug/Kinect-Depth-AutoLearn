import java.io.*;
import java.nio.*;
import java.util.*;
import java.awt.image.*;
import javax.imageio.ImageIO;
import ai.onnxruntime.*;

RgbdSourceHub sensor;
JavaPoseEngine teacherBody;
HandPoseEngine teacherHands;
BodyTracker3D tracker;
DatasetRecorder recorder;
StudentDepthEngine student;
RuntimeInfo runtimeInfo;
DashboardUI ui;

File projectRoot(){
  File sketchDir=new File(sketchPath(""));
  File appDir=sketchDir.getParentFile();
  return appDir!=null&&appDir.getParentFile()!=null?appDir.getParentFile():sketchDir;
}
File projectFile(String relative){ return new File(projectRoot(),relative.replace("/",File.separator)); }
File sketchFile(String relative){ return new File(sketchPath(relative)); }
File persistentRuntimeFile(String relative){
  String xdg=System.getenv("XDG_DATA_HOME");
  File base=(xdg!=null&&xdg.trim().length()>0)?new File(xdg):new File(System.getProperty("user.home"),".local/share");
  return new File(new File(base,"kinect-depth-autolearn"),relative.replace("/",File.separator));
}
File runtimeConfigFile(String name){ File local=projectFile(name); return local.isFile()?local:persistentRuntimeFile(name); }

boolean studentView=false;
PImage rgbView, depthView;
long rgbSeq=-1, depthSeq=-1;

void settings(){ size(1440,810,JAVA2D); smooth(4); }

void setup(){
  surface.setTitle("Kinect Depth AutoLearn");
  surface.setResizable(true);
  frameRate(30);
  textFont(createFont("SansSerif",24,true));
  rgbView=createImage(640,480,RGB);
  depthView=createImage(640,480,RGB);

  ui=new DashboardUI();
  runtimeInfo=new RuntimeInfo(); runtimeInfo.reload();
  sensor=new RgbdSourceHub();
  tracker=new BodyTracker3D();
  teacherBody=new JavaPoseEngine(); teacherBody.metricDepthMode=false;
  teacherHands=new HandPoseEngine();
  recorder=new DatasetRecorder(); recorder.setCaptureSource(sensor.sourceName());
  student=new StudentDepthEngine();

  sensor.start();
  teacherBody.start(sensor);
  teacherHands.start(sensor,teacherBody);
  student.start(sensor);
}

void draw(){
  ui.begin();
  DepthFrame d=sensor.latestDepth();
  RgbFrame r=sensor.latestRgb();
  PoseFrame p=teacherBody.latest();
  HandPoseFrame h=teacherHands.latest();

  if(d!=null && d.sequence!=tracker.lastFrameSequence) tracker.update(d,p,h);
  recorder.update(r,d,p,h,tracker);
  recorder.pollTrainingProgress();

  if(ui.windowTooSmall){
    ui.smallWindowNotice();
    return;
  }

  ui.header(runtimeInfo,studentView,sensor);
  drawCameraCards(r,d,p,h);
  ui.footer(runtimeInfo,recorder,student,studentView,sensor);
}

void drawCameraCards(RgbFrame r,DepthFrame d,PoseFrame p,HandPoseFrame h){
  boolean teacherLive=p!=null && millis()-p.receivedMs<700;
  boolean studentLive=student.latest()!=null && millis()-student.latest().receivedMs<800;
  ui.card(ui.leftX,"RGB • Teacher","MoveNet + RTMPose • pseudo-labels",teacherLive,ui.success);
  ui.card(ui.rightX,"Metric Depth • "+(studentView?"Student":"Labels"),studentView?"SynSkeleton.onnx • 58 landmarks + segments + endpoints":"structured metric labels • no eyes",studentView?studentLive:recorder.accepted>0,studentView?ui.accent:ui.magenta);

  int lx=ui.leftX+ui.imagePad, rx=ui.rightX+ui.imagePad, iy=ui.imageY, iw=ui.imageW, ih=ui.imageH;
  if(r!=null){
    if(r.sequence!=rgbSeq){ rgbView.loadPixels(); arrayCopy(r.argb,rgbView.pixels); rgbView.updatePixels(); rgbSeq=r.sequence; }
    image(rgbView,lx,iy,iw,ih);
    drawTeacherOverlay(p,h,lx,iy,iw,ih);
  } else drawEmptyFeed(lx,iy,iw,ih,"Waiting for RGB...");

  if(d!=null){
    if(d.sequence!=depthSeq){
      depthView.loadPixels();
      for(int i=0;i<d.mm.length;i++){
        float z=d.mm[i];
        int v=(z>=450&&z<=5000)?constrain(round(255*(1-(z-450)/4550.0f)),0,255):0;
        depthView.pixels[i]=color(v);
      }
      depthView.updatePixels(); depthSeq=d.sequence;
    }
    image(depthView,rx,iy,iw,ih);
    if(studentView) student.drawOverlay(rx,iy,iw,ih); else recorder.drawAcceptedOverlay(rx,iy,iw,ih);
  } else drawEmptyFeed(rx,iy,iw,ih,"Waiting for Depth...");
}

void drawEmptyFeed(int x,int y,int w,int h,String label){
  noStroke(); fill(8,10,14); rect(x,y,w,h,7);
  fill(ui.textMuted); textSize(ui.s(15)); textAlign(CENTER,CENTER); text(label,x+w/2,y+h/2); textAlign(LEFT,BASELINE);
}

void drawTeacherOverlay(PoseFrame p,HandPoseFrame h,int ox,int oy,int w,int hh){
  if(p!=null && millis()-p.receivedMs<700){
    int[][] e={{5,6},{5,7},{7,9},{6,8},{8,10},{5,11},{6,12},{11,12},{11,13},{13,15},{12,14},{14,16}};
    stroke(255,210,73); strokeWeight(max(2,ui.s(3)));
    for(int[] q:e){ PoseLandmark a=p.lm[q[0]],b=p.lm[q[1]]; if(a!=null&&b!=null&&a.score>.2&&b.score>.2) line(ox+a.x*w,oy+a.y*hh,ox+b.x*w,oy+b.y*hh); }
    noStroke(); fill(ui.success); int[] shown={0,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    for(int id:shown){PoseLandmark q=p.lm[id];if(q!=null&&q.score>.2)ellipse(ox+q.x*w,oy+q.y*hh,ui.s(9),ui.s(9));}
  }
  if(h!=null){ drawHand2D(h.left,ox,oy,w,hh); drawHand2D(h.right,ox,oy,w,hh); }
}

void drawHand2D(HandPose2D hand,int ox,int oy,int w,int hh){
  if(hand==null)return;
  int[][] fingers={{0,1,2,3,4},{0,5,6,7,8},{0,9,10,11,12},{0,13,14,15,16},{0,17,18,19,20}};
  strokeWeight(max(2,ui.s(2)));
  for(int f=0;f<5;f++){
    int c=ui.fingerColor(f); stroke(c);
    int[] ids=fingers[f];
    for(int j=0;j<ids.length-1;j++){
      HandLandmark2D a=hand.lm[ids[j]], b=hand.lm[ids[j+1]];
      if(a!=null&&b!=null&&a.score>.2&&b.score>.2) line(ox+a.x*w,oy+a.y*hh,ox+b.x*w,oy+b.y*hh);
    }
    noStroke(); fill(c);
    for(int j=1;j<ids.length;j++){
      HandLandmark2D q=hand.lm[ids[j]];
      if(q!=null&&q.score>.2) ellipse(ox+q.x*w,oy+q.y*hh,ui.s(7),ui.s(7));
    }
  }
  HandLandmark2D wrist=hand.lm[0];
  if(wrist!=null&&wrist.score>.2){ noStroke(); fill(ui.textPrimary); ellipse(ox+wrist.x*w,oy+wrist.y*hh,ui.s(8),ui.s(8)); }
}


void keyPressed(){ handleCommand(key); }

void mousePressed(){
  if(ui==null)return;
  char command=ui.commandAt(mouseX,mouseY);
  if(command!=0)handleCommand(command);
}

void handleCommand(char command){
  char c=Character.toUpperCase(command);
  if(c=='A') recorder.toggle();
  else if(c=='T') recorder.startTraining();
  else if(c=='P') recorder.toggleTrainingPause();
  else if(c=='X') recorder.stopCollecting();
  else if(c=='R') recorder.newSession();
  else if(c=='V'){ studentView=!studentView; student.setEnabled(studentView); }
  else if(c=='L'){ student.reloadLatest(); studentView=true; student.setEnabled(true); runtimeInfo.reload(); }
  else if(c>='1'&&c<='3'){
    int id=c-'0';
    recorder.stopCollecting();
    if(sensor.switchTo(id)){
      recorder.setCaptureSource(sensor.sourceName());
      recorder.newSession();
      studentView=false;
      student.setEnabled(false);
    }
  }
}

void exit(){
  recorder.close();
  if(student!=null)student.stop();
  if(teacherHands!=null)teacherHands.stop();
  if(teacherBody!=null)teacherBody.stop();
  if(sensor!=null)sensor.stop();
  super.exit();
}
