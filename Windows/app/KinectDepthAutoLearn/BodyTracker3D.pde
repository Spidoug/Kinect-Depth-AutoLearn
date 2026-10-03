class Joint3D {
  String name;
  PVector p=new PVector(), velocity=new PVector();
  float confidence=0; boolean valid=false; long lastSeen=0,lastUpdate=0;
  Joint3D(String n){name=n;}
  void update(PVector q,float c){
    if(q==null||!finite(q.x)||!finite(q.y)||!finite(q.z)){predict();return;}
    long now=millis();float dt=lastUpdate==0?.033f:max(.010f,min(.10f,(now-lastUpdate)/1000.0f));
    if(!valid){p.set(q);velocity.set(0,0,0);}else{
      PVector delta=PVector.sub(q,p);float distMm=delta.mag(),speed=distMm/dt;
      // Adaptive stability filter. Body joints get a slightly stronger stationary
      // dead-zone; the 21 fine hand joints use a smaller/faster one so finger
      // gestures stay responsive while depth/pose noise is suppressed.
      boolean fineHand=(name.startsWith("L")||name.startsWith("R"))&&name.length()<=3;
      float deadMm=fineHand?2.4f:4.8f;
      float slowMm=fineHand?9.0f:16.0f;
      float alpha=constrain(map(speed,0,2200,fineHand?.13f:.10f,fineHand?.68f:.62f),fineHand?.10f:.08f,fineHand?.72f:.66f);
      alpha*=constrain(map(c,.15f,1,.68f,1),.58f,1);
      if(distMm<=deadMm){
        // Do not chase sub-millimetric / few-millimetre jitter when stationary.
        alpha=fineHand?.055f:.035f;
        velocity.mult(fineHand?.72f:.66f);
      }else if(distMm<slowMm){
        float t=(distMm-deadMm)/max(.001f,slowMm-deadMm);
        alpha=lerp(fineHand?.09f:.06f,fineHand?.30f:.24f,t);
      }
      // Large discontinuities are more likely bad depth associations than real
      // one-frame motion; approach them cautiously.
      if(distMm>420)alpha=min(alpha,.18f);
      PVector old=p.copy();p.lerp(q,alpha);PVector nv=PVector.sub(p,old).div(dt);velocity.lerp(nv,fineHand?.38f:.30f);
    }
    confidence=max(c,confidence*.84f);valid=true;lastSeen=now;lastUpdate=now;
  }
  void predict(){if(!valid)return;long now=millis(),age=now-lastSeen;if(age<380){float dt=min(.035f,max(.008f,(now-lastUpdate)/1000.0f));p.add(PVector.mult(velocity,dt*.40f));velocity.mult(.88f);confidence*=.96f;lastUpdate=now;}else{confidence*=.90f;if(age>850)valid=false;}}
  boolean finite(float x){return !Float.isNaN(x)&&!Float.isInfinite(x);}
}

class Hand3D {
  Joint3D[] k=new Joint3D[21]; boolean valid=false; float confidence=0;
  Hand3D(String prefix){for(int i=0;i<21;i++)k[i]=new Joint3D(prefix+i);}
  void predict(){for(Joint3D q:k)q.predict();}
  int count(){int n=0;for(Joint3D q:k)if(q.valid&&q.confidence>.12f)n++;return n;}
}

class BodyTracker3D {
  final int W=640,H=480,SW=320,SH=240,STEP=2;
  boolean[] mask=new boolean[SW*SH], previousMask=new boolean[SW*SH];
  float[] z=new float[SW*SH]; DepthFrame frame; long lastFrameSequence=-1;
  boolean tracked=false,poseFused=false,bodyLock=false; int lockMisses=0,stableFrames=0;
  float confidence=0,bodyDepthM=0,lockedDepthMm=0; PVector lockedCentroidDs=new PVector(SW*.5f,SH*.5f);
  int minX,maxX,minY,maxY;
  Hand3D handL=new Hand3D("L"),handR=new Hand3D("R");

  LinkedHashMap<String,Joint3D> j=new LinkedHashMap<String,Joint3D>();
  String[][] bones={{"head","neck"},{"neck","chest"},{"chest","pelvis"},{"neck","shoulderL"},{"shoulderL","elbowL"},{"elbowL","wristL"},{"wristL","handL"},{"neck","shoulderR"},{"shoulderR","elbowR"},{"elbowR","wristR"},{"wristR","handR"},{"pelvis","hipL"},{"hipL","kneeL"},{"kneeL","ankleL"},{"ankleL","footL"},{"pelvis","hipR"},{"hipR","kneeR"},{"kneeR","ankleR"},{"ankleR","footR"},{"shoulderL","shoulderR"},{"hipL","hipR"}};
  int[][] handEdges={{0,1},{1,2},{2,3},{3,4},{0,5},{5,6},{6,7},{7,8},{0,9},{9,10},{10,11},{11,12},{0,13},{13,14},{14,15},{15,16},{0,17},{17,18},{18,19},{19,20},{5,9},{9,13},{13,17}};

  BodyTracker3D(){String[] n={"head","neck","chest","pelvis","shoulderL","elbowL","wristL","handL","shoulderR","elbowR","wristR","handR","hipL","kneeL","ankleL","footL","hipR","kneeR","ankleR","footR"};for(String s:n)j.put(s,new Joint3D(s));}

  void update(DepthFrame f,PoseFrame pose,HandPoseFrame hands){
    frame=f;lastFrameSequence=f.sequence;downsample();
    for(Joint3D q:j.values())q.predict();handL.predict();handR.predict();
    tracked=segmentPersonLocked(pose);
    if(!tracked){confidence*=.90f;poseFused=false;invalidateHand(handL);invalidateHand(handR);return;}
    boolean fresh=pose!=null&&millis()-pose.receivedMs<700&&pose.score>.20f;
    poseFused=fresh&&estimateFromPose(pose);
    if(!poseFused)estimateDepthFallback();
    applyAnatomicalStability();
    if(hands!=null&&millis()-hands.receivedMs<250)fuseHands3D(hands,pose);else{invalidateHand(handL);invalidateHand(handR);}
  }

  void downsample(){
    // Robust 2x2 reduction: use the median/center of valid values instead of one
    // single pixel. This reduces speckle and holes without over-blurring body edges.
    for(int y=0;y<SH;y++)for(int x=0;x<SW;x++){z[y*SW+x]=robustBlockDepth(x*STEP,y*STEP);mask[y*SW+x]=false;}
  }
  float robustBlockDepth(int u,int v){
    float[] a=new float[4];int n=0;
    for(int yy=v;yy<min(H,v+STEP);yy++)for(int xx=u;xx<min(W,u+STEP);xx++){float d=frame.mm[yy*W+xx];if(d>=400&&d<=6200)a[n++]=d;}
    if(n==0)return 0;
    for(int i=0;i<n;i++)for(int k=i+1;k<n;k++)if(a[k]<a[i]){float t=a[i];a[i]=a[k];a[k]=t;}
    if(n==1)return a[0];if((n&1)==1)return a[n/2];return (a[n/2-1]+a[n/2])*.5f;
  }

  boolean segmentPersonLocked(PoseFrame pose){
    float searchCenter=bodyLock?lockedDepthMm:estimateDepthPeak();
    if(searchCenter<=0)return retainLockOrFail();
    float near=bodyLock?max(420,searchCenter-520):max(420,searchCenter-700);
    float far=bodyLock?min(5200,searchCenter+520):min(5200,searchCenter+700);
    boolean[] cand=new boolean[SW*SH];
    for(int i=0;i<cand.length;i++){float d=z[i];cand[i]=d>=near&&d<=far;}

    float poseCx=.5f,poseCy=.48f,poseW=.42f,poseH=.82f;
    if(pose!=null&&millis()-pose.receivedMs<800){float ax=0,ay=0;int n=0;float mnx=1,mxx=0,mny=1,mxy=0;for(int id:new int[]{0,5,6,11,12,15,16}){PoseLandmark q=pose.lm[id];if(q!=null&&q.score>.20f){mnx=min(mnx,q.x);mxx=max(mxx,q.x);mny=min(mny,q.y);mxy=max(mxy,q.y);ax+=q.x;ay+=q.y;n++;}}if(n>=4){poseCx=ax/n;poseCy=ay/n;poseW=max(.15f,mxx-mnx);poseH=max(.30f,mxy-mny);}}

    int[] label=new int[SW*SH],queue=new int[SW*SH];int lid=0,bestL=0;float bestScore=-1e9f;int bestN=0;float bestMean=0,bestCx=0,bestCy=0;
    for(int idx=0;idx<cand.length;idx++){
      if(!cand[idx]||label[idx]!=0)continue;lid++;int qs=0,qe=0,n=0,over=0,poseHit=0;double sumD=0,sumX=0,sumY=0;int lx=SW,hx=0,ly=SH,hy=0;queue[qe++]=idx;label[idx]=lid;
      while(qs<qe){int p=queue[qs++],px=p%SW,py=p/SW;n++;sumD+=z[p];sumX+=px;sumY+=py;lx=min(lx,px);hx=max(hx,px);ly=min(ly,py);hy=max(hy,py);if(previousMask[p])over++;
        float nx=(px*STEP)/(float)W,ny=(py*STEP)/(float)H;if(abs(nx-poseCx)<poseW*.70f&&abs(ny-poseCy)<poseH*.64f)poseHit++;
        if(px>0)qe=pushNeighbor(p-1,p,cand,label,lid,queue,qe);if(px<SW-1)qe=pushNeighbor(p+1,p,cand,label,lid,queue,qe);if(py>0)qe=pushNeighbor(p-SW,p,cand,label,lid,queue,qe);if(py<SH-1)qe=pushNeighbor(p+SW,p,cand,label,lid,queue,qe);
      }
      if(n<180)continue;float cx=(float)(sumX/n),cy=(float)(sumY/n),mean=(float)(sumD/n);float bw=hx-lx+1,bh=hy-ly+1,aspect=bh/max(1,bw);
      float score=n + poseHit*5.0f + over*9.0f;
      score-=abs(aspect-1.9f)*120.0f;
      if(bodyLock){score-=abs(mean-lockedDepthMm)*3.5f;score-=dist(cx,cy,lockedCentroidDs.x,lockedCentroidDs.y)*25.0f;}
      else {score-=mean*.08f; score-=dist(cx,cy,SW*.5f,SH*.52f)*3.0f;}
      if(score>bestScore){bestScore=score;bestL=lid;bestN=n;bestMean=mean;bestCx=cx;bestCy=cy;}
    }

    if(bestL==0||bestN<430)return retainLockOrFail();
    if(bodyLock){float dz=abs(bestMean-lockedDepthMm),dc=dist(bestCx,bestCy,lockedCentroidDs.x,lockedCentroidDs.y);if((dz>520||dc>72)&&lockMisses<9)return retainLockOrFail();}

    int n=0;double sum=0;minX=SW;minY=SH;maxX=0;maxY=0;
    for(int i=0;i<label.length;i++)if(label[i]==bestL){mask[i]=true;n++;sum+=z[i];int x=i%SW,y=i/SW;minX=min(minX,x);maxX=max(maxX,x);minY=min(minY,y);maxY=max(maxY,y);}
    if(n<430||maxY-minY<70)return retainLockOrFail();

    bodyDepthM=(float)(sum/n/1000.0);confidence=constrain(map(n,430,12000,.46f,.995f),.40f,.995f);
    lockedDepthMm=bodyLock?lerp(lockedDepthMm,bodyDepthM*1000,.18f):bodyDepthM*1000;
    lockedCentroidDs.set(bodyLock?lerp(lockedCentroidDs.x,bestCx,.22f):bestCx,bodyLock?lerp(lockedCentroidDs.y,bestCy,.22f):bestCy);
    arrayCopy(mask,previousMask);bodyLock=true;lockMisses=0;stableFrames++;return true;
  }

  float estimateDepthPeak(){int[] hist=new int[55];for(int y=10;y<SH-8;y++)for(int x=28;x<SW-28;x++){float d=z[y*SW+x];if(d>=500&&d<=5200)hist[min(hist.length-1,(int)d/100)]++;}int best=0,c=0;for(int b=5;b<hist.length;b++)if(hist[b]>c){c=hist[b];best=b;}return c>25?best*100+50:0;}
  boolean retainLockOrFail(){lockMisses++;if(bodyLock&&lockMisses<=9){arrayCopy(previousMask,mask);bodyDepthM=lockedDepthMm/1000.0f;confidence*=.94f;return true;}bodyLock=false;stableFrames=0;return false;}
  int pushNeighbor(int ni,int p,boolean[] cand,int[] label,int lid,int[] q,int qe){
    float continuity=constrain(z[p]*.070f,105,210);
    if(cand[ni]&&label[ni]==0&&abs(z[ni]-z[p])<continuity){label[ni]=lid;q[qe++]=ni;}return qe;
  }

  boolean estimateFromPose(PoseFrame p){
    int[] ids={0,5,6,7,8,9,10,11,12,13,14,15,16};float rminx=1,rmaxx=0,rminy=1,rmaxy=0;int n=0;
    for(int id:ids){PoseLandmark q=p.lm[id];if(q!=null&&q.score>.22f){rminx=min(rminx,q.x);rmaxx=max(rmaxx,q.x);rminy=min(rminy,q.y);rmaxy=max(rmaxy,q.y);n++;}}
    if(n<7||rmaxx-rminx<.08f||rmaxy-rminy<.16f)return false;
    float mx=(rmaxx-rminx)*.10f,my=(rmaxy-rminy)*.06f;rminx-=mx;rmaxx+=mx;rminy-=my;rmaxy+=my;float dminx=minX*STEP,dmaxx=(maxX+1)*STEP,dminy=minY*STEP,dmaxy=(maxY+1)*STEP;
    putPose("shoulderL",p,5,rminx,rmaxx,rminy,rmaxy,dminx,dmaxx,dminy,dmaxy,16);putPose("shoulderR",p,6,rminx,rmaxx,rminy,rmaxy,dminx,dmaxx,dminy,dmaxy,16);putPose("elbowL",p,7,rminx,rmaxx,rminy,rmaxy,dminx,dmaxx,dminy,dmaxy,18);putPose("elbowR",p,8,rminx,rmaxx,rminy,rmaxy,dminx,dmaxx,dminy,dmaxy,18);putPose("wristL",p,9,rminx,rmaxx,rminy,rmaxy,dminx,dmaxx,dminy,dmaxy,18);putPose("wristR",p,10,rminx,rmaxx,rminy,rmaxy,dminx,dmaxx,dminy,dmaxy,18);putPose("hipL",p,11,rminx,rmaxx,rminy,rmaxy,dminx,dmaxx,dminy,dmaxy,17);putPose("hipR",p,12,rminx,rmaxx,rminy,rmaxy,dminx,dmaxx,dminy,dmaxy,17);putPose("kneeL",p,13,rminx,rmaxx,rminy,rmaxy,dminx,dmaxx,dminy,dmaxy,19);putPose("kneeR",p,14,rminx,rmaxx,rminy,rmaxy,dminx,dmaxx,dminy,dmaxy,19);putPose("ankleL",p,15,rminx,rmaxx,rminy,rmaxy,dminx,dmaxx,dminy,dmaxy,22);putPose("ankleR",p,16,rminx,rmaxx,rminy,rmaxy,dminx,dmaxx,dminy,dmaxy,22);putPose("head",p,0,rminx,rmaxx,rminy,rmaxy,dminx,dmaxx,dminy,dmaxy,16);
    if(p.metricDepthInput){refineMetricArm(true);refineMetricArm(false);}
    PVector sl=getP("shoulderL"),sr=getP("shoulderR"),hl=getP("hipL"),hr=getP("hipR");if(sl!=null&&sr!=null)put("neck",PVector.lerp(sl,sr,.5f),.92f);if(hl!=null&&hr!=null)put("pelvis",PVector.lerp(hl,hr,.5f),.92f);if(sl!=null&&sr!=null&&hl!=null&&hr!=null)put("chest",PVector.lerp(PVector.lerp(sl,sr,.5f),PVector.lerp(hl,hr,.5f),.40f),.90f);inferHandsFeet();return goodJointCount()>=11;
  }

  void putPose(String name,PoseFrame p,int id,float rminx,float rmaxx,float rminy,float rmaxy,float dminx,float dmaxx,float dminy,float dmaxy,int radius){
    PoseLandmark lm=p.lm[id];if(lm==null||lm.score<.12f)return;
    int[] uv;
    if(p.metricDepthInput){
      // The pose already came from the depth map: direct 1:1 correspondence in the same plane.
      uv=new int[]{constrain(round(lm.x*(W-1)),0,W-1),constrain(round(lm.y*(H-1)),0,H-1)};
    }else{
      uv=rgbNormToDepth(lm.x,lm.y,rminx,rmaxx,rminy,rmaxy,dminx,dmaxx,dminy,dmaxy);
    }
    PVector q=p.metricDepthInput?robustWorldMetric(uv[0],uv[1],radius):robustWorldNear(uv[0],uv[1],radius);
    if(q!=null)put(name,q,constrain(lm.score*.92f+.08f,.12f,1));
  }
  void refineMetricArm(boolean left){
    String sh=left?"shoulderL":"shoulderR",el=left?"elbowL":"elbowR",wr=left?"wristL":"wristR";
    Joint3D a=j.get(sh),b=j.get(el),c=j.get(wr);if(a==null||b==null||c==null||!a.valid||!b.valid||!c.valid)return;
    PVector rb=snapMetricLimbPoint(b.p,a.p,20,95,520);if(rb!=null)b.update(rb,.58f*confidence);
    PVector rc=snapMetricLimbPoint(c.p,b.p,28,120,480);if(rc!=null)c.update(rc,.54f*confidence);
  }

  PVector snapMetricLimbPoint(PVector target,PVector parent,int radius,float depthWeight,float maxLen){
    int tu=constrain(round(target.x*594.2f/max(1,target.z)+339.3f),0,W-1),tv=constrain(round(target.y*591.0f/max(1,target.z)+242.7f),0,H-1);
    float desired=PVector.dist(target,parent),best=Float.MAX_VALUE;PVector bestP=null;
    for(int y=max(0,tv-radius);y<=min(H-1,tv+radius);y+=2)for(int x=max(0,tu-radius);x<=min(W-1,tu+radius);x+=2){
      int sx=x/STEP,sy=y/STEP;if(sx<0||sx>=SW||sy<0||sy>=SH||!mask[sy*SW+sx])continue;
      float z0=frame.mm[y*W+x];if(z0<450||z0>5200||abs(z0-parent.z)>420)continue;
      PVector q=worldAtFull(x,y,z0);float limb=PVector.dist(q,parent);if(limb>maxLen)continue;
      float score=dist(x,y,tu,tv)*3.0f+abs(limb-desired)*.55f+abs(z0-target.z)*(depthWeight/100.0f);
      if(score<best){best=score;bestP=q;}
    }
    return bestP;
  }

  int[] rgbNormToDepth(float x,float y,float rminx,float rmaxx,float rminy,float rmaxy,float dminx,float dmaxx,float dminy,float dmaxy){float nx=constrain((x-rminx)/max(.001f,rmaxx-rminx),-.12f,1.12f),ny=constrain((y-rminy)/max(.001f,rmaxy-rminy),-.10f,1.10f);return new int[]{round(lerp(dminx,dmaxx,nx)),round(lerp(dminy,dmaxy,ny))};}

  Joint3D projectPoseFace(PoseFrame p,int id,int radius){
    if(p==null||id<0||id>=p.lm.length||frame==null||!tracked)return null;
    PoseLandmark lm=p.lm[id]; if(lm==null||lm.score<.18f)return null;
    int[] ids={0,3,4,5,6,7,8,9,10,11,12,13,14,15,16};float rminx=1,rmaxx=0,rminy=1,rmaxy=0;int n=0;
    for(int k:ids){PoseLandmark q=p.lm[k];if(q!=null&&q.score>.22f){rminx=min(rminx,q.x);rmaxx=max(rmaxx,q.x);rminy=min(rminy,q.y);rmaxy=max(rmaxy,q.y);n++;}}
    if(n<7)return null;float mx=(rmaxx-rminx)*.10f,my=(rmaxy-rminy)*.06f;rminx-=mx;rmaxx+=mx;rminy-=my;rmaxy+=my;
    float dminx=minX*STEP,dmaxx=(maxX+1)*STEP,dminy=minY*STEP,dmaxy=(maxY+1)*STEP;
    int[] uv=p.metricDepthInput?new int[]{constrain(round(lm.x*(W-1)),0,W-1),constrain(round(lm.y*(H-1)),0,H-1)}:rgbNormToDepth(lm.x,lm.y,rminx,rmaxx,rminy,rmaxy,dminx,dmaxx,dminy,dmaxy);
    PVector q=p.metricDepthInput?robustWorldMetric(uv[0],uv[1],radius):robustWorldNear(uv[0],uv[1],radius);if(q==null)return null;
    Joint3D out=new Joint3D("face"+id);out.p.set(q);out.confidence=constrain(lm.score*.90f,.12f,1);out.valid=true;out.lastSeen=millis();return out;
  }

  PVector robustWorldNear(int u,int v,int radius){ArrayList<Float> vals=new ArrayList<Float>();int x0=max(0,u-radius),x1=min(W-1,u+radius),y0=max(0,v-radius),y1=min(H-1,v+radius);for(int yy=y0;yy<=y1;yy+=2)for(int xx=x0;xx<=x1;xx+=2){int sx=xx/STEP,sy=yy/STEP;if(sx<0||sx>=SW||sy<0||sy>=SH||!mask[sy*SW+sx])continue;float d=frame.mm[yy*W+xx];if(d>400&&d<6200)vals.add(d);}if(vals.size()<3)return null;Collections.sort(vals);float med=vals.get(vals.size()/2),best=Float.MAX_VALUE;int bu=u,bv=v;float bz=med;for(int yy=y0;yy<=y1;yy++)for(int xx=x0;xx<=x1;xx++){int sx=xx/STEP,sy=yy/STEP;if(sx<0||sx>=SW||sy<0||sy>=SH||!mask[sy*SW+sx])continue;float d=frame.mm[yy*W+xx];if(d<=0)continue;float s=abs(d-med)+dist(xx,yy,u,v)*2.5f;if(s<best){best=s;bu=xx;bv=yy;bz=d;}}return worldAtFull(bu,bv,bz);}

  PVector robustWorldMetric(int u,int v,int radius){
    int x0=max(0,u-radius),x1=min(W-1,u+radius),y0=max(0,v-radius),y1=min(H-1,v+radius);
    ArrayList<Float> vals=new ArrayList<Float>();
    for(int yy=y0;yy<=y1;yy+=2)for(int xx=x0;xx<=x1;xx+=2){int sx=xx/STEP,sy=yy/STEP;if(sx<0||sx>=SW||sy<0||sy>=SH||!mask[sy*SW+sx])continue;float d=frame.mm[yy*W+xx];if(d>=400&&d<=6200)vals.add(d);}
    if(vals.size()<3)return null;Collections.sort(vals);float med=vals.get(vals.size()/2);
    // Weighted center in pixel space, rejecting surfaces behind/in front of
    // the joint. This is more stable than choosing one minimum-depth pixel.
    double sw=0,su=0,sv=0,sz=0;
    for(int yy=y0;yy<=y1;yy++)for(int xx=x0;xx<=x1;xx++){
      int sx=xx/STEP,sy=yy/STEP;if(sx<0||sx>=SW||sy<0||sy>=SH||!mask[sy*SW+sx])continue;
      float d=frame.mm[yy*W+xx];if(d<400||d>6200||abs(d-med)>150)continue;
      float ds=dist(xx,yy,u,v);double w=1.0/(1.0+ds*ds*.055+abs(d-med)*.018);
      sw+=w;su+=xx*w;sv+=yy*w;sz+=d*w;
    }
    if(sw<=0)return null;return worldAtFull((int)round((float)(su/sw)),(int)round((float)(sv/sw)),(float)(sz/sw));
  }

  void invalidateHand(Hand3D h){
    if(h==null)return;h.valid=false;h.confidence=0;
    for(int i=0;i<21;i++){h.k[i].valid=false;h.k[i].confidence=0;h.k[i].velocity.set(0,0,0);}
  }

  void fuseHands3D(HandPoseFrame hf,PoseFrame body){
    if(body==null)return;float rminx=1,rmaxx=0,rminy=1,rmaxy=0;int n=0;for(int id:new int[]{0,5,6,7,8,9,10,11,12,13,14,15,16}){PoseLandmark q=body.lm[id];if(q!=null&&q.score>.22f){rminx=min(rminx,q.x);rmaxx=max(rmaxx,q.x);rminy=min(rminy,q.y);rmaxy=max(rmaxy,q.y);n++;}}if(n<7)return;float mx=(rmaxx-rminx)*.10f,my=(rmaxy-rminy)*.06f;rminx-=mx;rmaxx+=mx;rminy-=my;rmaxy+=my;float dminx=minX*STEP,dmaxx=(maxX+1)*STEP,dminy=minY*STEP,dmaxy=(maxY+1)*STEP;
    fuseHand(hf.left,handL,getP("wristL"),rminx,rmaxx,rminy,rmaxy,dminx,dmaxx,dminy,dmaxy);fuseHand(hf.right,handR,getP("wristR"),rminx,rmaxx,rminy,rmaxy,dminx,dmaxx,dminy,dmaxy);
  }
  void fuseHand(HandPose2D hp,Hand3D out,PVector wrist3,float rminx,float rmaxx,float rminy,float rmaxy,float dminx,float dmaxx,float dminy,float dmaxy){
    if(hp==null||wrist3==null){invalidateHand(out);return;}int good=0,palmGood=1;
    int[] parent={-1,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19};
    out.k[0].update(wrist3,.88f);good++;
    for(int i=1;i<21;i++){HandLandmark2D lm=hp.lm[i];if(lm==null||lm.score<.065f)continue;
      int[] uv=hp.metricDepthInput?new int[]{constrain(round(lm.x*(W-1)),0,W-1),constrain(round(lm.y*(H-1)),0,H-1)}:rgbNormToDepth(lm.x,lm.y,rminx,rmaxx,rminy,rmaxy,dminx,dmaxx,dminy,dmaxy);
      PVector anchor=out.k[parent[i]].valid?out.k[parent[i]].p:wrist3;float expectedZ=anchor.z;
      PVector q=robustWorldNearHand(uv[0],uv[1],6,expectedZ);
      if(q==null)q=worldAtFull(constrain(uv[0],0,W-1),constrain(uv[1],0,H-1),expectedZ);
      if(q!=null){
        PVector dv=PVector.sub(q,anchor);float maxBone=(i==1||i==5||i==9||i==13||i==17)?78:62;
        if(dv.mag()>maxBone){dv.normalize().mult(maxBone);q=PVector.add(anchor,dv);}
        out.k[i].update(q,constrain(.34f+lm.score*.46f,.35f,.80f));good++; if(i==5||i==9||i==13||i==17)palmGood++;
      }
    }
    stabilizeFingerChains(out,wrist3);
    out.valid=(good>=3&&palmGood>=2);
    out.confidence=constrain((good+2*palmGood)/31.0f,.12f,1);
  }
  PVector robustWorldNearHand(int u,int v,int radius,float expectedZ){
    double sw=0,sx=0,sy=0,sz=0;
    for(int yy=max(0,v-radius);yy<=min(H-1,v+radius);yy++)for(int xx=max(0,u-radius);xx<=min(W-1,u+radius);xx++){
      float d=frame.mm[yy*W+xx];if(d<450||d>5200||abs(d-expectedZ)>190)continue;
      int mx=xx/STEP,my=yy/STEP;if(mx<0||mx>=SW||my<0||my>=SH)continue;
      boolean inBody=mask[my*SW+mx]; float pd=dist(xx,yy,u,v);double w=(inBody?1.0:.42)/(1.0+pd*pd*.11+abs(d-expectedZ)*.028);sw+=w;sx+=xx*w;sy+=yy*w;sz+=d*w;
    }
    return sw>0?worldAtFull(round((float)(sx/sw)),round((float)(sy/sw)),(float)(sz/sw)):null;
  }
  void stabilizeFingerChains(Hand3D h,PVector wrist){
    if(h==null)return;h.k[0].p.lerp(wrist,.72f);
    int[][] chains={{0,1,2,3,4},{0,5,6,7,8},{0,9,10,11,12},{0,13,14,15,16},{0,17,18,19,20}};
    for(int[] chain:chains)for(int n=1;n<chain.length;n++){Joint3D a=h.k[chain[n-1]],b=h.k[chain[n]];if(!a.valid||!b.valid)continue;PVector d=PVector.sub(b.p,a.p);float m=d.mag();float lo=n==1?6:4,hi=n==1?82:66;if(m<1)continue;if(m<lo||m>hi){d.mult(constrain(m,lo,hi)/m);b.p.lerp(PVector.add(a.p,d),.60f);}}
  }

  void inferHandsFeet(){inferEndpoint("elbowL","wristL","handL",.22f,95);inferEndpoint("elbowR","wristR","handR",.22f,95);inferEndpoint("kneeL","ankleL","footL",.16f,120);inferEndpoint("kneeR","ankleR","footR",.16f,120);}
  void inferEndpoint(String a,String b,String out,float fraction,float maxExtra){PVector p0=getP(a),p1=getP(b);if(p0==null||p1==null)return;PVector dir=PVector.sub(p1,p0);float m=dir.mag();if(m<1)return;dir.normalize().mult(min(maxExtra,m*fraction));put(out,PVector.add(p1,dir),.55f);}
  void estimateDepthFallback(){int h=maxY-minY+1;int ys=minY+(int)(h*.26f),yc=minY+(int)(h*.39f),yp=minY+(int)(h*.59f),yk=minY+(int)(h*.79f),yf=maxY-2;put("head",centroidBand(minY+(int)(h*.08f),5),.45f);put("neck",centroidBand(minY+(int)(h*.18f),4),.42f);put("chest",centroidBand(yc,5),.44f);put("pelvis",centroidBand(yp,5),.44f);int[] e=extentsBand(ys,5);put("shoulderL",worldDs(e[0]+2,ys),.36f);put("shoulderR",worldDs(e[1]-2,ys),.36f);PVector al=armEndpoint(true,ys,yp),ar=armEndpoint(false,ys,yp),sl=getP("shoulderL"),sr=getP("shoulderR");put("handL",al,.32f);put("handR",ar,.32f);if(sl!=null&&al!=null)put("elbowL",PVector.lerp(sl,al,.52f),.30f);if(sr!=null&&ar!=null)put("elbowR",PVector.lerp(sr,ar,.52f),.30f);put("wristL",al,.30f);put("wristR",ar,.30f);put("hipL",centroidSide(true,minY+(int)(h*.63f),4),.35f);put("hipR",centroidSide(false,minY+(int)(h*.63f),4),.35f);put("kneeL",centroidSide(true,yk,5),.34f);put("kneeR",centroidSide(false,yk,5),.34f);put("ankleL",centroidSide(true,yf,5),.32f);put("ankleR",centroidSide(false,yf,5),.32f);inferHandsFeet();}
  void applyAnatomicalStability(){softLimit("shoulderL","elbowL",180,500);softLimit("elbowL","wristL",160,470);softLimit("shoulderR","elbowR",180,500);softLimit("elbowR","wristR",160,470);softLimit("hipL","kneeL",260,650);softLimit("kneeL","ankleL",250,650);softLimit("hipR","kneeR",260,650);softLimit("kneeR","ankleR",250,650);}
  void softLimit(String a,String b,float lo,float hi){Joint3D ja=j.get(a),jb=j.get(b);if(ja==null||jb==null||!ja.valid||!jb.valid)return;PVector d=PVector.sub(jb.p,ja.p);float m=d.mag();if(m<1)return;if(m<lo||m>hi){float target=constrain(m,lo,hi);d.mult(target/m);jb.p.lerp(PVector.add(ja.p,d),.30f);}}
  void put(String n,PVector p,float c){if(p!=null)j.get(n).update(p,c*confidence);}PVector getP(String n){Joint3D q=j.get(n);return q!=null&&q.valid?q.p:null;}int goodJointCount(){int n=0;for(Joint3D q:j.values())if(q.valid&&q.confidence>.15f)n++;return n;}

  PVector centroidBand(int yy,int r){float sx=0,sy=0,sz=0;int n=0;for(int y=max(minY,yy-r);y<=min(maxY,yy+r);y++)for(int x=minX;x<=maxX;x++)if(mask[y*SW+x]){PVector p=worldDs(x,y);if(p!=null){sx+=p.x;sy+=p.y;sz+=p.z;n++;}}return n>0?new PVector(sx/n,sy/n,sz/n):null;}
  PVector centroidSide(boolean left,int yy,int r){int mid=(minX+maxX)/2;float sx=0,sy=0,sz=0;int n=0;for(int y=max(minY,yy-r);y<=min(maxY,yy+r);y++)for(int x=minX;x<=maxX;x++)if(mask[y*SW+x]&&((left&&x<=mid)||(!left&&x>=mid))){PVector p=worldDs(x,y);if(p!=null){sx+=p.x;sy+=p.y;sz+=p.z;n++;}}return n>0?new PVector(sx/n,sy/n,sz/n):null;}
  int[] extentsBand(int yy,int r){int lo=SW,hi=0;for(int y=max(minY,yy-r);y<=min(maxY,yy+r);y++)for(int x=minX;x<=maxX;x++)if(mask[y*SW+x]){lo=min(lo,x);hi=max(hi,x);}return lo<=hi?new int[]{lo,hi}:new int[]{minX,maxX};}
  PVector armEndpoint(boolean left,int y0,int y1){float cx=(minX+maxX)*.5f,best=-1;int bx=(int)cx,by=y0;for(int y=y0;y<=min(y1,maxY);y++)for(int x=minX;x<=maxX;x++){if(!mask[y*SW+x])continue;float side=left?(cx-x):(x-cx);if(side<0)continue;float s=side*2.7f+abs(y-y0)*.30f;if(s>best){best=s;bx=x;by=y;}}return worldDs(bx,by);}
  PVector worldDs(int x,int y){if(x<0||x>=SW||y<0||y>=SH)return null;float d=z[y*SW+x];return d>0?worldAtFull(x*STEP+1,y*STEP+1,d):null;}
  PVector worldAtFull(int u,int v,float d){float fx=594.2f,fy=591.0f,cx=339.3f,cy=242.7f;return new PVector((u-cx)*d/fx,(v-cy)*d/fy,d);}

  void drawSkeleton(boolean mirror){if(goodJointCount()<4)return;for(String[] b:bones){Joint3D a=j.get(b[0]),c=j.get(b[1]);if(a==null||c==null||!a.valid||!c.valid)continue;float cc=min(a.confidence,c.confidence);stroke(255,230,55,250);strokeWeight(7+4*cc);line(mirror?-a.p.x:a.p.x,a.p.y,a.p.z-2200,mirror?-c.p.x:c.p.x,c.p.y,c.p.z-2200);}noStroke();sphereDetail(10);for(Joint3D q:j.values())if(q.valid){pushMatrix();translate(mirror?-q.p.x:q.p.x,q.p.y,q.p.z-2200);fill(q.confidence>.55f?color(65,255,155):color(255,125,65));sphere(q.name.startsWith("hand")||q.name.startsWith("foot")?24:18);popMatrix();}drawHand3D(handL,mirror);drawHand3D(handR,mirror);}
  void drawHand3D(Hand3D h,boolean mirror){
    if(h==null||!h.valid||h.count()<4)return;
    int[][] chains={{0,1,2,3,4},{0,5,6,7,8},{0,9,10,11,12},{0,13,14,15,16},{0,17,18,19,20}};
    int[] colors={color(255,166,76),color(83,210,255),color(96,231,150),color(204,135,255),color(255,105,154)};
    for(int fi=0;fi<chains.length;fi++){
      int[] chain=chains[fi];stroke(colors[fi]);strokeWeight(3.8f);
      for(int n=0;n<chain.length-1;n++){Joint3D a=h.k[chain[n]],b=h.k[chain[n+1]];if(!a.valid||!b.valid)continue;line(mirror?-a.p.x:a.p.x,a.p.y,a.p.z-2200,mirror?-b.p.x:b.p.x,b.p.y,b.p.z-2200);}
      noStroke();fill(colors[fi]);
      for(int n=1;n<chain.length;n++){Joint3D q=h.k[chain[n]];if(!q.valid)continue;pushMatrix();translate(mirror?-q.p.x:q.p.x,q.p.y,q.p.z-2200);sphere(n==chain.length-1?11:7);popMatrix();}
    }
    Joint3D wrist=h.k[0];if(wrist.valid){noStroke();fill(235);pushMatrix();translate(mirror?-wrist.p.x:wrist.p.x,wrist.p.y,wrist.p.z-2200);sphere(9);popMatrix();}
  }
  void drawDebugMask(int ox,int oy,int ww,int hh){noStroke();fill(0,205);rect(ox,oy,ww,hh);float sx=ww/(float)SW,sy=hh/(float)SH;for(int y=0;y<SH;y+=2)for(int x=0;x<SW;x+=2)if(mask[y*SW+x]){float d=z[y*SW+x];float t=constrain(map(d,lockedDepthMm-500,lockedDepthMm+500,0,1),0,1);fill(35+220*(1-t),100+135*t,255,235);rect(ox+x*sx,oy+y*sy,max(1,sx*2),max(1,sy*2));}}
  void drawDepthJointDebug(int ox,int oy,int ww,int hh){
    if(frame==null)return;
    noStroke();fill(0,185);rect(ox,oy,ww,hh);
    fill(255);textSize(13);text("DEPTH/JOINTS (diagnostic)",ox+8,oy+15);
    for(Joint3D q:j.values()){
      if(!q.valid)continue;
      float u=q.p.x*594.2f/max(1,q.p.z)+339.3f;
      float v=q.p.y*591.0f/max(1,q.p.z)+242.7f;
      float px=ox+constrain(u/640.0f,0,1)*ww;
      float py=oy+constrain(v/480.0f,0,1)*hh;
      fill(q.confidence>.5f?color(80,255,150):color(255,180,70));
      ellipse(px,py,7,7);
    }
  }

}
