import argparse, csv, hashlib, json, math, os, random, sys, time, warnings, shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, Sampler

MODEL_VERSION=2
CHECKPOINT_VERSION=3
KP=58; BODY_KP=16; HAND_KP=21; IN_H=240; IN_W=320; HM_H=120; HM_W=160
HAND_IN=128; HAND_HM=64
BODY_EDGES=[(0,4),(0,5),(1,0),(2,0),(3,0),(4,5),(4,6),(6,8),(5,7),(7,9),(4,10),(5,11),(10,11),(10,12),(12,14),(11,13),(13,15)]
HAND_LOCAL=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
SEGMENTS=BODY_EDGES+[(16+a,16+b) for a,b in HAND_LOCAL]+[(37+a,37+b) for a,b in HAND_LOCAL]+[(8,16),(9,37)]
ENDPOINTS=[0,1,2,3,8,9,14,15,20,24,28,32,36,41,45,49,53,57]
LEFT_RIGHT_PAIRS=[(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15)]+[(16+i,37+i) for i in range(HAND_KP)]

class Tee:
    def __init__(self,*streams): self.streams=[s for s in streams if s is not None]
    def write(self,data):
        for s in self.streams:
            try: s.write(data); s.flush()
            except Exception: pass
        return len(data)
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except Exception: pass

def _make_user_writable(path):
    try:
        p=Path(path)
        if p.exists(): os.chmod(p,p.stat().st_mode|0o600)
        if p.parent.exists(): os.chmod(p.parent,p.parent.stat().st_mode|0o700)
    except Exception: pass

def _atomic_write_text(path,text,retries=30,nonfatal=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+'.tmp'); last=None
    for attempt in range(max(1,retries)):
        try: tmp.write_text(text,encoding='utf-8'); os.replace(tmp,path); return True
        except (PermissionError,OSError) as e:
            last=e; _make_user_writable(path); _make_user_writable(tmp); time.sleep(min(.25,.02*(attempt+1)))
    try:
        if tmp.exists(): shutil.copyfile(tmp,path); tmp.unlink(missing_ok=True)
        else: path.write_text(text,encoding='utf-8')
        return True
    except Exception as e: last=e
    if nonfatal:
        print(f'WARNING: could not update {path.name}: {type(last).__name__}: {last}',flush=True); return False
    raise last

def _torch_save_resilient(obj,path,retries=20,nonfatal=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+'.tmp'); last=None
    for attempt in range(max(1,retries)):
        try: torch.save(obj,tmp); os.replace(tmp,path); return True
        except Exception as e:
            last=e; _make_user_writable(path); _make_user_writable(tmp); time.sleep(min(.35,.03*(attempt+1)))
    try: torch.save(obj,path); tmp.unlink(missing_ok=True); return True
    except Exception as e: last=e
    if nonfatal:
        print(f'WARNING: checkpoint write failed for {path}: {type(last).__name__}: {last}',flush=True); return False
    raise last

class Progress:
    def __init__(self,out,interval_ms=100):
        self.out=Path(out); self.path=self.out/'progress.json'; self.t0=time.time(); self.interval=max(0,interval_ms)/1000.0; self.last=0.0
    def write(self,force=True,**kw):
        now=time.time()
        if not force and now-self.last<self.interval: return
        self.last=now; _atomic_write_text(self.path,json.dumps({'updated':now,**kw},ensure_ascii=False),retries=12,nonfatal=True)

def robust_downsample_depth(arr):
    if arr.shape==(IN_H,IN_W): return arr.astype(np.float32,copy=False)
    if arr.shape==(IN_H*2,IN_W*2):

        a=arr.reshape(IN_H,2,IN_W,2).transpose(0,2,1,3).reshape(IN_H,IN_W,4).astype(np.float32,copy=False)
        valid=(a>350)&(a<6000); ordered=np.sort(np.where(valid,a,np.float32(np.inf)),axis=-1); count=valid.sum(-1)
        d=np.zeros((IN_H,IN_W),dtype=np.float32)
        m=count==1; d[m]=ordered[...,0][m]
        m=count==2; d[m]=(ordered[...,0][m]+ordered[...,1][m])*.5
        m=count==3; d[m]=ordered[...,1][m]
        m=count==4; d[m]=(ordered[...,1][m]+ordered[...,2][m])*.5
        return d

    yy=np.clip(((np.arange(IN_H)+.5)*arr.shape[0]/IN_H).astype(np.int64),0,arr.shape[0]-1)
    xx=np.clip(((np.arange(IN_W)+.5)*arr.shape[1]/IN_W).astype(np.int64),0,arr.shape[1]-1)
    return arr[np.ix_(yy,xx)].astype(np.float32)


def affine_depth_and_labels(depth,xy,vis):
    angle=random.uniform(-12.0,12.0) if random.random()<.70 else 0.0
    scale=random.uniform(.92,1.08) if random.random()<.70 else 1.0
    tx=random.uniform(-.045,.045)*(IN_W-1) if random.random()<.65 else 0.0
    ty=random.uniform(-.045,.045)*(IN_H-1) if random.random()<.65 else 0.0
    if abs(angle)<1e-5 and abs(scale-1.0)<1e-5 and abs(tx)<1e-5 and abs(ty)<1e-5: return depth,xy,vis
    a=math.radians(angle); ca=math.cos(a)*scale; sa=math.sin(a)*scale; cx=(IN_W-1)*.5; cy=(IN_H-1)*.5

    M=np.array([[ca,-sa,cx+tx-ca*cx+sa*cy],[sa,ca,cy+ty-sa*cx-ca*cy],[0,0,1]],dtype=np.float64)
    inv=np.linalg.inv(M); coeff=(float(inv[0,0]),float(inv[0,1]),float(inv[0,2]),float(inv[1,0]),float(inv[1,1]),float(inv[1,2]))
    im=Image.fromarray(depth.astype(np.float32),mode='F')
    depth=np.asarray(im.transform((IN_W,IN_H),Image.Transform.AFFINE,coeff,resample=Image.Resampling.NEAREST,fillcolor=0.0),dtype=np.float32).copy()
    pts=np.ones((KP,3),dtype=np.float64); pts[:,0]=xy[:,0]*(IN_W-1); pts[:,1]=xy[:,1]*(IN_H-1); q=(M@pts.T).T
    xy[:,0]=(q[:,0]/(IN_W-1)).astype(np.float32); xy[:,1]=(q[:,1]/(IN_H-1)).astype(np.float32)
    inside=(xy[:,0]>=0)&(xy[:,0]<=1)&(xy[:,1]>=0)&(xy[:,1]<=1); vis*=inside.astype(np.float32); xy=np.clip(xy,0,1)
    return depth,xy,vis

def flip_labels(depth,xy,z,vis):
    depth=np.ascontiguousarray(depth[:,::-1]); xy[:,0]=1.0-xy[:,0]
    for a,b in LEFT_RIGHT_PAIRS:
        xy[[a,b]]=xy[[b,a]]; z[[a,b]]=z[[b,a]]; vis[[a,b]]=vis[[b,a]]
    return depth,xy,z,vis

class DepthDataset(Dataset):
    def __init__(self,root,augment=True):
        self.root=Path(root); self.augment=augment; self.rows=[]; self.labels=[]; self.meta=[]; self.hand_bits=[]
        with open(self.root/'labels.csv',newline='',encoding='utf-8') as f:
            rd=csv.reader(f); next(rd,None)
            for row in rd:
                if len(row)>=3+KP*4 and (self.root/row[0]).is_file():
                    try:
                        vals=np.asarray(row[3:3+KP*4],dtype=np.float32).reshape(KP,4)
                        if not np.isfinite(vals).all(): continue
                        if np.any(vals[:,3]<0) or np.any(vals[:,3]>1): continue
                        visible=vals[:,3]>.10
                        if np.any(visible & ((vals[:,0]<0)|(vals[:,0]>1)|(vals[:,1]<0)|(vals[:,1]>1)|(vals[:,2]<0)|(vals[:,2]>6000))): continue
                    except (ValueError,TypeError): continue
                    self.rows.append(row); self.labels.append(vals.copy())
                    try: seq=int(float(row[1])); tick=int(float(row[2]))
                    except Exception: seq=len(self.rows)-1; tick=(len(self.rows)-1)*100
                    self.meta.append((row[0],seq,tick))
                    vv=vals[:,3]; left=int((vv[BODY_KP:BODY_KP+HAND_KP]>.14).sum()>=4); right=int((vv[BODY_KP+HAND_KP:]>.14).sum()>=4); self.hand_bits.append((left,right))
    def __len__(self): return len(self.rows)
    def __getitem__(self,idx):
        row=self.rows[idx]; arr=np.asarray(Image.open(self.root/row[0])); d=robust_downsample_depth(arr)
        vals=self.labels[idx]; xy=vals[:,:2].copy(); z=vals[:,2].copy(); vis=np.clip(vals[:,3].copy(),0,1)
        if self.augment:
            if random.random()<.50: d,xy,z,vis=flip_labels(d,xy,z,vis)
            d,xy,vis=affine_depth_and_labels(d,xy,vis)
            if random.random()<.28:
                sf=np.float32(random.uniform(.985,1.015)); valid=(d>350)&(d<6000); d=np.where(valid,d*sf,d); z=np.where(vis>.10,z*sf,z)
            if random.random()<.55:
                noise=np.random.normal(0,8.0,d.shape).astype(np.float32); d=np.where((d>350)&(d<6000),d+noise,d)
            if random.random()<.24:
                ph=random.randint(8,34); pw=random.randint(8,40); y0=random.randint(0,max(0,IN_H-ph)); x0=random.randint(0,max(0,IN_W-pw)); d[y0:y0+ph,x0:x0+pw]=0
            if random.random()<.10:
                if random.random()<.5:
                    x0=random.randint(0,IN_W-2); d[:,x0:x0+random.randint(1,3)]=0
                else:
                    y0=random.randint(0,IN_H-2); d[y0:y0+random.randint(1,3),:]=0
        valid=(d>350)&(d<6000); x=np.full(d.shape,-2.0,dtype=np.float32); x[valid]=np.clip((d[valid]-2200.0)/1800.0,-1.0,1.8)


        return torch.from_numpy(x[None]),torch.from_numpy(z.astype(np.float32)),torch.from_numpy(vis.astype(np.float32)),torch.from_numpy(xy.astype(np.float32))


def _hand_crop_geometry(vals,side,width,height,jitter=False):
    xy=vals[:,:2]; vis=vals[:,3]
    elbow=6 if side==0 else 7; wrist=8 if side==0 else 9; base=BODY_KP if side==0 else BODY_KP+HAND_KP
    if vis[elbow]<.14 or vis[wrist]<.16 or int((vis[base:base+HAND_KP]>.10).sum())<4: return None
    wx=float(xy[wrist,0]*(width-1)); wy=float(xy[wrist,1]*(height-1)); ex=float(xy[elbow,0]*(width-1)); ey=float(xy[elbow,1]*(height-1))
    dx=wx-ex; dy=wy-ey; fore=math.hypot(dx,dy)
    if fore<10: return None
    dx/=fore; dy/=fore; cx=wx+dx*fore*.18; cy=wy+dy*fore*.18; side_px=float(np.clip(fore*1.60,72.0,190.0))
    if jitter:
        side_px*=random.uniform(.90,1.11); cx+=random.uniform(-.055,.055)*side_px; cy+=random.uniform(-.055,.055)*side_px
    return cx-side_px*.5,cy-side_px*.5,side_px,base

class HandCropDataset(Dataset):
    def __init__(self,base_dataset,indices,augment=True):
        self.base=base_dataset; self.root=base_dataset.root; self.augment=augment; self.items=[]
        for idx in indices:
            vals=base_dataset.labels[idx]
            for side in (0,1):
                geom=_hand_crop_geometry(vals,side,640,480,False)
                if geom is not None: self.items.append((idx,side))
    def __len__(self): return len(self.items)
    def __getitem__(self,item_idx):
        idx,side=self.items[item_idx]; row=self.base.rows[idx]; arr=np.asarray(Image.open(self.root/row[0])); h,w=arr.shape[:2]
        vals=self.base.labels[idx]; geom=_hand_crop_geometry(vals,side,w,h,self.augment)
        if geom is None: geom=_hand_crop_geometry(vals,side,w,h,False)
        x0,y0,side_px,base=geom; x1=x0+side_px; y1=y0+side_px
        im=Image.fromarray(arr.astype(np.float32),mode='F'); crop=np.asarray(im.transform((HAND_IN,HAND_IN),Image.Transform.EXTENT,(x0,y0,x1,y1),resample=Image.Resampling.NEAREST,fillcolor=0.0),dtype=np.float32).copy()
        hv=vals[base:base+HAND_KP].copy(); xy=hv[:,:2].copy(); z=hv[:,2].copy(); vis=np.clip(hv[:,3].copy(),0,1)
        xy[:,0]=(xy[:,0]*(w-1)-x0)/side_px; xy[:,1]=(xy[:,1]*(h-1)-y0)/side_px
        inside=(xy[:,0]>=0)&(xy[:,0]<=1)&(xy[:,1]>=0)&(xy[:,1]<=1); vis*=inside.astype(np.float32); xy=np.clip(xy,0,1)
        if self.augment and random.random()<.50:
            crop=np.ascontiguousarray(crop[:,::-1]); xy[:,0]=1.0-xy[:,0]
        if self.augment and random.random()<.30:
            sf=np.float32(random.uniform(.988,1.012)); ok=(crop>350)&(crop<6000); crop=np.where(ok,crop*sf,crop); z=np.where(vis>.10,z*sf,z)
        if self.augment and random.random()<.60:
            noise=np.random.normal(0,6.0,crop.shape).astype(np.float32); crop=np.where((crop>350)&(crop<6000),crop+noise,crop)
        if self.augment and random.random()<.24:
            ph=random.randint(5,18); pw=random.randint(5,18); yy=random.randint(0,HAND_IN-ph); xx=random.randint(0,HAND_IN-pw); crop[yy:yy+ph,xx:xx+pw]=0
        ok=(crop>350)&(crop<6000); x=np.full(crop.shape,-2.0,dtype=np.float32); x[ok]=np.clip((crop[ok]-2200.0)/1800.0,-1.0,1.8)
        crop_meta=np.asarray([x0/(w-1),y0/(h-1),side_px/(w-1),side_px/(h-1)],dtype=np.float32)
        return torch.from_numpy(x[None]),torch.from_numpy(z.astype(np.float32)),torch.from_numpy(vis.astype(np.float32)),torch.from_numpy(xy.astype(np.float32)),torch.from_numpy(crop_meta)

class Block(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(cin,cout,3,stride,1,bias=False),nn.BatchNorm2d(cout),nn.SiLU(),nn.Conv2d(cout,cout,3,1,1,bias=False),nn.BatchNorm2d(cout),nn.SiLU()); self.skip=nn.Identity() if cin==cout and stride==1 else nn.Conv2d(cin,cout,1,stride,bias=False)
    def forward(self,x): return self.net(x)+self.skip(x)

class DepthPoseNet(nn.Module):
    """SynSkeleton v2: 120x160 landmark maps + depth-conditioned local Z extraction."""
    def __init__(self):
        super().__init__()
        self.s0=nn.Sequential(nn.Conv2d(1,32,5,2,2,bias=False),nn.BatchNorm2d(32),nn.SiLU())
        self.s1=Block(32,48,2)
        self.s2=Block(48,96,2)
        self.s3=Block(96,160,2); self.mid=Block(160,160)
        self.up2=nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(160,96,3,1,1),nn.SiLU())
        self.d2=Block(192,112)
        self.up1=nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(112,72,3,1,1),nn.SiLU())
        self.d1=Block(120,80)
        self.up0=nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(80,48,3,1,1),nn.SiLU())
        self.d0=Block(80,64)
        self.body_hm=nn.Sequential(nn.Conv2d(64,64,3,1,1),nn.SiLU(),nn.Conv2d(64,BODY_KP,1))
        self.hand_hm=nn.Sequential(nn.Conv2d(64,96,3,1,1),nn.SiLU(),nn.Conv2d(96,2*HAND_KP,1))
        self.zoff=nn.Sequential(nn.Conv2d(64,64,3,1,1),nn.SiLU(),nn.Conv2d(64,KP,1))
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.zfallback=nn.Sequential(nn.Linear(160,256),nn.SiLU(),nn.Linear(256,KP))
        self.chead=nn.Sequential(nn.Linear(160,256),nn.SiLU(),nn.Linear(256,KP))
    def forward(self,x):
        a=self.s0(x); b=self.s1(a); c=self.s2(b); d=self.mid(self.s3(c))
        q=self.d2(torch.cat([self.up2(d),c],1)); q=self.d1(torch.cat([self.up1(q),b],1)); hi=self.d0(torch.cat([self.up0(q),a],1))
        hm=torch.cat([self.body_hm(hi),self.hand_hm(hi)],1)

        valid=(x>-1.75).float(); mm=torch.where(valid>0.5,x*1800.0+2200.0,torch.zeros_like(x))
        v=F.avg_pool2d(valid,2,2); m=F.avg_pool2d(mm,2,2)/(v+1e-4); local_valid=(v>.05).float()
        p=F.softmax((hm*1.8).flatten(2),dim=-1).reshape(hm.shape)
        off=torch.tanh(self.zoff(hi))*280.0; den=(p*local_valid).sum((2,3)); local=((p*local_valid)*(m+off)).sum((2,3))/(den+1e-4)
        g=self.pool(d).flatten(1); fallback=torch.sigmoid(self.zfallback(g))*5600.0+400.0
        support=torch.clamp(den,0.0,1.0); z=local*support+fallback*(1.0-support)
        conf=torch.sigmoid(self.chead(g))
        return hm,z,conf

class HandDepthNet(nn.Module):
    """High-resolution depth crop refiner shared by left and right hands."""
    def __init__(self):
        super().__init__(); self.s0=nn.Sequential(nn.Conv2d(1,24,5,2,2,bias=False),nn.BatchNorm2d(24),nn.SiLU()); self.s1=Block(24,40,2); self.s2=Block(40,72,2); self.s3=Block(72,112,2); self.mid=Block(112,112)
        self.u2=nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(112,72,3,1,1),nn.SiLU()); self.d2=Block(144,80)
        self.u1=nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(80,48,3,1,1),nn.SiLU()); self.d1=Block(88,56)
        self.u0=nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(56,36,3,1,1),nn.SiLU()); self.d0=Block(60,44)
        self.hm=nn.Sequential(nn.Conv2d(44,64,3,1,1),nn.SiLU(),nn.Conv2d(64,HAND_KP,1)); self.zoff=nn.Sequential(nn.Conv2d(44,44,3,1,1),nn.SiLU(),nn.Conv2d(44,HAND_KP,1)); self.pool=nn.AdaptiveAvgPool2d(1); self.zfallback=nn.Sequential(nn.Linear(112,128),nn.SiLU(),nn.Linear(128,HAND_KP)); self.chead=nn.Sequential(nn.Linear(112,128),nn.SiLU(),nn.Linear(128,HAND_KP))
    def forward(self,x):
        a=self.s0(x); b=self.s1(a); c=self.s2(b); d=self.mid(self.s3(c)); q=self.d2(torch.cat([self.u2(d),c],1)); q=self.d1(torch.cat([self.u1(q),b],1)); hi=self.d0(torch.cat([self.u0(q),a],1)); hm=self.hm(hi)
        valid=(x>-1.75).float(); mm=torch.where(valid>.5,x*1800.0+2200.0,torch.zeros_like(x)); v=F.avg_pool2d(valid,2,2); m=F.avg_pool2d(mm,2,2)/(v+1e-4); lv=(v>.05).float(); p=F.softmax((hm*2.0).flatten(2),dim=-1).reshape(hm.shape); off=torch.tanh(self.zoff(hi))*180.0; den=(p*lv).sum((2,3)); local=((p*lv)*(m+off)).sum((2,3))/(den+1e-4); g=self.pool(d).flatten(1); fallback=torch.sigmoid(self.zfallback(g))*5600.0+400.0; z=local*torch.clamp(den,0,1)+fallback*(1-torch.clamp(den,0,1)); conf=torch.sigmoid(self.chead(g)); return hm,z,conf

class HandExportModel(nn.Module):
    def __init__(self,m): super().__init__(); self.m=m
    def forward(self,x):
        hm,z,conf=self.m(x); xy=soft_xy(hm); return torch.cat([xy,z.unsqueeze(-1),conf.unsqueeze(-1)],dim=-1)

_CONST_CACHE={}

def _cached_tensor(name,values,device,dtype=torch.long):
    key=(name,str(device),str(dtype)); t=_CONST_CACHE.get(key)
    if t is None:
        t=torch.tensor(values,device=device,dtype=dtype); _CONST_CACHE[key]=t
    return t

def spatial_probs(logits): return F.softmax(logits.flatten(2),dim=-1).reshape(logits.shape)
def soft_xy(logits):
    b,k,h,w=logits.shape; p=spatial_probs(logits); key=('softxy',str(logits.device),str(logits.dtype),h,w); grids=_CONST_CACHE.get(key)
    if grids is None:
        gx=torch.linspace(0,1,w,device=logits.device,dtype=logits.dtype).view(1,1,1,w); gy=torch.linspace(0,1,h,device=logits.device,dtype=logits.dtype).view(1,1,h,1); grids=(gx,gy); _CONST_CACHE[key]=grids
    gx,gy=grids; return torch.stack([(p*gx).sum((2,3)),(p*gy).sum((2,3))],-1)

def target_heatmaps(xy,vis,h,w,hand_only=False,dtype=None):
    dtype=dtype or xy.dtype; device=xy.device; key=('hmgrid',str(device),str(dtype),h,w); grids=_CONST_CACHE.get(key)
    if grids is None:
        gx=torch.arange(w,device=device,dtype=dtype).view(1,1,1,w); gy=torch.arange(h,device=device,dtype=dtype).view(1,1,h,1); grids=(gx,gy); _CONST_CACHE[key]=grids
    gx,gy=grids; q=xy.to(dtype=dtype); cx=q[...,0,None,None]*(w-1); cy=q[...,1,None,None]*(h-1)
    if hand_only:
        sigma=q.new_full((1,q.shape[1],1,1),2.15)
    else:
        sig=[3.0 if i<BODY_KP else 2.2 for i in range(q.shape[1])]; sigma=_cached_tensor('body_hm_sigma',sig,device,dtype).view(1,-1,1,1)
    hm=torch.exp(-((gx-cx)**2+(gy-cy)**2)/(2*sigma*sigma)); return hm*(vis.to(dtype=dtype)[...,None,None]>.10).to(dtype)

def xyz_from_xy_z(xy,z):
    u=xy[...,0]*639.0; v=xy[...,1]*479.0; X=(u-339.3)*z/594.2; Y=(v-242.7)*z/591.0; return torch.stack([X,Y,z],dim=-1)
def safe_norm(x,dim=-1): return torch.sqrt(torch.clamp((x*x).sum(dim=dim),min=1e-12))
def robust_l1(a,b,delta=1.0):
    d=(a-b)/delta; return (delta*delta)*(torch.sqrt(1.0+d*d+1e-12)-1.0)

def structural_losses(pxy,pz,txy,tz,vis):
    pxyz=xyz_from_xy_z(pxy,pz); txyz=xyz_from_xy_z(txy,tz); dev=pxy.device
    ea=_cached_tensor('seg_a',[a for a,b in SEGMENTS],dev); eb=_cached_tensor('seg_b',[b for a,b in SEGMENTS],dev)
    m=((vis.index_select(1,ea)>.10)&(vis.index_select(1,eb)>.10)).float(); pv=pxyz.index_select(1,eb)-pxyz.index_select(1,ea); tv=txyz.index_select(1,eb)-txyz.index_select(1,ea)
    den=m.sum(); bone=(robust_l1(pv/500.0,tv/500.0).sum(-1)*m).sum()/(den+1); blen=(robust_l1(safe_norm(pv)/500.0,safe_norm(tv)/500.0)*m).sum()/(den+1)
    ep_idx=_cached_tensor('endpoints',ENDPOINTS,dev); em=(vis.index_select(1,ep_idx)>.10).float(); pe=pxyz.index_select(1,ep_idx); te=txyz.index_select(1,ep_idx); endpoint=(robust_l1(pe/500.0,te/500.0).sum(-1)*em).sum()/(em.sum()+1)
    return bone,blen,endpoint

def hand_full_xy(local_xy,crop_meta):
    x=crop_meta[:,None,0]+local_xy[...,0]*crop_meta[:,None,2]; y=crop_meta[:,None,1]+local_xy[...,1]*crop_meta[:,None,3]; return torch.stack([x,y],-1)

def hand_structural_losses(pxy,pz,txy,tz,vis,crop_meta):
    pfull=hand_full_xy(pxy,crop_meta); tfull=hand_full_xy(txy,crop_meta); pxyz=xyz_from_xy_z(pfull,pz); txyz=xyz_from_xy_z(tfull,tz); dev=pxy.device
    ea=_cached_tensor('hand_a',[a for a,b in HAND_LOCAL],dev); eb=_cached_tensor('hand_b',[b for a,b in HAND_LOCAL],dev); m=((vis.index_select(1,ea)>.10)&(vis.index_select(1,eb)>.10)).float(); pv=pxyz.index_select(1,eb)-pxyz.index_select(1,ea); tv=txyz.index_select(1,eb)-txyz.index_select(1,ea); den=m.sum()
    bone=(robust_l1(pv/120.0,tv/120.0).sum(-1)*m).sum()/(den+1); blen=(robust_l1(safe_norm(pv)/80.0,safe_norm(tv)/80.0)*m).sum()/(den+1)
    tips=_cached_tensor('hand_tips',[4,8,12,16,20],dev); tm=(vis.index_select(1,tips)>.10).float(); pe=pxyz.index_select(1,tips); te=txyz.index_select(1,tips); endpoint=(robust_l1(pe/120.0,te/120.0).sum(-1)*tm).sum()/(tm.sum()+1)
    return bone,blen,endpoint

def compute_hand_loss(model,x,z,vis,xy,crop_meta):
    ohm,oz,oc=model(x); hm=target_heatmaps(xy,vis,HAND_HM,HAND_HM,True,ohm.dtype); mask=(vis>.10).float(); weights=_cached_tensor('hand_weights',[1.0,1.0,1.0,1.0,1.45,1.0,1.0,1.0,1.45,1.0,1.0,1.0,1.45,1.0,1.0,1.0,1.45,1.0,1.0,1.0,1.45],vis.device,vis.dtype).view(1,-1).expand_as(vis); cw=(.16+.84*mask)*weights; pix=(torch.sigmoid(ohm)-hm)**2*(.12+hm*3.8); lhm=(pix*cw.unsqueeze(-1).unsqueeze(-1)).sum()/(cw.sum()*HAND_HM*HAND_HM+1); pxy=soft_xy(ohm); lxy=(robust_l1(pxy,xy).sum(-1)*mask*weights).sum()/((mask*weights).sum()+1); lz=(robust_l1(oz/6000.0,z/6000.0)*mask*weights).sum()/((mask*weights).sum()+1); lc=(((oc-vis)*(oc-vis))*weights).sum()/(weights.sum()+1); lb,ll,le=hand_structural_losses(pxy,oz,xy,z,vis,crop_meta); loss=1.15*lhm+2.3*lxy+.95*lz+.20*lc+.48*lb+.16*ll+.72*le; return loss,pxy,(lhm,lxy,lz,lc,lb,ll,le)

class ExportModel(nn.Module):
    def __init__(self,m): super().__init__(); self.m=m
    def forward(self,x):
        hm,z,conf=self.m(x); xy=soft_xy(hm); kp=torch.cat([xy,z.unsqueeze(-1),conf.unsqueeze(-1)],dim=-1); seg=[]
        for a,b in SEGMENTS:
            ca=torch.minimum(conf[:,a],conf[:,b]).unsqueeze(-1); pa=torch.cat([xy[:,a],z[:,a].unsqueeze(-1)],-1); pb=torch.cat([xy[:,b],z[:,b].unsqueeze(-1)],-1); length=safe_norm(xyz_from_xy_z(xy[:,b],z[:,b])-xyz_from_xy_z(xy[:,a],z[:,a])).unsqueeze(-1); seg.append(torch.cat([pa,pb,ca,length],-1))
        return kp,torch.stack(seg,dim=1),kp[:,ENDPOINTS,:]

def fmt_eta(sec): sec=max(0,int(sec)); return f'{sec//3600:02d}:{(sec%3600)//60:02d}:{sec%60:02d}'

class MutableIndexSampler(Sampler):
    def __init__(self,indices=()): self.indices=list(indices)
    def set_indices(self,indices): self.indices=list(indices)
    def __iter__(self): return iter(self.indices)
    def __len__(self): return len(self.indices)

def _seed_worker(worker_id):
    seed=torch.initial_seed()%2**32; random.seed(seed); np.random.seed(seed)

def auto_workers(requested,backend):
    if requested>=0: return requested
    cpu=max(1,os.cpu_count() or 1)
    if cpu<=2: return 0
    cap=6 if os.name=='nt' else 8
    if backend=='CPU': return min(cap,max(1,cpu-1))
    return min(cap,max(2,cpu//2))

def make_loader(dataset,batch,backend,workers,sampler=None,shuffle=False,prefetch=3):
    kw={'batch_size':batch,'shuffle':shuffle if sampler is None else False,'sampler':sampler,'num_workers':workers,'pin_memory':backend in ('CUDA','ROCM'),'worker_init_fn':_seed_worker if workers>0 else None}
    if workers>0: kw.update(persistent_workers=True,prefetch_factor=max(1,prefetch))
    return DataLoader(dataset,**kw)

def configure_backend(backend):
    try: torch.set_float32_matmul_precision('high')
    except Exception: pass
    if backend in ('CUDA','ROCM'):
        try: torch.backends.cudnn.benchmark=True
        except Exception: pass
        try: torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
        except Exception: pass

def select_device():
    status_path=Path(__file__).resolve().parent.parent/'runtime_status.json'
    if not status_path.is_file() and os.name=='nt':
        base=os.environ.get('LOCALAPPDATA'); status_path=Path(base)/'KinectDepthAutoLearn'/'runtime_status.json' if base else status_path
    if not status_path.is_file() and os.name!='nt':
        base=Path(os.environ.get('XDG_DATA_HOME',str(Path.home()/'.local'/'share')))/'kinect-depth-autolearn'; status_path=base/'runtime_status.json'
    if not status_path.is_file(): raise RuntimeError('runtime_status.json is missing. Run the platform install_autolearn installer first.')
    status=json.loads(status_path.read_text(encoding='utf-8-sig')); backend=str(status.get('backend','')).upper()
    if backend=='CPU': return torch.device('cpu'),'CPU','CPU'
    if backend in ('CUDA','ROCM'):
        if not torch.cuda.is_available(): raise RuntimeError(f'{backend} selected, but torch.cuda is unavailable in this runtime.')
        return torch.device('cuda'),backend,torch.cuda.get_device_name(0)
    if backend=='DIRECTML':
        import torch_directml; dev=torch_directml.device(); probe=(torch.ones(1,device=dev)+1).cpu().item()
        if probe!=2: raise RuntimeError(f'DirectML test returned {probe}')
        return dev,'DirectML',str(status.get('gpu') or 'GPU DirectX 12')
    if backend=='MPS':
        if not hasattr(torch.backends,'mps') or not torch.backends.mps.is_available(): raise RuntimeError('MPS selected, but it is unavailable in this runtime.')
        return torch.device('mps'),'MPS','Apple Metal'
    raise RuntimeError(f'Unknown backend: {backend!r}')

def move_batch(items,dev,backend): return [t.to(dev,non_blocking=(backend in ('CUDA','ROCM'))) for t in items]

def landmark_weights(vis):
    vals=[1.0]*KP
    for i in range(BODY_KP,KP): vals[i]=1.35
    for i in ENDPOINTS: vals[i]=1.55
    return _cached_tensor('landmark_weights',vals,vis.device,vis.dtype).view(1,-1).expand_as(vis)

def compute_loss(model,x,z,vis,xy):
    ohm,oz,oc=model(x); hm=target_heatmaps(xy,vis,HM_H,HM_W,False,ohm.dtype); mask=(vis>.10).float(); lw=landmark_weights(vis); channel_w=(.18+.82*mask)*lw
    pixel_err=(torch.sigmoid(ohm)-hm)**2*(.15+hm*3.4); lhm=(pixel_err*channel_w.unsqueeze(-1).unsqueeze(-1)).sum()/(channel_w.sum()*HM_H*HM_W+1)
    lz=(robust_l1(oz/6000.0,z/6000.0)*mask*lw).sum()/((mask*lw).sum()+1)
    lc=(((oc-vis)*(oc-vis))*lw).sum()/(lw.sum()+1)
    pxy=soft_xy(ohm); lxy=(robust_l1(pxy,xy).sum(-1)*mask*lw).sum()/((mask*lw).sum()+1)
    lb,ll,le=structural_losses(pxy,oz,xy,z,vis); loss=1.15*lhm+2.0*lxy+.85*lz+.22*lc+.42*lb+.18*ll+.62*le
    return loss,pxy,(lhm,lxy,lz,lc,lb,ll,le)

def cpu_tree(obj):
    if torch.is_tensor(obj): return obj.detach().cpu()
    if isinstance(obj,dict): return {k:cpu_tree(v) for k,v in obj.items()}
    if isinstance(obj,list): return [cpu_tree(v) for v in obj]
    if isinstance(obj,tuple): return tuple(cpu_tree(v) for v in obj)
    return obj

def optimizer_to(opt,dev):
    for st in opt.state.values():
        for k,v in list(st.items()):
            if torch.is_tensor(v): st[k]=v.to(dev)

def read_control(out):
    p=Path(out)/'control.json'
    try: return json.loads(p.read_text(encoding='utf-8')) if p.is_file() else {}
    except Exception: return {}
def write_control(out,paused=False): _atomic_write_text(Path(out)/'control.json',json.dumps({'paused':bool(paused),'updated':time.time()}),retries=20,nonfatal=False)

def split_fingerprint(ds,tr_idx,va_idx):
    h=hashlib.sha256()
    for tag,indices in ((b'T',tr_idx),(b'V',va_idx)):
        h.update(tag)
        for i in indices:
            fn,seq,tick=ds.meta[i]; h.update(f'{fn}|{seq}|{tick}\n'.encode())
    return h.hexdigest()

def temporal_group_split(ds,val_fraction=.12,group_ms=30000,guard_ms=1500):
    n=len(ds); raw_ticks=np.asarray([m[2] for m in ds.meta],dtype=np.int64)


    ticks=np.zeros_like(raw_ticks); offset=0; prev=int(raw_ticks[0]) if n else 0
    for i,tv in enumerate(raw_ticks):
        t=int(tv)
        if i>0 and t+offset < ticks[i-1]-1000: offset=int(ticks[i-1]+100-t)
        ticks[i]=t+offset; prev=t
    if n<12:
        cut=max(1,int(round(n*val_fraction))); return list(range(0,n-cut)),list(range(n-cut,n)),{'strategy':'tail-small-dataset','group_ms':0,'guard_ms':0}
    t0=int(ticks.min()); rel=np.maximum(0,ticks-t0); groups=(rel//max(1000,group_ms)).astype(np.int64)
    uniq=np.unique(groups)
    if len(uniq)<4:
        groups=np.arange(n,dtype=np.int64)//max(12,int(round(group_ms/100.0))); uniq=np.unique(groups); rel=np.arange(n,dtype=np.int64)*100
    nvg=max(1,int(round(len(uniq)*val_fraction))); nvg=min(nvg,max(1,len(uniq)-1))
    pos=np.linspace(0,len(uniq)-1,nvg+2)[1:-1] if nvg>1 else np.array([(len(uniq)-1)*.5])
    val_groups=set(int(uniq[int(round(p))]) for p in pos)

    for g in uniq:
        if len(val_groups)>=nvg: break
        if int(g) not in val_groups: val_groups.add(int(g))
    va=[i for i,g in enumerate(groups) if int(g) in val_groups]
    tr=[]; guard_units=float(guard_ms)
    intervals=[]
    for g in sorted(val_groups):
        ids=np.where(groups==g)[0]
        if ids.size: intervals.append((float(rel[ids].min()),float(rel[ids].max())))
    for i,g in enumerate(groups):
        if int(g) in val_groups: continue
        t=float(rel[i]); near=any((lo-guard_units)<=t<=(hi+guard_units) for lo,hi in intervals)
        if not near: tr.append(i)
    if len(tr)<max(2,n//2) or len(va)<1:

        cut=max(1,int(round(n*val_fraction))); va=list(range(n-cut,n)); guard=max(1,int(round(guard_ms/100.0))); tr=list(range(0,max(1,n-cut-guard))); val_groups={'tail'}
    info={'strategy':'temporal-groups','group_ms':group_ms,'guard_ms':guard_ms,'val_fraction':val_fraction,'groups_total':int(len(uniq)),'val_groups':[str(v) for v in sorted(val_groups,key=str)],'train_samples':len(tr),'val_samples':len(va),'guarded_samples':n-len(tr)-len(va)}
    return tr,va,info


def save_checkpoint(path,model,opt,sched,epoch,batch,best,no_improve,done,ema_train,ema_val,elapsed,backend,split_hash):
    state={'version':CHECKPOINT_VERSION,'architecture':'SynSkeletonV2','model_version':MODEL_VERSION,'split_hash':split_hash,'model':cpu_tree(model.state_dict()),'optimizer':cpu_tree(opt.state_dict()),'scheduler':sched.state_dict(),'epoch':epoch,'batch':batch,'best':best,'no_improve':no_improve,'done':done,'ema_train':ema_train,'ema_val':ema_val,'elapsed':elapsed,'backend':backend,'saved':time.time()}
    _torch_save_resilient(state,path,retries=20,nonfatal=True)

def latest_dataset(base):
    ds=Path(base)/'dataset'; cand=[p for p in ds.glob('session_*') if (p/'labels.csv').is_file()]
    if not cand: raise RuntimeError('No dataset/session_* with labels.csv was found.')
    return max(cand,key=lambda p:p.name)
def deterministic_epoch_indices(indices,epoch,ds=None):
    g=torch.Generator().manual_seed(0x4B1D+epoch*1009); n=len(indices)
    if n<=1: return list(indices)
    order=torch.randperm(n,generator=g).tolist(); shuffled=[indices[i] for i in order]
    if ds is None: return shuffled
    hands=[i for i in indices if i<len(ds.hand_bits) and (ds.hand_bits[i][0] or ds.hand_bits[i][1])]
    if not hands or len(hands)>=int(.45*n): return shuffled
    target_hand=max(len(hands),int(round(.45*n))); target_hand=min(target_hand,n)
    hand_set=set(hands); body=[i for i in indices if i not in hand_set]; bo=torch.randperm(len(body),generator=g).tolist() if body else []; ho=torch.randperm(len(hands),generator=g).tolist()
    chosen=[body[j] for j in bo[:max(0,n-target_hand)]]
    chosen.extend(hands[j] for j in ho)
    while len(chosen)<n: chosen.append(hands[int(torch.randint(0,len(hands),(1,),generator=g).item())])
    mix=torch.randperm(len(chosen),generator=g).tolist(); return [chosen[j] for j in mix]

def validation_metrics(model,loader,dev,backend):
    model.eval(); sums={'loss':0.0,'xy_body':0.0,'xy_hand':0.0,'z_body':0.0,'z_hand':0.0}; counts={'batches':0,'body':0.0,'hand':0.0}
    with torch.inference_mode():
        for x,z,vis,xy in loader:
            x,z,vis,xy=move_batch((x,z,vis,xy),dev,backend); ohm,oz,oc=model(x); pxy=soft_xy(ohm); lb,ll,le=structural_losses(pxy,oz,xy,z,vis); mask=(vis>.10).float(); xyerr=(pxy-xy).abs().sum(-1); zerr=(oz-z).abs()
            score=(xyerr*mask).sum()/(mask.sum()+1)+.20*lb+.10*ll+.25*le; sums['loss']+=float(score.item()); counts['batches']+=1
            bm=mask[:,:BODY_KP]; hm_=mask[:,BODY_KP:]; sums['xy_body']+=float((xyerr[:,:BODY_KP]*bm).sum().item()); sums['z_body']+=float((zerr[:,:BODY_KP]*bm).sum().item()); counts['body']+=float(bm.sum().item())
            sums['xy_hand']+=float((xyerr[:,BODY_KP:]*hm_).sum().item()); sums['z_hand']+=float((zerr[:,BODY_KP:]*hm_).sum().item()); counts['hand']+=float(hm_.sum().item())
    return {'val_struct':sums['loss']/max(1,counts['batches']),'body_xy_l1_norm':sums['xy_body']/max(1,counts['body']),'hand_xy_l1_norm':sums['xy_hand']/max(1,counts['hand']),'body_z_mae_mm':sums['z_body']/max(1,counts['body']),'hand_z_mae_mm':sums['z_hand']/max(1,counts['hand']),'body_points':int(counts['body']),'hand_points':int(counts['hand'])}


def hand_validation_metrics(model,loader,dev,backend):
    model.eval(); xy_sum=z_sum=tip_sum=0.0; count=tip_count=0.0; loss_sum=0.0; batches=0
    with torch.inference_mode():
        for x,z,vis,xy,crop in loader:
            x,z,vis,xy,crop=move_batch((x,z,vis,xy,crop),dev,backend); loss,pxy,_=compute_hand_loss(model,x,z,vis,xy,crop); ohm,oz,oc=model(x); mask=(vis>.10).float(); err=(pxy-xy).abs().sum(-1); ze=(oz-z).abs(); xy_sum+=float((err*mask).sum().item()); z_sum+=float((ze*mask).sum().item()); count+=float(mask.sum().item()); tips=_cached_tensor('hand_tips',[4,8,12,16,20],dev); tm=mask.index_select(1,tips); tip_sum+=float((err.index_select(1,tips)*tm).sum().item()); tip_count+=float(tm.sum().item()); loss_sum+=float(loss.item()); batches+=1
    return {'val_loss':loss_sum/max(1,batches),'xy_l1_local':xy_sum/max(1,count),'tip_xy_l1_local':tip_sum/max(1,tip_count),'z_mae_mm':z_sum/max(1,count),'points':int(count),'tip_points':int(tip_count)}


def verify_onnx(path,input_shape,first_output_shape):
    import onnxruntime as ort
    sess=ort.InferenceSession(str(path),providers=['CPUExecutionProvider']); name=sess.get_inputs()[0].name; x=np.full(input_shape,-2.0,dtype=np.float32); out=sess.run(None,{name:x})
    if not out or tuple(out[0].shape)!=tuple(first_output_shape): raise RuntimeError(f'ONNX self-test shape mismatch for {Path(path).name}: got {None if not out else out[0].shape}, expected {first_output_shape}')
    if not np.isfinite(out[0]).all(): raise RuntimeError(f'ONNX self-test found non-finite output in {Path(path).name}')
    return True

def train_hand_refiner(train_ds,val_ds,out,dev,backend,gpu,args,prog,split_hash,elapsed_fn,progress_start=.74,progress_end=.995):
    if len(train_ds)<64 or len(val_ds)<12:
        print(f'HAND REFINER skipped: train crops={len(train_ds)} val crops={len(val_ds)} (minimum 64/12).',flush=True); return None
    defaults={'CPU':8,'CUDA':32,'ROCM':32,'DirectML':8,'MPS':16}; batch=args.hand_batch if args.hand_batch>0 else defaults.get(backend,8); workers=auto_workers(args.workers,backend); configure_backend(backend)
    model=HandDepthNet().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=args.hand_lr,weight_decay=1.5e-4,foreach=False); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.hand_epochs); amp=backend in ('CUDA','ROCM'); scaler=torch.amp.GradScaler('cuda') if amp else None; ckpt=Path(out)/'hand_last.pt'; bestp=Path(out)/'hand_best.pt'; best=1e9; no_improve=0; start_ep=1; start_batch=0; done=0; ema=None
    if backend=='DirectML':
        try:
            sx=torch.full((1,1,HAND_IN,HAND_IN),-2.0,device=dev); sz=torch.full((1,HAND_KP),1800.0,device=dev); sv=torch.ones(1,HAND_KP,device=dev); sxy=torch.full((1,HAND_KP,2),0.5,device=dev); scrop=torch.tensor([[.25,.25,.25,.33]],device=dev); opt.zero_grad(set_to_none=True); sloss,_,_=compute_hand_loss(model,sx,sz,sv,sxy,scrop); sloss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5,foreach=False); opt.zero_grad(set_to_none=True); print('DirectML hand-refiner self-test: OK',flush=True)
        except Exception as e:
            print(f'WARNING: DirectML hand-refiner self-test failed: {type(e).__name__}: {e}',flush=True); print('WARNING: hand refiner will train on CPU while the body model remains valid.',flush=True); dev=torch.device('cpu');backend='CPU';gpu='CPU fallback for SynHand';batch=args.hand_batch if args.hand_batch>0 else defaults['CPU'];workers=auto_workers(args.workers,backend);model=HandDepthNet().to(dev);opt=torch.optim.AdamW(model.parameters(),lr=args.hand_lr,weight_decay=1.5e-4,foreach=False);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.hand_epochs);amp=False;scaler=None
    item_hash=hashlib.sha256((split_hash+f'|{len(train_ds)}|{len(val_ds)}').encode()).hexdigest()
    if args.resume!='never' and ckpt.is_file():
        try:
            st=torch.load(ckpt,map_location='cpu')
            if st.get('version')!=1 or st.get('architecture')!='SynHandV1' or st.get('item_hash')!=item_hash: raise RuntimeError('incompatible hand checkpoint')
            model.load_state_dict(st['model']); opt.load_state_dict(st['optimizer']); optimizer_to(opt,dev); sched.load_state_dict(st['scheduler']); start_ep=int(st['epoch']); start_batch=int(st['batch']); best=float(st.get('best',1e9)); no_improve=int(st.get('no_improve',0)); done=int(st.get('done',0)); ema=st.get('ema')
            full=math.ceil(len(train_ds)/batch)
            if start_batch>=full: start_ep+=1;start_batch=0
            if best<1e8 and not bestp.is_file(): raise RuntimeError('hand best checkpoint missing')
            print(f'HAND RESUME epoch={start_ep} batch={start_batch} best={best:.5f}',flush=True)
        except Exception as e:
            if args.resume=='required': raise
            print(f'WARNING: hand checkpoint not resumed: {e}',flush=True); start_ep=1;start_batch=0;done=0;best=1e9;no_improve=0
    if start_ep==1 and start_batch==0 and done==0:
        bestp.unlink(missing_ok=True)
    def save_hand(ep,bi):
        st={'version':1,'architecture':'SynHandV1','item_hash':item_hash,'model':cpu_tree(model.state_dict()),'optimizer':cpu_tree(opt.state_dict()),'scheduler':sched.state_dict(),'epoch':ep,'batch':bi,'best':best,'no_improve':no_improve,'done':done,'ema':ema,'saved':time.time()}; _torch_save_resilient(st,ckpt,retries=20,nonfatal=True)
    vl=make_loader(val_ds,batch,backend,workers,prefetch=args.prefetch); sampler=MutableIndexSampler(); tl=make_loader(train_ds,batch,backend,workers,sampler=sampler,prefetch=args.prefetch); total=max(1,args.hand_epochs*math.ceil(len(train_ds)/batch)); last_loss=0.0; last_metrics={}; ep=max(1,start_ep)
    print(f'HAND REFINER crops train={len(train_ds)} val={len(val_ds)} params={sum(p.numel() for p in model.parameters()):,} batch={batch} epochs={args.hand_epochs} workers={workers} prefetch={args.prefetch} amp={amp}',flush=True)
    for ep in range(start_ep,args.hand_epochs+1):
        g=torch.Generator().manual_seed(0x71A9+ep*997); order=torch.randperm(len(train_ds),generator=g).tolist(); skip=start_batch if ep==start_ep else 0; sampler.set_indices(order[skip*batch:]); model.train(); total_loss=0.0; seen=0; full=math.ceil(len(train_ds)/batch)
        for lbi,batch_data in enumerate(tl,1):
            bi=skip+lbi; t0=time.time(); x,z,vis,xy,crop=move_batch(batch_data,dev,backend); opt.zero_grad(set_to_none=True)
            if amp:
                with torch.amp.autocast('cuda'): loss,_,_=compute_hand_loss(model,x,z,vis,xy,crop)
                scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),5,foreach=False);scaler.step(opt);scaler.update()
            else:
                loss,_,_=compute_hand_loss(model,x,z,vis,xy,crop);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5,foreach=False);opt.step()
            total_loss+=loss.item();seen+=1;done+=1;last_loss=total_loss/max(1,seen);dt=time.time()-t0;ema=dt if ema is None else ema*.92+dt*.08
            if bi%max(1,args.checkpoint_batches)==0: save_hand(ep,bi)
            frac=min(1.0,done/max(1,total)); overall=progress_start+(progress_end-progress_start)*frac; eta=(ema or 0)*max(0,total-done)
            prog.write(force=(bi==skip+1 or bi==full),state='training-hand-refiner',phase='hand_refiner',device=backend,gpu=gpu,epoch=ep,epochs=args.hand_epochs,batch=bi,batches=full,progress=overall,loss=last_loss,val=last_metrics.get('val_loss',0),best=best if best<1e8 else 0,eta=fmt_eta(eta),elapsed=fmt_eta(elapsed_fn()),samples=len(train_ds),patience=no_improve,workers=workers)
            if read_control(out).get('paused',False):
                save_hand(ep,bi); print(f'HAND PAUSED at epoch {ep}/{args.hand_epochs}, batch {bi}.',flush=True)
                while read_control(out).get('paused',False): time.sleep(.5)
                print('HAND RESUMING...',flush=True)
        sched.step(); last_metrics=hand_validation_metrics(model,vl,dev,backend); v=last_metrics['val_loss']; print(f'Hand epoch {ep:03d}/{args.hand_epochs} loss={last_loss:.5f} val={v:.5f} xy={last_metrics["xy_l1_local"]:.4f} tip={last_metrics["tip_xy_l1_local"]:.4f} z={last_metrics["z_mae_mm"]:.1f}mm',flush=True)
        if v<best-args.hand_min_delta: best=v;no_improve=0;_torch_save_resilient(cpu_tree(model.state_dict()),bestp,retries=20,nonfatal=False)
        else: no_improve+=1
        save_hand(ep,full);start_batch=0
        if args.hand_patience>0 and ep>=args.hand_min_epochs and no_improve>=args.hand_patience: print(f'Hand refiner early stopping after {no_improve} epochs without improvement.',flush=True);break
    if not bestp.is_file(): raise RuntimeError('No validated hand checkpoint available for ONNX export')
    state=torch.load(bestp,map_location='cpu'); cpu=HandDepthNet().cpu(); cpu.load_state_dict(state); wrapper=HandExportModel(cpu.eval()).eval(); hand_onnx=Path(out)/'SynHand.onnx'; dummy=torch.full((1,1,HAND_IN,HAND_IN),-2.0); torch.onnx.export(wrapper,dummy,hand_onnx,input_names=['hand_depth_metric'],output_names=['keypoints'],opset_version=17,dynamic_axes={'hand_depth_metric':{0:'batch'},'keypoints':{0:'batch'}},dynamo=False); verify_onnx(hand_onnx,(1,1,HAND_IN,HAND_IN),(1,HAND_KP,4)); print('HAND READY:',hand_onnx,flush=True); return {'model':'SynHand.onnx','train_crops':len(train_ds),'val_crops':len(val_ds),'best_val':best,'metrics':last_metrics}

def run(args):
    if args.epochs<1 or args.hand_epochs<1 or args.min_epochs<1 or args.hand_min_epochs<1:
        raise ValueError('Epoch limits must be positive integers.')
    if args.patience<0 or args.hand_patience<0 or args.checkpoint_batches<1:
        raise ValueError('Patience must be nonnegative and checkpoint-batches >= 1.')
    if not (0<args.val_fraction<0.5) or args.val_group_ms<1000 or args.val_guard_ms<0:
        raise ValueError('Invalid temporal validation split settings.')
    root=Path(args.dataset).resolve(); out=root/'output'; out.mkdir(parents=True,exist_ok=True); _make_user_writable(out)
    probe=out/'.write_probe'; _atomic_write_text(probe,'ok',retries=10,nonfatal=False); probe.unlink(missing_ok=True)
    prog=Progress(out,args.progress_ms); ds_train=DepthDataset(root,True)
    if len(ds_train)<2: raise RuntimeError('Dataset is too small or its labels are invalid.')
    ds_val=DepthDataset(root,False); tr_idx,va_idx,split_info=temporal_group_split(ds_train,args.val_fraction,args.val_group_ms,args.val_guard_ms)
    if len(tr_idx)<2 or len(va_idx)<1: raise RuntimeError('Temporal split produced too few samples. Collect a longer session.')
    split_hash=split_fingerprint(ds_train,tr_idx,va_idx); split_info['fingerprint']=split_hash; split_info['model_version']=MODEL_VERSION
    _atomic_write_text(out/'dataset_split.json',json.dumps(split_info,indent=2),nonfatal=True)
    hand_train_ds=HandCropDataset(ds_train,tr_idx,True); hand_val_ds=HandCropDataset(ds_val,va_idx,False); hand_ready=len(hand_train_ds)>=64 and len(hand_val_ds)>=12; body_progress_max=.74 if hand_ready else .99
    dev,backend,gpu=select_device(); configure_backend(backend); defaults={'CPU':4,'CUDA':8,'ROCM':8,'DirectML':4,'MPS':6}; batch=args.batch if args.batch>0 else defaults.get(backend,4); workers=auto_workers(args.workers,backend)
    def make_runtime(cur_dev,cur_backend):
        mdl=DepthPoseNet().to(cur_dev); optimizer=torch.optim.AdamW(mdl.parameters(),lr=args.lr,weight_decay=1.5e-4,foreach=False); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs); use_amp=cur_backend in ('CUDA','ROCM'); amp_scaler=torch.amp.GradScaler('cuda') if use_amp else None; return mdl,optimizer,scheduler,use_amp,amp_scaler
    model,opt,sched,amp,scaler=make_runtime(dev,backend)
    if backend=='DirectML':
        try:
            sx=torch.full((1,1,IN_H,IN_W),-2.0,device=dev); sz=torch.full((1,KP),1800.0,device=dev); sv=torch.ones(1,KP,device=dev); sxy=torch.full((1,KP,2),0.5,device=dev); opt.zero_grad(set_to_none=True); sloss,_,_=compute_loss(model,sx,sz,sv,sxy); sloss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5,foreach=False); opt.zero_grad(set_to_none=True); print('DirectML full training self-test: OK',flush=True)
        except Exception as e:
            print(f'WARNING: DirectML self-test failed: {type(e).__name__}: {e}',flush=True); print('WARNING: continuing safely on CPU.',flush=True); dev=torch.device('cpu');backend='CPU';gpu='CPU fallback (DirectML self-test failed)';batch=args.batch if args.batch>0 else defaults['CPU'];workers=auto_workers(args.workers,backend);model,opt,sched,amp,scaler=make_runtime(dev,backend)
    val_sampler=MutableIndexSampler(va_idx); vl=make_loader(ds_val,batch,backend,workers,sampler=val_sampler,prefetch=args.prefetch); train_sampler=MutableIndexSampler(); tl=make_loader(ds_train,batch,backend,workers,sampler=train_sampler,prefetch=args.prefetch)
    best=1e9; no_improve=0; start_ep=1; start_batch=0; done=0; ema_train=None; ema_val=None; elapsed_before=0.0; ckpt=out/'last.pt'
    if args.resume!='never' and ckpt.is_file():
        try:
            state=torch.load(ckpt,map_location='cpu'); required=('version','model','optimizer','scheduler','epoch','batch','split_hash')
            missing=[key for key in required if key not in state]
            if state.get('version')!=CHECKPOINT_VERSION or state.get('architecture')!='SynSkeletonV2' or missing: raise RuntimeError('checkpoint architecture does not match the current model')
            if state.get('split_hash')!=split_hash: raise RuntimeError('dataset/split changed since the checkpoint was created')
            model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); optimizer_to(opt,dev); sched.load_state_dict(state['scheduler']); start_ep=int(state['epoch']); start_batch=int(state['batch']); best=float(state.get('best',1e9)); no_improve=int(state.get('no_improve',0)); ema_train=state.get('ema_train'); ema_val=state.get('ema_val'); elapsed_before=float(state.get('elapsed',0.0)); done=int(state.get('done',max(0,(start_ep-1)*math.ceil(len(tr_idx)/batch)+start_batch)))
            if start_batch>=math.ceil(len(tr_idx)/batch): start_ep+=1; start_batch=0
            if best<1e8 and not (out/'best.pt').is_file(): raise RuntimeError('body best checkpoint missing')
            print(f'RESUME checkpoint={ckpt} epoch={start_ep} batch={start_batch} best={best:.5f}',flush=True)
        except Exception as e:
            if args.resume=='required': raise
            print(f'WARNING: checkpoint not resumed: {e}',flush=True); print('Starting a new training run because the checkpoint cannot be resumed.',flush=True); start_ep=1;start_batch=0;done=0;best=1e9;no_improve=0
    if start_ep==1 and start_batch==0 and done==0:
        (out/'best.pt').unlink(missing_ok=True)
        (out/'hand_best.pt').unlink(missing_ok=True)
        (out/'hand_last.pt').unlink(missing_ok=True)
    total_train_batches=max(1,args.epochs*math.ceil(len(tr_idx)/batch)); started=time.time(); write_control(out,False)
    params=sum(p.numel() for p in model.parameters()); train_hands=sum(1 for i in tr_idx if ds_train.hand_bits[i][0] or ds_train.hand_bits[i][1]); train_left=sum(ds_train.hand_bits[i][0] for i in tr_idx); train_right=sum(ds_train.hand_bits[i][1] for i in tr_idx)
    print(f'DATASET samples={len(ds_train)} train={len(tr_idx)} val={len(va_idx)} guarded={split_info.get("guarded_samples",0)} hands(train)={train_hands} left={train_left} right={train_right} backend={backend} device={gpu} batch={batch} epochs={args.epochs} workers={workers} prefetch={args.prefetch} amp={amp}',flush=True); print(f'MODEL SynSkeleton v2 params={params:,} heatmaps={HM_W}x{HM_H} preprocess=v2(invalid=-2.0) target_heatmaps=backend-generated',flush=True)
    last_console=0.0; paused_accum=0.0
    def elapsed(): return elapsed_before+(time.time()-started)-paused_accum
    def eta_seconds(ep,bi):
        remain_train=max(0,total_train_batches-done); rem_epochs=max(0,args.epochs-ep+1); return (ema_train or 0)*remain_train+(ema_val if ema_val is not None else (ema_train or 0))*len(vl)*rem_epochs
    def maybe_pause(ep,bi,avg):
        nonlocal paused_accum
        if not read_control(out).get('paused',False): return
        save_checkpoint(ckpt,model,opt,sched,ep,bi,best,no_improve,done,ema_train,ema_val,elapsed(),backend,split_hash); paused_elapsed=elapsed(); p0=time.time(); print(f'PAUSED safely at epoch {ep}/{args.epochs}, batch {bi}. Resume from the UI or platform resume script.',flush=True)
        while read_control(out).get('paused',False):
            prog.write(state='paused',device=backend,gpu=gpu,epoch=ep,epochs=args.epochs,batch=bi,batches=math.ceil(len(tr_idx)/batch),progress=body_progress_max*min(1.0,done/total_train_batches),loss=avg,val=0,best=best if best<1e8 else 0,eta=fmt_eta(eta_seconds(ep,bi)),elapsed=fmt_eta(paused_elapsed),samples=len(ds_train),patience=no_improve,workers=workers); time.sleep(.5)
        paused_accum+=time.time()-p0; print('RESUMING training...',flush=True)
    prog.write(state='resuming' if ckpt.is_file() and (start_ep>1 or start_batch>0) else 'starting',device=backend,gpu=gpu,epoch=start_ep,epochs=args.epochs,batch=start_batch,batches=math.ceil(len(tr_idx)/batch),progress=body_progress_max*min(1.0,done/total_train_batches),loss=0,val=0,best=best if best<1e8 else 0,eta='--:--:--',elapsed=fmt_eta(elapsed()),samples=len(ds_train))
    last_avg=0.0;last_v=0.0;ep=max(1,start_ep);last_metrics={}
    for ep in range(start_ep,args.epochs+1):
        ep_indices=deterministic_epoch_indices(tr_idx,ep,ds_train); skip=start_batch if ep==start_ep else 0; train_sampler.set_indices(ep_indices[skip*batch:])
        model.train();total=0.0;processed=0;full_batches=math.ceil(len(tr_idx)/batch)
        if skip: print(f'Epoch {ep:03d}/{args.epochs}: continuing at batch {skip+1}/{full_batches}',flush=True)
        for local_bi,(x,z,vis,xy) in enumerate(tl,1):
            bi=skip+local_bi;bt=time.time();x,z,vis,xy=move_batch((x,z,vis,xy),dev,backend);opt.zero_grad(set_to_none=True)
            if amp:
                with torch.amp.autocast('cuda'): loss,_,parts=compute_loss(model,x,z,vis,xy)
                scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),5,foreach=False);scaler.step(opt);scaler.update()
            else:
                loss,_,parts=compute_loss(model,x,z,vis,xy);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5,foreach=False);opt.step()
            total+=loss.item();processed+=1;done+=1;dt=time.time()-bt;ema_train=dt if ema_train is None else ema_train*.92+dt*.08;last_avg=total/max(1,processed);now=time.time();eta=eta_seconds(ep,bi)
            if bi%max(1,args.checkpoint_batches)==0: save_checkpoint(ckpt,model,opt,sched,ep,bi,best,no_improve,done,ema_train,ema_val,elapsed(),backend,split_hash)
            prog.write(force=(bi==skip+1 or bi==full_batches),state='training',device=backend,gpu=gpu,epoch=ep,epochs=args.epochs,batch=bi,batches=full_batches,progress=body_progress_max*min(1.0,done/total_train_batches),loss=last_avg,val=last_v,best=best if best<1e8 else 0,eta=fmt_eta(eta),elapsed=fmt_eta(elapsed()),samples=len(ds_train),patience=no_improve,workers=workers)
            if now-last_console>=args.console_seconds:
                last_console=now;print(f'[TRAIN] epoch {ep:03d}/{args.epochs} batch {bi:04d}/{full_batches} {100*done/total_train_batches:6.2f}% loss={last_avg:.5f} elapsed={fmt_eta(elapsed())} ETA={fmt_eta(eta)}',flush=True)
            maybe_pause(ep,bi,last_avg)
        sched.step();vt0=time.time();last_metrics=validation_metrics(model,vl,dev,backend);ema_val=(time.time()-vt0)/max(1,len(vl)) if ema_val is None else ema_val*.90+((time.time()-vt0)/max(1,len(vl)))*.10;v=last_metrics['val_struct'];last_v=v;avg=last_avg
        print(f'Epoch {ep:03d}/{args.epochs} loss={avg:.5f} val={v:.5f} body_xy={last_metrics["body_xy_l1_norm"]:.4f} hand_xy={last_metrics["hand_xy_l1_norm"]:.4f} body_z={last_metrics["body_z_mae_mm"]:.1f}mm hand_z={last_metrics["hand_z_mae_mm"]:.1f}mm',flush=True)
        if v<best-args.min_delta: best=v;no_improve=0;_torch_save_resilient(cpu_tree(model.state_dict()),out/'best.pt',retries=20,nonfatal=False)
        else: no_improve+=1
        save_checkpoint(ckpt,model,opt,sched,ep,full_batches,best,no_improve,done,ema_train,ema_val,elapsed(),backend,split_hash)
        prog.write(state='validating',device=backend,gpu=gpu,epoch=ep,epochs=args.epochs,batch=full_batches,batches=full_batches,progress=body_progress_max*min(1.0,done/total_train_batches),loss=avg,val=v,best=best,eta=fmt_eta(eta_seconds(ep+1,0)),samples=len(ds_train),elapsed=fmt_eta(elapsed()),patience=no_improve,body_xy=last_metrics['body_xy_l1_norm'],hand_xy=last_metrics['hand_xy_l1_norm'],body_z_mm=last_metrics['body_z_mae_mm'],hand_z_mm=last_metrics['hand_z_mae_mm']);start_batch=0
        if args.patience>0 and ep>=args.min_epochs and no_improve>=args.patience: print(f'Early stopping after {no_improve} epochs without improvement.',flush=True);break
    prog.write(state='exporting-body',phase='body',device=backend,gpu=gpu,epoch=ep,epochs=args.epochs,progress=body_progress_max,loss=last_avg,val=last_v,best=best,eta='00:00:00',elapsed=fmt_eta(elapsed()),samples=len(ds_train))
    if not (out/'best.pt').is_file(): raise RuntimeError('No validated body checkpoint available for ONNX export')
    state=torch.load(out/'best.pt',map_location='cpu');cpu=DepthPoseNet().cpu();cpu.load_state_dict(state);wrapper=ExportModel(cpu.eval()).eval();dummy=torch.full((1,1,IN_H,IN_W),-2.0);onnx=out/'SynSkeleton.onnx'
    torch.onnx.export(wrapper,dummy,onnx,input_names=['depth_metric'],output_names=['keypoints','segments','endpoints'],opset_version=17,dynamic_axes={'depth_metric':{0:'batch'},'keypoints':{0:'batch'},'segments':{0:'batch'},'endpoints':{0:'batch'}},dynamo=False); verify_onnx(onnx,(1,1,IN_H,IN_W),(1,KP,4))
    try: model.cpu()
    except Exception: pass
    if backend in ('CUDA','ROCM'):
        try: torch.cuda.empty_cache()
        except Exception: pass
    hand_info=train_hand_refiner(hand_train_ds,hand_val_ds,out,dev,backend,gpu,args,prog,split_hash,elapsed,.74,.995) if hand_ready else None
    if hand_info is None:
        try: (out/'SynHand.onnx').unlink(missing_ok=True)
        except Exception: pass
    meta={'model_version':MODEL_VERSION,'architecture':'SynSkeletonV2','preprocess_version':2,'input_shape':[1,1,IN_H,IN_W],'invalid_depth_value':-2.0,'normalization':'valid_mm: clip((mm-2200)/1800,-1.0,1.8); invalid=-2.0','heatmap_shape':[HM_H,HM_W],'keypoints':KP,'body_keypoints':BODY_KP,'hand_keypoints_each':HAND_KP,'segments':SEGMENTS,'endpoints':ENDPOINTS,'hand_refiner':hand_info,'layout':['head','nose','earL','earR','shoulderL','shoulderR','elbowL','elbowR','wristL','wristR','hipL','hipR','kneeL','kneeR','ankleL','ankleR']+[f'handL_{i}' for i in range(21)]+[f'handR_{i}' for i in range(21)],'validation_split':split_info,'validation_metrics':last_metrics}
    _atomic_write_text(out/'SynSkeleton_layout.json',json.dumps(meta,indent=2),nonfatal=False);_atomic_write_text(out/'quality_report.json',json.dumps({'model_version':MODEL_VERSION,'dataset_samples':len(ds_train),'train_samples':len(tr_idx),'validation_samples':len(va_idx),'hand_train_crops':len(hand_train_ds),'hand_validation_crops':len(hand_val_ds),'split':split_info,'metrics':last_metrics,'hand_refiner':hand_info,'best_val':best},indent=2),nonfatal=True)
    prog.write(state='done',device=backend,gpu=gpu,epoch=ep,epochs=args.epochs,batch=math.ceil(len(tr_idx)/batch),batches=math.ceil(len(tr_idx)/batch),progress=1.0,loss=last_avg,val=last_v,best=best,eta='00:00:00',elapsed=fmt_eta(elapsed()),samples=len(ds_train),onnx=str(onnx),hand_onnx=str(out/'SynHand.onnx') if hand_info else '',body_xy=last_metrics.get('body_xy_l1_norm',0),hand_xy=last_metrics.get('hand_xy_l1_norm',0),body_z_mm=last_metrics.get('body_z_mae_mm',0),hand_z_mm=last_metrics.get('hand_z_mae_mm',0));print('\nREADY:',onnx,(' + SynHand.onnx' if hand_info else ''),flush=True)

def resolve_dataset(value): return str(Path(value)) if value else str(latest_dataset(Path.cwd()))

if __name__=='__main__':
    import traceback
    ap=argparse.ArgumentParser();ap.add_argument('dataset',nargs='?',default='');ap.add_argument('--epochs',type=int,default=80);ap.add_argument('--batch',type=int,default=0);ap.add_argument('--lr',type=float,default=1.6e-3);ap.add_argument('--workers',type=int,default=-1,help='DataLoader workers; -1=automatic');ap.add_argument('--patience',type=int,default=18);ap.add_argument('--min-epochs',type=int,default=30);ap.add_argument('--min-delta',type=float,default=.00010);ap.add_argument('--progress-ms',type=int,default=100);ap.add_argument('--log',default='');ap.add_argument('--mirror-log',default='');ap.add_argument('--resume',choices=['auto','never','required'],default='auto');ap.add_argument('--checkpoint-batches',type=int,default=100);ap.add_argument('--console-seconds',type=float,default=15.0);ap.add_argument('--prefetch',type=int,default=3,help='batches prefetched per worker');ap.add_argument('--val-fraction',type=float,default=.12);ap.add_argument('--val-group-ms',type=int,default=30000);ap.add_argument('--val-guard-ms',type=int,default=1500);ap.add_argument('--hand-epochs',type=int,default=80);ap.add_argument('--hand-batch',type=int,default=0);ap.add_argument('--hand-lr',type=float,default=1.4e-3);ap.add_argument('--hand-patience',type=int,default=14);ap.add_argument('--hand-min-epochs',type=int,default=25);ap.add_argument('--hand-min-delta',type=float,default=.00010);args=ap.parse_args();args.dataset=resolve_dataset(args.dataset)
    _log_files=[]
    try:
        for lp in (args.log,args.mirror_log):
            if lp:
                path=Path(lp);path.parent.mkdir(parents=True,exist_ok=True);_log_files.append(open(path,'a',encoding='utf-8',buffering=1))
        if _log_files: sys.stdout=Tee(sys.__stdout__,*_log_files);sys.stderr=Tee(sys.__stderr__,*_log_files)
        print(f'TRAINER START dataset={args.dataset}',flush=True);run(args)
    except Exception as e:
        try: out=Path(args.dataset)/'output';out.mkdir(parents=True,exist_ok=True);Progress(out).write(state='error',progress=0.0,error=f'{type(e).__name__}: {e}',eta='--:--:--')
        except Exception: pass
        traceback.print_exc();raise
    finally:
        for _f in _log_files:
            try:_f.close()
            except Exception:pass
