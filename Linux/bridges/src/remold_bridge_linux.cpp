#include "BridgeProtocolLinux.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cerrno>
#include <csignal>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

namespace {
std::atomic<bool> running{true};
void stop_sig(int){ running=false; }

constexpr uint32_t REQ_MAGIC=0x43534D52u;
constexpr uint32_t FRAME_MAGIC=0x46534D52u;
constexpr uint32_t VERSION=1u;
constexpr uint32_t CMD_SUBSCRIBE=1u;
constexpr uint32_t STREAM_RGB=1u;
constexpr uint32_t STREAM_DEPTH=4u;
constexpr uint32_t MODE_RGB=0u;
constexpr uint32_t MODE_DEPTH=2u;
constexpr uint32_t PIXEL_BAYER_GRBG8=4u;
constexpr uint32_t PIXEL_DEPTH11=6u;
constexpr size_t RGB_BYTES=640u*480u;
constexpr size_t DEPTH_PACKED_BYTES=640u*480u*11u/8u;

#pragma pack(push,1)
struct ScannerRequest{uint32_t magic=REQ_MAGIC,version=VERSION,command=CMD_SUBSCRIBE,streamMask=STREAM_RGB|STREAM_DEPTH;};
struct ScannerReply{
  uint32_t magic=REQ_MAGIC,version=VERSION; int32_t result=0; uint32_t acceptedMask=0,width=640,height=480,capabilities=0,maxPayloadBytes=0,depthCalibrationValid=0;
  double depthConstShift=0,depthEmitterDistance=0,depthReferenceDistance=0,depthReferencePixelSize=0;
};
struct MotionSample{uint32_t flags;int32_t accelX,accelY,accelZ,tiltTenths;uint64_t tickMs;};
struct ScannerFrameHeader{uint32_t magic,version,mode,width,height,pixelFormat,payloadBytes,flags;uint64_t frameNumber,tickMs;MotionSample motion;};
#pragma pack(pop)
static_assert(sizeof(ScannerRequest)==16,"request ABI");
static_assert(sizeof(ScannerReply)==68,"reply ABI");
static_assert(sizeof(ScannerFrameHeader)==76,"frame ABI");

bool read_all(int fd,void* p,size_t n){
  auto* b=static_cast<uint8_t*>(p);size_t off=0;
  while(off<n){ssize_t r=::read(fd,b+off,n-off);if(r==0)return false;if(r<0){if(errno==EINTR)continue;return false;}off+=(size_t)r;}return true;
}
bool write_all(int fd,const void* p,size_t n){
  auto* b=static_cast<const uint8_t*>(p);size_t off=0;
  while(off<n){ssize_t w=::send(fd,b+off,n-off,MSG_NOSIGNAL);if(w<=0){if(w<0&&errno==EINTR)continue;return false;}off+=(size_t)w;}return true;
}
int connect_unix(const std::string& path){
  int fd=::socket(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC,0);if(fd<0)return -1;
  sockaddr_un a{};a.sun_family=AF_UNIX;
  if(path.size()>=sizeof(a.sun_path)){::close(fd);errno=ENAMETOOLONG;return -1;}
  std::memcpy(a.sun_path,path.c_str(),path.size()+1);
  if(::connect(fd,reinterpret_cast<sockaddr*>(&a),sizeof(a))<0){int e=errno;::close(fd);errno=e;return -1;}
  timeval tv{};tv.tv_sec=4;::setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&tv,sizeof(tv));::setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&tv,sizeof(tv));
  return fd;
}
std::string runtime_dir(){
  const char* base=std::getenv("REMOLD_RUNDIR");
  if(base&&*base){std::string s(base);while(s.size()>1&&s.back()=='/')s.pop_back();if(s.size()>=18&&s.substr(s.size()-18)=="kinect360-remold")return s;return s+"/kinect360-remold";}
  return "/run/kinect360-remold";
}
std::vector<std::string> split_tabs(const std::string& row){
  std::vector<std::string> v;std::stringstream ss(row);std::string f;while(std::getline(ss,f,'\t'))v.push_back(f);return v;
}
std::string sdk_query(const std::string& command){
  int fd=connect_unix(runtime_dir()+"/sdk.sock");if(fd<0)return {};
  if(!write_all(fd,command.data(),command.size())){::close(fd);return {};}
  ::shutdown(fd,SHUT_WR);std::string out;char buf[4096];
  while(true){ssize_t n=::read(fd,buf,sizeof(buf));if(n>0)out.append(buf,(size_t)n);else break;}
  ::close(fd);return out;
}
bool discover_ready(std::string& deviceId,std::string& cameraEndpoint){
  std::string list=sdk_query("LIST\n");if(list.empty())return false;
  std::stringstream ss(list);std::string row;
  while(std::getline(ss,row)){
    if(row.empty()||row.rfind("ERR ",0)==0)continue;auto f=split_tabs(row);
    if(f.size()>4&&f[2]=="Ready"&&!f[4].empty()){deviceId=f[0];cameraEndpoint=f[4];return true;}
  }
  return false;
}
int sample(const std::vector<uint8_t>& s,int w,int h,int x,int y){x=std::max(0,std::min(w-1,x));y=std::max(0,std::min(h-1,y));return s[(size_t)y*w+x];}
int avg2(int a,int b){return(a+b)>>1;}int avg4(int a,int b,int c,int d){return(a+b+c+d)>>2;}
void demosaic_grbg(const std::vector<uint8_t>& src,std::vector<uint8_t>& bgra,int w,int h){
  bgra.resize((size_t)w*h*4);
  for(int y=0;y<h;y++)for(int x=0;x<w;x++){
    int c=sample(src,w,h,x,y),r,g,b;bool ye=(y&1)==0,xe=(x&1)==0;
    if(ye&&xe){g=c;r=avg2(sample(src,w,h,x-1,y),sample(src,w,h,x+1,y));b=avg2(sample(src,w,h,x,y-1),sample(src,w,h,x,y+1));}
    else if(ye){r=c;g=avg4(sample(src,w,h,x-1,y),sample(src,w,h,x+1,y),sample(src,w,h,x,y-1),sample(src,w,h,x,y+1));b=avg4(sample(src,w,h,x-1,y-1),sample(src,w,h,x+1,y-1),sample(src,w,h,x-1,y+1),sample(src,w,h,x+1,y+1));}
    else if(xe){b=c;g=avg4(sample(src,w,h,x-1,y),sample(src,w,h,x+1,y),sample(src,w,h,x,y-1),sample(src,w,h,x,y+1));r=avg4(sample(src,w,h,x-1,y-1),sample(src,w,h,x+1,y-1),sample(src,w,h,x-1,y+1),sample(src,w,h,x+1,y+1));}
    else{g=c;r=avg2(sample(src,w,h,x,y-1),sample(src,w,h,x,y+1));b=avg2(sample(src,w,h,x-1,y),sample(src,w,h,x+1,y));}
    size_t p=((size_t)y*w+x)*4;bgra[p]=(uint8_t)b;bgra[p+1]=(uint8_t)g;bgra[p+2]=(uint8_t)r;bgra[p+3]=255;
  }
}
void build_depth_lut(const ScannerReply& rep,std::vector<uint16_t>& lut){
  lut.assign(2048,0);const bool valid=rep.depthCalibrationValid!=0&&std::isfinite(rep.depthConstShift)&&std::isfinite(rep.depthEmitterDistance)&&std::isfinite(rep.depthReferenceDistance)&&std::isfinite(rep.depthReferencePixelSize)&&rep.depthEmitterDistance>0&&rep.depthReferenceDistance>0&&rep.depthReferencePixelSize>0;
  for(int raw=1;raw<2047;raw++){
    double mm=0;
    if(valid){double fixedRefX=((raw-4.0*rep.depthConstShift)/4.0)-0.375;double metric=fixedRefX*rep.depthReferencePixelSize;double denom=rep.depthEmitterDistance-metric;if(std::abs(denom)>1e-9)mm=10.0*((metric*rep.depthReferenceDistance/denom)+rep.depthReferenceDistance);}
    else mm=1000.0*0.1236*std::tan(raw/2842.5+1.1863);
    if(std::isfinite(mm)&&mm>=300&&mm<=10000)lut[(size_t)raw]=(uint16_t)std::lround(mm);
  }
}
void unpack_depth_mm(const std::vector<uint8_t>& packed,const std::vector<uint16_t>& lut,std::vector<uint16_t>& mm){
  mm.resize(640u*480u);size_t src=0;int bits=0;uint64_t acc=0;
  for(size_t i=0;i<mm.size();i++){
    while(bits<11&&src<packed.size()){acc=(acc<<8)|packed[src++];bits+=8;}
    if(bits<11){mm[i]=0;continue;}bits-=11;uint16_t raw=(uint16_t)((acc>>bits)&0x7ffu);if(bits==0)acc=0;else acc&=((1ULL<<bits)-1ULL);mm[i]=lut[raw];
  }
}
}

int main(int argc,char** argv){
  uint16_t port=38174;if(argc>1)port=(uint16_t)std::atoi(argv[1]);
  std::signal(SIGINT,stop_sig);std::signal(SIGTERM,stop_sig);std::signal(SIGPIPE,SIG_IGN);
  std::fprintf(stdout,"Kinect Depth AutoLearn Remold bridge\nRuntime: %s\n",runtime_dir().c_str());std::fflush(stdout);
  while(running){
    TcpWriter out(port);
    if(!out.connect()){std::perror("TCP bridge listen/accept");usleep(500000);continue;}
    std::fprintf(stdout,"Processing connected on 127.0.0.1:%u. Waiting for Remold device if necessary...\n",(unsigned)port);std::fflush(stdout);
    bool processingAlive=true;
    while(running && processingAlive){
      std::string id,endpoint;
      if(!discover_ready(id,endpoint)){std::fprintf(stdout,"Waiting for Ready Kinect in Remold SDK...\n");std::fflush(stdout);usleep(700000);continue;}
      std::fprintf(stdout,"Remold device=%s camera=%s\n",id.c_str(),endpoint.c_str());std::fflush(stdout);
      int cam=connect_unix(endpoint);if(cam<0){std::perror("Remold camera socket");usleep(600000);continue;}
      ScannerRequest req{};ScannerReply rep{};
      if(!write_all(cam,&req,sizeof(req))||!read_all(cam,&rep,sizeof(rep))||rep.magic!=REQ_MAGIC||rep.version!=VERSION||rep.result<0||((rep.acceptedMask&(STREAM_RGB|STREAM_DEPTH))!=(STREAM_RGB|STREAM_DEPTH))){
        std::fprintf(stderr,"Remold subscribe failed result=%d mask=0x%x\n",rep.result,rep.acceptedMask);::close(cam);usleep(600000);continue;
      }
      std::vector<uint16_t> lut;build_depth_lut(rep,lut);std::vector<uint8_t> payload,bgra;std::vector<uint16_t> mm;
      while(running){
        ScannerFrameHeader h{};if(!read_all(cam,&h,sizeof(h)))break;
        if(h.magic!=FRAME_MAGIC||h.version!=VERSION||h.payloadBytes>4u*1024u*1024u){std::fprintf(stderr,"Invalid Remold frame header\n");break;}
        payload.resize(h.payloadBytes);if(!read_all(cam,payload.data(),payload.size()))break;bool ok=true;
        if(h.mode==MODE_RGB&&h.pixelFormat==PIXEL_BAYER_GRBG8&&h.width==640&&h.height==480&&payload.size()==RGB_BYTES){demosaic_grbg(payload,bgra,640,480);ok=out.writeFrame(1,1,bgra.data(),(uint32_t)bgra.size(),h.frameNumber,h.tickMs,h.flags);}
        else if(h.mode==MODE_DEPTH&&h.pixelFormat==PIXEL_DEPTH11&&h.width==640&&h.height==480&&payload.size()==DEPTH_PACKED_BYTES){unpack_depth_mm(payload,lut,mm);ok=out.writeFrame(2,2,mm.data(),(uint32_t)(mm.size()*sizeof(uint16_t)),h.frameNumber,h.tickMs,h.flags);}
        if(!ok){processingAlive=false;break;}
      }
      ::close(cam);if(running&&processingAlive)usleep(450000);
    }
    if(running){std::fprintf(stdout,"Processing disconnected; reopening localhost bridge...\n");std::fflush(stdout);usleep(250000);}
  }
  return 0;
}
