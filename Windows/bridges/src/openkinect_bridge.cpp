#include "BridgeProtocol.h"
#include <libfreenect_sync.h>
#include <vector>
#include <cstdio>

int main(){
    PipeWriter out(L"\\\\.\\pipe\\KinectDepthAutoLearn-OpenKinect"); std::puts("Waiting for Processing to connect..."); if(!out.connect())return 2;
    std::vector<uint8_t> bgra(640*480*4); uint64_t seq=0;
    while(true){
        void* v=nullptr; void* d=nullptr; uint32_t tv=0,td=0;
        if(freenect_sync_get_video(&v,&tv,0,FREENECT_VIDEO_RGB)<0)break;
        const uint8_t* rgb=(const uint8_t*)v; for(size_t i=0;i<640u*480u;i++){bgra[i*4+0]=rgb[i*3+2];bgra[i*4+1]=rgb[i*3+1];bgra[i*4+2]=rgb[i*3+0];bgra[i*4+3]=255;}
        if(!out.writeFrame(1,1,bgra.data(),(uint32_t)bgra.size(),++seq,GetTickCount64()))break;
        if(freenect_sync_get_depth(&d,&td,0,FREENECT_DEPTH_MM)<0)break;
        if(!out.writeFrame(2,2,d,640u*480u*2u,++seq,GetTickCount64()))break;
    }
    freenect_sync_stop(); return 0;
}
