#include "BridgeProtocolLinux.h"
#include <libfreenect_sync.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <atomic>

static std::atomic<bool> running{true};
static void stop_sig(int){ running=false; }

int main(int argc,char** argv){
    uint16_t port=38173; if(argc>1)port=(uint16_t)std::atoi(argv[1]);
    std::signal(SIGINT,stop_sig); std::signal(SIGTERM,stop_sig);
    while(running){
        TcpWriter out(port); if(!out.connect()){std::perror("bridge connect");return 2;}
        std::vector<uint8_t> bgra(640u*480u*4u); uint64_t seq=0;
        while(running){
            void* v=nullptr; void* d=nullptr; uint32_t tv=0,td=0;
            if(freenect_sync_get_video(&v,&tv,0,FREENECT_VIDEO_RGB)<0)break;
            const uint8_t* rgb=(const uint8_t*)v;
            for(size_t i=0;i<640u*480u;i++){bgra[i*4+0]=rgb[i*3+2];bgra[i*4+1]=rgb[i*3+1];bgra[i*4+2]=rgb[i*3+0];bgra[i*4+3]=255;}
            if(!out.writeFrame(1,1,bgra.data(),(uint32_t)bgra.size(),++seq,kd_now_ms()))break;
            if(freenect_sync_get_depth(&d,&td,0,FREENECT_DEPTH_MM)<0)break;
            if(!out.writeFrame(2,2,d,640u*480u*2u,++seq,kd_now_ms()))break;
        }
        freenect_sync_stop(); if(running) usleep(400000);
    }
    freenect_sync_stop(); return 0;
}
