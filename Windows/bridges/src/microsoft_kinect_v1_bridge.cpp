#include "BridgeProtocol.h"
#include <NuiApi.h>
#include <vector>
#include <cstdio>

static bool sendColor(HANDLE stream,PipeWriter& out,uint64_t& seq){
    NUI_IMAGE_FRAME f{}; if(FAILED(NuiImageStreamGetNextFrame(stream,80,&f)))return true;
    INuiFrameTexture* tex=f.pFrameTexture; NUI_LOCKED_RECT lr{}; tex->LockRect(0,&lr,nullptr,0);
    bool ok=true; if(lr.Pitch>0 && lr.pBits){
        std::vector<uint8_t> bgra(640*480*4); const uint8_t* src=(const uint8_t*)lr.pBits;
        for(int y=0;y<480;y++)memcpy(bgra.data()+y*640*4,src+y*lr.Pitch,640*4);
        ok=out.writeFrame(1,1,bgra.data(),(uint32_t)bgra.size(),++seq,GetTickCount64());
    }
    tex->UnlockRect(0); NuiImageStreamReleaseFrame(stream,&f); return ok;
}
static bool sendDepth(HANDLE stream,PipeWriter& out,uint64_t& seq){
    NUI_IMAGE_FRAME f{}; if(FAILED(NuiImageStreamGetNextFrame(stream,80,&f)))return true;
    INuiFrameTexture* tex=f.pFrameTexture; NUI_LOCKED_RECT lr{}; tex->LockRect(0,&lr,nullptr,0);
    bool ok=true; if(lr.Pitch>0 && lr.pBits){
        std::vector<uint16_t> mm(640*480); const USHORT* src=(const USHORT*)lr.pBits; int stride=lr.Pitch/sizeof(USHORT);
        for(int y=0;y<480;y++)for(int x=0;x<640;x++)mm[y*640+x]=(uint16_t)NuiDepthPixelToDepth(src[y*stride+x]);
        ok=out.writeFrame(2,2,mm.data(),(uint32_t)(mm.size()*2),++seq,GetTickCount64());
    }
    tex->UnlockRect(0); NuiImageStreamReleaseFrame(stream,&f); return ok;
}
int wmain(){
    HRESULT hr=NuiInitialize(NUI_INITIALIZE_FLAG_USES_COLOR|NUI_INITIALIZE_FLAG_USES_DEPTH); if(FAILED(hr)){std::fprintf(stderr,"NuiInitialize failed: 0x%08lx\n",hr);return 2;}
    HANDLE color=nullptr,depth=nullptr;
    hr=NuiImageStreamOpen(NUI_IMAGE_TYPE_COLOR,NUI_IMAGE_RESOLUTION_640x480,0,2,nullptr,&color); if(FAILED(hr)){NuiShutdown();return 3;}
    hr=NuiImageStreamOpen(NUI_IMAGE_TYPE_DEPTH,NUI_IMAGE_RESOLUTION_640x480,0,2,nullptr,&depth); if(FAILED(hr)){NuiShutdown();return 4;}
    PipeWriter out(L"\\\\.\\pipe\\KinectDepthAutoLearn-MicrosoftV1"); std::puts("Waiting for Processing to connect..."); if(!out.connect()){NuiShutdown();return 5;}
    uint64_t seq=0; while(true){if(!sendColor(color,out,seq)||!sendDepth(depth,out,seq))break;Sleep(1);} NuiShutdown(); return 0;
}
