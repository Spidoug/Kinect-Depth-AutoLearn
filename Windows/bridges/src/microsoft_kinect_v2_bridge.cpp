#include "BridgeProtocol.h"
#include <Kinect.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <limits>

namespace {
constexpr int OUT_W = 640;
constexpr int OUT_H = 480;
constexpr int DEPTH_W = 512;
constexpr int DEPTH_H = 424;
constexpr int COLOR_W = 1920;
constexpr int COLOR_H = 1080;

constexpr float FX = 594.2f;
constexpr float FY = 591.0f;
constexpr float CX = 339.3f;
constexpr float CY = 242.7f;

template<class T> void safeRelease(T*& p){ if(p){ p->Release(); p=nullptr; } }

bool acquireFrames(IMultiSourceFrameReader* reader, ICoordinateMapper* mapper, PipeWriter& out, uint64_t& seq){
    IMultiSourceFrame* multi=nullptr;
    HRESULT hr=reader->AcquireLatestFrame(&multi);
    if(FAILED(hr) || !multi) return true;

    IDepthFrameReference* depthRef=nullptr;
    IColorFrameReference* colorRef=nullptr;
    IDepthFrame* depthFrame=nullptr;
    IColorFrame* colorFrame=nullptr;
    multi->get_DepthFrameReference(&depthRef);
    multi->get_ColorFrameReference(&colorRef);
    if(depthRef) depthRef->AcquireFrame(&depthFrame);
    if(colorRef) colorRef->AcquireFrame(&colorFrame);

    bool ok=true;
    if(depthFrame && colorFrame){
        UINT depthCount=0;
        UINT16* depthData=nullptr;
        if(SUCCEEDED(depthFrame->AccessUnderlyingBuffer(&depthCount,&depthData)) && depthCount==DEPTH_W*DEPTH_H){
            std::vector<RGBQUAD> color(COLOR_W*COLOR_H);
            hr=colorFrame->CopyConvertedFrameDataToArray((UINT)color.size()*sizeof(RGBQUAD),reinterpret_cast<BYTE*>(color.data()),ColorImageFormat_Bgra);
            if(SUCCEEDED(hr)){
                std::vector<ColorSpacePoint> depthToColor(depthCount);
                std::vector<CameraSpacePoint> depthToCamera(depthCount);
                hr=mapper->MapDepthFrameToColorSpace(depthCount,depthData,(UINT)depthToColor.size(),depthToColor.data());
                if(SUCCEEDED(hr)) hr=mapper->MapDepthFrameToCameraSpace(depthCount,depthData,(UINT)depthToCamera.size(),depthToCamera.data());
                if(SUCCEEDED(hr)){
                    std::vector<uint16_t> depthOut(OUT_W*OUT_H,0);
                    std::vector<uint8_t> colorOut(OUT_W*OUT_H*4,0);
                    std::vector<float> zBuffer(OUT_W*OUT_H,std::numeric_limits<float>::infinity());

                    for(UINT di=0;di<depthCount;di++){
                        const CameraSpacePoint& cam=depthToCamera[di];
                        if(!std::isfinite(cam.X)||!std::isfinite(cam.Y)||!std::isfinite(cam.Z)||cam.Z<=0.0f) continue;
                        const float zmm=cam.Z*1000.0f;
                        if(zmm<300.0f||zmm>10000.0f) continue;

                        const int ox=(int)std::lround(FX*(cam.X/cam.Z)+CX);
                        const int oy=(int)std::lround(FY*(cam.Y/cam.Z)+CY);
                        if(ox<0||oy<0||ox>=OUT_W||oy>=OUT_H) continue;
                        const int oi=oy*OUT_W+ox;
                        if(zmm>=zBuffer[oi]) continue;
                        zBuffer[oi]=zmm;
                        depthOut[oi]=(uint16_t)std::lround(zmm);

                        const ColorSpacePoint& cp=depthToColor[di];
                        if(std::isfinite(cp.X)&&std::isfinite(cp.Y)){
                            const int cx=(int)std::lround(cp.X), cy=(int)std::lround(cp.Y);
                            if(cx>=0&&cy>=0&&cx<COLOR_W&&cy<COLOR_H){
                                const RGBQUAD& q=color[cy*COLOR_W+cx];
                                const size_t p=(size_t)oi*4;
                                colorOut[p+0]=q.rgbBlue;
                                colorOut[p+1]=q.rgbGreen;
                                colorOut[p+2]=q.rgbRed;
                                colorOut[p+3]=255;
                            }
                        }
                    }

                    std::vector<uint16_t> filledDepth=depthOut;
                    std::vector<uint8_t> filledColor=colorOut;
                    for(int y=1;y<OUT_H-1;y++) for(int x=1;x<OUT_W-1;x++){
                        const int i=y*OUT_W+x; if(depthOut[i]!=0) continue;
                        const int n[4]={i-1,i+1,i-OUT_W,i+OUT_W};
                        int best=-1; uint16_t ref=0;
                        for(int k=0;k<4;k++) if(depthOut[n[k]]!=0){ if(best<0){best=n[k];ref=depthOut[n[k]];} else if(std::abs((int)depthOut[n[k]]-(int)ref)<35){best=n[k];break;} }
                        if(best>=0){ filledDepth[i]=depthOut[best]; for(int c=0;c<4;c++)filledColor[(size_t)i*4+c]=colorOut[(size_t)best*4+c]; }
                    }

                    const uint64_t tick=GetTickCount64();
                    const uint64_t frameSeq=++seq;
                    ok=out.writeFrame(1,1,filledColor.data(),(uint32_t)filledColor.size(),frameSeq,tick,1) &&
                       out.writeFrame(2,2,filledDepth.data(),(uint32_t)(filledDepth.size()*sizeof(uint16_t)),frameSeq,tick,1);
                }
            }
        }
    }

    safeRelease(depthFrame); safeRelease(colorFrame); safeRelease(depthRef); safeRelease(colorRef); safeRelease(multi);
    return ok;
}
}

int wmain(){
    IKinectSensor* sensor=nullptr;
    HRESULT hr=GetDefaultKinectSensor(&sensor);
    if(FAILED(hr)||!sensor){ std::fprintf(stderr,"Kinect v2 sensor not found. HRESULT=0x%08lx\n",hr); return 2; }
    hr=sensor->Open();
    if(FAILED(hr)){ std::fprintf(stderr,"IKinectSensor::Open failed. HRESULT=0x%08lx\n",hr); safeRelease(sensor); return 3; }

    ICoordinateMapper* mapper=nullptr;
    IMultiSourceFrameReader* reader=nullptr;
    hr=sensor->get_CoordinateMapper(&mapper);
    if(SUCCEEDED(hr)) hr=sensor->OpenMultiSourceFrameReader(FrameSourceTypes_Depth|FrameSourceTypes_Color,&reader);
    if(FAILED(hr)||!mapper||!reader){
        std::fprintf(stderr,"Failed to initialize Kinect v2 color/depth reader. HRESULT=0x%08lx\n",hr);
        safeRelease(reader); safeRelease(mapper); sensor->Close(); safeRelease(sensor); return 4;
    }

    PipeWriter out(L"\\\\.\\pipe\\KinectDepthAutoLearn-MicrosoftV2");
    std::puts("Waiting for Processing to connect...");
    if(!out.connect()){ safeRelease(reader); safeRelease(mapper); sensor->Close(); safeRelease(sensor); return 5; }

    uint64_t seq=0;
    while(true){ if(!acquireFrames(reader,mapper,out,seq)) break; Sleep(2); }

    safeRelease(reader); safeRelease(mapper); sensor->Close(); safeRelease(sensor);
    return 0;
}
