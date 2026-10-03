#pragma once
#include <windows.h>
#include <cstdint>
#include <string>

#pragma pack(push,1)
struct KdabHeader {
    uint32_t magic = 0x4241444B; // bytes "KDAB" little-endian
    uint32_t version = 1;
    uint32_t stream = 0;        // 1 RGB, 2 Depth
    uint32_t width = 640;
    uint32_t height = 480;
    uint32_t format = 0;        // 1 BGRA32, 2 DEPTH_MM16
    uint32_t bytes = 0;
    uint32_t flags = 0;
    uint64_t sequence = 0;
    uint64_t tick_ms = 0;
};
#pragma pack(pop)

class PipeWriter {
public:
    explicit PipeWriter(const wchar_t* name) : pipeName_(name) {}
    ~PipeWriter(){ close(); }
    bool connect(){
        close();
        pipe_=CreateNamedPipeW(pipeName_.c_str(),PIPE_ACCESS_OUTBOUND,PIPE_TYPE_BYTE|PIPE_WAIT,1,4*1024*1024,4*1024*1024,0,nullptr);
        if(pipe_==INVALID_HANDLE_VALUE)return false;
        BOOL ok=ConnectNamedPipe(pipe_,nullptr); if(!ok && GetLastError()!=ERROR_PIPE_CONNECTED){close();return false;} return true;
    }
    bool writeFrame(uint32_t stream,uint32_t format,const void* data,uint32_t bytes,uint64_t seq,uint64_t tick,uint32_t flags=0){
        if(pipe_==INVALID_HANDLE_VALUE)return false; KdabHeader h; h.stream=stream;h.format=format;h.bytes=bytes;h.sequence=seq;h.tick_ms=tick;h.flags=flags;
        return writeAll(&h,sizeof(h)) && writeAll(data,bytes);
    }
    void close(){if(pipe_!=INVALID_HANDLE_VALUE){FlushFileBuffers(pipe_);DisconnectNamedPipe(pipe_);CloseHandle(pipe_);pipe_=INVALID_HANDLE_VALUE;}}
private:
    bool writeAll(const void* p,DWORD n){const uint8_t* b=(const uint8_t*)p;DWORD off=0;while(off<n){DWORD w=0;if(!WriteFile(pipe_,b+off,n-off,&w,nullptr)||w==0)return false;off+=w;}return true;}
    HANDLE pipe_=INVALID_HANDLE_VALUE; std::wstring pipeName_;
};
