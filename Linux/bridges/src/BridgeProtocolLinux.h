#pragma once
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#pragma pack(push,1)
struct KdabHeader {
    uint32_t magic = 0x4241444B;
    uint32_t version = 1;
    uint32_t stream = 0;
    uint32_t width = 640;
    uint32_t height = 480;
    uint32_t format = 0;
    uint32_t bytes = 0;
    uint32_t flags = 0;
    uint64_t sequence = 0;
    uint64_t tick_ms = 0;
};
#pragma pack(pop)

inline uint64_t kd_now_ms(){
    using namespace std::chrono;
    return (uint64_t)duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

class TcpWriter {
public:
    explicit TcpWriter(uint16_t port):port_(port){}
    ~TcpWriter(){closeAll();}
    bool connect(){
        closeAll();
        listen_=::socket(AF_INET,SOCK_STREAM,0); if(listen_<0)return false;
        int yes=1; setsockopt(listen_,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
        sockaddr_in a{}; a.sin_family=AF_INET; a.sin_addr.s_addr=htonl(INADDR_LOOPBACK); a.sin_port=htons(port_);
        if(::bind(listen_,(sockaddr*)&a,sizeof(a))<0){closeAll();return false;}
        if(::listen(listen_,1)<0){closeAll();return false;}
        std::printf("Waiting for Processing on 127.0.0.1:%u...\n",(unsigned)port_); std::fflush(stdout);
        client_=::accept(listen_,nullptr,nullptr); return client_>=0;
    }
    bool writeFrame(uint32_t stream,uint32_t format,const void* data,uint32_t bytes,uint64_t seq,uint64_t tick,uint32_t flags=0){
        if(client_<0)return false; KdabHeader h; h.stream=stream;h.format=format;h.bytes=bytes;h.sequence=seq;h.tick_ms=tick;h.flags=flags;
        return writeAll(&h,sizeof(h))&&writeAll(data,bytes);
    }
private:
    bool writeAll(const void* p,size_t n){
        const uint8_t* b=(const uint8_t*)p; size_t off=0;
        while(off<n){ssize_t w=::send(client_,b+off,n-off,MSG_NOSIGNAL);if(w<=0)return false;off+=(size_t)w;}return true;
    }
    void closeAll(){if(client_>=0){::close(client_);client_=-1;}if(listen_>=0){::close(listen_);listen_=-1;}}
    uint16_t port_; int listen_=-1,client_=-1;
};
