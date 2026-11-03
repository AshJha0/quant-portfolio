// Thin POSIX socket wrappers for the QPBF feed: UDP multicast send and
// receive (non-blocking, for busy-poll loops), and a length-prefixed TCP
// stream for snapshots. Linux only. Every call's failure is returned, not
// thrown, so a test can skip cleanly where multicast is not routable.
//
// Loopback multicast: the sender sets IP_MULTICAST_IF to 127.0.0.1 and
// IP_MULTICAST_LOOP to 1, TTL 0 so nothing leaves the host; the receiver
// joins the group on 127.0.0.1. Groups and ports are the caller's choice;
// 239.0.0.0/8 is the administratively scoped range (RFC 2365).
#pragma once

#include <arpa/inet.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

namespace ull::mcast::net {

class Fd {
public:
    Fd() = default;
    explicit Fd(int fd) : fd_(fd) {}
    Fd(Fd&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }
    Fd& operator=(Fd&& o) noexcept {
        if (this != &o) { reset(); fd_ = o.fd_; o.fd_ = -1; }
        return *this;
    }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    ~Fd() { reset(); }
    void reset() { if (fd_ >= 0) ::close(fd_); fd_ = -1; }
    [[nodiscard]] int get() const { return fd_; }
    [[nodiscard]] bool ok() const { return fd_ >= 0; }

private:
    int fd_{-1};
};

inline sockaddr_in addr_of(const char* ip, std::uint16_t port) {
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    ::inet_pton(AF_INET, ip, &a.sin_addr);
    return a;
}

class McastSender {
public:
    // Returns an empty string on success, else the failing call and errno.
    std::string open(const char* group, std::uint16_t port, const char* iface = "127.0.0.1") {
        fd_ = Fd(::socket(AF_INET, SOCK_DGRAM, 0));
        if (!fd_.ok()) return err("socket");
        in_addr ifa{};
        ::inet_pton(AF_INET, iface, &ifa);
        if (::setsockopt(fd_.get(), IPPROTO_IP, IP_MULTICAST_IF, &ifa, sizeof ifa) != 0) return err("IP_MULTICAST_IF");
        unsigned char loop = 1, ttl = 0;
        if (::setsockopt(fd_.get(), IPPROTO_IP, IP_MULTICAST_LOOP, &loop, 1) != 0) return err("IP_MULTICAST_LOOP");
        if (::setsockopt(fd_.get(), IPPROTO_IP, IP_MULTICAST_TTL, &ttl, 1) != 0) return err("IP_MULTICAST_TTL");
        dst_ = addr_of(group, port);
        return {};
    }
    bool send(const std::uint8_t* p, std::size_t n) {
        return ::sendto(fd_.get(), p, n, 0, reinterpret_cast<const sockaddr*>(&dst_), sizeof dst_) ==
               static_cast<ssize_t>(n);
    }

private:
    static std::string err(const char* what) { return std::string(what) + ": " + std::strerror(errno); }
    Fd fd_;
    sockaddr_in dst_{};
};

class McastReceiver {
public:
    std::string open(const char* group, std::uint16_t port, const char* iface = "127.0.0.1",
                     int rcvbuf = 8 << 20) {
        fd_ = Fd(::socket(AF_INET, SOCK_DGRAM, 0));
        if (!fd_.ok()) return err("socket");
        int one = 1;
        ::setsockopt(fd_.get(), SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        ::setsockopt(fd_.get(), SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof rcvbuf);
        sockaddr_in a = addr_of(group, port);  // bind to the group: only its traffic
        if (::bind(fd_.get(), reinterpret_cast<sockaddr*>(&a), sizeof a) != 0) return err("bind");
        ip_mreq mr{};
        ::inet_pton(AF_INET, group, &mr.imr_multiaddr);
        ::inet_pton(AF_INET, iface, &mr.imr_interface);
        if (::setsockopt(fd_.get(), IPPROTO_IP, IP_ADD_MEMBERSHIP, &mr, sizeof mr) != 0)
            return err("IP_ADD_MEMBERSHIP");
        if (::fcntl(fd_.get(), F_SETFL, O_NONBLOCK) != 0) return err("O_NONBLOCK");
        return {};
    }
    // Non-blocking: returns bytes read, 0 if nothing is waiting, -1 on error.
    ssize_t poll(std::uint8_t* buf, std::size_t cap) {
        const ssize_t n = ::recv(fd_.get(), buf, cap, 0);
        if (n < 0) return (errno == EAGAIN || errno == EWOULDBLOCK) ? 0 : -1;
        return n;
    }
    [[nodiscard]] int fd() const { return fd_.get(); }

private:
    static std::string err(const char* what) { return std::string(what) + ": " + std::strerror(errno); }
    Fd fd_;
};

// ---- TCP snapshot stream: each frame is u16 little-endian length + packet.
inline bool write_all(int fd, const std::uint8_t* p, std::size_t n) {
    while (n > 0) {
        const ssize_t k = ::send(fd, p, n, MSG_NOSIGNAL);
        if (k <= 0) { if (k < 0 && errno == EINTR) continue; return false; }
        p += k;
        n -= static_cast<std::size_t>(k);
    }
    return true;
}
inline bool read_all(int fd, std::uint8_t* p, std::size_t n) {
    while (n > 0) {
        const ssize_t k = ::recv(fd, p, n, 0);
        if (k <= 0) { if (k < 0 && errno == EINTR) continue; return false; }
        p += k;
        n -= static_cast<std::size_t>(k);
    }
    return true;
}
inline bool write_frame(int fd, const std::uint8_t* p, std::size_t n) {
    const std::uint8_t h[2] = {static_cast<std::uint8_t>(n), static_cast<std::uint8_t>(n >> 8)};
    return write_all(fd, h, 2) && write_all(fd, p, n);
}
// Returns frame length, or -1 on EOF/error/oversize.
inline long read_frame(int fd, std::uint8_t* buf, std::size_t cap) {
    std::uint8_t h[2];
    if (!read_all(fd, h, 2)) return -1;
    const std::size_t n = h[0] | (h[1] << 8);
    if (n > cap) return -1;
    return read_all(fd, buf, n) ? static_cast<long>(n) : -1;
}

inline Fd tcp_listen(std::uint16_t port) {
    Fd s(::socket(AF_INET, SOCK_STREAM, 0));
    if (!s.ok()) return s;
    int one = 1;
    ::setsockopt(s.get(), SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a = addr_of("127.0.0.1", port);
    if (::bind(s.get(), reinterpret_cast<sockaddr*>(&a), sizeof a) != 0 || ::listen(s.get(), 4) != 0)
        return Fd();
    return s;
}
inline std::uint16_t local_port(int fd) {
    sockaddr_in a{};
    socklen_t l = sizeof a;
    ::getsockname(fd, reinterpret_cast<sockaddr*>(&a), &l);
    return ntohs(a.sin_port);
}
inline Fd tcp_connect(std::uint16_t port) {
    Fd s(::socket(AF_INET, SOCK_STREAM, 0));
    if (!s.ok()) return s;
    int one = 1;
    ::setsockopt(s.get(), IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    sockaddr_in a = addr_of("127.0.0.1", port);
    if (::connect(s.get(), reinterpret_cast<sockaddr*>(&a), sizeof a) != 0) return Fd();
    return s;
}

}  // namespace ull::mcast::net
