// Read back, from the kernel, the latency-relevant settings actually in
// effect for this process, rather than trusting what we asked for.
//
// Everything here comes from /proc, /sys or a syscall result: CPU model and
// TSC flags (/proc/cpuinfo), the affinity mask the scheduler holds for us
// (sched_getaffinity), transparent huge pages mode and huge page pool
// (/sys/kernel/mm/transparent_hugepage/enabled, /proc/meminfo), whether a
// given mapping is really backed by huge pages (AnonHugePages in
// /proc/self/smaps_rollup), socket busy-poll settings (/proc/sys/net/core/
// busy_poll and busy_read), the CPU frequency governor, and whether we are
// in a VM (the "hypervisor" cpuinfo flag). A field reads "unavailable" when
// the file does not exist, which is itself recorded.
#pragma once

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

namespace ull::bench {

inline std::string read_first_line(const char* path) {
    std::ifstream f(path);
    std::string s;
    if (!f || !std::getline(f, s)) return "unavailable";
    return s;
}

inline std::string field_from(const char* path, const std::string& key) {
    std::ifstream f(path);
    std::string line;
    while (f && std::getline(f, line)) {
        if (line.rfind(key, 0) == 0) {
            auto p = line.find(':');
            if (p == std::string::npos) return line;
            auto v = line.substr(p + 1);
            while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.erase(v.begin());
            return v;
        }
    }
    return "unavailable";
}

struct SysInfo {
    std::string cpu_model, cpu_flags_tsc, hypervisor, affinity, thp_enabled, hugepages_total,
        anon_huge_kb, busy_poll, busy_read, governor;
    int online_cpus{0};
    int current_cpu{-1};

    std::string to_json() const {
        std::ostringstream o;
        auto q = [](const std::string& s) {
            std::string r = "\"";
            for (char c : s) {
                if (c == '"' || c == '\\') r += '\\';
                r += c;
            }
            return r + "\"";
        };
        o << "{\"cpu_model\":" << q(cpu_model) << ",\"tsc_flags\":" << q(cpu_flags_tsc)
          << ",\"hypervisor\":" << q(hypervisor) << ",\"online_cpus\":" << online_cpus
          << ",\"current_cpu\":" << current_cpu << ",\"affinity\":" << q(affinity)
          << ",\"thp_enabled\":" << q(thp_enabled) << ",\"hugepages_total\":" << q(hugepages_total)
          << ",\"anon_huge_kb\":" << q(anon_huge_kb) << ",\"busy_poll\":" << q(busy_poll)
          << ",\"busy_read\":" << q(busy_read) << ",\"governor\":" << q(governor) << "}";
        return o.str();
    }
};

inline SysInfo read_sysinfo() {
    SysInfo s;
    s.cpu_model = field_from("/proc/cpuinfo", "model name");
    const std::string flags = field_from("/proc/cpuinfo", "flags");
    std::string tf;
    for (const char* f : {"constant_tsc", "nonstop_tsc", "rdtscp"}) {
        const bool has = (" " + flags + " ").find(std::string(" ") + f + " ") != std::string::npos;
        tf += std::string(f) + (has ? "=yes " : "=no ");
    }
    s.cpu_flags_tsc = flags == "unavailable" ? flags : tf;
    s.hypervisor =
        flags == "unavailable"
            ? flags
            : ((" " + flags + " ").find(" hypervisor ") != std::string::npos ? "yes" : "no");
    s.thp_enabled = read_first_line("/sys/kernel/mm/transparent_hugepage/enabled");
    s.hugepages_total = field_from("/proc/meminfo", "HugePages_Total");
    s.anon_huge_kb = field_from("/proc/self/smaps_rollup", "AnonHugePages");
    s.busy_poll = read_first_line("/proc/sys/net/core/busy_poll");
    s.busy_read = read_first_line("/proc/sys/net/core/busy_read");
    s.governor = read_first_line("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
#if defined(__linux__)
    s.online_cpus = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
    s.current_cpu = sched_getcpu();
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        std::string a;
        for (int i = 0; i < CPU_SETSIZE; ++i)
            if (CPU_ISSET(i, &set)) a += (a.empty() ? "" : ",") + std::to_string(i);
        s.affinity = a;
    } else {
        s.affinity = "unavailable";
    }
#else
    s.affinity = "unavailable";
#endif
    return s;
}

// Pin the calling thread to one CPU; returns true only if the kernel's
// read-back mask is exactly that CPU.
inline bool pin_to_cpu(int cpu) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) return false;
    cpu_set_t back;
    CPU_ZERO(&back);
    if (sched_getaffinity(0, sizeof(back), &back) != 0) return false;
    return CPU_COUNT(&back) == 1 && CPU_ISSET(cpu, &back);
#else
    (void)cpu;
    return false;
#endif
}

}  // namespace ull::bench
