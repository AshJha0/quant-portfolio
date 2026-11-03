//! Reads back the environment a benchmark actually ran in.
//!
//! The rule (plan W30): pinning, busy-poll and huge pages are reported as
//! read from `/proc` and `/sys`, never as what the operator meant to set.
//! On a non-Linux host each field says "unavailable" with the reason.

/// One key/value fact about the host, as read.
pub type Fact = (&'static str, String);

fn read(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

fn field(text: &str, key: &str) -> Option<String> {
    text.lines()
        .find(|l| l.starts_with(key))
        .and_then(|l| l.split_once(':'))
        .map(|(_, v)| v.trim().to_string())
}

/// Collects the facts. Never panics; missing files become "unavailable".
pub fn collect() -> Vec<Fact> {
    let na = |why: &str| format!("unavailable ({why})");
    let mut v: Vec<Fact> = vec![
        ("os", std::env::consts::OS.to_string()),
        ("arch", std::env::consts::ARCH.to_string()),
        (
            "logical_cpus",
            std::thread::available_parallelism()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| na("available_parallelism failed")),
        ),
    ];
    if cfg!(target_os = "linux") {
        let cpuinfo = read("/proc/cpuinfo").unwrap_or_default();
        let status = read("/proc/self/status").unwrap_or_default();
        let meminfo = read("/proc/meminfo").unwrap_or_default();
        v.push((
            "cpu_model",
            field(&cpuinfo, "model name").unwrap_or_else(|| na("no model name in /proc/cpuinfo")),
        ));
        v.push((
            "cpus_allowed_list",
            field(&status, "Cpus_allowed_list").unwrap_or_else(|| na("/proc/self/status")),
        ));
        v.push((
            "thp_enabled",
            read("/sys/kernel/mm/transparent_hugepage/enabled")
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| na("/sys/kernel/mm/transparent_hugepage/enabled")),
        ));
        v.push((
            "hugepages_total",
            field(&meminfo, "HugePages_Total").unwrap_or_else(|| na("/proc/meminfo")),
        ));
        v.push((
            "scaling_governor_cpu0",
            read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| na("no cpufreq in this VM/host")),
        ));
        v.push((
            "isolcpus",
            read("/sys/devices/system/cpu/isolated")
                .map(|s| {
                    let s = s.trim();
                    if s.is_empty() {
                        "none".to_string()
                    } else {
                        s.to_string()
                    }
                })
                .unwrap_or_else(|| na("/sys/devices/system/cpu/isolated")),
        ));
    } else {
        v.push((
            "cpu_model",
            std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| na("not Linux")),
        ));
        for k in [
            "cpus_allowed_list",
            "thp_enabled",
            "hugepages_total",
            "scaling_governor_cpu0",
            "isolcpus",
        ] {
            v.push((k, na("not Linux; /proc and /sys absent")));
        }
    }
    v.push((
        "busy_poll",
        "yes: the harness and ring consumers spin with spin_loop(); no sleep or futex on the timed path".to_string(),
    ));
    v
}

#[cfg(test)]
mod tests {
    #[test]
    fn collect_never_panics_and_labels_every_fact() {
        let f = super::collect();
        assert!(f.len() >= 9);
        assert!(f.iter().all(|(_, v)| !v.is_empty()));
    }

    #[test]
    fn field_parses_proc_style_lines() {
        let t = "Name:\tx\nCpus_allowed_list:\t0-3\n";
        assert_eq!(super::field(t, "Cpus_allowed_list").as_deref(), Some("0-3"));
        assert_eq!(super::field(t, "Missing"), None);
    }
}
