//! CPU and RAM sampling via `sysinfo`.

use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

/// Static CPU/RAM summary.
pub struct HardwareSummary {
    pub cpu_count: usize,
    pub ram_total_gb: f64,
}

/// Stateful CPU/RAM sampler.
pub struct CpuRamSampler {
    sys: System,
}

impl Default for CpuRamSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuRamSampler {
    pub fn new() -> Self {
        let sys = System::new_with_specifics(
            RefreshKind::nothing()
                .with_cpu(CpuRefreshKind::nothing().with_cpu_usage())
                .with_memory(MemoryRefreshKind::nothing().with_ram()),
        );
        Self { sys }
    }

    pub fn sample(&mut self, out: &mut Vec<(String, f64)>) {
        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();
        let cpu_pct = f64::from(self.sys.global_cpu_usage());
        let used_gb = (self.sys.used_memory() as f64) / (1024.0 * 1024.0 * 1024.0);
        let total_gb = (self.sys.total_memory() as f64) / (1024.0 * 1024.0 * 1024.0);
        out.push(("sys/cpu_pct".into(), cpu_pct));
        out.push(("sys/ram_used_gb".into(), used_gb));
        if total_gb > 0.0 {
            out.push(("sys/ram_pct".into(), (used_gb / total_gb) * 100.0));
        }
    }
}

/// Query static CPU core count and total RAM in GiB.
pub fn hardware_summary() -> HardwareSummary {
    let sys = System::new_with_specifics(
        RefreshKind::nothing()
            .with_cpu(CpuRefreshKind::nothing())
            .with_memory(MemoryRefreshKind::nothing().with_ram()),
    );
    HardwareSummary {
        cpu_count: sys.cpus().len(),
        ram_total_gb: (sys.total_memory() as f64) / (1024.0 * 1024.0 * 1024.0),
    }
}
