//! Optional runtime-loaded NVIDIA GPU telemetry via `nvml-wrapper`.

use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
use nvml_wrapper::Nvml;

/// Stateful NVML GPU sampler. Gracefully unavailable on CPU-only hosts.
pub struct GpuSampler {
    nvml: Nvml,
    device_count: u32,
}

impl GpuSampler {
    /// Attempt to dynamically initialize NVML. Returns `None` if unavailable.
    pub fn try_init() -> Option<Self> {
        let nvml = Nvml::init().ok()?;
        let device_count = nvml.device_count().ok()?;
        if device_count == 0 {
            return None;
        }
        Some(Self { nvml, device_count })
    }

    pub fn sample(&mut self, out: &mut Vec<(String, f64)>) {
        for idx in 0..self.device_count {
            let Ok(dev) = self.nvml.device_by_index(idx) else {
                continue;
            };
            if let Ok(util) = dev.utilization_rates() {
                out.push((format!("sys/gpu{idx}_util"), f64::from(util.gpu)));
            }
            if let Ok(mem) = dev.memory_info() {
                let used_gb = (mem.used as f64) / (1024.0 * 1024.0 * 1024.0);
                let total_gb = (mem.total as f64) / (1024.0 * 1024.0 * 1024.0);
                out.push((format!("sys/gpu{idx}_vram_gb"), used_gb));
                if total_gb > 0.0 {
                    out.push((format!("sys/gpu{idx}_vram_pct"), (used_gb / total_gb) * 100.0));
                }
            }
            if let Ok(temp) = dev.temperature(TemperatureSensor::Gpu) {
                out.push((format!("sys/gpu{idx}_temp_c"), f64::from(temp)));
            }
            if let Ok(power_mw) = dev.power_usage() {
                out.push((format!("sys/gpu{idx}_power_w"), f64::from(power_mw) / 1000.0));
            }
        }
    }
}

/// Return names of detected NVIDIA GPUs (empty vec when NVML is absent).
pub fn gpu_names() -> Vec<String> {
    let Ok(nvml) = Nvml::init() else {
        return Vec::new();
    };
    let Ok(count) = nvml.device_count() else {
        return Vec::new();
    };
    (0..count)
        .filter_map(|i| nvml.device_by_index(i).ok()?.name().ok())
        .collect()
}
