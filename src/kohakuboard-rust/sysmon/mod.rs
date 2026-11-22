//! Low-overhead background system metrics sampler (CPU, RAM, optional NVML GPUs).

pub mod cpu_ram;
pub mod gpu;
#[cfg(test)]
mod tests;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crossbeam_channel::{bounded, Sender};

use crate::engine::events::LogEvent;

/// Background thread periodically sampling `sys/*` metrics into the logger channel.
pub struct SystemMonitor {
    stop_tx: Option<Sender<()>>,
    stopped: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl SystemMonitor {
    /// Start sampling system metrics every `interval_ms` milliseconds.
    pub fn start(
        interval_ms: u64,
        event_tx: Sender<LogEvent>,
        step_counter: Arc<std::sync::atomic::AtomicI64>,
        global_step_counter: Arc<std::sync::atomic::AtomicI64>,
    ) -> Self {
        let interval = Duration::from_millis(interval_ms.max(250));
        let (stop_tx, stop_rx) = bounded::<()>(1);
        let stopped = Arc::new(AtomicBool::new(false));
        let stopped_bg = Arc::clone(&stopped);

        let handle = thread::Builder::new()
            .name("kobo-sysmon".into())
            .spawn(move || {
                let mut cpu = cpu_ram::CpuRamSampler::new();
                let mut nvml = gpu::GpuSampler::try_init();

                while !stopped_bg.load(Ordering::Acquire) {
                    if stop_rx.recv_timeout(interval).is_ok() {
                        break;
                    }
                    let mut metrics = Vec::with_capacity(16);
                    cpu.sample(&mut metrics);
                    if let Some(ref mut gpus) = nvml {
                        gpus.sample(&mut metrics);
                    }
                    if metrics.is_empty() {
                        continue;
                    }
                    let ts_ms = now_ms();
                    let step = step_counter.load(Ordering::Relaxed);
                    let global_step = global_step_counter.load(Ordering::Relaxed);
                    // Non-blocking send: never stall sysmon if writer is busy or shutting down.
                    let _ =
                        event_tx.try_send(LogEvent::Scalars { step, global_step, ts_ms, metrics });
                }
            })
            .ok();

        Self { stop_tx: Some(stop_tx), stopped, handle }
    }

    /// Stop the sampler thread cleanly.
    pub fn stop(&mut self) {
        self.stopped.store(true, Ordering::Release);
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.try_send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for SystemMonitor {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Snapshot static hardware info (CPU count, RAM GB, GPU names) for `run/env`.
pub fn collect_hardware_env() -> serde_json::Value {
    let cpu_info = cpu_ram::hardware_summary();
    let gpus = gpu::gpu_names();
    serde_json::json!({
        "cpu_count": cpu_info.cpu_count,
        "ram_total_gb": cpu_info.ram_total_gb,
        "gpus": gpus,
    })
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
