//! Unit tests for CPU/RAM and hardware environment sampling.

use super::collect_hardware_env;
use super::cpu_ram::CpuRamSampler;

#[test]
fn cpu_ram_sampler_emits_expected_keys() {
    let mut sampler = CpuRamSampler::new();
    let mut out = Vec::new();
    sampler.sample(&mut out);
    assert!(out.iter().any(|(k, _)| k == "sys/cpu_pct"));
    assert!(out.iter().any(|(k, _)| k == "sys/ram_used_gb"));
    let env = collect_hardware_env();
    assert!(env["cpu_count"].as_u64().unwrap_or(0) >= 1);
}
