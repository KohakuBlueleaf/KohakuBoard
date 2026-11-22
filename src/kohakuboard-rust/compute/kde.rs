//! Gaussian kernel density estimation (exact and FFT-binned via `rustfft`).

use std::f64::consts::PI;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::error::{BoardError, BoardResult};

/// Computed 1D KDE evaluation ready for storage.
#[derive(Debug, Clone)]
pub struct ComputedKde {
    pub range_min: f64,
    pub range_max: f64,
    pub bandwidth: f64,
    pub sample_count: u64,
    pub grid: Vec<f32>,
    pub density: Vec<f32>,
}

fn percentile_sorted(sorted: &[f32], pct: f64) -> f64 {
    let p = pct.clamp(0.0, 100.0) / 100.0;
    let pos = p * (sorted.len().saturating_sub(1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - (lo as f64);
    let a = f64::from(sorted[lo]);
    let b = f64::from(sorted[hi.min(sorted.len() - 1)]);
    a + (b - a) * frac
}

fn std_dev(values: &[f32]) -> f64 {
    if values.len() < 2 {
        return 1.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let var = values
        .iter()
        .map(|&v| {
            let d = f64::from(v) - mean;
            d * d
        })
        .sum::<f64>()
        / (n - 1.0);
    var.sqrt()
}

/// Compute a 1D Gaussian KDE from raw `f32` samples.
///
/// Uses exact evaluation for small inputs (`<= 2048` and `!approximate`) and FFT-binned
/// linear-binning convolution for larger sample counts.
pub fn compute_kde(
    values: &[f32],
    num_points: usize,
    bandwidth: Option<f64>,
    approximate: bool,
    range: Option<(f64, f64)>,
    percentiles: (f64, f64),
) -> BoardResult<Option<ComputedKde>> {
    if !(8..=4096).contains(&num_points) {
        return Err(BoardError::Invalid(format!(
            "KDE num_points must be in 8..=4096, got {num_points}"
        )));
    }

    let mut finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return Ok(None);
    }
    let n = finite.len();

    let (mut range_min, mut range_max) = match range {
        Some((lo, hi)) if lo.is_finite() && hi.is_finite() && lo < hi => (lo, hi),
        Some(_) => return Err(BoardError::Invalid("KDE range requires finite min < max".into())),
        None => {
            finite.sort_unstable_by(|a, b| a.total_cmp(b));
            let lo = percentile_sorted(&finite, percentiles.0);
            let hi = percentile_sorted(&finite, percentiles.1);
            if (hi - lo).abs() < 1e-9 {
                (f64::from(finite[0]), f64::from(*finite.last().unwrap()))
            } else {
                (lo, hi)
            }
        }
    };

    if (range_max - range_min).abs() < 1e-12 {
        range_min -= 0.5;
        range_max += 0.5;
    }

    // Scott's rule-of-thumb bandwidth when unspecified.
    let bw = match bandwidth {
        Some(b) if b.is_finite() && b > 0.0 => b,
        Some(_) => return Err(BoardError::Invalid("KDE bandwidth must be finite and > 0".into())),
        None => {
            let sd = std_dev(&finite).max((range_max - range_min) * 1e-3);
            (1.06 * sd * (n as f64).powf(-0.2)).max(1e-6)
        }
    };

    let step = (range_max - range_min) / ((num_points - 1) as f64);
    let grid: Vec<f32> = (0..num_points)
        .map(|i| (range_min + (i as f64) * step) as f32)
        .collect();

    let density = if !approximate && n <= 2048 {
        exact_gaussian(&finite, &grid, bw)
    } else {
        fft_gaussian(&finite, num_points, range_min, range_max, bw)
    };

    Ok(Some(ComputedKde {
        range_min,
        range_max,
        bandwidth: bw,
        sample_count: n as u64,
        grid,
        density,
    }))
}

fn exact_gaussian(samples: &[f32], grid: &[f32], bw: f64) -> Vec<f32> {
    let norm = 1.0 / ((samples.len() as f64) * bw * (2.0 * PI).sqrt());
    let inv_2bw2 = 0.5 / (bw * bw);
    grid.iter()
        .map(|&gx| {
            let gx64 = f64::from(gx);
            let sum: f64 = samples
                .iter()
                .map(|&x| {
                    let d = gx64 - f64::from(x);
                    (-d * d * inv_2bw2).exp()
                })
                .sum();
            (sum * norm) as f32
        })
        .collect()
}

fn fft_gaussian(samples: &[f32], num_points: usize, lo: f64, hi: f64, bw: f64) -> Vec<f32> {
    let dx = (hi - lo) / ((num_points - 1) as f64);
    let mut weights = vec![0.0f64; num_points];
    for &x in samples {
        let xf = f64::from(x);
        if xf < lo || xf > hi {
            continue;
        }
        let pos = (xf - lo) / dx;
        let idx = (pos.floor() as usize).min(num_points - 1);
        let frac = pos - (idx as f64);
        if idx + 1 < num_points {
            weights[idx] += 1.0 - frac;
            weights[idx + 1] += frac;
        } else {
            weights[idx] += 1.0;
        }
    }

    let pad = (4.0 * bw / dx).ceil() as usize;
    let fft_len = (num_points + 2 * pad).next_power_of_two().max(64);
    let mut planner = FftPlanner::<f64>::new();
    let fft_fwd = planner.plan_fft_forward(fft_len);
    let fft_inv = planner.plan_fft_inverse(fft_len);

    let mut sig = vec![Complex::new(0.0, 0.0); fft_len];
    for (i, &w) in weights.iter().enumerate() {
        sig[i] = Complex::new(w, 0.0);
    }

    let mut ker = vec![Complex::new(0.0, 0.0); fft_len];
    let inv_2bw2 = 0.5 / (bw * bw);
    let norm = 1.0 / ((samples.len() as f64) * bw * (2.0 * PI).sqrt());
    for (k, slot) in ker.iter_mut().enumerate() {
        let signed = if k <= fft_len / 2 {
            k as f64
        } else {
            (k as f64) - (fft_len as f64)
        };
        let dist = signed * dx;
        *slot = Complex::new((-dist * dist * inv_2bw2).exp() * norm, 0.0);
    }

    fft_fwd.process(&mut sig);
    fft_fwd.process(&mut ker);
    for (s, k) in sig.iter_mut().zip(ker.iter()) {
        *s *= *k;
    }
    fft_inv.process(&mut sig);

    let scale = 1.0 / (fft_len as f64);
    sig.iter()
        .take(num_points)
        .map(|c| (c.re * scale).max(0.0) as f32)
        .collect()
}
