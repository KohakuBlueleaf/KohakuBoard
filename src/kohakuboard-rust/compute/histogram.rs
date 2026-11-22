//! Fast off-GIL histogram binning with percentile clipping and u8/i32 quantization.

use crate::error::{BoardError, BoardResult};
use crate::schema::HistPrecision;

/// Computed histogram result ready for columnar packing.
#[derive(Debug, Clone)]
pub struct ComputedHistogram {
    pub min_val: f64,
    pub max_val: f64,
    pub bins: usize,
    pub precision: HistPrecision,
    pub counts_bytes: Vec<u8>,
}

fn percentile_sorted(sorted: &[f32], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let p = pct.clamp(0.0, 100.0) / 100.0;
    let pos = p * (sorted.len().saturating_sub(1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - (lo as f64);
    let a = f64::from(sorted[lo]);
    let b = f64::from(sorted[hi.min(sorted.len() - 1)]);
    a + (b - a) * frac
}

fn encode_counts(counts: &[i32], precision: HistPrecision) -> Vec<u8> {
    match precision {
        HistPrecision::Exact => {
            let mut out = Vec::with_capacity(counts.len() * 4);
            for &c in counts {
                out.extend_from_slice(&c.to_le_bytes());
            }
            out
        }
        HistPrecision::Compact => {
            let max_c = counts.iter().copied().max().unwrap_or(0).max(0) as f64;
            if max_c <= 0.0 {
                return vec![0u8; counts.len()];
            }
            counts
                .iter()
                .map(|&c| {
                    ((c.max(0) as f64) / max_c * 255.0)
                        .round()
                        .clamp(0.0, 255.0) as u8
                })
                .collect()
        }
    }
}

/// Compute a histogram from raw `f32` samples with optional range or percentile clipping.
pub fn compute_histogram(
    values: &[f32],
    bins: usize,
    precision: HistPrecision,
    range: Option<(f64, f64)>,
    percentiles: (f64, f64),
) -> BoardResult<Option<ComputedHistogram>> {
    if bins == 0 || bins > 4096 {
        return Err(BoardError::Invalid(format!("histogram bins must be in 1..=4096, got {bins}")));
    }

    let mut finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return Ok(None);
    }

    let (mut min_val, mut max_val) = match range {
        Some((lo, hi)) if lo.is_finite() && hi.is_finite() && lo < hi => (lo, hi),
        Some(_) => {
            return Err(BoardError::Invalid("histogram range requires finite min < max".into()))
        }
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

    if (max_val - min_val).abs() < 1e-12 {
        min_val -= 0.5;
        max_val += 0.5;
    }

    let mut counts = vec![0i32; bins];
    let inv_span = (bins as f64) / (max_val - min_val);
    for &v in &finite {
        let vf = f64::from(v);
        if vf < min_val || vf > max_val {
            continue;
        }
        let mut idx = ((vf - min_val) * inv_span) as usize;
        if idx >= bins {
            idx = bins - 1;
        }
        counts[idx] = counts[idx].saturating_add(1);
    }

    Ok(Some(ComputedHistogram {
        min_val,
        max_val,
        bins,
        precision,
        counts_bytes: encode_counts(&counts, precision),
    }))
}

/// Quantize precomputed bin edges and integer counts into storage format.
pub fn pack_precomputed(
    edges: &[f64],
    counts: &[i32],
    precision: HistPrecision,
) -> BoardResult<ComputedHistogram> {
    if counts.is_empty() || edges.len() != counts.len() + 1 {
        return Err(BoardError::Invalid(format!(
            "precomputed histogram requires len(edges) == len(counts) + 1 (got {} and {})",
            edges.len(),
            counts.len()
        )));
    }
    let min_val = edges[0];
    let max_val = *edges.last().unwrap();
    if !min_val.is_finite() || !max_val.is_finite() || min_val >= max_val {
        return Err(BoardError::Invalid(
            "histogram bin edges must be finite and increasing".into(),
        ));
    }
    Ok(ComputedHistogram {
        min_val,
        max_val,
        bins: counts.len(),
        precision,
        counts_bytes: encode_counts(counts, precision),
    })
}
