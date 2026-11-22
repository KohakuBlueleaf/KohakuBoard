//! Little-endian pack/unpack codecs for columnar binary records.

use serde::Serialize;

use crate::error::{BoardError, BoardResult};

use super::{HistHeader, HistPrecision, KdeHeader, ScalarRow};

/// Decoded scalar columns separated into contiguous arrays.
#[derive(Debug, Clone)]
pub struct ScalarColumns {
    pub steps: Vec<i64>,
    pub global_steps: Vec<i64>,
    pub ts_ms: Vec<i64>,
    pub values: Vec<f64>,
}

/// Decoded histogram record.
#[derive(Debug, Clone, Serialize)]
pub struct DecodedHistogram {
    pub step: i64,
    pub global_step: i64,
    pub ts_ms: i64,
    pub min_val: f64,
    pub max_val: f64,
    pub precision: HistPrecision,
    pub counts: Vec<i32>,
}

/// Decoded KDE record.
#[derive(Debug, Clone, Serialize)]
pub struct DecodedKde {
    pub step: i64,
    pub global_step: i64,
    pub ts_ms: i64,
    pub range_min: f64,
    pub range_max: f64,
    pub bandwidth: f64,
    pub sample_count: u64,
    pub grid: Vec<f32>,
    pub density: Vec<f32>,
}

/// Pack `ScalarRow` slice into contiguous little-endian bytes.
pub fn pack_scalar_rows(rows: &[ScalarRow]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows.len() * ScalarRow::BYTE_SIZE);
    for r in rows {
        out.extend_from_slice(&r.step.to_le_bytes());
        out.extend_from_slice(&r.global_step.to_le_bytes());
        out.extend_from_slice(&r.ts_ms.to_le_bytes());
        out.extend_from_slice(&r.value.to_le_bytes());
    }
    out
}

/// Unpack raw little-endian bytes into `ScalarRow` values.
pub fn unpack_scalar_rows(bytes: &[u8]) -> BoardResult<Vec<ScalarRow>> {
    if !bytes.len().is_multiple_of(ScalarRow::BYTE_SIZE) {
        return Err(BoardError::Codec(format!(
            "scalar buffer length {} is not a multiple of {}",
            bytes.len(),
            ScalarRow::BYTE_SIZE
        )));
    }
    let count = bytes.len() / ScalarRow::BYTE_SIZE;
    let mut rows = Vec::with_capacity(count);
    for chunk in bytes.chunks_exact(ScalarRow::BYTE_SIZE) {
        rows.push(ScalarRow {
            step: i64::from_le_bytes(chunk[0..8].try_into().unwrap()),
            global_step: i64::from_le_bytes(chunk[8..16].try_into().unwrap()),
            ts_ms: i64::from_le_bytes(chunk[16..24].try_into().unwrap()),
            value: f64::from_le_bytes(chunk[24..32].try_into().unwrap()),
        });
    }
    Ok(rows)
}

/// De-interleave packed `ScalarRow` bytes directly into 4 contiguous typed vectors.
pub fn deinterleave_scalars(bytes: &[u8]) -> BoardResult<ScalarColumns> {
    if !bytes.len().is_multiple_of(ScalarRow::BYTE_SIZE) {
        return Err(BoardError::Codec(format!(
            "scalar buffer length {} is not a multiple of {}",
            bytes.len(),
            ScalarRow::BYTE_SIZE
        )));
    }
    let count = bytes.len() / ScalarRow::BYTE_SIZE;
    let mut steps = Vec::with_capacity(count);
    let mut global_steps = Vec::with_capacity(count);
    let mut ts_ms = Vec::with_capacity(count);
    let mut values = Vec::with_capacity(count);

    for chunk in bytes.chunks_exact(ScalarRow::BYTE_SIZE) {
        steps.push(i64::from_le_bytes(chunk[0..8].try_into().unwrap()));
        global_steps.push(i64::from_le_bytes(chunk[8..16].try_into().unwrap()));
        ts_ms.push(i64::from_le_bytes(chunk[16..24].try_into().unwrap()));
        values.push(f64::from_le_bytes(chunk[24..32].try_into().unwrap()));
    }
    Ok(ScalarColumns { steps, global_steps, ts_ms, values })
}

/// Pack one histogram record (40-byte header + u8 or i32 little-endian bin counts).
pub fn pack_histogram_record(
    header: &HistHeader,
    precision: HistPrecision,
    counts_bytes: &[u8],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(HistHeader::BYTE_SIZE + counts_bytes.len());
    out.extend_from_slice(&header.step.to_le_bytes());
    out.extend_from_slice(&header.global_step.to_le_bytes());
    out.extend_from_slice(&header.ts_ms.to_le_bytes());
    out.extend_from_slice(&header.min_val.to_le_bytes());
    out.extend_from_slice(&header.max_val.to_le_bytes());
    let _ = precision;
    out.extend_from_slice(counts_bytes);
    out
}

/// Unpack a sequence of fixed-width histogram records.
pub fn unpack_histogram_records(
    bytes: &[u8],
    bins: usize,
    precision: HistPrecision,
) -> BoardResult<Vec<DecodedHistogram>> {
    let rec_size = HistHeader::BYTE_SIZE + bins * precision.elem_bytes();
    if !bytes.len().is_multiple_of(rec_size) {
        return Err(BoardError::Codec(format!(
            "histogram buffer length {} not divisible by record size {rec_size}",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / rec_size);
    for rec in bytes.chunks_exact(rec_size) {
        let step = i64::from_le_bytes(rec[0..8].try_into().unwrap());
        let global_step = i64::from_le_bytes(rec[8..16].try_into().unwrap());
        let ts_ms = i64::from_le_bytes(rec[16..24].try_into().unwrap());
        let min_val = f64::from_le_bytes(rec[24..32].try_into().unwrap());
        let max_val = f64::from_le_bytes(rec[32..40].try_into().unwrap());
        let payload = &rec[40..];
        let counts = match precision {
            HistPrecision::Compact => payload.iter().map(|&b| i32::from(b)).collect(),
            HistPrecision::Exact => payload
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        };
        out.push(DecodedHistogram {
            step,
            global_step,
            ts_ms,
            min_val,
            max_val,
            precision,
            counts,
        });
    }
    Ok(out)
}

/// Pack one KDE record (56-byte header + `f32` grid + `f32` density).
pub fn pack_kde_record(header: &KdeHeader, grid: &[f32], density: &[f32]) -> BoardResult<Vec<u8>> {
    if grid.len() != density.len() {
        return Err(BoardError::Invalid(format!(
            "KDE grid length {} != density length {}",
            grid.len(),
            density.len()
        )));
    }
    let mut out = Vec::with_capacity(KdeHeader::BYTE_SIZE + grid.len() * 8);
    out.extend_from_slice(&header.step.to_le_bytes());
    out.extend_from_slice(&header.global_step.to_le_bytes());
    out.extend_from_slice(&header.ts_ms.to_le_bytes());
    out.extend_from_slice(&header.range_min.to_le_bytes());
    out.extend_from_slice(&header.range_max.to_le_bytes());
    out.extend_from_slice(&header.bandwidth.to_le_bytes());
    out.extend_from_slice(&header.sample_count.to_le_bytes());
    for &g in grid {
        out.extend_from_slice(&g.to_le_bytes());
    }
    for &d in density {
        out.extend_from_slice(&d.to_le_bytes());
    }
    Ok(out)
}

/// Unpack a sequence of fixed-width KDE records.
pub fn unpack_kde_records(bytes: &[u8], num_points: usize) -> BoardResult<Vec<DecodedKde>> {
    let rec_size = KdeHeader::BYTE_SIZE + num_points * 8;
    if !bytes.len().is_multiple_of(rec_size) {
        return Err(BoardError::Codec(format!(
            "KDE buffer length {} not divisible by record size {rec_size}",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / rec_size);
    for rec in bytes.chunks_exact(rec_size) {
        let step = i64::from_le_bytes(rec[0..8].try_into().unwrap());
        let global_step = i64::from_le_bytes(rec[8..16].try_into().unwrap());
        let ts_ms = i64::from_le_bytes(rec[16..24].try_into().unwrap());
        let range_min = f64::from_le_bytes(rec[24..32].try_into().unwrap());
        let range_max = f64::from_le_bytes(rec[32..40].try_into().unwrap());
        let bandwidth = f64::from_le_bytes(rec[40..48].try_into().unwrap());
        let sample_count = u64::from_le_bytes(rec[48..56].try_into().unwrap());
        let grid_end = 56 + num_points * 4;
        let grid = rec[56..grid_end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let density = rec[grid_end..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        out.push(DecodedKde {
            step,
            global_step,
            ts_ms,
            range_min,
            range_max,
            bandwidth,
            sample_count,
            grid,
            density,
        });
    }
    Ok(out)
}
