//! Newest-first reverse-paged binary chunk framing for zero-copy JS/Python scalar loading.
//!
//! Frame layout:
//! ```text
//! ┌───────────────────┬───────────────────────────────────────┬─────────────────────────────────┐
//! │ header_len: u32le │ header_json (UTF-8, space-padded so   │ contiguous little-endian arrays │
//! │                   │ (4 + header_len) % 8 == 0 for aligned │ step, global_step, ts_ms, value │
//! │                   │ Float64Array / BigInt64Array views)   │ each of length (end - start)    │
//! └───────────────────┴───────────────────────────────────────┴─────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};

use crate::error::{BoardError, BoardResult};
use crate::storage::RunReader;

/// Element dtype used for `step`, `global_step`, and `ts_ms` in binary chunk frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkXType {
    /// Cast index columns to `f64` so JS uPlot can create zero-copy `Float64Array` views.
    F64,
    /// Preserve exact 64-bit signed integers (`i64`) for lossless Python/analytics consumers.
    I64,
}

impl ChunkXType {
    pub fn from_name(s: &str) -> BoardResult<Self> {
        match s {
            "f64" | "float64" => Ok(Self::F64),
            "i64" | "int64" => Ok(Self::I64),
            other => Err(BoardError::Invalid(format!("unsupported x_dtype '{other}'"))),
        }
    }

    pub fn dtype_str(self) -> &'static str {
        match self {
            Self::F64 => "f64",
            Self::I64 => "i64",
        }
    }
}

/// JSON header embedded at the start of every binary scalar chunk frame.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkHeader {
    pub run_id: String,
    pub metric: String,
    pub total_rows: usize,
    pub start: usize,
    pub end: usize,
    pub columns: [String; 4],
    pub dtypes: [String; 4],
    pub done: bool,
    pub next_before: Option<usize>,
}

/// Compute adaptive newest-first `(start, end)` row ranges covering `[0 .. total_rows)`.
///
/// Starts at `first_chunk` (e.g. 8,192 rows) from the tail and doubles each step up to `max_chunk`
/// (e.g. 65,536 rows), walking cleanly down to row `0` with zero gaps or overlaps.
pub fn plan_newest_first_ranges(
    total_rows: usize,
    first_chunk: usize,
    max_chunk: usize,
) -> Vec<(usize, usize)> {
    if total_rows == 0 {
        return Vec::new();
    }
    let first = first_chunk.max(1);
    let cap = max_chunk.max(first);
    let mut ranges = Vec::new();
    let mut end = total_rows;
    let mut chunk_size = first;
    while end > 0 {
        let start = end.saturating_sub(chunk_size);
        ranges.push((start, end));
        end = start;
        chunk_size = chunk_size.saturating_mul(2).min(cap);
    }
    ranges
}

/// Encode an explicit `[start .. end)` slice of a scalar metric into one binary chunk frame.
pub fn encode_scalar_range_frame(
    reader: &RunReader,
    run_id: &str,
    metric: &str,
    start: usize,
    end: usize,
    x_dtype: ChunkXType,
) -> BoardResult<Vec<u8>> {
    let want = end.saturating_sub(start);
    let (total_rows, cols) = reader.scalar_slice(metric, start, Some(want))?;
    let actual_end = start + cols.values.len();
    let done = start == 0 || cols.values.is_empty();
    let next_before = if start > 0 { Some(start) } else { None };
    let x_str = x_dtype.dtype_str();

    let header = ChunkHeader {
        run_id: run_id.to_string(),
        metric: metric.to_string(),
        total_rows,
        start,
        end: actual_end,
        columns: [
            "step".into(),
            "global_step".into(),
            "ts_ms".into(),
            "value".into(),
        ],
        dtypes: [x_str.into(), x_str.into(), x_str.into(), "f64".into()],
        done,
        next_before,
    };

    let mut header_bytes = serde_json::to_vec(&header)?;
    // Pad header JSON with ASCII spaces so (4 + header_bytes.len()) is 8-byte aligned.
    let rem = (4 + header_bytes.len()) % 8;
    if rem != 0 {
        let pad = 8 - rem;
        header_bytes.extend(std::iter::repeat_n(b' ', pad));
    }

    let n = cols.values.len();
    let payload_bytes = n * 4 * 8;
    let mut frame = Vec::with_capacity(4 + header_bytes.len() + payload_bytes);
    frame.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
    frame.extend_from_slice(&header_bytes);

    match x_dtype {
        ChunkXType::F64 => {
            for &v in &cols.steps {
                frame.extend_from_slice(&(v as f64).to_le_bytes());
            }
            for &v in &cols.global_steps {
                frame.extend_from_slice(&(v as f64).to_le_bytes());
            }
            for &v in &cols.ts_ms {
                frame.extend_from_slice(&(v as f64).to_le_bytes());
            }
        }
        ChunkXType::I64 => {
            for &v in &cols.steps {
                frame.extend_from_slice(&v.to_le_bytes());
            }
            for &v in &cols.global_steps {
                frame.extend_from_slice(&v.to_le_bytes());
            }
            for &v in &cols.ts_ms {
                frame.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    for &v in &cols.values {
        frame.extend_from_slice(&v.to_le_bytes());
    }
    Ok(frame)
}

/// Encode either a newest-first backward page (`before`) or live-tail forward page (`after`).
pub fn encode_scalar_page_frame(
    reader: &RunReader,
    run_id: &str,
    metric: &str,
    before: Option<usize>,
    after: Option<usize>,
    limit: usize,
    x_dtype: ChunkXType,
) -> BoardResult<Vec<u8>> {
    if before.is_some() && after.is_some() {
        return Err(BoardError::Invalid("cannot specify both 'before' and 'after'".into()));
    }
    let total = reader.scalar_len(metric)?;
    let lim = limit.max(1);
    let (start, end) = if let Some(a) = after {
        let s = a.min(total);
        let e = s.saturating_add(lim).min(total);
        (s, e)
    } else {
        let e = before.unwrap_or(total).min(total);
        let s = e.saturating_sub(lim);
        (s, e)
    };
    encode_scalar_range_frame(reader, run_id, metric, start, end, x_dtype)
}

/// Parse a binary chunk frame back into `(ChunkHeader, raw_payload_slice)` for verification/testing.
pub fn decode_chunk_frame(frame: &[u8]) -> BoardResult<(ChunkHeader, &[u8])> {
    if frame.len() < 4 {
        return Err(BoardError::Codec("chunk frame shorter than 4-byte header length".into()));
    }
    let header_len = u32::from_le_bytes([frame[0], frame[1], frame[2], frame[3]]) as usize;
    let payload_offset = 4 + header_len;
    if payload_offset > frame.len() {
        return Err(BoardError::Codec("chunk frame truncated inside header".into()));
    }
    if !payload_offset.is_multiple_of(8) {
        return Err(BoardError::Codec("chunk payload offset is not 8-byte aligned".into()));
    }
    let header: ChunkHeader = serde_json::from_slice(&frame[4..payload_offset])?;
    Ok((header, &frame[payload_offset..]))
}
