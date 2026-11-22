//! On-disk record schemas, constants, and binary codecs for KohakuBoard v2.

pub mod codec;
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// Current KohakuBoard v2 storage schema version.
pub const SCHEMA_VERSION: u32 = 2;

/// Target block size for scalar ColumnVault columns: 32 KiB = 1,024 `ScalarRow`s per block.
///
/// Sized so wide runs (1,000+ metrics) stay compact on disk while 8,192-row newest-first
/// chunk reads still touch only 8 contiguous SQLite BLOB blocks.
pub const SCALAR_BLOCK_BYTES: usize = 32 * 1024;

/// Target block size for histogram and KDE columns (64 KiB).
pub const DIST_BLOCK_BYTES: usize = 64 * 1024;

/// Fixed-width 32-byte little-endian scalar sample stored atomically in one ColumnVault column.
///
/// Storing all four fields in a single fixed-width column eliminates cross-column SWMR tearing
/// and preserves IEEE-754 `NaN` / `±Inf` verbatim.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct ScalarRow {
    pub step: i64,
    pub global_step: i64,
    pub ts_ms: i64,
    pub value: f64,
}

impl ScalarRow {
    pub const BYTE_SIZE: usize = std::mem::size_of::<Self>();
}

/// Fixed-width 40-byte header preceding histogram bin counts.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct HistHeader {
    pub step: i64,
    pub global_step: i64,
    pub ts_ms: i64,
    pub min_val: f64,
    pub max_val: f64,
}

impl HistHeader {
    pub const BYTE_SIZE: usize = std::mem::size_of::<Self>();
}

/// Fixed-width 56-byte header preceding KDE `f32` grid + density arrays.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct KdeHeader {
    pub step: i64,
    pub global_step: i64,
    pub ts_ms: i64,
    pub range_min: f64,
    pub range_max: f64,
    pub bandwidth: f64,
    pub sample_count: u64,
}

impl KdeHeader {
    pub const BYTE_SIZE: usize = std::mem::size_of::<Self>();
}

/// Histogram count storage precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HistPrecision {
    /// 32-bit signed integer counts (exact).
    Exact,
    /// 8-bit unsigned normalized counts (compact, 0..=255).
    Compact,
}

impl HistPrecision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "i32",
            Self::Compact => "u8",
        }
    }

    pub fn elem_bytes(self) -> usize {
        match self {
            Self::Exact => 4,
            Self::Compact => 1,
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "exact" | "i32" => Some(Self::Exact),
            "compact" | "u8" => Some(Self::Compact),
            _ => None,
        }
    }
}

/// Run lifecycle state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunState {
    Running,
    Finished,
    Crashed,
    Interrupted,
}

impl RunState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Finished => "finished",
            Self::Crashed => "crashed",
            Self::Interrupted => "interrupted",
        }
    }
}

/// Top-level run manifest persisted both in `run.json` and `kb_meta["run/info"]`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunInfo {
    pub schema_version: u32,
    pub run_id: String,
    pub name: String,
    pub project: String,
    pub annotation: Option<String>,
    pub group: Option<String>,
    pub job_type: Option<String>,
    pub tags: Vec<String>,
    pub notes: String,
    pub state: RunState,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    pub finished_at_ms: Option<i64>,
    pub exit_code: Option<i32>,
    pub pid: Option<u32>,
    pub hostname: String,
    pub last_step: i64,
    pub last_global_step: i64,
}

/// Running incremental summary statistics for one scalar metric.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricSummary {
    pub last: f64,
    pub last_step: i64,
    pub last_global_step: i64,
    pub min: f64,
    pub max: f64,
    pub count: u64,
    pub nan_count: u64,
    pub updated_at_ms: i64,
}

impl MetricSummary {
    pub fn new(step: i64, global_step: i64, ts_ms: i64, value: f64) -> Self {
        let is_fin = value.is_finite();
        Self {
            last: value,
            last_step: step,
            last_global_step: global_step,
            min: if is_fin { value } else { f64::INFINITY },
            max: if is_fin { value } else { f64::NEG_INFINITY },
            count: 1,
            nan_count: u64::from(!is_fin),
            updated_at_ms: ts_ms,
        }
    }

    pub fn update(&mut self, step: i64, global_step: i64, ts_ms: i64, value: f64) {
        self.last = value;
        self.last_step = step;
        self.last_global_step = global_step;
        self.count += 1;
        self.updated_at_ms = ts_ms;
        if value.is_finite() {
            if value < self.min {
                self.min = value;
            }
            if value > self.max {
                self.max = value;
            }
        } else {
            self.nan_count += 1;
        }
    }
}

/// Catalog entry mapping a logical scalar metric name to its `ColumnVault` column id.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScalarCatalogEntry {
    pub col_name: String,
    pub col_id: i64,
    pub count: u64,
    pub first_step: i64,
    pub last_step: i64,
    pub first_global_step: i64,
    pub last_global_step: i64,
    pub has_non_finite: bool,
}

/// Catalog entry mapping a histogram name to its `ColumnVault` column id.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistCatalogEntry {
    pub col_name: String,
    pub col_id: i64,
    pub bins: u32,
    pub precision: HistPrecision,
    pub count: u64,
}

/// Catalog entry mapping a KDE name to its `ColumnVault` column id.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KdeCatalogEntry {
    pub col_name: String,
    pub col_id: i64,
    pub num_points: u32,
    pub count: u64,
}

/// Per-metric display and summary preference registered via `board.define_metric`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricDefinition {
    pub pattern: String,
    pub step_metric: Option<String>,
    pub summary: String,
    pub hidden: bool,
}

/// Persisted run alert / milestone event.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlertRecord {
    pub id: i64,
    pub step: i64,
    pub global_step: i64,
    pub ts_ms: i64,
    pub level: String,
    pub title: String,
    pub text: String,
}

/// Full metric catalog stored in `kb_meta`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RunCatalogMeta {
    pub scalars: BTreeMap<String, ScalarCatalogEntry>,
    pub histograms: BTreeMap<String, HistCatalogEntry>,
    pub kdes: BTreeMap<String, KdeCatalogEntry>,
    pub next_col_seq: u32,
}
