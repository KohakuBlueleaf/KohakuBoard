//! Event protocol passed from caller threads to the background Rust logger thread.

use std::collections::BTreeMap;

use crossbeam_channel::Sender;
use serde_json::Value as JsonValue;

use crate::error::BoardResult;
use crate::schema::{HistPrecision, MetricDefinition, RunState};

/// Backpressure policy when the bounded logger event queue is full.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverflowPolicy {
    /// Block caller thread (with GIL released in Python) until space is available.
    Block,
    /// Evict the oldest non-control event in the queue so the caller never blocks.
    DropOldest,
}

impl OverflowPolicy {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "block" => Some(Self::Block),
            "drop_oldest" => Some(Self::DropOldest),
            _ => None,
        }
    }
}

/// Events enqueued onto the single-writer logger thread.
pub enum LogEvent {
    Scalars {
        step: i64,
        global_step: i64,
        ts_ms: i64,
        metrics: Vec<(String, f64)>,
    },
    MediaPixels {
        step: i64,
        global_step: i64,
        ts_ms: i64,
        name: String,
        idx: i32,
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        channels: u8,
        format: String,
        quality: u8,
        caption: String,
        meta_json: String,
    },
    MediaEncoded {
        step: i64,
        global_step: i64,
        ts_ms: i64,
        name: String,
        idx: i32,
        media_type: String,
        bytes: Vec<u8>,
        ext: String,
        duration_ms: Option<i64>,
        caption: String,
        meta_json: String,
    },
    PutBlob {
        blob_key: String,
        bytes: Vec<u8>,
    },
    Table {
        step: i64,
        global_step: i64,
        ts_ms: i64,
        name: String,
        columns_json: String,
        row_count: i64,
        rows_json: String,
    },
    HistogramRaw {
        step: i64,
        global_step: i64,
        ts_ms: i64,
        name: String,
        values: Vec<f32>,
        bins: usize,
        precision: HistPrecision,
        range: Option<(f64, f64)>,
        percentiles: (f64, f64),
    },
    HistogramPrecomputed {
        step: i64,
        global_step: i64,
        ts_ms: i64,
        name: String,
        edges: Vec<f64>,
        counts: Vec<i32>,
        precision: HistPrecision,
    },
    KdeRaw {
        step: i64,
        global_step: i64,
        ts_ms: i64,
        name: String,
        values: Vec<f32>,
        num_points: usize,
        bandwidth: Option<f64>,
        approximate: bool,
        range: Option<(f64, f64)>,
        percentiles: (f64, f64),
    },
    KdePrecomputed {
        step: i64,
        global_step: i64,
        ts_ms: i64,
        name: String,
        range_min: f64,
        range_max: f64,
        bandwidth: f64,
        sample_count: u64,
        grid: Vec<f32>,
        density: Vec<f32>,
    },
    Tensor {
        step: i64,
        global_step: i64,
        ts_ms: i64,
        name: String,
        raw_bytes: Vec<u8>,
        dtype: String,
        shape_json: String,
        compress: bool,
        meta_json: String,
    },
    Artifact {
        step: i64,
        global_step: i64,
        ts_ms: i64,
        name: String,
        kind: String,
        uri: String,
        digest: Option<String>,
        size_bytes: Option<i64>,
        meta_json: String,
    },
    Alert {
        step: i64,
        global_step: i64,
        ts_ms: i64,
        level: String,
        title: String,
        text: String,
    },
    LogLines {
        lines: Vec<(i64, i64, String, String)>,
    },
    ConfigPatch {
        patch: JsonValue,
    },
    SummaryPatch {
        step: i64,
        global_step: i64,
        ts_ms: i64,
        values: BTreeMap<String, f64>,
    },
    MetaPatch {
        name: Option<String>,
        group: Option<String>,
        add_tags: Vec<String>,
        notes: Option<String>,
    },
    DefineMetric(MetricDefinition),
    Progress {
        label: String,
        current: u64,
        total: Option<u64>,
        unit: String,
        ts_ms: i64,
    },
    Flush(Sender<BoardResult<()>>),
    Shutdown {
        state: RunState,
        exit_code: i32,
        ack: Sender<BoardResult<()>>,
    },
}

impl LogEvent {
    /// True for synchronous control barriers that must never be dropped under `DropOldest`.
    pub fn is_control(&self) -> bool {
        matches!(self, Self::Flush(_) | Self::Shutdown { .. })
    }
}
