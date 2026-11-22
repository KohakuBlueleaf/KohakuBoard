//! Live in-memory run snapshot updated by the logger thread for zero-I/O live inspection & TUI.

use std::collections::{BTreeMap, VecDeque};

use serde::Serialize;

use crate::schema::{RunInfo, RunState};

const PREVIEW_RING_CAP: usize = 512;
const ALERT_RING_CAP: usize = 32;
const EMA_ALPHA: f64 = 0.1;

/// Live rolling statistics for one scalar metric.
#[derive(Clone, Debug, Serialize)]
pub struct MetricLiveStats {
    pub last: f64,
    pub ema: f64,
    pub min: f64,
    pub max: f64,
    pub count: u64,
    pub nan_count: u64,
    pub last_step: i64,
    pub last_global_step: i64,
    pub updated_at_ms: i64,
    pub preview: VecDeque<(i64, f64)>,
}

impl MetricLiveStats {
    fn new(step: i64, global_step: i64, ts_ms: i64, value: f64) -> Self {
        let is_fin = value.is_finite();
        let mut preview = VecDeque::with_capacity(64);
        if is_fin {
            preview.push_back((step, value));
        }
        Self {
            last: value,
            ema: if is_fin { value } else { f64::NAN },
            min: if is_fin { value } else { f64::INFINITY },
            max: if is_fin { value } else { f64::NEG_INFINITY },
            count: 1,
            nan_count: u64::from(!is_fin),
            last_step: step,
            last_global_step: global_step,
            updated_at_ms: ts_ms,
            preview,
        }
    }

    fn update(&mut self, step: i64, global_step: i64, ts_ms: i64, value: f64) {
        self.last = value;
        self.last_step = step;
        self.last_global_step = global_step;
        self.updated_at_ms = ts_ms;
        self.count += 1;
        if value.is_finite() {
            if self.ema.is_finite() {
                self.ema = EMA_ALPHA * value + (1.0 - EMA_ALPHA) * self.ema;
            } else {
                self.ema = value;
            }
            if value < self.min {
                self.min = value;
            }
            if value > self.max {
                self.max = value;
            }
            if self.preview.len() >= PREVIEW_RING_CAP {
                self.preview.pop_front();
            }
            self.preview.push_back((step, value));
        } else {
            self.nan_count += 1;
        }
    }
}

/// Live progress tracker entry reported via `board.progress(...)`.
#[derive(Clone, Debug, Serialize)]
pub struct ProgressSnapshot {
    pub label: String,
    pub current: u64,
    pub total: Option<u64>,
    pub unit: String,
    pub updated_at_ms: i64,
}

/// Recent alert toast entry.
#[derive(Clone, Debug, Serialize)]
pub struct AlertSnapshot {
    pub step: i64,
    pub ts_ms: i64,
    pub level: String,
    pub title: String,
    pub text: String,
}

/// Thread-safe snapshot state exposed via `LoggerEngine::snapshot_json()`.
#[derive(Clone, Debug, Serialize)]
pub struct LiveSnapshot {
    pub seq: u64,
    pub run_id: String,
    pub name: String,
    pub project: String,
    pub state: RunState,
    pub last_step: i64,
    pub last_global_step: i64,
    pub updated_at_ms: i64,
    pub dropped_events: u64,
    pub metrics: BTreeMap<String, MetricLiveStats>,
    pub progress: BTreeMap<String, ProgressSnapshot>,
    pub recent_alerts: VecDeque<AlertSnapshot>,
}

impl LiveSnapshot {
    pub fn from_info(info: &RunInfo) -> Self {
        Self {
            seq: 1,
            run_id: info.run_id.clone(),
            name: info.name.clone(),
            project: info.project.clone(),
            state: info.state,
            last_step: info.last_step,
            last_global_step: info.last_global_step,
            updated_at_ms: info.updated_at_ms,
            dropped_events: 0,
            metrics: BTreeMap::new(),
            progress: BTreeMap::new(),
            recent_alerts: VecDeque::with_capacity(ALERT_RING_CAP),
        }
    }

    pub fn record_scalar(
        &mut self,
        metric: &str,
        step: i64,
        global_step: i64,
        ts_ms: i64,
        val: f64,
    ) {
        if step > self.last_step {
            self.last_step = step;
        }
        if global_step > self.last_global_step {
            self.last_global_step = global_step;
        }
        self.updated_at_ms = self.updated_at_ms.max(ts_ms);
        match self.metrics.get_mut(metric) {
            Some(entry) => entry.update(step, global_step, ts_ms, val),
            None => {
                self.metrics.insert(
                    metric.to_string(),
                    MetricLiveStats::new(step, global_step, ts_ms, val),
                );
            }
        }
        self.seq = self.seq.wrapping_add(1);
    }

    pub fn record_progress(
        &mut self,
        label: String,
        current: u64,
        total: Option<u64>,
        unit: String,
        ts_ms: i64,
    ) {
        self.updated_at_ms = self.updated_at_ms.max(ts_ms);
        self.progress.insert(
            label.clone(),
            ProgressSnapshot { label, current, total, unit, updated_at_ms: ts_ms },
        );
        self.seq = self.seq.wrapping_add(1);
    }

    pub fn record_alert(
        &mut self,
        step: i64,
        ts_ms: i64,
        level: String,
        title: String,
        text: String,
    ) {
        if self.recent_alerts.len() >= ALERT_RING_CAP {
            self.recent_alerts.pop_front();
        }
        self.recent_alerts
            .push_back(AlertSnapshot { step, ts_ms, level, title, text });
        self.updated_at_ms = self.updated_at_ms.max(ts_ms);
        self.seq = self.seq.wrapping_add(1);
    }
}
