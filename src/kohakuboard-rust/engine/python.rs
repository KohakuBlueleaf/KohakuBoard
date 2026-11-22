//! PyO3 bindings exposing `_LoggerEngine` and hardware/media compute helpers.

use std::collections::BTreeMap;
use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::compute::media::{encode_image_pixels, wrap_encoded_media};
use crate::error::BoardError;
use crate::schema::{HistPrecision, MetricDefinition, RunState};
use crate::storage::RunOpenOptions;
use crate::sysmon::collect_hardware_env;

use super::{EngineConfig, LogEvent, LoggerEngine, OverflowPolicy};

fn parse_run_state(s: &str) -> Result<RunState, BoardError> {
    match s {
        "running" => Ok(RunState::Running),
        "finished" => Ok(RunState::Finished),
        "crashed" => Ok(RunState::Crashed),
        "interrupted" => Ok(RunState::Interrupted),
        other => Err(BoardError::Invalid(format!("invalid run state '{other}'"))),
    }
}

fn parse_precision(s: &str) -> Result<HistPrecision, BoardError> {
    HistPrecision::from_name(s)
        .ok_or_else(|| BoardError::Invalid(format!("invalid histogram precision '{s}'")))
}

type StagedMediaMeta = (String, String, Option<u32>, Option<u32>, i64);

/// Native logger engine owned by `kohakuboard.Board`.
#[pyclass(name = "_LoggerEngine")]
pub struct PyLoggerEngine {
    inner: LoggerEngine,
}

#[pymethods]
impl PyLoggerEngine {
    #[new]
    #[pyo3(signature = (
        run_dir,
        run_id,
        name,
        project,
        annotation = None,
        group = None,
        job_type = None,
        tags = Vec::new(),
        notes = String::new(),
        config_json = "{}".to_string(),
        env_json = "{}".to_string(),
        resume = false,
        flush_interval_ms = 500,
        queue_capacity = 65536,
        overflow = "block".to_string(),
        system_metrics_interval_ms = Some(2000),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        run_dir: Option<String>,
        run_id: String,
        name: String,
        project: String,
        annotation: Option<String>,
        group: Option<String>,
        job_type: Option<String>,
        tags: Vec<String>,
        notes: String,
        config_json: String,
        env_json: String,
        resume: bool,
        flush_interval_ms: u64,
        queue_capacity: usize,
        overflow: String,
        system_metrics_interval_ms: Option<u64>,
    ) -> PyResult<Self> {
        let policy = OverflowPolicy::from_name(&overflow)
            .ok_or_else(|| BoardError::Invalid(format!("invalid overflow policy '{overflow}'")))?;
        let config: serde_json::Value = serde_json::from_str(&config_json)
            .map_err(|e| BoardError::Invalid(format!("invalid config_json: {e}")))?;
        let mut env_val: serde_json::Value = serde_json::from_str(&env_json)
            .map_err(|e| BoardError::Invalid(format!("invalid env_json: {e}")))?;
        let hw = collect_hardware_env();
        if let Some(obj) = env_val.as_object_mut() {
            obj.entry("hardware".to_string()).or_insert(hw);
        }

        let cfg = EngineConfig {
            open: RunOpenOptions {
                run_dir: run_dir.map(PathBuf::from),
                run_id,
                name,
                project,
                annotation,
                group,
                job_type,
                tags,
                notes,
                config,
                env: env_val,
                resume,
            },
            flush_interval_ms,
            queue_capacity,
            overflow: policy,
            system_metrics_interval_ms,
        };
        let inner = py.allow_threads(|| LoggerEngine::start(cfg))?;
        Ok(Self { inner })
    }

    /// Initial last committed step (useful when resuming an existing run).
    #[getter]
    fn initial_last_step(&self) -> i64 {
        self.inner.initial_info().last_step
    }

    /// Initial last committed global_step (useful when resuming an existing run).
    #[getter]
    fn initial_last_global_step(&self) -> i64 {
        self.inner.initial_info().last_global_step
    }

    /// True if `finish()` has already been called.
    #[getter]
    fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }

    /// Number of events evicted under `overflow="drop_oldest"`.
    #[getter]
    fn dropped_events(&self) -> u64 {
        self.inner.dropped_events()
    }

    /// Update atomic step counters read by the background system monitor.
    fn set_step_counters(&self, step: i64, global_step: i64) {
        self.inner.set_step_counters(step, global_step);
    }

    /// Enqueue a batch of scalar metrics for one step.
    fn log_scalars(
        &self,
        py: Python<'_>,
        step: i64,
        global_step: i64,
        ts_ms: i64,
        metrics: Vec<(String, f64)>,
    ) -> PyResult<()> {
        self.inner.set_step_counters(step, global_step);
        py.allow_threads(|| {
            self.inner
                .send(LogEvent::Scalars { step, global_step, ts_ms, metrics })
        })?;
        Ok(())
    }

    /// Enqueue raw HWC u8 pixel bytes for off-GIL PNG/WebP/JPEG encoding and dedup storage.
    #[allow(clippy::too_many_arguments)]
    fn log_media_pixels(
        &self,
        py: Python<'_>,
        step: i64,
        global_step: i64,
        ts_ms: i64,
        name: String,
        idx: i32,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: u8,
        format: String,
        quality: u8,
        caption: String,
        meta_json: String,
    ) -> PyResult<()> {
        let owned = pixels.to_vec();
        py.allow_threads(|| {
            self.inner.send(LogEvent::MediaPixels {
                step,
                global_step,
                ts_ms,
                name,
                idx,
                pixels: owned,
                width,
                height,
                channels,
                format,
                quality,
                caption,
                meta_json,
            })
        })?;
        Ok(())
    }

    /// Enqueue pre-encoded media bytes (image/video/audio).
    #[allow(clippy::too_many_arguments)]
    fn log_media_encoded(
        &self,
        py: Python<'_>,
        step: i64,
        global_step: i64,
        ts_ms: i64,
        name: String,
        idx: i32,
        media_type: String,
        bytes: &[u8],
        ext: String,
        duration_ms: Option<i64>,
        caption: String,
        meta_json: String,
    ) -> PyResult<()> {
        let owned = bytes.to_vec();
        py.allow_threads(|| {
            self.inner.send(LogEvent::MediaEncoded {
                step,
                global_step,
                ts_ms,
                name,
                idx,
                media_type,
                bytes: owned,
                ext,
                duration_ms,
                caption,
                meta_json,
            })
        })?;
        Ok(())
    }

    /// Encode raw pixels off-GIL, queue blob write, and return `(blob_key, mime, width, height, size)`.
    #[allow(clippy::too_many_arguments)]
    fn stage_media_pixels(
        &self,
        py: Python<'_>,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: u8,
        format: String,
        quality: u8,
    ) -> PyResult<StagedMediaMeta> {
        let owned = pixels.to_vec();
        let (blob_key, mime, w, h, size) = py.allow_threads(|| -> Result<_, BoardError> {
            let enc = encode_image_pixels(&owned, width, height, channels, &format, quality)?;
            let size = enc.bytes.len() as i64;
            let key = enc.blob_key.clone();
            self.inner
                .send(LogEvent::PutBlob { blob_key: enc.blob_key, bytes: enc.bytes })?;
            Ok((key, enc.mime, enc.width, enc.height, size))
        })?;
        Ok((blob_key, mime, w, h, size))
    }

    /// Hash pre-encoded media bytes off-GIL, queue blob write, and return `(blob_key, mime, w, h, size)`.
    fn stage_media_encoded(
        &self,
        py: Python<'_>,
        bytes: &[u8],
        ext: String,
        media_type: String,
    ) -> PyResult<StagedMediaMeta> {
        let owned = bytes.to_vec();
        let (blob_key, mime, w, h, size) = py.allow_threads(|| -> Result<_, BoardError> {
            let enc = wrap_encoded_media(owned, &ext, &media_type);
            let size = enc.bytes.len() as i64;
            let key = enc.blob_key.clone();
            self.inner
                .send(LogEvent::PutBlob { blob_key: enc.blob_key, bytes: enc.bytes })?;
            Ok((key, enc.mime, enc.width, enc.height, size))
        })?;
        Ok((blob_key, mime, w, h, size))
    }

    /// Enqueue a table snapshot.
    #[allow(clippy::too_many_arguments)]
    fn log_table(
        &self,
        py: Python<'_>,
        step: i64,
        global_step: i64,
        ts_ms: i64,
        name: String,
        columns_json: String,
        row_count: i64,
        rows_json: String,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            self.inner.send(LogEvent::Table {
                step,
                global_step,
                ts_ms,
                name,
                columns_json,
                row_count,
                rows_json,
            })
        })?;
        Ok(())
    }

    /// Enqueue raw float32 bytes (`f32` little-endian contiguous buffer) for off-GIL histogramming.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (step, global_step, ts_ms, name, values_f32_bytes, bins, precision, range = None, percentiles = (1.0, 99.0)))]
    fn log_histogram_raw(
        &self,
        py: Python<'_>,
        step: i64,
        global_step: i64,
        ts_ms: i64,
        name: String,
        values_f32_bytes: &[u8],
        bins: usize,
        precision: String,
        range: Option<(f64, f64)>,
        percentiles: (f64, f64),
    ) -> PyResult<()> {
        if !values_f32_bytes.len().is_multiple_of(4) {
            return Err(BoardError::Invalid(
                "values_f32_bytes length must be multiple of 4".into(),
            )
            .into());
        }
        let prec = parse_precision(&precision)?;
        let values: Vec<f32> = values_f32_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        py.allow_threads(|| {
            self.inner.send(LogEvent::HistogramRaw {
                step,
                global_step,
                ts_ms,
                name,
                values,
                bins,
                precision: prec,
                range,
                percentiles,
            })
        })?;
        Ok(())
    }

    /// Enqueue precomputed histogram bin edges and integer counts.
    #[allow(clippy::too_many_arguments)]
    fn log_histogram_precomputed(
        &self,
        py: Python<'_>,
        step: i64,
        global_step: i64,
        ts_ms: i64,
        name: String,
        edges: Vec<f64>,
        counts: Vec<i32>,
        precision: String,
    ) -> PyResult<()> {
        let prec = parse_precision(&precision)?;
        py.allow_threads(|| {
            self.inner.send(LogEvent::HistogramPrecomputed {
                step,
                global_step,
                ts_ms,
                name,
                edges,
                counts,
                precision: prec,
            })
        })?;
        Ok(())
    }

    /// Enqueue raw float32 bytes for off-GIL exact/FFT Gaussian KDE.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (step, global_step, ts_ms, name, values_f32_bytes, num_points, bandwidth = None, approximate = true, range = None, percentiles = (1.0, 99.0)))]
    fn log_kde_raw(
        &self,
        py: Python<'_>,
        step: i64,
        global_step: i64,
        ts_ms: i64,
        name: String,
        values_f32_bytes: &[u8],
        num_points: usize,
        bandwidth: Option<f64>,
        approximate: bool,
        range: Option<(f64, f64)>,
        percentiles: (f64, f64),
    ) -> PyResult<()> {
        if !values_f32_bytes.len().is_multiple_of(4) {
            return Err(BoardError::Invalid(
                "values_f32_bytes length must be multiple of 4".into(),
            )
            .into());
        }
        let values: Vec<f32> = values_f32_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        py.allow_threads(|| {
            self.inner.send(LogEvent::KdeRaw {
                step,
                global_step,
                ts_ms,
                name,
                values,
                num_points,
                bandwidth,
                approximate,
                range,
                percentiles,
            })
        })?;
        Ok(())
    }

    /// Enqueue precomputed KDE grid and density arrays.
    #[allow(clippy::too_many_arguments)]
    fn log_kde_precomputed(
        &self,
        py: Python<'_>,
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
    ) -> PyResult<()> {
        py.allow_threads(|| {
            self.inner.send(LogEvent::KdePrecomputed {
                step,
                global_step,
                ts_ms,
                name,
                range_min,
                range_max,
                bandwidth,
                sample_count,
                grid,
                density,
            })
        })?;
        Ok(())
    }

    /// Enqueue raw tensor bytes for off-GIL compression + blake3 dedup storage.
    #[allow(clippy::too_many_arguments)]
    fn log_tensor(
        &self,
        py: Python<'_>,
        step: i64,
        global_step: i64,
        ts_ms: i64,
        name: String,
        raw_bytes: &[u8],
        dtype: String,
        shape_json: String,
        compress: bool,
        meta_json: String,
    ) -> PyResult<()> {
        let owned = raw_bytes.to_vec();
        py.allow_threads(|| {
            self.inner.send(LogEvent::Tensor {
                step,
                global_step,
                ts_ms,
                name,
                raw_bytes: owned,
                dtype,
                shape_json,
                compress,
                meta_json,
            })
        })?;
        Ok(())
    }

    /// Enqueue an artifact pointer / metadata event.
    #[allow(clippy::too_many_arguments)]
    fn log_artifact(
        &self,
        py: Python<'_>,
        step: i64,
        global_step: i64,
        ts_ms: i64,
        name: String,
        kind: String,
        uri: String,
        digest: Option<String>,
        size_bytes: Option<i64>,
        meta_json: String,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            self.inner.send(LogEvent::Artifact {
                step,
                global_step,
                ts_ms,
                name,
                kind,
                uri,
                digest,
                size_bytes,
                meta_json,
            })
        })?;
        Ok(())
    }

    /// Enqueue an alert / milestone record.
    #[allow(clippy::too_many_arguments)]
    fn log_alert(
        &self,
        py: Python<'_>,
        step: i64,
        global_step: i64,
        ts_ms: i64,
        level: String,
        title: String,
        text: String,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            self.inner
                .send(LogEvent::Alert { step, global_step, ts_ms, level, title, text })
        })?;
        Ok(())
    }

    /// Enqueue captured console log lines `(ts_ms, step, stream, text)`.
    fn log_lines(&self, py: Python<'_>, lines: Vec<(i64, i64, String, String)>) -> PyResult<()> {
        py.allow_threads(|| self.inner.send(LogEvent::LogLines { lines }))?;
        Ok(())
    }

    /// Merge a JSON config patch into `run/config`.
    fn patch_config(&self, py: Python<'_>, patch_json: String) -> PyResult<()> {
        let patch: serde_json::Value = serde_json::from_str(&patch_json)
            .map_err(|e| BoardError::Invalid(format!("invalid patch_json: {e}")))?;
        py.allow_threads(|| self.inner.send(LogEvent::ConfigPatch { patch }))?;
        Ok(())
    }

    /// Update explicit scalar summary overrides.
    fn patch_summary(
        &self,
        py: Python<'_>,
        step: i64,
        global_step: i64,
        ts_ms: i64,
        values: BTreeMap<String, f64>,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            self.inner
                .send(LogEvent::SummaryPatch { step, global_step, ts_ms, values })
        })?;
        Ok(())
    }

    /// Patch mutable run metadata fields.
    #[pyo3(signature = (name = None, group = None, add_tags = Vec::new(), notes = None))]
    fn patch_meta(
        &self,
        py: Python<'_>,
        name: Option<String>,
        group: Option<String>,
        add_tags: Vec<String>,
        notes: Option<String>,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            self.inner
                .send(LogEvent::MetaPatch { name, group, add_tags, notes })
        })?;
        Ok(())
    }

    /// Register a metric display / summary preference pattern.
    #[pyo3(signature = (pattern, step_metric = None, summary = "last".to_string(), hidden = false))]
    fn define_metric(
        &self,
        py: Python<'_>,
        pattern: String,
        step_metric: Option<String>,
        summary: String,
        hidden: bool,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            self.inner.send(LogEvent::DefineMetric(MetricDefinition {
                pattern,
                step_metric,
                summary,
                hidden,
            }))
        })?;
        Ok(())
    }

    /// Update live progress state (in-memory snapshot for TUI / observability).
    #[pyo3(signature = (label, current, total = None, unit = "step".to_string(), ts_ms = 0))]
    fn progress(
        &self,
        py: Python<'_>,
        label: String,
        current: u64,
        total: Option<u64>,
        unit: String,
        ts_ms: i64,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            self.inner
                .send(LogEvent::Progress { label, current, total, unit, ts_ms })
        })?;
        Ok(())
    }

    /// Block (with GIL released) until all prior queued events are durably committed.
    #[pyo3(signature = (timeout_ms = 30000))]
    fn flush(&self, py: Python<'_>, timeout_ms: u64) -> PyResult<()> {
        py.allow_threads(|| self.inner.flush(timeout_ms))?;
        Ok(())
    }

    /// Idempotently drain queue, finalize run state, checkpoint WAL, and join writer thread.
    #[pyo3(signature = (state = "finished".to_string(), exit_code = 0, timeout_ms = 30000))]
    fn finish(
        &self,
        py: Python<'_>,
        state: String,
        exit_code: i32,
        timeout_ms: u64,
    ) -> PyResult<()> {
        let st = parse_run_state(&state)?;
        py.allow_threads(|| self.inner.finish(st, exit_code, timeout_ms))?;
        Ok(())
    }

    /// Return current live snapshot JSON string.
    fn snapshot_json(&self) -> PyResult<String> {
        Ok(self.inner.snapshot_json()?)
    }
}

/// Decompress gzip-compressed tensor bytes returned from `RunReader.blob(...)`.
#[pyfunction]
fn decompress_blob<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let raw = py.allow_threads(|| crate::compute::media::decompress_bytes(data))?;
    Ok(PyBytes::new(py, &raw))
}

/// Compute a 32-char blake3 hex digest for a byte slice.
#[pyfunction]
fn blake3_hex(data: &[u8]) -> String {
    crate::compute::media::blake3_short_hex(data)
}

/// Register engine classes and helper functions on `kohakuboard._rs`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLoggerEngine>()?;
    m.add_function(wrap_pyfunction!(decompress_blob, m)?)?;
    m.add_function(wrap_pyfunction!(blake3_hex, m)?)?;
    Ok(())
}
