//! Read-only SWMR `RunReader` over KohakuVault 1.2.0 (`run.db` + lazy `blobs.db`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use kohakuvault::common::open_readonly_connection;
use kohakuvault::rusqlite::{params, Connection};
use kohakuvault::{ColumnVault, KVault, KVaultOptions, TextVault};
use parking_lot::Mutex;
use serde_json::Value as JsonValue;

use crate::error::{BoardError, BoardResult};
use crate::schema::codec::{
    deinterleave_scalars, unpack_histogram_records, unpack_kde_records, DecodedHistogram,
    DecodedKde, ScalarColumns,
};
use crate::schema::{
    AlertRecord, MetricDefinition, MetricSummary, RunCatalogMeta, RunInfo, ScalarRow,
};

use super::{ArtifactRecord, LogEntry, MediaEvent, TableRecord, TensorRecord};

/// Read-only handle to one KohakuBoard v2 run directory.
///
/// Opens SQLite strictly `read_only = true` with pooled read snapshots so active writers
/// are never blocked and schema/WAL metadata is never mutated.
pub struct RunReader {
    run_dir: PathBuf,
    meta_kv: KVault,
    columns: ColumnVault,
    logs: TextVault,
    events_conn: Mutex<Connection>,
    blobs_kv: Mutex<Option<KVault>>,
}

impl RunReader {
    /// Open an existing run directory read-only.
    pub fn open(run_dir: &Path) -> BoardResult<Self> {
        let db_path = run_dir.join("run.db");
        if !db_path.exists() {
            return Err(BoardError::NotFound(format!("run.db not found in {}", run_dir.display())));
        }
        let db_str = db_path.to_string_lossy().into_owned();
        let meta_kv = KVault::open_with(
            &db_str,
            KVaultOptions::default()
                .with_table("kb_meta")
                .read_only(true),
        )?;
        let columns = ColumnVault::open(&db_str, true)?;
        let logs = TextVault::open(&db_str, "kb_logs", Some(vec!["content".into()]), true)?;
        let events_conn = open_readonly_connection(&db_str, 64 * 1024 * 1024, 8192)?;

        Ok(Self {
            run_dir: run_dir.to_path_buf(),
            meta_kv,
            columns,
            logs,
            events_conn: Mutex::new(events_conn),
            blobs_kv: Mutex::new(None),
        })
    }

    /// Return the run directory path.
    pub fn run_dir(&self) -> &Path {
        &self.run_dir
    }

    pub fn info(&self) -> BoardResult<RunInfo> {
        self.meta_kv
            .get_msgpack(b"run/info")?
            .ok_or_else(|| BoardError::NotFound("run/info missing".into()))
    }

    pub fn config(&self) -> BoardResult<JsonValue> {
        Ok(self
            .meta_kv
            .get_msgpack(b"run/config")?
            .unwrap_or_else(|| serde_json::json!({})))
    }

    pub fn env(&self) -> BoardResult<JsonValue> {
        Ok(self
            .meta_kv
            .get_msgpack(b"run/env")?
            .unwrap_or_else(|| serde_json::json!({})))
    }

    pub fn summary(&self) -> BoardResult<BTreeMap<String, MetricSummary>> {
        Ok(self
            .meta_kv
            .get_msgpack(b"run/summary")?
            .unwrap_or_default())
    }

    pub fn catalog(&self) -> BoardResult<RunCatalogMeta> {
        Ok(self
            .meta_kv
            .get_msgpack(b"catalog/meta")?
            .unwrap_or_default())
    }

    pub fn metric_definitions(&self) -> BoardResult<Vec<MetricDefinition>> {
        Ok(self
            .meta_kv
            .get_msgpack(b"run/metric_defs")?
            .unwrap_or_default())
    }

    /// Committed row count for scalar metric `name`.
    pub fn scalar_len(&self, name: &str) -> BoardResult<usize> {
        let cat = self.catalog()?;
        let Some(entry) = cat.scalars.get(name) else {
            return Ok(0);
        };
        Ok(self.columns.length(entry.col_id)?)
    }

    /// Read raw packed `ScalarRow` bytes for `[start .. start + count)`.
    pub fn scalar_raw_slice(
        &self,
        name: &str,
        start: usize,
        count: Option<usize>,
    ) -> BoardResult<(usize, Vec<u8>)> {
        let cat = self.catalog()?;
        let Some(entry) = cat.scalars.get(name) else {
            return Ok((0, Vec::new()));
        };
        let total = self.columns.length(entry.col_id)?;
        if start >= total {
            return Ok((total, Vec::new()));
        }
        let clamped = count.map(|c| c.min(total - start));
        let bytes = self.columns.read_fixed(entry.col_id, start, clamped)?;
        Ok((total, bytes))
    }

    /// Read and de-interleave scalar slice `[start .. start + count)` into typed columns.
    pub fn scalar_slice(
        &self,
        name: &str,
        start: usize,
        count: Option<usize>,
    ) -> BoardResult<(usize, ScalarColumns)> {
        let (total, bytes) = self.scalar_raw_slice(name, start, count)?;
        if bytes.is_empty() {
            return Ok((
                total,
                ScalarColumns {
                    steps: Vec::new(),
                    global_steps: Vec::new(),
                    ts_ms: Vec::new(),
                    values: Vec::new(),
                },
            ));
        }
        Ok((total, deinterleave_scalars(&bytes)?))
    }

    /// Read full scalar series for `name`.
    pub fn scalar_all(&self, name: &str) -> BoardResult<ScalarColumns> {
        Ok(self.scalar_slice(name, 0, None)?.1)
    }

    /// Read histogram records `[start .. start + count)` (or newest-first when caller computes range).
    pub fn histogram_slice(
        &self,
        name: &str,
        start: usize,
        count: Option<usize>,
    ) -> BoardResult<(usize, Vec<DecodedHistogram>)> {
        let cat = self.catalog()?;
        let Some(entry) = cat.histograms.get(name) else {
            return Ok((0, Vec::new()));
        };
        let total = self.columns.length(entry.col_id)?;
        if start >= total {
            return Ok((total, Vec::new()));
        }
        let clamped = count.map(|c| c.min(total - start));
        let bytes = self.columns.read_fixed(entry.col_id, start, clamped)?;
        let records = unpack_histogram_records(&bytes, entry.bins as usize, entry.precision)?;
        Ok((total, records))
    }

    /// Read KDE records `[start .. start + count)`.
    pub fn kde_slice(
        &self,
        name: &str,
        start: usize,
        count: Option<usize>,
    ) -> BoardResult<(usize, Vec<DecodedKde>)> {
        let cat = self.catalog()?;
        let Some(entry) = cat.kdes.get(name) else {
            return Ok((0, Vec::new()));
        };
        let total = self.columns.length(entry.col_id)?;
        if start >= total {
            return Ok((total, Vec::new()));
        }
        let clamped = count.map(|c| c.min(total - start));
        let bytes = self.columns.read_fixed(entry.col_id, start, clamped)?;
        let records = unpack_kde_records(&bytes, entry.num_points as usize)?;
        Ok((total, records))
    }

    /// List distinct media log names.
    pub fn media_names(&self) -> BoardResult<Vec<String>> {
        let conn = self.events_conn.lock();
        let mut stmt =
            conn.prepare("SELECT DISTINCT name FROM kb_media_events ORDER BY name ASC")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Fetch media events for `name` ordered newest-first (`before_id` cursor).
    pub fn media_events(
        &self,
        name: &str,
        before_id: Option<i64>,
        limit: usize,
    ) -> BoardResult<Vec<MediaEvent>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let conn = self.events_conn.lock();
        let map_row =
            |r: &kohakuvault::rusqlite::Row<'_>| -> kohakuvault::rusqlite::Result<MediaEvent> {
                Ok(MediaEvent {
                    id: r.get(0)?,
                    step: r.get(1)?,
                    global_step: r.get(2)?,
                    ts_ms: r.get(3)?,
                    name: r.get(4)?,
                    idx: r.get(5)?,
                    media_type: r.get(6)?,
                    blob_key: r.get(7)?,
                    mime: r.get(8)?,
                    size_bytes: r.get(9)?,
                    width: r.get::<_, Option<i64>>(10)?.map(|v| v as u32),
                    height: r.get::<_, Option<i64>>(11)?.map(|v| v as u32),
                    duration_ms: r.get(12)?,
                    caption: r.get(13)?,
                    meta_json: r.get(14)?,
                })
            };
        let mut stmt = match before_id {
            Some(_) => conn.prepare(
                "SELECT id, step, global_step, ts_ms, name, idx, media_type, blob_key, mime,
                        size_bytes, width, height, duration_ms, caption, meta_json
                 FROM kb_media_events WHERE name = ?1 AND id < ?2
                 ORDER BY id DESC LIMIT ?3",
            )?,
            None => conn.prepare(
                "SELECT id, step, global_step, ts_ms, name, idx, media_type, blob_key, mime,
                        size_bytes, width, height, duration_ms, caption, meta_json
                 FROM kb_media_events WHERE name = ?1
                 ORDER BY id DESC LIMIT ?2",
            )?,
        };
        let rows = match before_id {
            Some(cursor) => stmt.query_map(params![name, cursor, limit as i64], map_row)?,
            None => stmt.query_map(params![name, limit as i64], map_row)?,
        };
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Read a raw blob by content-addressed key from `blobs.db`.
    pub fn blob(&self, key: &str) -> BoardResult<Option<Vec<u8>>> {
        let mut guard = self.blobs_kv.lock();
        if guard.is_none() {
            let blobs_path = self.run_dir.join("blobs.db");
            if !blobs_path.exists() {
                return Ok(None);
            }
            let kv = KVault::open_with(
                &blobs_path.to_string_lossy(),
                KVaultOptions::default().with_table("blobs").read_only(true),
            )?;
            *guard = Some(kv);
        }
        Ok(guard.as_ref().unwrap().get(key.as_bytes())?)
    }

    /// List distinct table log names.
    pub fn table_names(&self) -> BoardResult<Vec<String>> {
        let conn = self.events_conn.lock();
        let mut stmt = conn.prepare("SELECT DISTINCT name FROM kb_tables ORDER BY name ASC")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Fetch logged table snapshots newest-first (`before_id` cursor).
    pub fn tables(
        &self,
        name: &str,
        before_id: Option<i64>,
        limit: usize,
    ) -> BoardResult<Vec<TableRecord>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        type RawTableRow =
            (i64, i64, i64, i64, String, String, i64, Option<String>, Option<String>);
        let raw_rows: Vec<RawTableRow> = {
            let conn = self.events_conn.lock();
            let mut stmt = match before_id {
                Some(_) => conn.prepare(
                    "SELECT id, step, global_step, ts_ms, name, columns_json, row_count,
                            rows_blob_key, inline_json
                     FROM kb_tables WHERE name = ?1 AND id < ?2
                     ORDER BY id DESC LIMIT ?3",
                )?,
                None => conn.prepare(
                    "SELECT id, step, global_step, ts_ms, name, columns_json, row_count,
                            rows_blob_key, inline_json
                     FROM kb_tables WHERE name = ?1
                     ORDER BY id DESC LIMIT ?2",
                )?,
            };
            let map_fn = |r: &kohakuvault::rusqlite::Row<'_>| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                    r.get(6)?,
                    r.get(7)?,
                    r.get(8)?,
                ))
            };
            let mapped = match before_id {
                Some(cursor) => stmt.query_map(params![name, cursor, limit as i64], map_fn)?,
                None => stmt.query_map(params![name, limit as i64], map_fn)?,
            };
            mapped.collect::<Result<Vec<_>, _>>()?
        };

        let mut out = Vec::with_capacity(raw_rows.len());
        for (id, step, global_step, ts_ms, name, columns_json, row_count, blob_key, inline_json) in
            raw_rows
        {
            let rows_json = match (inline_json, blob_key) {
                (Some(s), _) => s,
                (None, Some(key)) => {
                    let bytes = self.blob(&key)?.ok_or_else(|| {
                        BoardError::NotFound(format!("table blob '{key}' missing"))
                    })?;
                    String::from_utf8(bytes)
                        .map_err(|e| BoardError::Codec(format!("invalid UTF-8 table blob: {e}")))?
                }
                (None, None) => "[]".into(),
            };
            out.push(TableRecord {
                id,
                step,
                global_step,
                ts_ms,
                name,
                columns_json,
                row_count,
                rows_json,
            });
        }
        Ok(out)
    }

    /// List distinct tensor log names.
    pub fn tensor_names(&self) -> BoardResult<Vec<String>> {
        let conn = self.events_conn.lock();
        let mut stmt = conn.prepare("SELECT DISTINCT name FROM kb_tensors ORDER BY name ASC")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Fetch tensor metadata newest-first.
    pub fn tensors(
        &self,
        name: &str,
        before_id: Option<i64>,
        limit: usize,
    ) -> BoardResult<Vec<TensorRecord>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let conn = self.events_conn.lock();
        let mut stmt = match before_id {
            Some(_) => conn.prepare(
                "SELECT id, step, global_step, ts_ms, name, blob_key, dtype, shape_json,
                        size_bytes, compressed, meta_json
                 FROM kb_tensors WHERE name = ?1 AND id < ?2 ORDER BY id DESC LIMIT ?3",
            )?,
            None => conn.prepare(
                "SELECT id, step, global_step, ts_ms, name, blob_key, dtype, shape_json,
                        size_bytes, compressed, meta_json
                 FROM kb_tensors WHERE name = ?1 ORDER BY id DESC LIMIT ?2",
            )?,
        };
        let map_fn = |r: &kohakuvault::rusqlite::Row<'_>| {
            Ok(TensorRecord {
                id: r.get(0)?,
                step: r.get(1)?,
                global_step: r.get(2)?,
                ts_ms: r.get(3)?,
                name: r.get(4)?,
                blob_key: r.get(5)?,
                dtype: r.get(6)?,
                shape_json: r.get(7)?,
                size_bytes: r.get(8)?,
                compressed: r.get::<_, i64>(9)? != 0,
                meta_json: r.get(10)?,
            })
        };
        let rows = match before_id {
            Some(cursor) => stmt.query_map(params![name, cursor, limit as i64], map_fn)?,
            None => stmt.query_map(params![name, limit as i64], map_fn)?,
        };
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Fetch artifact records newest-first.
    pub fn artifacts(
        &self,
        before_id: Option<i64>,
        limit: usize,
    ) -> BoardResult<Vec<ArtifactRecord>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let conn = self.events_conn.lock();
        let mut stmt = match before_id {
            Some(_) => conn.prepare(
                "SELECT id, step, global_step, ts_ms, name, kind, uri, digest, size_bytes, meta_json
                 FROM kb_artifacts WHERE id < ?1 ORDER BY id DESC LIMIT ?2",
            )?,
            None => conn.prepare(
                "SELECT id, step, global_step, ts_ms, name, kind, uri, digest, size_bytes, meta_json
                 FROM kb_artifacts ORDER BY id DESC LIMIT ?1",
            )?,
        };
        let map_fn = |r: &kohakuvault::rusqlite::Row<'_>| {
            Ok(ArtifactRecord {
                id: r.get(0)?,
                step: r.get(1)?,
                global_step: r.get(2)?,
                ts_ms: r.get(3)?,
                name: r.get(4)?,
                kind: r.get(5)?,
                uri: r.get(6)?,
                digest: r.get(7)?,
                size_bytes: r.get(8)?,
                meta_json: r.get(9)?,
            })
        };
        let rows = match before_id {
            Some(cursor) => stmt.query_map(params![cursor, limit as i64], map_fn)?,
            None => stmt.query_map(params![limit as i64], map_fn)?,
        };
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Fetch alerts newest-first.
    pub fn alerts(&self, before_id: Option<i64>, limit: usize) -> BoardResult<Vec<AlertRecord>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let conn = self.events_conn.lock();
        let mut stmt = match before_id {
            Some(_) => conn.prepare(
                "SELECT id, step, global_step, ts_ms, level, title, text
                 FROM kb_alerts WHERE id < ?1 ORDER BY id DESC LIMIT ?2",
            )?,
            None => conn.prepare(
                "SELECT id, step, global_step, ts_ms, level, title, text
                 FROM kb_alerts ORDER BY id DESC LIMIT ?1",
            )?,
        };
        let map_fn = |r: &kohakuvault::rusqlite::Row<'_>| {
            Ok(AlertRecord {
                id: r.get(0)?,
                step: r.get(1)?,
                global_step: r.get(2)?,
                ts_ms: r.get(3)?,
                level: r.get(4)?,
                title: r.get(5)?,
                text: r.get(6)?,
            })
        };
        let rows = match before_id {
            Some(cursor) => stmt.query_map(params![cursor, limit as i64], map_fn)?,
            None => stmt.query_map(params![limit as i64], map_fn)?,
        };
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Fetch captured console log lines newest-first via `TextVault::tail`.
    pub fn logs_tail(&self, before_id: Option<i64>, limit: usize) -> BoardResult<Vec<LogEntry>> {
        let records = self.logs.tail(before_id, limit)?;
        Ok(records
            .into_iter()
            .map(|rec| decode_log_entry(rec.doc_id, rec.texts, &rec.value))
            .collect())
    }

    /// Full-text BM25 search across captured console logs via `TextVault::search`.
    pub fn logs_search(&self, query: &str, k: usize) -> BoardResult<Vec<(f64, LogEntry)>> {
        let hits = self.logs.search(query, k, None, true)?;
        Ok(hits
            .into_iter()
            .map(|h| (h.score, decode_log_entry(h.doc_id, h.texts, &h.value)))
            .collect())
    }

    /// Close all underlying read handles cleanly.
    pub fn close(&self) {
        self.columns.close();
        let _ = self.meta_kv.close();
        self.logs.close();
        if let Some(blobs) = self.blobs_kv.lock().take() {
            let _ = blobs.close();
        }
    }
}

fn decode_log_entry(id: i64, mut texts: Vec<String>, meta_bytes: &[u8]) -> LogEntry {
    let text = if texts.is_empty() {
        String::new()
    } else {
        texts.remove(0)
    };
    let meta: JsonValue = serde_json::from_slice(meta_bytes).unwrap_or(JsonValue::Null);
    LogEntry {
        id,
        ts_ms: meta.get("ts_ms").and_then(JsonValue::as_i64).unwrap_or(0),
        step: meta.get("step").and_then(JsonValue::as_i64).unwrap_or(0),
        stream: meta
            .get("stream")
            .and_then(JsonValue::as_str)
            .unwrap_or("stdout")
            .to_string(),
        text,
    }
}

/// Helper verifying row byte size matches `ScalarRow::BYTE_SIZE`.
#[allow(dead_code)]
pub(crate) const _SCALAR_BYTES: usize = ScalarRow::BYTE_SIZE;
