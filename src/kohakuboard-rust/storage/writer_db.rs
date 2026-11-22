//! Single-writer `RunStore` over KohakuVault 1.2.0 (`run.db` + lazy `blobs.db` + `run.json`).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use kohakuvault::rusqlite::{params, Connection, TransactionBehavior};
use kohakuvault::{common::open_connection, ColumnVault, KVault, KVaultOptions, TextVault};
use serde_json::Value as JsonValue;

use crate::error::{BoardError, BoardResult};
use crate::schema::codec::{pack_histogram_record, pack_kde_record, pack_scalar_rows};
use crate::schema::{
    HistCatalogEntry, HistHeader, HistPrecision, KdeCatalogEntry, KdeHeader, MetricDefinition,
    MetricSummary, RunCatalogMeta, RunInfo, RunState, ScalarCatalogEntry, ScalarRow,
    DIST_BLOCK_BYTES, SCALAR_BLOCK_BYTES, SCHEMA_VERSION,
};

const INLINE_TABLE_MAX_BYTES: usize = 64 * 1024;

/// Configuration for creating or resuming a [`RunStore`].
#[derive(Clone, Debug)]
pub struct RunOpenOptions {
    pub run_dir: Option<PathBuf>,
    pub run_id: String,
    pub name: String,
    pub project: String,
    pub annotation: Option<String>,
    pub group: Option<String>,
    pub job_type: Option<String>,
    pub tags: Vec<String>,
    pub notes: String,
    pub config: JsonValue,
    pub env: JsonValue,
    pub resume: bool,
}

/// Single-writer handle owning `run.db`, lazy `blobs.db`, and atomic `run.json`.
pub struct RunStore {
    run_dir: Option<PathBuf>,
    meta_kv: KVault,
    columns: ColumnVault,
    logs: TextVault,
    events_conn: Connection,
    blobs_kv: Option<KVault>,
    pub info: RunInfo,
    pub config: JsonValue,
    pub env: JsonValue,
    pub summary: BTreeMap<String, MetricSummary>,
    pub catalog: RunCatalogMeta,
    pub metric_defs: Vec<MetricDefinition>,
    meta_dirty: bool,
}

impl RunStore {
    /// Open or create a run store on disk (`run_dir = Some(...)`) or in memory (`None`).
    pub fn open(opts: RunOpenOptions) -> BoardResult<Self> {
        let now = now_ms();
        let (run_db_str, existing) = match &opts.run_dir {
            Some(dir) => {
                fs::create_dir_all(dir)?;
                let db_path = dir.join("run.db");
                let exists = db_path.exists();
                (db_path.to_string_lossy().into_owned(), exists)
            }
            None => (":memory:".to_string(), false),
        };

        let meta_kv =
            KVault::open_with(&run_db_str, KVaultOptions::default().with_table("kb_meta"))?;
        let columns = ColumnVault::open(&run_db_str, false)?;
        let logs = TextVault::open(&run_db_str, "kb_logs", Some(vec!["content".into()]), false)?;
        let events_conn = open_connection(&run_db_str, true, 32768, 64 * 1024 * 1024, 16_384)?;
        init_event_tables(&events_conn)?;

        let (info, config, env, summary, catalog, metric_defs) = if existing && opts.resume {
            let mut info: RunInfo = meta_kv
                .get_msgpack(b"run/info")?
                .ok_or_else(|| BoardError::Storage("missing run/info in resumed run.db".into()))?;
            info.state = RunState::Running;
            info.updated_at_ms = now;
            info.finished_at_ms = None;
            info.exit_code = None;
            info.pid = Some(std::process::id());
            let mut config: JsonValue = meta_kv
                .get_msgpack(b"run/config")?
                .unwrap_or_else(|| serde_json::json!({}));
            merge_json(&mut config, &opts.config);
            let env: JsonValue = meta_kv.get_msgpack(b"run/env")?.unwrap_or(opts.env);
            let summary: BTreeMap<String, MetricSummary> =
                meta_kv.get_msgpack(b"run/summary")?.unwrap_or_default();
            let catalog: RunCatalogMeta = meta_kv.get_msgpack(b"catalog/meta")?.unwrap_or_default();
            let metric_defs: Vec<MetricDefinition> =
                meta_kv.get_msgpack(b"run/metric_defs")?.unwrap_or_default();
            (info, config, env, summary, catalog, metric_defs)
        } else {
            let info = RunInfo {
                schema_version: SCHEMA_VERSION,
                run_id: opts.run_id,
                name: opts.name,
                project: opts.project,
                annotation: opts.annotation,
                group: opts.group,
                job_type: opts.job_type,
                tags: opts.tags,
                notes: opts.notes,
                state: RunState::Running,
                created_at_ms: now,
                updated_at_ms: now,
                finished_at_ms: None,
                exit_code: None,
                pid: Some(std::process::id()),
                hostname: hostname_str(),
                last_step: -1,
                last_global_step: 0,
            };
            (
                info,
                opts.config,
                opts.env,
                BTreeMap::new(),
                RunCatalogMeta::default(),
                Vec::new(),
            )
        };

        let mut store = Self {
            run_dir: opts.run_dir,
            meta_kv,
            columns,
            logs,
            events_conn,
            blobs_kv: None,
            info,
            config,
            env,
            summary,
            catalog,
            metric_defs,
            meta_dirty: true,
        };
        store.flush_metadata()?;
        Ok(store)
    }

    fn ensure_blobs_kv(&mut self) -> BoardResult<&KVault> {
        if self.blobs_kv.is_none() {
            let path_str = match &self.run_dir {
                Some(dir) => dir.join("blobs.db").to_string_lossy().into_owned(),
                None => ":memory:".to_string(),
            };
            let kv = KVault::open_with(&path_str, KVaultOptions::default().with_table("blobs"))?;
            self.blobs_kv = Some(kv);
        }
        Ok(self.blobs_kv.as_ref().unwrap())
    }

    /// Store a content-addressed blob in `blobs.db` (skips write if already present).
    pub fn put_blob_dedup(&mut self, blob_key: &str, bytes: &[u8]) -> BoardResult<bool> {
        let kv = self.ensure_blobs_kv()?;
        if kv.exists(blob_key.as_bytes())? {
            return Ok(false);
        }
        kv.put(blob_key.as_bytes(), bytes)?;
        Ok(true)
    }

    /// Append a batch of `ScalarRow`s grouped by metric name.
    pub fn append_scalar_batches(
        &mut self,
        batches: &BTreeMap<String, Vec<ScalarRow>>,
    ) -> BoardResult<()> {
        if batches.is_empty() {
            return Ok(());
        }
        for (metric, rows) in batches {
            if rows.is_empty() {
                continue;
            }
            let col_id = self.ensure_scalar_column(metric)?;
            let packed = pack_scalar_rows(rows);
            self.columns.append_fixed(col_id, &packed)?;

            let entry = self.catalog.scalars.get_mut(metric).unwrap();
            let first = rows.first().unwrap();
            let last = rows.last().unwrap();
            if entry.count == 0 {
                entry.first_step = first.step;
                entry.first_global_step = first.global_step;
            }
            entry.count += rows.len() as u64;
            entry.last_step = last.step;
            entry.last_global_step = last.global_step;

            for r in rows {
                if !r.value.is_finite() {
                    entry.has_non_finite = true;
                }
                if r.step > self.info.last_step {
                    self.info.last_step = r.step;
                }
                if r.global_step > self.info.last_global_step {
                    self.info.last_global_step = r.global_step;
                }
                self.info.updated_at_ms = self.info.updated_at_ms.max(r.ts_ms);
                match self.summary.get_mut(metric) {
                    Some(sum) => sum.update(r.step, r.global_step, r.ts_ms, r.value),
                    None => {
                        self.summary.insert(
                            metric.clone(),
                            MetricSummary::new(r.step, r.global_step, r.ts_ms, r.value),
                        );
                    }
                }
            }
            self.meta_dirty = true;
        }
        Ok(())
    }

    fn ensure_scalar_column(&mut self, metric: &str) -> BoardResult<i64> {
        if let Some(entry) = self.catalog.scalars.get(metric) {
            return Ok(entry.col_id);
        }
        let seq = self.catalog.next_col_seq;
        self.catalog.next_col_seq += 1;
        let col_name = format!("s_{seq}");
        let col_id =
            self.columns
                .ensure(&col_name, "bytes:32", ScalarRow::BYTE_SIZE, SCALAR_BLOCK_BYTES)?;
        self.catalog.scalars.insert(
            metric.to_string(),
            ScalarCatalogEntry {
                col_name,
                col_id,
                count: 0,
                first_step: 0,
                last_step: 0,
                first_global_step: 0,
                last_global_step: 0,
                has_non_finite: false,
            },
        );
        self.meta_dirty = true;
        Ok(col_id)
    }

    /// Append one pre-computed histogram record into its dedicated fixed-width ColumnVault column.
    #[allow(clippy::too_many_arguments)]
    pub fn append_histogram(
        &mut self,
        name: &str,
        step: i64,
        global_step: i64,
        ts_ms: i64,
        min_val: f64,
        max_val: f64,
        bins: usize,
        precision: HistPrecision,
        counts_bytes: &[u8],
    ) -> BoardResult<()> {
        let rec_size = HistHeader::BYTE_SIZE + bins * precision.elem_bytes();
        let col_id = if let Some(existing) = self.catalog.histograms.get(name) {
            if existing.bins as usize != bins || existing.precision != precision {
                return Err(BoardError::Invalid(format!(
                    "histogram '{name}' already registered with bins={} precision={:?}",
                    existing.bins, existing.precision
                )));
            }
            existing.col_id
        } else {
            let seq = self.catalog.next_col_seq;
            self.catalog.next_col_seq += 1;
            let col_name = format!("h_{seq}");
            let dtype = format!("bytes:{rec_size}");
            let id = self
                .columns
                .ensure(&col_name, &dtype, rec_size, DIST_BLOCK_BYTES)?;
            self.catalog.histograms.insert(
                name.to_string(),
                HistCatalogEntry { col_name, col_id: id, bins: bins as u32, precision, count: 0 },
            );
            id
        };

        let header = HistHeader { step, global_step, ts_ms, min_val, max_val };
        let rec = pack_histogram_record(&header, precision, counts_bytes);
        self.columns.append_fixed(col_id, &rec)?;
        if let Some(entry) = self.catalog.histograms.get_mut(name) {
            entry.count += 1;
        }
        self.touch_step(step, global_step, ts_ms);
        Ok(())
    }

    /// Append one pre-computed KDE record into its dedicated fixed-width ColumnVault column.
    #[allow(clippy::too_many_arguments)]
    pub fn append_kde(
        &mut self,
        name: &str,
        step: i64,
        global_step: i64,
        ts_ms: i64,
        range_min: f64,
        range_max: f64,
        bandwidth: f64,
        sample_count: u64,
        grid: &[f32],
        density: &[f32],
    ) -> BoardResult<()> {
        let num_points = grid.len();
        let rec_size = KdeHeader::BYTE_SIZE + num_points * 8;
        let col_id = if let Some(existing) = self.catalog.kdes.get(name) {
            if existing.num_points as usize != num_points {
                return Err(BoardError::Invalid(format!(
                    "KDE '{name}' already registered with num_points={}",
                    existing.num_points
                )));
            }
            existing.col_id
        } else {
            let seq = self.catalog.next_col_seq;
            self.catalog.next_col_seq += 1;
            let col_name = format!("k_{seq}");
            let dtype = format!("bytes:{rec_size}");
            let id = self
                .columns
                .ensure(&col_name, &dtype, rec_size, DIST_BLOCK_BYTES)?;
            self.catalog.kdes.insert(
                name.to_string(),
                KdeCatalogEntry { col_name, col_id: id, num_points: num_points as u32, count: 0 },
            );
            id
        };

        let header =
            KdeHeader { step, global_step, ts_ms, range_min, range_max, bandwidth, sample_count };
        let rec = pack_kde_record(&header, grid, density)?;
        self.columns.append_fixed(col_id, &rec)?;
        if let Some(entry) = self.catalog.kdes.get_mut(name) {
            entry.count += 1;
        }
        self.touch_step(step, global_step, ts_ms);
        Ok(())
    }

    /// Insert a media event row (blob must already be written to `blobs.db`).
    #[allow(clippy::too_many_arguments)]
    pub fn insert_media_event(
        &mut self,
        step: i64,
        global_step: i64,
        ts_ms: i64,
        name: &str,
        idx: i32,
        media_type: &str,
        blob_key: &str,
        mime: &str,
        size_bytes: i64,
        width: Option<u32>,
        height: Option<u32>,
        duration_ms: Option<i64>,
        caption: &str,
        meta_json: &str,
    ) -> BoardResult<i64> {
        self.events_conn.execute(
            "INSERT INTO kb_media_events
             (step, global_step, ts_ms, name, idx, media_type, blob_key, mime,
              size_bytes, width, height, duration_ms, caption, meta_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            params![
                step,
                global_step,
                ts_ms,
                name,
                idx,
                media_type,
                blob_key,
                mime,
                size_bytes,
                width.map(i64::from),
                height.map(i64::from),
                duration_ms,
                caption,
                meta_json
            ],
        )?;
        let id = self.events_conn.last_insert_rowid();
        self.touch_step(step, global_step, ts_ms);
        Ok(id)
    }

    /// Insert a logged table snapshot (spills large row payloads into `blobs.db`).
    #[allow(clippy::too_many_arguments)]
    pub fn insert_table(
        &mut self,
        step: i64,
        global_step: i64,
        ts_ms: i64,
        name: &str,
        columns_json: &str,
        row_count: i64,
        rows_json: &str,
    ) -> BoardResult<i64> {
        let (rows_blob_key, inline_json): (Option<String>, Option<&str>) =
            if rows_json.len() > INLINE_TABLE_MAX_BYTES {
                let hash = crate::compute::media::blake3_short_hex(rows_json.as_bytes());
                let key = format!("{hash}.table.json");
                self.put_blob_dedup(&key, rows_json.as_bytes())?;
                (Some(key), None)
            } else {
                (None, Some(rows_json))
            };
        self.events_conn.execute(
            "INSERT INTO kb_tables
             (step, global_step, ts_ms, name, columns_json, row_count, rows_blob_key, inline_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                step,
                global_step,
                ts_ms,
                name,
                columns_json,
                row_count,
                rows_blob_key,
                inline_json
            ],
        )?;
        let id = self.events_conn.last_insert_rowid();
        self.touch_step(step, global_step, ts_ms);
        Ok(id)
    }

    /// Insert a tensor metadata row (payload already stored in `blobs.db`).
    #[allow(clippy::too_many_arguments)]
    pub fn insert_tensor(
        &mut self,
        step: i64,
        global_step: i64,
        ts_ms: i64,
        name: &str,
        blob_key: &str,
        dtype: &str,
        shape_json: &str,
        size_bytes: i64,
        compressed: bool,
        meta_json: &str,
    ) -> BoardResult<i64> {
        self.events_conn.execute(
            "INSERT INTO kb_tensors
             (step, global_step, ts_ms, name, blob_key, dtype, shape_json, size_bytes, compressed, meta_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                step,
                global_step,
                ts_ms,
                name,
                blob_key,
                dtype,
                shape_json,
                size_bytes,
                i64::from(compressed),
                meta_json
            ],
        )?;
        let id = self.events_conn.last_insert_rowid();
        self.touch_step(step, global_step, ts_ms);
        Ok(id)
    }

    /// Insert an artifact reference row.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_artifact(
        &mut self,
        step: i64,
        global_step: i64,
        ts_ms: i64,
        name: &str,
        kind: &str,
        uri: &str,
        digest: Option<&str>,
        size_bytes: Option<i64>,
        meta_json: &str,
    ) -> BoardResult<i64> {
        self.events_conn.execute(
            "INSERT INTO kb_artifacts
             (step, global_step, ts_ms, name, kind, uri, digest, size_bytes, meta_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                step,
                global_step,
                ts_ms,
                name,
                kind,
                uri,
                digest,
                size_bytes,
                meta_json
            ],
        )?;
        let id = self.events_conn.last_insert_rowid();
        self.touch_step(step, global_step, ts_ms);
        Ok(id)
    }

    /// Insert a run alert / milestone event.
    pub fn insert_alert(
        &mut self,
        step: i64,
        global_step: i64,
        ts_ms: i64,
        level: &str,
        title: &str,
        text: &str,
    ) -> BoardResult<i64> {
        self.events_conn.execute(
            "INSERT INTO kb_alerts (step, global_step, ts_ms, level, title, text)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![step, global_step, ts_ms, level, title, text],
        )?;
        let id = self.events_conn.last_insert_rowid();
        self.touch_step(step, global_step, ts_ms);
        Ok(id)
    }

    /// Batch-insert captured console log lines into `TextVault("kb_logs")`.
    pub fn insert_log_lines(&mut self, lines: &[(i64, i64, String, String)]) -> BoardResult<()> {
        if lines.is_empty() {
            return Ok(());
        }
        let items: Vec<(Vec<String>, Vec<u8>)> = lines
            .iter()
            .map(|(ts_ms, step, stream, text)| {
                let meta = serde_json::json!({
                    "ts_ms": ts_ms,
                    "step": step,
                    "stream": stream,
                });
                (vec![text.clone()], serde_json::to_vec(&meta).unwrap_or_default())
            })
            .collect();
        self.logs.insert_many(&items)?;
        if let Some((last_ts, _, _, _)) = lines.last() {
            self.info.updated_at_ms = self.info.updated_at_ms.max(*last_ts);
            self.meta_dirty = true;
        }
        Ok(())
    }

    /// Merge a JSON config patch into `run/config`.
    pub fn patch_config(&mut self, patch: &JsonValue) {
        merge_json(&mut self.config, patch);
        self.meta_dirty = true;
    }

    /// Update explicit scalar summary overrides.
    pub fn patch_summary(
        &mut self,
        patch: &BTreeMap<String, f64>,
        step: i64,
        global_step: i64,
        ts_ms: i64,
    ) {
        for (k, &v) in patch {
            match self.summary.get_mut(k) {
                Some(sum) => sum.update(step, global_step, ts_ms, v),
                None => {
                    self.summary
                        .insert(k.clone(), MetricSummary::new(step, global_step, ts_ms, v));
                }
            }
        }
        self.meta_dirty = true;
    }

    /// Update mutable run metadata (tags, notes, group, name).
    pub fn patch_meta(
        &mut self,
        name: Option<String>,
        group: Option<String>,
        add_tags: Vec<String>,
        notes: Option<String>,
    ) {
        if let Some(n) = name {
            self.info.name = n;
        }
        if let Some(g) = group {
            self.info.group = Some(g);
        }
        for t in add_tags {
            if !self.info.tags.contains(&t) {
                self.info.tags.push(t);
            }
        }
        if let Some(n) = notes {
            self.info.notes = n;
        }
        self.info.updated_at_ms = now_ms();
        self.meta_dirty = true;
    }

    /// Register or replace a metric definition pattern.
    pub fn define_metric(&mut self, def: MetricDefinition) {
        if let Some(existing) = self
            .metric_defs
            .iter_mut()
            .find(|d| d.pattern == def.pattern)
        {
            *existing = def;
        } else {
            self.metric_defs.push(def);
        }
        self.meta_dirty = true;
    }

    fn touch_step(&mut self, step: i64, global_step: i64, ts_ms: i64) {
        if step > self.info.last_step {
            self.info.last_step = step;
        }
        if global_step > self.info.last_global_step {
            self.info.last_global_step = global_step;
        }
        self.info.updated_at_ms = self.info.updated_at_ms.max(ts_ms);
        self.meta_dirty = true;
    }

    /// Persist dirty metadata to `kb_meta` and atomically refresh `run.json`.
    pub fn flush_metadata(&mut self) -> BoardResult<()> {
        if !self.meta_dirty {
            return Ok(());
        }
        self.meta_kv.put_msgpack(b"run/info", &self.info)?;
        self.meta_kv.put_msgpack(b"run/config", &self.config)?;
        self.meta_kv.put_msgpack(b"run/env", &self.env)?;
        self.meta_kv.put_msgpack(b"run/summary", &self.summary)?;
        self.meta_kv.put_msgpack(b"catalog/meta", &self.catalog)?;
        self.meta_kv
            .put_msgpack(b"run/metric_defs", &self.metric_defs)?;
        self.write_manifest()?;
        self.meta_dirty = false;
        Ok(())
    }

    /// Mark the run finished/interrupted/crashed, flush metadata, checkpoint WAL, and close handles.
    pub fn finalize(&mut self, state: RunState, exit_code: i32) -> BoardResult<()> {
        let now = now_ms();
        self.info.state = state;
        self.info.updated_at_ms = now;
        self.info.finished_at_ms = Some(now);
        self.info.exit_code = Some(exit_code);
        self.meta_dirty = true;
        self.flush_metadata()?;
        let _ = self.columns.checkpoint();
        let _ = self.meta_kv.checkpoint();
        if let Some(blobs) = &self.blobs_kv {
            let _ = blobs.checkpoint();
        }
        Ok(())
    }

    /// Close all underlying SQLite handles cleanly.
    pub fn close(mut self) {
        let _ = self.flush_metadata();
        self.columns.close();
        let _ = self.meta_kv.close();
        self.logs.close();
        if let Some(blobs) = self.blobs_kv.take() {
            let _ = blobs.close();
        }
    }

    fn write_manifest(&self) -> BoardResult<()> {
        let Some(dir) = &self.run_dir else {
            return Ok(());
        };
        let manifest_path = dir.join("run.json");
        let tmp_path = dir.join("run.json.tmp");
        let bytes = serde_json::to_vec_pretty(&self.info)?;
        fs::write(&tmp_path, bytes)?;
        // Windows rename fails if target exists unless overwritten; handle cleanly.
        if let Err(err) = fs::rename(&tmp_path, &manifest_path) {
            if manifest_path.exists() {
                let _ = fs::remove_file(&manifest_path);
                fs::rename(&tmp_path, &manifest_path)?;
            } else {
                return Err(BoardError::Io(err));
            }
        }
        Ok(())
    }
}

fn init_event_tables(conn: &Connection) -> BoardResult<()> {
    let tx = conn.unchecked_transaction()?;
    tx.execute_batch(
        "CREATE TABLE IF NOT EXISTS kb_media_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            step        INTEGER NOT NULL,
            global_step INTEGER NOT NULL,
            ts_ms       INTEGER NOT NULL,
            name        TEXT NOT NULL,
            idx         INTEGER NOT NULL DEFAULT 0,
            media_type  TEXT NOT NULL,
            blob_key    TEXT NOT NULL,
            mime        TEXT NOT NULL,
            size_bytes  INTEGER NOT NULL,
            width       INTEGER,
            height      INTEGER,
            duration_ms INTEGER,
            caption     TEXT NOT NULL DEFAULT '',
            meta_json   TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_kb_media_name_step ON kb_media_events(name, step DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_kb_media_gstep     ON kb_media_events(name, global_step DESC);

        CREATE TABLE IF NOT EXISTS kb_tables (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            step          INTEGER NOT NULL,
            global_step   INTEGER NOT NULL,
            ts_ms         INTEGER NOT NULL,
            name          TEXT NOT NULL,
            columns_json  TEXT NOT NULL,
            row_count     INTEGER NOT NULL,
            rows_blob_key TEXT,
            inline_json   TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_kb_tables_name_step ON kb_tables(name, step DESC, id DESC);

        CREATE TABLE IF NOT EXISTS kb_tensors (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            step        INTEGER NOT NULL,
            global_step INTEGER NOT NULL,
            ts_ms       INTEGER NOT NULL,
            name        TEXT NOT NULL,
            blob_key    TEXT NOT NULL,
            dtype       TEXT NOT NULL,
            shape_json  TEXT NOT NULL,
            size_bytes  INTEGER NOT NULL,
            compressed  INTEGER NOT NULL DEFAULT 0,
            meta_json   TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_kb_tensors_name_step ON kb_tensors(name, step DESC, id DESC);

        CREATE TABLE IF NOT EXISTS kb_artifacts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            step        INTEGER NOT NULL,
            global_step INTEGER NOT NULL,
            ts_ms       INTEGER NOT NULL,
            name        TEXT NOT NULL,
            kind        TEXT NOT NULL,
            uri         TEXT NOT NULL,
            digest      TEXT,
            size_bytes  INTEGER,
            meta_json   TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_kb_artifacts_step ON kb_artifacts(step DESC, id DESC);

        CREATE TABLE IF NOT EXISTS kb_alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            step        INTEGER NOT NULL,
            global_step INTEGER NOT NULL,
            ts_ms       INTEGER NOT NULL,
            level       TEXT NOT NULL,
            title       TEXT NOT NULL,
            text        TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_kb_alerts_step ON kb_alerts(step DESC, id DESC);",
    )?;
    tx.commit()?;
    let _ = TransactionBehavior::Immediate;
    Ok(())
}

fn merge_json(base: &mut JsonValue, patch: &JsonValue) {
    match (base, patch) {
        (JsonValue::Object(b), JsonValue::Object(p)) => {
            for (k, v) in p {
                merge_json(b.entry(k.clone()).or_insert(JsonValue::Null), v);
            }
        }
        (b, p) => {
            *b = p.clone();
        }
    }
}

pub(crate) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

pub(crate) fn hostname_str() -> String {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "localhost".into())
}

/// Ensure a path is safe as a run directory path.
pub fn run_db_path(run_dir: &Path) -> PathBuf {
    run_dir.join("run.db")
}
