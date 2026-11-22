//! Project-wide `.catalog.db` SQLite index for fast listing, filtering, and sorting of runs.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use kohakuvault::common::open_connection;
use kohakuvault::rusqlite::{params, Connection};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::error::BoardResult;
use crate::schema::{RunInfo, RunState};

use super::reader_db::RunReader;
use super::writer_db::hostname_str;

/// Summary row for one project in the catalog.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CatalogProject {
    pub name: String,
    pub run_count: i64,
    pub running_count: i64,
    pub latest_updated_ms: i64,
}

/// Summary row for one run in the catalog.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CatalogRun {
    pub project: String,
    pub run_id: String,
    pub dir_name: String,
    pub path: String,
    pub name: String,
    pub annotation: Option<String>,
    pub group: Option<String>,
    pub job_type: Option<String>,
    pub tags: Vec<String>,
    pub notes: String,
    pub state: String,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    pub finished_at_ms: Option<i64>,
    pub exit_code: Option<i32>,
    pub last_step: i64,
    pub last_global_step: i64,
    pub config: JsonValue,
    pub summary: JsonValue,
}

/// Rebuildable SQLite index at `{data_dir}/.catalog.db`.
pub struct Catalog {
    data_dir: PathBuf,
    conn: Mutex<Connection>,
}

impl Catalog {
    /// Open (or create) `{data_dir}/.catalog.db`.
    pub fn open(data_dir: &Path) -> BoardResult<Self> {
        fs::create_dir_all(data_dir)?;
        let db_path = data_dir.join(".catalog.db");
        let conn =
            open_connection(&db_path.to_string_lossy(), true, 16384, 32 * 1024 * 1024, 8192)?;
        init_catalog_schema(&conn)?;
        Ok(Self { data_dir: data_dir.to_path_buf(), conn: Mutex::new(conn) })
    }

    /// Incrementally refresh dirty/new runs and prune deleted ones.
    pub fn refresh(&self, force_rebuild: bool) -> BoardResult<usize> {
        let mut conn = self.conn.lock();
        if force_rebuild {
            conn.execute("DELETE FROM kb_catalog_runs", [])?;
        }

        let mut cached: HashMap<String, (i64, i64, String)> = HashMap::new();
        {
            let mut stmt =
                conn.prepare("SELECT rel_path, mtime_ns, file_size, state FROM kb_catalog_runs")?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    (r.get::<_, i64>(1)?, r.get::<_, i64>(2)?, r.get::<_, String>(3)?),
                ))
            })?;
            for row in rows {
                let (k, v) = row?;
                cached.insert(k, v);
            }
        }

        let mut seen = HashSet::new();
        let mut updated = 0usize;
        let local_host = hostname_str();

        if let Ok(projects) = fs::read_dir(&self.data_dir) {
            for proj_entry in projects.flatten() {
                let proj_path = proj_entry.path();
                if !proj_path.is_dir() {
                    continue;
                }
                let Some(proj_name) = proj_path.file_name().and_then(|s| s.to_str()) else {
                    continue;
                };
                if proj_name.starts_with('.') {
                    continue;
                }
                let Ok(runs) = fs::read_dir(&proj_path) else {
                    continue;
                };
                for run_entry in runs.flatten() {
                    let run_path = run_entry.path();
                    if !run_path.is_dir() {
                        continue;
                    }
                    let manifest = run_path.join("run.json");
                    if !manifest.exists() {
                        continue;
                    }
                    let Some(dir_name) = run_path.file_name().and_then(|s| s.to_str()) else {
                        continue;
                    };
                    let rel_path = format!("{proj_name}/{dir_name}");
                    seen.insert(rel_path.clone());

                    let Ok(meta) = fs::metadata(&manifest) else {
                        continue;
                    };
                    let mtime_ns = meta
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                        .map(|d| d.as_nanos() as i64)
                        .unwrap_or(0);
                    let file_size = meta.len() as i64;

                    if let Some((old_mtime, old_size, old_state)) = cached.get(&rel_path) {
                        if *old_mtime == mtime_ns
                            && *old_size == file_size
                            && old_state != "running"
                        {
                            continue;
                        }
                    }

                    if let Ok(info_bytes) = fs::read(&manifest) {
                        if let Ok(info) = serde_json::from_slice::<RunInfo>(&info_bytes) {
                            let (config_val, summary_val) = match RunReader::open(&run_path) {
                                Ok(reader) => {
                                    let cfg =
                                        reader.config().unwrap_or_else(|_| serde_json::json!({}));
                                    let sum = reader
                                        .summary()
                                        .ok()
                                        .and_then(|s| serde_json::to_value(s).ok())
                                        .unwrap_or_else(|| serde_json::json!({}));
                                    reader.close();
                                    (cfg, sum)
                                }
                                Err(_) => (serde_json::json!({}), serde_json::json!({})),
                            };

                            let effective_state = resolve_effective_state(&info, &local_host)
                                .as_str()
                                .to_string();
                            let tags_json =
                                serde_json::to_string(&info.tags).unwrap_or_else(|_| "[]".into());
                            let config_json =
                                serde_json::to_string(&config_val).unwrap_or_else(|_| "{}".into());
                            let summary_json =
                                serde_json::to_string(&summary_val).unwrap_or_else(|_| "{}".into());

                            conn.execute(
                                "INSERT OR REPLACE INTO kb_catalog_runs (
                                    rel_path, project, run_id, dir_name, abs_path, name,
                                    annotation, group_name, job_type, tags_json, notes, state,
                                    created_at_ms, updated_at_ms, finished_at_ms, exit_code,
                                    last_step, last_global_step, config_json, summary_json,
                                    mtime_ns, file_size
                                 ) VALUES (
                                    ?1, ?2, ?3, ?4, ?5, ?6,
                                    ?7, ?8, ?9, ?10, ?11, ?12,
                                    ?13, ?14, ?15, ?16,
                                    ?17, ?18, ?19, ?20,
                                    ?21, ?22
                                 )",
                                params![
                                    rel_path,
                                    proj_name,
                                    info.run_id,
                                    dir_name,
                                    run_path.to_string_lossy().as_ref(),
                                    info.name,
                                    info.annotation,
                                    info.group,
                                    info.job_type,
                                    tags_json,
                                    info.notes,
                                    effective_state,
                                    info.created_at_ms,
                                    info.updated_at_ms,
                                    info.finished_at_ms,
                                    info.exit_code,
                                    info.last_step,
                                    info.last_global_step,
                                    config_json,
                                    summary_json,
                                    mtime_ns,
                                    file_size
                                ],
                            )?;
                            updated += 1;
                        }
                    }
                }
            }
        }

        // Remove catalog rows whose run folders no longer exist.
        let tx = conn.transaction()?;
        for old_rel in cached.keys() {
            if !seen.contains(old_rel) {
                tx.execute("DELETE FROM kb_catalog_runs WHERE rel_path = ?1", [old_rel])?;
            }
        }
        tx.commit()?;
        Ok(updated)
    }

    /// List all indexed projects with run and active-run counts.
    pub fn list_projects(&self) -> BoardResult<Vec<CatalogProject>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT project,
                    COUNT(*) AS run_count,
                    SUM(CASE WHEN state = 'running' THEN 1 ELSE 0 END) AS running_count,
                    MAX(updated_at_ms) AS latest_updated_ms
             FROM kb_catalog_runs
             GROUP BY project
             ORDER BY latest_updated_ms DESC, project ASC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(CatalogProject {
                name: r.get(0)?,
                run_count: r.get(1)?,
                running_count: r.get(2)?,
                latest_updated_ms: r.get(3)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// List runs in `project` (or across all projects when `None`) with optional filters.
    #[allow(clippy::too_many_arguments)]
    pub fn list_runs(
        &self,
        project: Option<&str>,
        state: Option<&str>,
        group: Option<&str>,
        search: Option<&str>,
        sort_by: Option<&str>,
        descending: bool,
        limit: Option<usize>,
        offset: usize,
    ) -> BoardResult<Vec<CatalogRun>> {
        let conn = self.conn.lock();
        let mut sql = String::from(
            "SELECT project, run_id, dir_name, abs_path, name, annotation, group_name,
                    job_type, tags_json, notes, state, created_at_ms, updated_at_ms,
                    finished_at_ms, exit_code, last_step, last_global_step,
                    config_json, summary_json
             FROM kb_catalog_runs WHERE 1 = 1",
        );
        let mut vals: Vec<kohakuvault::rusqlite::types::Value> = Vec::new();
        if let Some(p) = project {
            sql.push_str(" AND project = ?");
            vals.push(p.to_string().into());
        }
        if let Some(st) = state {
            sql.push_str(" AND state = ?");
            vals.push(st.to_string().into());
        }
        if let Some(g) = group {
            sql.push_str(" AND group_name = ?");
            vals.push(g.to_string().into());
        }
        if let Some(q) = search {
            if !q.trim().is_empty() {
                let pat = format!("%{}%", q.trim());
                sql.push_str(
                    " AND (name LIKE ? OR run_id LIKE ? OR COALESCE(annotation, '') LIKE ?
                           OR tags_json LIKE ? OR notes LIKE ?)",
                );
                for _ in 0..5 {
                    vals.push(pat.clone().into());
                }
            }
        }

        let order_col = match sort_by.unwrap_or("updated_at_ms") {
            "created_at_ms" | "created_at" => "created_at_ms",
            "name" => "name",
            "last_step" | "step" => "last_step",
            "last_global_step" | "global_step" => "last_global_step",
            _ => "updated_at_ms",
        };
        let dir = if descending { "DESC" } else { "ASC" };
        sql.push_str(&format!(" ORDER BY {order_col} {dir}, run_id ASC"));

        if let Some(lim) = limit {
            sql.push_str(" LIMIT ? OFFSET ?");
            vals.push((lim as i64).into());
            vals.push((offset as i64).into());
        }

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(kohakuvault::rusqlite::params_from_iter(vals), |r| {
            let tags_str: String = r.get(8)?;
            let cfg_str: String = r.get(17)?;
            let sum_str: String = r.get(18)?;
            Ok(CatalogRun {
                project: r.get(0)?,
                run_id: r.get(1)?,
                dir_name: r.get(2)?,
                path: r.get(3)?,
                name: r.get(4)?,
                annotation: r.get(5)?,
                group: r.get(6)?,
                job_type: r.get(7)?,
                tags: serde_json::from_str(&tags_str).unwrap_or_default(),
                notes: r.get(9)?,
                state: r.get(10)?,
                created_at_ms: r.get(11)?,
                updated_at_ms: r.get(12)?,
                finished_at_ms: r.get(13)?,
                exit_code: r.get(14)?,
                last_step: r.get(15)?,
                last_global_step: r.get(16)?,
                config: serde_json::from_str(&cfg_str).unwrap_or_else(|_| serde_json::json!({})),
                summary: serde_json::from_str(&sum_str).unwrap_or_else(|_| serde_json::json!({})),
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Resolve a `(project, run_id)` to its on-disk directory path.
    pub fn resolve_run_path(&self, project: &str, run_id: &str) -> BoardResult<Option<PathBuf>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT abs_path FROM kb_catalog_runs
             WHERE project = ?1 AND (run_id = ?2 OR dir_name = ?2) LIMIT 1",
        )?;
        let mut rows = stmt.query(params![project, run_id])?;
        if let Some(row) = rows.next()? {
            let p: String = row.get(0)?;
            return Ok(Some(PathBuf::from(p)));
        }
        Ok(None)
    }
}

fn init_catalog_schema(conn: &Connection) -> BoardResult<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS kb_catalog_runs (
            rel_path         TEXT PRIMARY KEY,
            project          TEXT NOT NULL,
            run_id           TEXT NOT NULL,
            dir_name         TEXT NOT NULL,
            abs_path         TEXT NOT NULL,
            name             TEXT NOT NULL,
            annotation       TEXT,
            group_name       TEXT,
            job_type         TEXT,
            tags_json        TEXT NOT NULL,
            notes            TEXT NOT NULL,
            state            TEXT NOT NULL,
            created_at_ms    INTEGER NOT NULL,
            updated_at_ms    INTEGER NOT NULL,
            finished_at_ms   INTEGER,
            exit_code        INTEGER,
            last_step        INTEGER NOT NULL,
            last_global_step INTEGER NOT NULL,
            config_json      TEXT NOT NULL,
            summary_json     TEXT NOT NULL,
            mtime_ns         INTEGER NOT NULL,
            file_size        INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cat_project_updated
            ON kb_catalog_runs(project, updated_at_ms DESC);
        CREATE INDEX IF NOT EXISTS idx_cat_project_runid
            ON kb_catalog_runs(project, run_id);",
    )?;
    Ok(())
}

fn resolve_effective_state(info: &RunInfo, local_host: &str) -> RunState {
    if info.state != RunState::Running {
        return info.state;
    }
    if info.hostname == local_host {
        if let Some(pid) = info.pid {
            if !is_pid_alive(pid) {
                return RunState::Crashed;
            }
        }
    }
    RunState::Running
}

fn is_pid_alive(pid: u32) -> bool {
    if pid == std::process::id() {
        return true;
    }
    let mut sys = sysinfo::System::new();
    let target = sysinfo::Pid::from_u32(pid);
    sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[target]), true);
    sys.process(target).is_some()
}
