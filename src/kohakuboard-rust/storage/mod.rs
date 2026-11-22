//! KohakuVault 1.2.0 storage layer: single-file `run.db` + lazy `blobs.db` + `.catalog.db`.

pub mod catalog;
pub mod reader_db;
#[cfg(test)]
mod tests;
pub mod writer_db;

use serde::{Deserialize, Serialize};

pub use catalog::{Catalog, CatalogProject, CatalogRun};
pub use reader_db::RunReader;
pub use writer_db::{RunOpenOptions, RunStore};

/// One recorded media event in `kb_media_events` (decoupled from `blobs.db` deduplication).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MediaEvent {
    pub id: i64,
    pub step: i64,
    pub global_step: i64,
    pub ts_ms: i64,
    pub name: String,
    pub idx: i32,
    pub media_type: String,
    pub blob_key: String,
    pub mime: String,
    pub size_bytes: i64,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub duration_ms: Option<i64>,
    pub caption: String,
    pub meta_json: String,
}

/// One logged table snapshot in `kb_tables`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TableRecord {
    pub id: i64,
    pub step: i64,
    pub global_step: i64,
    pub ts_ms: i64,
    pub name: String,
    pub columns_json: String,
    pub row_count: i64,
    pub rows_json: String,
}

/// One logged tensor metadata record in `kb_tensors`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorRecord {
    pub id: i64,
    pub step: i64,
    pub global_step: i64,
    pub ts_ms: i64,
    pub name: String,
    pub blob_key: String,
    pub dtype: String,
    pub shape_json: String,
    pub size_bytes: i64,
    pub compressed: bool,
    pub meta_json: String,
}

/// One logged artifact reference in `kb_artifacts`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactRecord {
    pub id: i64,
    pub step: i64,
    pub global_step: i64,
    pub ts_ms: i64,
    pub name: String,
    pub kind: String,
    pub uri: String,
    pub digest: Option<String>,
    pub size_bytes: Option<i64>,
    pub meta_json: String,
}

/// One captured console log line stored in `TextVault("kb_logs")`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogEntry {
    pub id: i64,
    pub ts_ms: i64,
    pub step: i64,
    pub stream: String,
    pub text: String,
}
