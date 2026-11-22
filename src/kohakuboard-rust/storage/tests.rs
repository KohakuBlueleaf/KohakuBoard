//! Unit and concurrency tests for `RunStore`, `RunReader`, and `Catalog`.

use std::collections::BTreeMap;

use tempfile::tempdir;

use crate::compute::histogram::compute_histogram;
use crate::compute::media::encode_image_pixels;
use crate::schema::{HistPrecision, RunState, ScalarRow};

use super::{Catalog, RunOpenOptions, RunReader, RunStore};

fn default_opts(run_dir: Option<std::path::PathBuf>, run_id: &str) -> RunOpenOptions {
    RunOpenOptions {
        run_dir,
        run_id: run_id.into(),
        name: format!("run-{run_id}"),
        project: "vision".into(),
        annotation: Some("test".into()),
        group: Some("sweep-a".into()),
        job_type: Some("train".into()),
        tags: vec!["dit".into()],
        notes: "initial note".into(),
        config: serde_json::json!({"lr": 3e-4, "batch_size": 128}),
        env: serde_json::json!({"os": "test"}),
        resume: false,
    }
}

#[test]
fn writer_reader_swmr_and_media_dedup() {
    let dir = tempdir().unwrap();
    let run_dir = dir.path().join("vision").join("ab12_test");
    let mut store = RunStore::open(default_opts(Some(run_dir.clone()), "ab12")).unwrap();

    // Write scalars including NaN and Inf.
    let mut batch = BTreeMap::new();
    batch.insert(
        "train/loss".to_string(),
        vec![
            ScalarRow { step: 0, global_step: 1, ts_ms: 1000, value: 2.0 },
            ScalarRow { step: 1, global_step: 2, ts_ms: 2000, value: f64::NAN },
            ScalarRow { step: 2, global_step: 3, ts_ms: 3000, value: 0.8 },
        ],
    );
    store.append_scalar_batches(&batch).unwrap();

    // Log the EXACT same image at two different steps: blob deduped once, 2 media events recorded!
    let img = encode_image_pixels(&[42u8; 4 * 4 * 3], 4, 4, 3, "png", 90).unwrap();
    assert!(store.put_blob_dedup(&img.blob_key, &img.bytes).unwrap());
    assert!(!store.put_blob_dedup(&img.blob_key, &img.bytes).unwrap());
    store
        .insert_media_event(
            0,
            1,
            1000,
            "samples",
            0,
            "image",
            &img.blob_key,
            &img.mime,
            img.bytes.len() as i64,
            img.width,
            img.height,
            None,
            "step 1",
            "{}",
        )
        .unwrap();
    store
        .insert_media_event(
            2,
            3,
            3000,
            "samples",
            0,
            "image",
            &img.blob_key,
            &img.mime,
            img.bytes.len() as i64,
            img.width,
            img.height,
            None,
            "step 3",
            "{}",
        )
        .unwrap();

    // Histogram + logs.
    let hist =
        compute_histogram(&[0.0, 0.2, 0.5, 0.9], 8, HistPrecision::Exact, None, (0.0, 100.0))
            .unwrap()
            .unwrap();
    store
        .append_histogram(
            "grad/w",
            2,
            3,
            3000,
            hist.min_val,
            hist.max_val,
            hist.bins,
            hist.precision,
            &hist.counts_bytes,
        )
        .unwrap();
    store
        .insert_log_lines(&[
            (1000, 0, "stdout".into(), "training started".into()),
            (2000, 1, "stderr".into(), "NaN loss detected at step 1".into()),
        ])
        .unwrap();
    store.flush_metadata().unwrap();

    // Open SWMR read-only reader while writer is still open!
    let reader = RunReader::open(&run_dir).unwrap();
    let (total, cols) = reader.scalar_slice("train/loss", 0, None).unwrap();
    assert_eq!(total, 3);
    assert_eq!(cols.values[0], 2.0);
    assert!(cols.values[1].is_nan());
    assert_eq!(cols.values[2], 0.8);

    let events = reader.media_events("samples", None, 10).unwrap();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].caption, "step 3");
    assert_eq!(events[1].caption, "step 1");
    assert_eq!(reader.blob(&img.blob_key).unwrap().unwrap(), img.bytes);

    let (htotal, hrecs) = reader.histogram_slice("grad/w", 0, None).unwrap();
    assert_eq!(htotal, 1);
    assert_eq!(hrecs[0].counts.iter().sum::<i32>(), 4);

    let hits = reader.logs_search("NaN", 5).unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0].1.text.contains("NaN"));

    // Catalog index check.
    let cat = Catalog::open(dir.path()).unwrap();
    assert_eq!(cat.refresh(false).unwrap(), 1);
    let projects = cat.list_projects().unwrap();
    assert_eq!(projects.len(), 1);
    assert_eq!(projects[0].name, "vision");
    let runs = cat
        .list_runs(Some("vision"), None, None, None, None, true, None, 0)
        .unwrap();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].run_id, "ab12");
    assert_eq!(runs[0].state, "running");

    reader.close();
    store.finalize(RunState::Finished, 0).unwrap();
    store.close();
}

#[test]
fn memory_mode_store_works_without_disk() {
    let mut store = RunStore::open(default_opts(None, "mem1")).unwrap();
    let mut batch = BTreeMap::new();
    batch.insert("loss".into(), vec![ScalarRow { step: 0, global_step: 0, ts_ms: 1, value: 0.5 }]);
    store.append_scalar_batches(&batch).unwrap();
    store.finalize(RunState::Finished, 0).unwrap();
    store.close();
}

#[test]
fn catalog_rechecks_running_runs_even_when_manifest_unchanged() {
    let dir = tempdir().unwrap();
    let run_dir = dir.path().join("proj").join("dead1");
    let store = RunStore::open(default_opts(Some(run_dir.clone()), "dead1")).unwrap();
    store.close();

    // First refresh sees live current PID -> state "running".
    let cat = Catalog::open(dir.path()).unwrap();
    assert_eq!(cat.refresh(false).unwrap(), 1);
    let runs = cat
        .list_runs(Some("proj"), None, None, None, None, true, None, 0)
        .unwrap();
    assert_eq!(runs[0].state, "running");

    // Second refresh with unchanged manifest still re-evaluates "running" entries.
    assert_eq!(cat.refresh(false).unwrap(), 1);

    // Replace PID in run.json with a guaranteed non-existent PID without changing state.
    let manifest_path = run_dir.join("run.json");
    let mut info: crate::schema::RunInfo =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    info.pid = Some(u32::MAX - 1);
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&info).unwrap()).unwrap();

    cat.refresh(false).unwrap();
    let runs_after = cat
        .list_runs(Some("proj"), None, None, None, None, true, None, 0)
        .unwrap();
    assert_eq!(runs_after[0].state, "crashed");
}
