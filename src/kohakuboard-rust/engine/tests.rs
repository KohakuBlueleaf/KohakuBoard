use tempfile::TempDir;

use crate::schema::{HistPrecision, RunState};
use crate::storage::{RunOpenOptions, RunReader};

use super::{EngineConfig, LogEvent, LoggerEngine, OverflowPolicy};

fn test_config(dir: &std::path::Path, overflow: OverflowPolicy, cap: usize) -> EngineConfig {
    EngineConfig {
        open: RunOpenOptions {
            run_dir: Some(dir.to_path_buf()),
            run_id: "eng1".into(),
            name: "engine-test".into(),
            project: "proj".into(),
            annotation: None,
            group: None,
            job_type: None,
            tags: vec!["v2".into()],
            notes: String::new(),
            config: serde_json::json!({"lr": 1e-3}),
            env: serde_json::json!({}),
            resume: false,
        },
        flush_interval_ms: 100,
        queue_capacity: cap,
        overflow,
        system_metrics_interval_ms: None,
    }
}

#[test]
fn engine_end_to_end_flush_and_finish() {
    let tmp = TempDir::new().unwrap();
    let run_dir = tmp.path().join("run_1");
    let engine = LoggerEngine::start(test_config(&run_dir, OverflowPolicy::Block, 1024)).unwrap();

    for i in 0..500 {
        let val = if i == 100 {
            f64::NAN
        } else {
            (i as f64) * 0.25
        };
        engine
            .send(LogEvent::Scalars {
                step: i,
                global_step: i * 2,
                ts_ms: 10_000 + i,
                metrics: vec![("train/loss".into(), val), ("train/lr".into(), 1e-3)],
            })
            .unwrap();
    }

    engine
        .send(LogEvent::HistogramRaw {
            step: 10,
            global_step: 20,
            ts_ms: 11_000,
            name: "grad/w".into(),
            values: (0..256).map(|i| i as f32).collect(),
            bins: 32,
            precision: HistPrecision::Exact,
            range: None,
            percentiles: (0.0, 100.0),
        })
        .unwrap();

    engine
        .send(LogEvent::KdeRaw {
            step: 10,
            global_step: 20,
            ts_ms: 11_000,
            name: "kde/w".into(),
            values: (0..256).map(|i| (i as f32) * 0.1).collect(),
            num_points: 64,
            bandwidth: None,
            approximate: true,
            range: None,
            percentiles: (0.0, 100.0),
        })
        .unwrap();

    engine
        .send(LogEvent::MediaPixels {
            step: 10,
            global_step: 20,
            ts_ms: 11_000,
            name: "sample".into(),
            idx: 0,
            pixels: vec![128u8; 4 * 4 * 3],
            width: 4,
            height: 4,
            channels: 3,
            format: "png".into(),
            quality: 90,
            caption: "tiny".into(),
            meta_json: "{}".into(),
        })
        .unwrap();

    engine
        .send(LogEvent::LogLines {
            lines: vec![
                (11_000, 10, "stdout".into(), "step 10 loss=2.5".into()),
                (11_001, 11, "stderr".into(), "WARNING NaN encountered".into()),
            ],
        })
        .unwrap();

    engine
        .send(LogEvent::Alert {
            step: 100,
            global_step: 200,
            ts_ms: 12_000,
            level: "warn".into(),
            title: "NaN detected".into(),
            text: "train/loss became NaN".into(),
        })
        .unwrap();

    engine.flush(10_000).unwrap();

    // Verify live SWMR reader sees committed rows while engine is still running.
    let live_reader = RunReader::open(&run_dir).unwrap();
    assert_eq!(live_reader.scalar_len("train/loss").unwrap(), 500);
    assert_eq!(live_reader.info().unwrap().state, RunState::Running);
    live_reader.close();

    let snap_str = engine.snapshot_json().unwrap();
    assert!(snap_str.contains("train/loss"));
    assert!(snap_str.contains("NaN detected"));

    engine.finish(RunState::Finished, 0, 10_000).unwrap();
    assert!(engine.is_closed());
    // Idempotent second finish.
    engine.finish(RunState::Finished, 0, 10_000).unwrap();
    // Sending after close fails cleanly.
    assert!(engine.send(LogEvent::LogLines { lines: vec![] }).is_err());

    let reader = RunReader::open(&run_dir).unwrap();
    assert_eq!(reader.info().unwrap().state, RunState::Finished);
    let cols = reader.scalar_all("train/loss").unwrap();
    assert_eq!(cols.values.len(), 500);
    assert!(cols.values[100].is_nan());
    assert_eq!(reader.histogram_slice("grad/w", 0, None).unwrap().1.len(), 1);
    assert_eq!(reader.kde_slice("kde/w", 0, None).unwrap().1.len(), 1);
    let media = reader.media_events("sample", None, 10).unwrap();
    assert_eq!(media.len(), 1);
    assert!(reader.blob(&media[0].blob_key).unwrap().is_some());
    assert_eq!(reader.alerts(None, 10).unwrap().len(), 1);
    assert_eq!(reader.logs_search("NaN", 10).unwrap().len(), 1);
    reader.close();
}

#[test]
fn engine_drop_oldest_never_blocks_and_finishes_cleanly() {
    let tmp = TempDir::new().unwrap();
    let run_dir = tmp.path().join("run_drop");
    let engine =
        LoggerEngine::start(test_config(&run_dir, OverflowPolicy::DropOldest, 16)).unwrap();

    for i in 0..2000 {
        engine
            .send(LogEvent::Scalars {
                step: i,
                global_step: i,
                ts_ms: i,
                metrics: vec![("m".into(), i as f64)],
            })
            .unwrap();
    }
    engine.finish(RunState::Finished, 0, 10_000).unwrap();

    let reader = RunReader::open(&run_dir).unwrap();
    let len = reader.scalar_len("m").unwrap();
    assert!(len > 0);
    assert_eq!(len as u64 + engine.dropped_events(), 2000);
    reader.close();
}
