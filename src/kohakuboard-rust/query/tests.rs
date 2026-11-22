use std::collections::BTreeMap;

use tempfile::TempDir;

use crate::schema::{RunState, ScalarRow};
use crate::storage::{RunOpenOptions, RunReader, RunStore};

use super::chunks::{
    decode_chunk_frame, encode_scalar_page_frame, encode_scalar_range_frame,
    plan_newest_first_ranges, ChunkXType,
};

#[test]
fn newest_first_planner_covers_exact_range_with_ramp() {
    let ranges = plan_newest_first_ranges(200_000, 8_192, 65_536);
    assert_eq!(ranges[0], (200_000 - 8_192, 200_000));
    assert_eq!(ranges[1], (200_000 - 8_192 - 16_384, 200_000 - 8_192));
    assert_eq!(ranges[2].1 - ranges[2].0, 32_768);
    assert_eq!(ranges[3].1 - ranges[3].0, 65_536);
    assert_eq!(ranges.last().unwrap().0, 0);

    // Check continuity: no gaps, no overlaps.
    for pair in ranges.windows(2) {
        assert_eq!(pair[0].0, pair[1].1);
    }
    let total: usize = ranges.iter().map(|(s, e)| e - s).sum();
    assert_eq!(total, 200_000);
}

#[test]
fn binary_chunk_frames_are_aligned_and_lossless() {
    let tmp = TempDir::new().unwrap();
    let run_dir = tmp.path().join("r_chunks");
    let mut store = RunStore::open(RunOpenOptions {
        run_dir: Some(run_dir.clone()),
        run_id: "chk1".into(),
        name: "chunk-test".into(),
        project: "p".into(),
        annotation: None,
        group: None,
        job_type: None,
        tags: vec![],
        notes: String::new(),
        config: serde_json::json!({}),
        env: serde_json::json!({}),
        resume: false,
    })
    .unwrap();

    let rows: Vec<ScalarRow> = (0..1_000)
        .map(|i| ScalarRow {
            step: i,
            global_step: i * 10,
            ts_ms: 50_000 + i,
            value: if i == 500 {
                f64::NAN
            } else if i == 501 {
                f64::INFINITY
            } else {
                i as f64
            },
        })
        .collect();
    let mut batch = BTreeMap::new();
    batch.insert("train/loss".into(), rows);
    store.append_scalar_batches(&batch).unwrap();
    store.finalize(RunState::Finished, 0).unwrap();
    store.close();

    let reader = RunReader::open(&run_dir).unwrap();
    let plan = plan_newest_first_ranges(1_000, 128, 512);
    let mut reconstructed = vec![f64::NEG_INFINITY; 1_000];

    for (start, end) in plan {
        let frame =
            encode_scalar_range_frame(&reader, "chk1", "train/loss", start, end, ChunkXType::F64)
                .unwrap();
        let (hdr, payload) = decode_chunk_frame(&frame).unwrap();
        assert_eq!(hdr.start, start);
        assert_eq!(hdr.end, end);
        assert_eq!(hdr.total_rows, 1_000);
        assert_eq!(hdr.done, start == 0);
        let n = end - start;
        assert_eq!(payload.len(), n * 4 * 8);

        // Extract `step` (column 0) and `value` (column 3).
        let vals_offset = n * 3 * 8;
        for i in 0..n {
            let s_bytes = &payload[i * 8..(i + 1) * 8];
            let v_bytes = &payload[vals_offset + i * 8..vals_offset + (i + 1) * 8];
            let step = f64::from_le_bytes(s_bytes.try_into().unwrap()) as usize;
            let val = f64::from_le_bytes(v_bytes.try_into().unwrap());
            assert_eq!(step, start + i);
            reconstructed[start + i] = val;
        }
    }

    assert!(reconstructed[500].is_nan());
    assert!(reconstructed[501].is_infinite() && reconstructed[501].is_sign_positive());
    assert_eq!(reconstructed[0], 0.0);
    assert_eq!(reconstructed[999], 999.0);

    // Forward tail query (`after=990`) returns the final 10 rows.
    let tail = encode_scalar_page_frame(
        &reader,
        "chk1",
        "train/loss",
        None,
        Some(990),
        50,
        ChunkXType::I64,
    )
    .unwrap();
    let (hdr, payload) = decode_chunk_frame(&tail).unwrap();
    assert_eq!(hdr.start, 990);
    assert_eq!(hdr.end, 1_000);
    assert_eq!(
        hdr.dtypes,
        [
            "i64".to_string(),
            "i64".to_string(),
            "i64".to_string(),
            "f64".to_string()
        ]
    );
    assert_eq!(payload.len(), 10 * 4 * 8);
    reader.close();
}
