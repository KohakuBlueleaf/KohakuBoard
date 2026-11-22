//! Unit tests for binary record codecs and IEEE-754 edge cases.

use super::codec::{
    deinterleave_scalars, pack_histogram_record, pack_kde_record, pack_scalar_rows,
    unpack_histogram_records, unpack_kde_records, unpack_scalar_rows,
};
use super::{HistHeader, HistPrecision, KdeHeader, MetricSummary, ScalarRow};

#[test]
fn scalar_roundtrip_preserves_nan_and_inf() {
    let rows = vec![
        ScalarRow { step: 0, global_step: 1, ts_ms: 1000, value: 1.5 },
        ScalarRow { step: 1, global_step: 2, ts_ms: 2000, value: f64::NAN },
        ScalarRow { step: 2, global_step: 3, ts_ms: 3000, value: f64::INFINITY },
        ScalarRow { step: 3, global_step: 4, ts_ms: 4000, value: f64::NEG_INFINITY },
    ];
    let packed = pack_scalar_rows(&rows);
    assert_eq!(packed.len(), 4 * 32);

    let unpacked = unpack_scalar_rows(&packed).unwrap();
    assert_eq!(unpacked[0].value, 1.5);
    assert!(unpacked[1].value.is_nan());
    assert!(unpacked[2].value.is_infinite() && unpacked[2].value.is_sign_positive());
    assert!(unpacked[3].value.is_infinite() && unpacked[3].value.is_sign_negative());

    let cols = deinterleave_scalars(&packed).unwrap();
    assert_eq!(cols.steps, vec![0, 1, 2, 3]);
    assert_eq!(cols.global_steps, vec![1, 2, 3, 4]);
    assert!(cols.values[1].is_nan());
}

#[test]
fn metric_summary_tracks_non_finite_without_poisoning_min_max() {
    let mut sum = MetricSummary::new(0, 1, 100, f64::NAN);
    assert_eq!(sum.nan_count, 1);
    sum.update(1, 2, 200, 0.5);
    sum.update(2, 3, 300, 2.5);
    sum.update(3, 4, 400, f64::INFINITY);
    assert_eq!(sum.count, 4);
    assert_eq!(sum.nan_count, 2);
    assert_eq!(sum.min, 0.5);
    assert_eq!(sum.max, 2.5);
}

#[test]
fn histogram_and_kde_record_roundtrip() {
    let hdr = HistHeader { step: 10, global_step: 5, ts_ms: 99, min_val: -1.0, max_val: 1.0 };
    let counts_bytes: Vec<u8> = [10i32, 20, 30, 40]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let rec = pack_histogram_record(&hdr, HistPrecision::Exact, &counts_bytes);
    let decoded = unpack_histogram_records(&rec, 4, HistPrecision::Exact).unwrap();
    assert_eq!(decoded.len(), 1);
    assert_eq!(decoded[0].counts, vec![10, 20, 30, 40]);

    let khdr = KdeHeader {
        step: 10,
        global_step: 5,
        ts_ms: 99,
        range_min: -2.0,
        range_max: 2.0,
        bandwidth: 0.25,
        sample_count: 1000,
    };
    let krec = pack_kde_record(&khdr, &[-1.0, 0.0, 1.0], &[0.1, 0.8, 0.1]).unwrap();
    let kdec = unpack_kde_records(&krec, 3).unwrap();
    assert_eq!(kdec[0].grid, vec![-1.0, 0.0, 1.0]);
    assert_eq!(kdec[0].density, vec![0.1, 0.8, 0.1]);
}
