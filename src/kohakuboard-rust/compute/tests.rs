//! Unit tests for histogram binning, exact/FFT KDE, and media encoding.

use crate::schema::HistPrecision;

use super::histogram::{compute_histogram, pack_precomputed};
use super::kde::compute_kde;
use super::media::{blake3_short_hex, compress_bytes, decompress_bytes, encode_image_pixels};

#[test]
fn histogram_exact_and_compact() {
    let vals: Vec<f32> = (0..1000).map(|i| i as f32 / 100.0).collect();
    let exact = compute_histogram(&vals, 16, HistPrecision::Exact, None, (0.0, 100.0))
        .unwrap()
        .unwrap();
    assert_eq!(exact.counts_bytes.len(), 16 * 4);
    let total: i32 = exact
        .counts_bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .sum();
    assert_eq!(total, 1000);

    let compact = compute_histogram(&vals, 16, HistPrecision::Compact, None, (0.0, 100.0))
        .unwrap()
        .unwrap();
    assert_eq!(compact.counts_bytes.len(), 16);
    assert_eq!(*compact.counts_bytes.iter().max().unwrap(), 255);

    let pre = pack_precomputed(&[0.0, 1.0, 2.0], &[3, 7], HistPrecision::Exact).unwrap();
    assert_eq!(pre.bins, 2);
}

#[test]
fn kde_exact_and_fft_peak_at_center() {
    let vals: Vec<f32> = (0..500)
        .map(|i| {
            let t = (i as f32 / 500.0) * 2.0 - 1.0;
            t * t * t
        })
        .collect();
    let exact = compute_kde(&vals, 64, Some(0.2), false, Some((-1.0, 1.0)), (1.0, 99.0))
        .unwrap()
        .unwrap();
    let fft = compute_kde(&vals, 64, Some(0.2), true, Some((-1.0, 1.0)), (1.0, 99.0))
        .unwrap()
        .unwrap();
    assert_eq!(exact.grid.len(), 64);
    assert_eq!(fft.density.len(), 64);
    let peak_exact = exact
        .density
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    let peak_fft = fft
        .density
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    assert!((peak_exact as isize - peak_fft as isize).abs() <= 2);
}

#[test]
fn png_webp_and_gzip_roundtrip() {
    let pixels = vec![128u8; 8 * 6 * 3];
    let png = encode_image_pixels(&pixels, 8, 6, 3, "png", 90).unwrap();
    assert_eq!(png.ext, "png");
    assert_eq!(png.width, Some(8));
    assert_eq!(png.height, Some(6));
    assert!(png.blob_key.ends_with(".png"));
    assert_eq!(blake3_short_hex(&png.bytes).len(), 32);

    let webp = encode_image_pixels(&pixels, 8, 6, 3, "webp", 90).unwrap();
    assert_eq!(webp.ext, "webp");

    let raw = b"kohakuboard-tensor-payload-repeat-repeat-repeat";
    let gz = compress_bytes(raw).unwrap();
    assert_eq!(decompress_bytes(&gz).unwrap(), raw);
}
