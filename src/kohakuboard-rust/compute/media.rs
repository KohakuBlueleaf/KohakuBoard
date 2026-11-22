//! Off-GIL blake3 content hashing, PNG/WebP/JPEG encoding, and tensor compression.

use std::io::{Cursor, Read, Write};

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use image::codecs::webp::WebPEncoder;
use image::{ColorType, ImageEncoder, ImageReader};

use crate::error::{BoardError, BoardResult};

/// Encoded media payload ready for `blobs.db`.
#[derive(Debug, Clone)]
pub struct EncodedMedia {
    pub blob_key: String,
    pub bytes: Vec<u8>,
    pub ext: String,
    pub mime: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

/// Compute a 32-char lowercase hex blake3 digest prefix (128-bit collision resistance).
pub fn blake3_short_hex(bytes: &[u8]) -> String {
    let hash = blake3::hash(bytes);
    hash.to_hex()[..32].to_string()
}

/// Encode raw HWC `u8` pixels (`channels` = 1, 3, or 4) into PNG, WebP, or JPEG.
pub fn encode_image_pixels(
    pixels: &[u8],
    width: u32,
    height: u32,
    channels: u8,
    format: &str,
    quality: u8,
) -> BoardResult<EncodedMedia> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|v| v.checked_mul(usize::from(channels)))
        .ok_or_else(|| BoardError::Invalid("image dimensions overflow".into()))?;
    if pixels.len() != expected {
        return Err(BoardError::Invalid(format!(
            "pixel buffer size {} != {}x{}x{}",
            pixels.len(),
            width,
            height,
            channels
        )));
    }

    let color = match channels {
        1 => ColorType::L8,
        3 => ColorType::Rgb8,
        4 => ColorType::Rgba8,
        c => return Err(BoardError::Invalid(format!("unsupported channel count {c}"))),
    };

    let fmt = match format.to_ascii_lowercase().as_str() {
        "auto" | "png" => "png",
        "webp" => "webp",
        "jpg" | "jpeg" => "jpg",
        other => return Err(BoardError::Invalid(format!("unsupported image format '{other}'"))),
    };

    let mut out = Vec::new();
    match fmt {
        "png" => {
            PngEncoder::new(&mut out)
                .write_image(pixels, width, height, color.into())
                .map_err(|e| BoardError::Compute(format!("PNG encode failed: {e}")))?;
        }
        "webp" => {
            let (rgba_owned, enc_color) = match color {
                ColorType::L8 => {
                    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
                    for &p in pixels {
                        rgb.extend_from_slice(&[p, p, p]);
                    }
                    (Some(rgb), ColorType::Rgb8)
                }
                _ => (None, color),
            };
            let slice = rgba_owned.as_deref().unwrap_or(pixels);
            WebPEncoder::new_lossless(&mut out)
                .write_image(slice, width, height, enc_color.into())
                .map_err(|e| BoardError::Compute(format!("WebP encode failed: {e}")))?;
        }
        "jpg" => {
            let rgb_owned = match color {
                ColorType::Rgba8 => {
                    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
                    for px in pixels.chunks_exact(4) {
                        rgb.extend_from_slice(&px[..3]);
                    }
                    Some(rgb)
                }
                _ => None,
            };
            let slice = rgb_owned.as_deref().unwrap_or(pixels);
            let jpg_color = if color == ColorType::L8 {
                ColorType::L8
            } else {
                ColorType::Rgb8
            };
            JpegEncoder::new_with_quality(&mut out, quality.clamp(1, 100))
                .write_image(slice, width, height, jpg_color.into())
                .map_err(|e| BoardError::Compute(format!("JPEG encode failed: {e}")))?;
        }
        _ => unreachable!(),
    }

    let mime = match fmt {
        "png" => "image/png",
        "webp" => "image/webp",
        "jpg" => "image/jpeg",
        _ => "application/octet-stream",
    };
    let hash = blake3_short_hex(&out);
    Ok(EncodedMedia {
        blob_key: format!("{hash}.{fmt}"),
        bytes: out,
        ext: fmt.to_string(),
        mime: mime.to_string(),
        width: Some(width),
        height: Some(height),
    })
}

/// Hash pre-encoded media bytes (images/video/audio) and detect image dimensions when possible.
pub fn wrap_encoded_media(bytes: Vec<u8>, ext: &str, media_type: &str) -> EncodedMedia {
    let ext_clean = ext.trim_start_matches('.').to_ascii_lowercase();
    let dims = if media_type == "image" {
        ImageReader::new(Cursor::new(&bytes))
            .with_guessed_format()
            .ok()
            .and_then(|r| r.into_dimensions().ok())
    } else {
        None
    };
    let mime = mime_for_ext(&ext_clean).to_string();
    let hash = blake3_short_hex(&bytes);
    EncodedMedia {
        blob_key: format!("{hash}.{ext_clean}"),
        bytes,
        ext: ext_clean,
        mime,
        width: dims.map(|(w, _)| w),
        height: dims.map(|(_, h)| h),
    }
}

/// Map common file extensions to MIME types.
pub fn mime_for_ext(ext: &str) -> &'static str {
    match ext.to_ascii_lowercase().as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "mov" => "video/quicktime",
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "ogg" => "audio/ogg",
        "flac" => "audio/flac",
        _ => "application/octet-stream",
    }
}

/// Optionally gzip-compress raw tensor bytes.
pub fn compress_bytes(raw: &[u8]) -> BoardResult<Vec<u8>> {
    let mut enc = GzEncoder::new(Vec::new(), Compression::fast());
    enc.write_all(raw)?;
    Ok(enc.finish()?)
}

/// Decompress gzip-compressed bytes.
pub fn decompress_bytes(compressed: &[u8]) -> BoardResult<Vec<u8>> {
    let mut dec = GzDecoder::new(compressed);
    let mut out = Vec::new();
    dec.read_to_end(&mut out)?;
    Ok(out)
}
