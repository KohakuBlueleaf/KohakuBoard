//! Query engine: newest-first reverse-paged binary chunk framing and Python reader bindings.

pub mod chunks;
#[cfg(feature = "python")]
pub mod python;
#[cfg(test)]
mod tests;

pub use chunks::{
    decode_chunk_frame, encode_scalar_page_frame, encode_scalar_range_frame,
    plan_newest_first_ranges, ChunkHeader, ChunkXType,
};
