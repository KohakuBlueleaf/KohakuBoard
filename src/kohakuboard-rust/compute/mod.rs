//! Off-GIL native compute: histogram binning, exact/FFT KDE, media encoding, and blake3 hashing.

pub mod histogram;
pub mod kde;
pub mod media;
#[cfg(test)]
mod tests;
