// Allow PyO3-specific false positive warnings
#![allow(clippy::useless_conversion)]

//! KohakuBoard native core: Rust-first logging engine, KohakuVault 1.2 storage,
//! off-GIL compute (histograms, FFT-KDE, media encoding), system metrics, and
//! newest-first reverse-paged query engine.
//!
//! With the `python` feature this crate builds the `kohakuboard._rs` extension module.

pub mod compute;
pub mod engine;
pub mod error;
pub mod query;
pub mod schema;
pub mod storage;
pub mod sysmon;

/// Crate version, the single source of truth for the package version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(feature = "python")]
mod py {
    use pyo3::prelude::*;

    /// Return the native extension version string.
    #[pyfunction]
    fn version() -> &'static str {
        crate::VERSION
    }

    #[pymodule]
    fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(version, m)?)?;
        m.add("SCHEMA_VERSION", crate::schema::SCHEMA_VERSION)?;
        crate::engine::python::register(m)?;
        crate::query::python::register(m)?;
        Ok(())
    }
}
