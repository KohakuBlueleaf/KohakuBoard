//! Unified error hierarchy for KohakuBoard native modules.

#[cfg(feature = "python")]
pub mod python;

use kohakuvault::{rusqlite, ColError, VaultError};
use thiserror::Error;

/// Crate-wide result alias.
pub type BoardResult<T> = Result<T, BoardError>;

/// Errors produced by KohakuBoard's native storage, compute, engine, and query layers.
#[derive(Debug, Error)]
pub enum BoardError {
    #[error("storage error: {0}")]
    Storage(String),

    #[error("invalid argument: {0}")]
    Invalid(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("board is closed")]
    Closed,

    #[error("flush timed out after {0} ms")]
    FlushTimeout(u64),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("codec error: {0}")]
    Codec(String),

    #[error("compute error: {0}")]
    Compute(String),
}

impl From<VaultError> for BoardError {
    fn from(err: VaultError) -> Self {
        match err {
            VaultError::NotFound(msg) => Self::NotFound(msg),
            VaultError::Invalid(msg) => Self::Invalid(msg),
            VaultError::Closed => Self::Closed,
            VaultError::Codec(msg) => Self::Codec(msg),
            other => Self::Storage(other.to_string()),
        }
    }
}

impl From<ColError> for BoardError {
    fn from(err: ColError) -> Self {
        match err {
            ColError::NotFound => Self::NotFound(err.to_string()),
            ColError::Invalid(msg) => Self::Invalid(msg),
            ColError::Closed => Self::Closed,
            ColError::Bounds => Self::Invalid(err.to_string()),
            other => Self::Storage(other.to_string()),
        }
    }
}

impl From<rusqlite::Error> for BoardError {
    fn from(err: rusqlite::Error) -> Self {
        Self::Storage(err.to_string())
    }
}

impl From<serde_json::Error> for BoardError {
    fn from(err: serde_json::Error) -> Self {
        Self::Codec(err.to_string())
    }
}
