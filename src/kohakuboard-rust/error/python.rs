//! PyO3 exception mapping for `BoardError`.

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::PyErr;

use super::BoardError;

impl From<BoardError> for PyErr {
    fn from(err: BoardError) -> Self {
        match err {
            BoardError::Invalid(_) | BoardError::Closed => PyValueError::new_err(err.to_string()),
            BoardError::NotFound(_) => PyKeyError::new_err(err.to_string()),
            BoardError::FlushTimeout(_) => PyTimeoutError::new_err(err.to_string()),
            _ => PyRuntimeError::new_err(err.to_string()),
        }
    }
}
