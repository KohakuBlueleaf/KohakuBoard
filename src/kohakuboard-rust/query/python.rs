//! PyO3 bindings exposing `_RunReader` and `_Catalog` with GIL released during SQLite I/O.

use std::path::Path;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::storage::{Catalog, RunReader};

use super::chunks::{
    encode_scalar_page_frame, encode_scalar_range_frame, plan_newest_first_ranges, ChunkXType,
};

fn i64_slice_to_bytes(vals: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 8);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn f64_slice_to_bytes(vals: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 8);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Read-only native handle to a single KohakuBoard v2 run directory.
#[pyclass(name = "_RunReader")]
pub struct PyRunReader {
    inner: RunReader,
}

#[pymethods]
impl PyRunReader {
    #[new]
    fn new(py: Python<'_>, run_dir: String) -> PyResult<Self> {
        let inner = py.allow_threads(|| RunReader::open(Path::new(&run_dir)))?;
        Ok(Self { inner })
    }

    #[getter]
    fn run_dir(&self) -> String {
        self.inner.run_dir().to_string_lossy().into_owned()
    }

    fn info_json(&self, py: Python<'_>) -> PyResult<String> {
        let info = py.allow_threads(|| self.inner.info())?;
        Ok(serde_json::to_string(&info).map_err(crate::error::BoardError::from)?)
    }

    fn config_json(&self, py: Python<'_>) -> PyResult<String> {
        let val = py.allow_threads(|| self.inner.config())?;
        Ok(serde_json::to_string(&val).map_err(crate::error::BoardError::from)?)
    }

    fn env_json(&self, py: Python<'_>) -> PyResult<String> {
        let val = py.allow_threads(|| self.inner.env())?;
        Ok(serde_json::to_string(&val).map_err(crate::error::BoardError::from)?)
    }

    fn summary_json(&self, py: Python<'_>) -> PyResult<String> {
        let val = py.allow_threads(|| self.inner.summary())?;
        Ok(serde_json::to_string(&val).map_err(crate::error::BoardError::from)?)
    }

    fn catalog_json(&self, py: Python<'_>) -> PyResult<String> {
        let val = py.allow_threads(|| self.inner.catalog())?;
        Ok(serde_json::to_string(&val).map_err(crate::error::BoardError::from)?)
    }

    fn metric_definitions_json(&self, py: Python<'_>) -> PyResult<String> {
        let val = py.allow_threads(|| self.inner.metric_definitions())?;
        Ok(serde_json::to_string(&val).map_err(crate::error::BoardError::from)?)
    }

    fn scalar_len(&self, py: Python<'_>, name: String) -> PyResult<usize> {
        Ok(py.allow_threads(|| self.inner.scalar_len(&name))?)
    }

    /// Return `(total_rows, step_i64_le, global_step_i64_le, ts_ms_i64_le, value_f64_le)`.
    #[pyo3(signature = (name, start = 0, count = None))]
    #[allow(clippy::type_complexity)]
    fn scalar_slice_bytes<'py>(
        &self,
        py: Python<'py>,
        name: String,
        start: usize,
        count: Option<usize>,
    ) -> PyResult<(
        usize,
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
    )> {
        let (total, sb, gb, tb, vb) = py.allow_threads(|| -> crate::error::BoardResult<_> {
            let (total, cols) = self.inner.scalar_slice(&name, start, count)?;
            Ok((
                total,
                i64_slice_to_bytes(&cols.steps),
                i64_slice_to_bytes(&cols.global_steps),
                i64_slice_to_bytes(&cols.ts_ms),
                f64_slice_to_bytes(&cols.values),
            ))
        })?;
        Ok((
            total,
            PyBytes::new(py, &sb),
            PyBytes::new(py, &gb),
            PyBytes::new(py, &tb),
            PyBytes::new(py, &vb),
        ))
    }

    /// Return adaptive newest-first `(start, end)` row ranges covering `[0 .. total_rows)`.
    #[pyo3(signature = (name, first_chunk = 8192, max_chunk = 65536))]
    fn scalar_newest_first_plan(
        &self,
        py: Python<'_>,
        name: String,
        first_chunk: usize,
        max_chunk: usize,
    ) -> PyResult<Vec<(usize, usize)>> {
        let total = py.allow_threads(|| self.inner.scalar_len(&name))?;
        Ok(plan_newest_first_ranges(total, first_chunk, max_chunk))
    }

    /// Encode one explicit `[start .. end)` binary chunk frame for streaming endpoints.
    #[pyo3(signature = (run_id, metric, start, end, x_dtype = "f64".to_string()))]
    fn scalar_range_frame<'py>(
        &self,
        py: Python<'py>,
        run_id: String,
        metric: String,
        start: usize,
        end: usize,
        x_dtype: String,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let xdt = ChunkXType::from_name(&x_dtype)?;
        let bytes = py.allow_threads(|| {
            encode_scalar_range_frame(&self.inner, &run_id, &metric, start, end, xdt)
        })?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Encode a newest-first backward page (`before`) or forward live-tail page (`after`).
    #[pyo3(signature = (run_id, metric, before = None, after = None, limit = 8192, x_dtype = "f64".to_string()))]
    #[allow(clippy::too_many_arguments)]
    fn scalar_page_frame<'py>(
        &self,
        py: Python<'py>,
        run_id: String,
        metric: String,
        before: Option<usize>,
        after: Option<usize>,
        limit: usize,
        x_dtype: String,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let xdt = ChunkXType::from_name(&x_dtype)?;
        let bytes = py.allow_threads(|| {
            encode_scalar_page_frame(&self.inner, &run_id, &metric, before, after, limit, xdt)
        })?;
        Ok(PyBytes::new(py, &bytes))
    }

    #[pyo3(signature = (name, start = 0, count = None))]
    fn histogram_slice_json(
        &self,
        py: Python<'_>,
        name: String,
        start: usize,
        count: Option<usize>,
    ) -> PyResult<(usize, String)> {
        let (total, recs) = py.allow_threads(|| self.inner.histogram_slice(&name, start, count))?;
        let json = serde_json::to_string(&recs).map_err(crate::error::BoardError::from)?;
        Ok((total, json))
    }

    #[pyo3(signature = (name, start = 0, count = None))]
    fn kde_slice_json(
        &self,
        py: Python<'_>,
        name: String,
        start: usize,
        count: Option<usize>,
    ) -> PyResult<(usize, String)> {
        let (total, recs) = py.allow_threads(|| self.inner.kde_slice(&name, start, count))?;
        let json = serde_json::to_string(&recs).map_err(crate::error::BoardError::from)?;
        Ok((total, json))
    }

    fn media_names(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        Ok(py.allow_threads(|| self.inner.media_names())?)
    }

    #[pyo3(signature = (name, before_id = None, limit = 24))]
    fn media_events_json(
        &self,
        py: Python<'_>,
        name: String,
        before_id: Option<i64>,
        limit: usize,
    ) -> PyResult<String> {
        let items = py.allow_threads(|| self.inner.media_events(&name, before_id, limit))?;
        Ok(serde_json::to_string(&items).map_err(crate::error::BoardError::from)?)
    }

    fn blob<'py>(&self, py: Python<'py>, key: String) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let bytes = py.allow_threads(|| self.inner.blob(&key))?;
        Ok(bytes.map(|b| PyBytes::new(py, &b)))
    }

    fn table_names(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        Ok(py.allow_threads(|| self.inner.table_names())?)
    }

    #[pyo3(signature = (name, before_id = None, limit = 10))]
    fn tables_json(
        &self,
        py: Python<'_>,
        name: String,
        before_id: Option<i64>,
        limit: usize,
    ) -> PyResult<String> {
        let items = py.allow_threads(|| self.inner.tables(&name, before_id, limit))?;
        Ok(serde_json::to_string(&items).map_err(crate::error::BoardError::from)?)
    }

    fn tensor_names(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        Ok(py.allow_threads(|| self.inner.tensor_names())?)
    }

    #[pyo3(signature = (name, before_id = None, limit = 20))]
    fn tensors_json(
        &self,
        py: Python<'_>,
        name: String,
        before_id: Option<i64>,
        limit: usize,
    ) -> PyResult<String> {
        let items = py.allow_threads(|| self.inner.tensors(&name, before_id, limit))?;
        Ok(serde_json::to_string(&items).map_err(crate::error::BoardError::from)?)
    }

    #[pyo3(signature = (before_id = None, limit = 50))]
    fn artifacts_json(
        &self,
        py: Python<'_>,
        before_id: Option<i64>,
        limit: usize,
    ) -> PyResult<String> {
        let items = py.allow_threads(|| self.inner.artifacts(before_id, limit))?;
        Ok(serde_json::to_string(&items).map_err(crate::error::BoardError::from)?)
    }

    #[pyo3(signature = (before_id = None, limit = 50))]
    fn alerts_json(
        &self,
        py: Python<'_>,
        before_id: Option<i64>,
        limit: usize,
    ) -> PyResult<String> {
        let items = py.allow_threads(|| self.inner.alerts(before_id, limit))?;
        Ok(serde_json::to_string(&items).map_err(crate::error::BoardError::from)?)
    }

    #[pyo3(signature = (before_id = None, limit = 200))]
    fn logs_tail_json(
        &self,
        py: Python<'_>,
        before_id: Option<i64>,
        limit: usize,
    ) -> PyResult<String> {
        let items = py.allow_threads(|| self.inner.logs_tail(before_id, limit))?;
        Ok(serde_json::to_string(&items).map_err(crate::error::BoardError::from)?)
    }

    #[pyo3(signature = (query, limit = 50))]
    fn logs_search_json(&self, py: Python<'_>, query: String, limit: usize) -> PyResult<String> {
        let items = py.allow_threads(|| self.inner.logs_search(&query, limit))?;
        Ok(serde_json::to_string(&items).map_err(crate::error::BoardError::from)?)
    }

    fn close(&self, py: Python<'_>) {
        py.allow_threads(|| self.inner.close());
    }
}

/// Native handle to the project-wide `.catalog.db` SQLite index.
#[pyclass(name = "_Catalog")]
pub struct PyCatalog {
    inner: Catalog,
}

#[pymethods]
impl PyCatalog {
    #[new]
    fn new(py: Python<'_>, data_dir: String) -> PyResult<Self> {
        let inner = py.allow_threads(|| Catalog::open(Path::new(&data_dir)))?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (force_rebuild = false))]
    fn refresh(&self, py: Python<'_>, force_rebuild: bool) -> PyResult<usize> {
        Ok(py.allow_threads(|| self.inner.refresh(force_rebuild))?)
    }

    fn list_projects_json(&self, py: Python<'_>) -> PyResult<String> {
        let items = py.allow_threads(|| self.inner.list_projects())?;
        Ok(serde_json::to_string(&items).map_err(crate::error::BoardError::from)?)
    }

    #[pyo3(signature = (
        project = None,
        state = None,
        group = None,
        search = None,
        sort_by = None,
        descending = true,
        limit = None,
        offset = 0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn list_runs_json(
        &self,
        py: Python<'_>,
        project: Option<String>,
        state: Option<String>,
        group: Option<String>,
        search: Option<String>,
        sort_by: Option<String>,
        descending: bool,
        limit: Option<usize>,
        offset: usize,
    ) -> PyResult<String> {
        let items = py.allow_threads(|| {
            self.inner.list_runs(
                project.as_deref(),
                state.as_deref(),
                group.as_deref(),
                search.as_deref(),
                sort_by.as_deref(),
                descending,
                limit,
                offset,
            )
        })?;
        Ok(serde_json::to_string(&items).map_err(crate::error::BoardError::from)?)
    }

    fn resolve_run_path(
        &self,
        py: Python<'_>,
        project: String,
        run_id: String,
    ) -> PyResult<Option<String>> {
        let path = py.allow_threads(|| self.inner.resolve_run_path(&project, &run_id))?;
        Ok(path.map(|p| p.to_string_lossy().into_owned()))
    }
}

/// Register query reader/catalog classes on `kohakuboard._rs`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRunReader>()?;
    m.add_class::<PyCatalog>()?;
    Ok(())
}
