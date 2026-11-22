"""Type stubs for the native `kohakuboard._rs` extension module."""

from __future__ import annotations

SCHEMA_VERSION: int

def version() -> str: ...
def blake3_hex(data: bytes) -> str: ...
def decompress_blob(data: bytes) -> bytes: ...

class _LoggerEngine:
    def __init__(
        self,
        run_dir: str | None,
        run_id: str,
        name: str,
        project: str,
        annotation: str | None = None,
        group: str | None = None,
        job_type: str | None = None,
        tags: list[str] = ...,
        notes: str = "",
        config_json: str = "{}",
        env_json: str = "{}",
        resume: bool = False,
        flush_interval_ms: int = 500,
        queue_capacity: int = 65536,
        overflow: str = "block",
        system_metrics_interval_ms: int | None = 2000,
    ) -> None: ...
    @property
    def initial_last_step(self) -> int: ...
    @property
    def initial_last_global_step(self) -> int: ...
    @property
    def is_closed(self) -> bool: ...
    @property
    def dropped_events(self) -> int: ...
    def set_step_counters(self, step: int, global_step: int) -> None: ...
    def log_scalars(
        self,
        step: int,
        global_step: int,
        ts_ms: int,
        metrics: list[tuple[str, float]],
    ) -> None: ...
    def log_media_pixels(
        self,
        step: int,
        global_step: int,
        ts_ms: int,
        name: str,
        idx: int,
        pixels: bytes,
        width: int,
        height: int,
        channels: int,
        format: str,
        quality: int,
        caption: str,
        meta_json: str,
    ) -> None: ...
    def log_media_encoded(
        self,
        step: int,
        global_step: int,
        ts_ms: int,
        name: str,
        idx: int,
        media_type: str,
        bytes: bytes,
        ext: str,
        duration_ms: int | None,
        caption: str,
        meta_json: str,
    ) -> None: ...
    def stage_media_pixels(
        self,
        pixels: bytes,
        width: int,
        height: int,
        channels: int,
        format: str,
        quality: int,
    ) -> tuple[str, str, int | None, int | None, int]: ...
    def stage_media_encoded(
        self,
        bytes: bytes,
        ext: str,
        media_type: str,
    ) -> tuple[str, str, int | None, int | None, int]: ...
    def log_table(
        self,
        step: int,
        global_step: int,
        ts_ms: int,
        name: str,
        columns_json: str,
        row_count: int,
        rows_json: str,
    ) -> None: ...
    def log_histogram_raw(
        self,
        step: int,
        global_step: int,
        ts_ms: int,
        name: str,
        values_f32_bytes: bytes,
        bins: int,
        precision: str,
        range: tuple[float, float] | None = None,
        percentiles: tuple[float, float] = (1.0, 99.0),
    ) -> None: ...
    def log_histogram_precomputed(
        self,
        step: int,
        global_step: int,
        ts_ms: int,
        name: str,
        edges: list[float],
        counts: list[int],
        precision: str,
    ) -> None: ...
    def log_kde_raw(
        self,
        step: int,
        global_step: int,
        ts_ms: int,
        name: str,
        values_f32_bytes: bytes,
        num_points: int,
        bandwidth: float | None = None,
        approximate: bool = True,
        range: tuple[float, float] | None = None,
        percentiles: tuple[float, float] = (1.0, 99.0),
    ) -> None: ...
    def log_kde_precomputed(
        self,
        step: int,
        global_step: int,
        ts_ms: int,
        name: str,
        range_min: float,
        range_max: float,
        bandwidth: float,
        sample_count: int,
        grid: list[float],
        density: list[float],
    ) -> None: ...
    def log_tensor(
        self,
        step: int,
        global_step: int,
        ts_ms: int,
        name: str,
        raw_bytes: bytes,
        dtype: str,
        shape_json: str,
        compress: bool,
        meta_json: str,
    ) -> None: ...
    def log_artifact(
        self,
        step: int,
        global_step: int,
        ts_ms: int,
        name: str,
        kind: str,
        uri: str,
        digest: str | None,
        size_bytes: int | None,
        meta_json: str,
    ) -> None: ...
    def log_alert(
        self,
        step: int,
        global_step: int,
        ts_ms: int,
        level: str,
        title: str,
        text: str,
    ) -> None: ...
    def log_lines(self, lines: list[tuple[int, int, str, str]]) -> None: ...
    def patch_config(self, patch_json: str) -> None: ...
    def patch_summary(
        self,
        step: int,
        global_step: int,
        ts_ms: int,
        values: dict[str, float],
    ) -> None: ...
    def patch_meta(
        self,
        name: str | None = None,
        group: str | None = None,
        add_tags: list[str] = ...,
        notes: str | None = None,
    ) -> None: ...
    def define_metric(
        self,
        pattern: str,
        step_metric: str | None = None,
        summary: str = "last",
        hidden: bool = False,
    ) -> None: ...
    def progress(
        self,
        label: str,
        current: int,
        total: int | None = None,
        unit: str = "step",
        ts_ms: int = 0,
    ) -> None: ...
    def flush(self, timeout_ms: int = 30000) -> None: ...
    def finish(
        self,
        state: str = "finished",
        exit_code: int = 0,
        timeout_ms: int = 30000,
    ) -> None: ...
    def snapshot_json(self) -> str: ...

class _RunReader:
    def __init__(self, run_dir: str) -> None: ...
    @property
    def run_dir(self) -> str: ...
    def info_json(self) -> str: ...
    def config_json(self) -> str: ...
    def env_json(self) -> str: ...
    def summary_json(self) -> str: ...
    def catalog_json(self) -> str: ...
    def metric_definitions_json(self) -> str: ...
    def scalar_len(self, name: str) -> int: ...
    def scalar_slice_bytes(
        self,
        name: str,
        start: int = 0,
        count: int | None = None,
    ) -> tuple[int, bytes, bytes, bytes, bytes]: ...
    def scalar_newest_first_plan(
        self,
        name: str,
        first_chunk: int = 8192,
        max_chunk: int = 65536,
    ) -> list[tuple[int, int]]: ...
    def scalar_range_frame(
        self,
        run_id: str,
        metric: str,
        start: int,
        end: int,
        x_dtype: str = "f64",
    ) -> bytes: ...
    def scalar_page_frame(
        self,
        run_id: str,
        metric: str,
        before: int | None = None,
        after: int | None = None,
        limit: int = 8192,
        x_dtype: str = "f64",
    ) -> bytes: ...
    def histogram_slice_json(
        self,
        name: str,
        start: int = 0,
        count: int | None = None,
    ) -> tuple[int, str]: ...
    def kde_slice_json(
        self,
        name: str,
        start: int = 0,
        count: int | None = None,
    ) -> tuple[int, str]: ...
    def media_names(self) -> list[str]: ...
    def media_events_json(
        self,
        name: str,
        before_id: int | None = None,
        limit: int = 24,
    ) -> str: ...
    def blob(self, key: str) -> bytes | None: ...
    def table_names(self) -> list[str]: ...
    def tables_json(
        self,
        name: str,
        before_id: int | None = None,
        limit: int = 10,
    ) -> str: ...
    def tensor_names(self) -> list[str]: ...
    def tensors_json(
        self,
        name: str,
        before_id: int | None = None,
        limit: int = 20,
    ) -> str: ...
    def artifacts_json(
        self,
        before_id: int | None = None,
        limit: int = 50,
    ) -> str: ...
    def alerts_json(
        self,
        before_id: int | None = None,
        limit: int = 50,
    ) -> str: ...
    def logs_tail_json(
        self,
        before_id: int | None = None,
        limit: int = 200,
    ) -> str: ...
    def logs_search_json(self, query: str, limit: int = 50) -> str: ...
    def close(self) -> None: ...

class _Catalog:
    def __init__(self, data_dir: str) -> None: ...
    def refresh(self, force_rebuild: bool = False) -> int: ...
    def list_projects_json(self) -> str: ...
    def list_runs_json(
        self,
        project: str | None = None,
        state: str | None = None,
        group: str | None = None,
        search: str | None = None,
        sort_by: str | None = None,
        descending: bool = True,
        limit: int | None = None,
        offset: int = 0,
    ) -> str: ...
    def resolve_run_path(self, project: str, run_id: str) -> str | None: ...
