"""Declarative configuration dataclasses for KohakuBoard's KohakuTUI integration.

Pure stdlib module with zero dependency on `kohakutui`, so `TuiConfig` and
`ChartSpec` can always be imported from `kohakuboard` even when the `[tui]`
extra is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from kohakuboard.errors import ConfigError


@dataclass(frozen=True)
class ChartSpec:
    """Specification for a pinned scalar chart inside the live training TUI."""

    metrics: str | tuple[str, ...] | list[str]
    label: str | None = None
    height: int = 8
    digits: int | None = None
    ema: float | None = 0.97
    y_min: float | None = None
    y_max: float | None = None
    alert_above: float | None = None
    alert_below: float | None = None

    def metric_names(self) -> tuple[str, ...]:
        if isinstance(self.metrics, str):
            return (self.metrics,)
        names = tuple(str(m) for m in self.metrics if str(m).strip())
        if not names:
            raise ConfigError("ChartSpec.metrics must contain at least one metric name")
        return names

    def display_label(self) -> str:
        if self.label:
            return self.label
        names = self.metric_names()
        return names[0] if len(names) == 1 else " / ".join(names[:3])


@dataclass
class TuiConfig:
    """Declarative configuration for `Board(tui=TuiConfig(...))`."""

    mode: Literal["inline", "fullscreen", "auto"] = "inline"
    theme: str = "kohaku-dark"
    refresh_hz: float = 12.0
    total_steps: int | None = None
    charts: list[ChartSpec | str] = field(default_factory=list)
    watch: list[str] = field(default_factory=list)
    hide: list[str] = field(default_factory=list)
    collapse_namespaces: list[str] = field(default_factory=lambda: ["grad/", "act/", "sys/"])
    log_height: int = 4
    capture_stdio: bool = True
    show_system: bool = True
    on_finish: Literal["seal", "clear", "hold"] = "seal"
    force: bool = False
    backend: Literal["auto", "tty", "plain", "test"] = "auto"
    size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("inline", "fullscreen", "auto"):
            raise ConfigError(f"Unsupported TuiConfig.mode {self.mode!r}")
        if self.on_finish not in ("seal", "clear", "hold"):
            raise ConfigError(f"Unsupported TuiConfig.on_finish {self.on_finish!r}")
        if self.backend not in ("auto", "tty", "plain", "test"):
            raise ConfigError(f"Unsupported TuiConfig.backend {self.backend!r}")
        if self.refresh_hz <= 0 or self.refresh_hz > 60:
            raise ConfigError("TuiConfig.refresh_hz must be in (0, 60]")
        if self.log_height < 0:
            raise ConfigError("TuiConfig.log_height must be >= 0")

    def normalized_charts(self) -> list[ChartSpec]:
        out: list[ChartSpec] = []
        for item in self.charts:
            if isinstance(item, ChartSpec):
                out.append(item)
            elif isinstance(item, str) and item.strip():
                out.append(ChartSpec(item.strip()))
            else:
                raise ConfigError(f"Unsupported chart spec {item!r}")
        return out


def resolve_tui_config(tui: Any) -> TuiConfig | None:
    """Coerce `Board(tui=...)` argument into a validated `TuiConfig` or `None`."""
    if tui is False or tui is None:
        return None
    if tui is True:
        return TuiConfig()
    if isinstance(tui, str):
        if tui in ("inline", "fullscreen", "auto"):
            return TuiConfig(mode=tui)  # type: ignore[arg-type]
        raise ConfigError(
            f"Unsupported tui mode {tui!r}; expected bool, 'inline', 'fullscreen', or TuiConfig."
        )
    if isinstance(tui, TuiConfig):
        return tui
    raise ConfigError(
        f"Unsupported tui value of type {type(tui).__name__}; "
        "expected bool, 'inline', 'fullscreen', or TuiConfig."
    )
