"""Pure helper utilities for KohakuBoard's KohakuTUI live adapter and SSH viewer."""

from __future__ import annotations

import fnmatch
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def format_num(value: float | int | None, digits: int = 4) -> str:
    """Format a scalar value concisely, preserving NaN and ±Inf markers."""
    if value is None:
        return "-"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return "NaN"
    if math.isinf(f):
        return "+Inf" if f > 0 else "-Inf"
    af = abs(f)
    if af == 0.0:
        return "0"
    if af < 1e-3 or af >= 1e5:
        return f"{f:.{max(1, digits - 1)}e}"
    return f"{f:.{digits}f}".rstrip("0").rstrip(".")


def format_bytes(size: int | float | None) -> str:
    if not size:
        return "0 B"
    val = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(val) < 1024.0 or unit == "TiB":
            return f"{val:.1f} {unit}" if unit != "B" else f"{int(val)} B"
        val /= 1024.0
    return f"{val:.1f} TiB"


def format_duration(
    created_ms: int | None, finished_ms: int | None, now_ms: int | None = None
) -> str:
    if not created_ms:
        return "-"
    end = finished_ms if finished_ms else now_ms
    if not end or end < created_ms:
        return "0s"
    secs = int((end - created_ms) / 1000)
    if secs < 60:
        return f"{secs}s"
    mins, rem_s = divmod(secs, 60)
    if mins < 60:
        return f"{mins}m{rem_s:02d}s"
    hrs, rem_m = divmod(mins, 60)
    return f"{hrs}h{rem_m:02d}m"


def is_hidden(metric: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(metric, pat) for pat in patterns)


def split_namespace(name: str) -> tuple[str, str]:
    idx = name.find("/")
    if idx <= 0:
        return ("metrics", name)
    return (name[:idx], name[idx + 1 :])


def group_by_namespace(names: Iterable[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for name in sorted(names):
        ns, _ = split_namespace(name)
        grouped.setdefault(ns, []).append(name)
    return grouped


def auto_select_charts(
    metric_names: Sequence[str],
    hide_patterns: Sequence[str] = (),
    *,
    max_charts: int = 2,
) -> list[str]:
    """Choose 1-2 high-signal default metrics for inline/overview charts."""
    visible = [m for m in metric_names if not is_hidden(m, hide_patterns)]
    if not visible or max_charts <= 0:
        return []

    picked: list[str] = []

    def _pick(predicate: Any) -> None:
        if len(picked) >= max_charts:
            return
        for m in visible:
            if m not in picked and predicate(m):
                picked.append(m)
                return

    # Priority 1: loss metric (train/loss, loss, *loss*)
    _pick(lambda m: m in ("train/loss", "loss"))
    _pick(lambda m: "loss" in m.lower() and not m.startswith("sys/"))

    # Priority 2: secondary training signal (val/loss, grad norm, lr, acc)
    _pick(lambda m: "val" in m.lower() and "loss" in m.lower())
    _pick(lambda m: "grad" in m.lower() and "norm" in m.lower())
    _pick(lambda m: m in ("train/lr", "lr") or m.endswith("/lr"))
    _pick(lambda m: not m.startswith("sys/"))
    _pick(lambda _m: True)
    return picked[:max_charts]


def compute_ema(values: Sequence[float], alpha: float) -> list[float]:
    """Bias-corrected EMA preserving NaN gaps."""
    if not values:
        return []
    a = max(0.001, min(0.999, float(alpha)))
    out: list[float] = []
    acc = 0.0
    denom = 0.0
    for v in values:
        if not math.isfinite(v):
            out.append(float("nan"))
            continue
        acc = acc * a + v * (1.0 - a)
        denom = denom * a + (1.0 - a)
        out.append(acc / denom if denom > 0.0 else v)
    return out


def extract_system_fields(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Summarize `sys/*` live metrics into compact KV strings."""
    fields: dict[str, str] = {}
    for name in sorted(metrics):
        if not name.startswith("sys/"):
            continue
        short = name[4:]
        val = metrics[name].get("last")
        fields[short] = format_num(val, digits=2)
    return fields


def format_distribution_bars(
    record: Mapping[str, Any],
    *,
    max_bins: int = 16,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Convert a histogram or KDE record into KohakuTUI `Bars(labels, series)` inputs."""
    if "counts" in record and isinstance(record["counts"], list):
        raw = [float(x) for x in record["counts"]]
        lo = float(record.get("min_val", 0.0))
        hi = float(record.get("max_val", 1.0))
        series_name = "count"
    elif "density" in record and isinstance(record["density"], list):
        raw = [float(x) for x in record["density"]]
        lo = float(record.get("range_min", 0.0))
        hi = float(record.get("range_max", 1.0))
        series_name = "density"
    else:
        return ([], [])

    n = len(raw)
    if n == 0:
        return ([], [])

    # Rebin if more bins than terminal height budget
    if n > max_bins:
        bucket = n / max_bins
        binned: list[float] = []
        for i in range(max_bins):
            a = int(i * bucket)
            b = max(a + 1, int((i + 1) * bucket))
            chunk = raw[a:b]
            binned.append(sum(chunk) if series_name == "count" else sum(chunk) / len(chunk))
        vals = binned
    else:
        vals = raw

    span = hi - lo if hi > lo else 1.0
    m = len(vals)
    labels = [
        f"{format_num(lo + (i / m) * span, 2)}..{format_num(lo + ((i + 1) / m) * span, 2)}"
        for i in range(m)
    ]
    return (labels, [{"name": series_name, "data": vals}])


def status_tone(state: str) -> str:
    return {
        "running": "ok",
        "finished": "accent",
        "crashed": "danger",
        "interrupted": "attention",
    }.get(state, "muted")
