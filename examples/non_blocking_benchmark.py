"""KohakuBoard non-blocking benchmark suite.

Runs seven scenarios so we can understand how the client queue behaves for each
data type.  Each phase records throughput in both values/sec and objects/sec,
and prints the exact data volume that was logged.

Scenarios
---------
1. Scalar only
2. Histogram only
3. Media only (random RGB images)
4. Tensor only
5. KDE only (raw samples auto-converted)
6. Scalars + tensors
7. Scalars + tensors + KDE
"""

import argparse
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kohakuboard.client import Board, Histogram, Media
from kohakuboard.logger import LoggerFactory

# Silence noisy loggers globally for the benchmark
for _logger_name in ("BOARD", "WRITER", "STORAGE"):
    try:
        LoggerFactory.get_logger(_logger_name, drop=True)
    except Exception:
        pass


@dataclass
class PhaseResult:
    name: str
    objects: int
    values: int
    elapsed: float

    @property
    def values_per_sec(self) -> float:
        return self.values / self.elapsed if self.elapsed > 0 else float("inf")

    @property
    def objects_per_sec(self) -> float:
        return self.objects / self.elapsed if self.elapsed > 0 else float("inf")


def remove_board_dir(board_dir: Path) -> None:
    if board_dir.exists():
        shutil.rmtree(board_dir, ignore_errors=True)


def build_board(args: argparse.Namespace) -> Board:
    return Board(
        name="non_blocking_bench",
        board_id=args.board_id,
        base_dir=args.base_dir,
        sync_enabled=False,
        capture_output=False,
    )


# ---------------------- Phase runners ----------------------
def run_scalar_only(board: Board, steps: int, scalars_per_step: int) -> PhaseResult:
    metric_names = [f"bench/scalar_{i:02d}" for i in range(scalars_per_step)]
    start = time.perf_counter()
    for _ in range(steps):
        payload = {name: np.random.random() for name in metric_names}
        board.log(**payload)
    elapsed = time.perf_counter() - start
    objects = steps
    values = steps * scalars_per_step
    return PhaseResult("scalar_only", objects, values, elapsed)


def run_hist_only(
    board: Board, histograms: int, values_per_hist: int, bins: int
) -> PhaseResult:
    start = time.perf_counter()
    for _ in range(histograms):
        samples = np.random.randn(values_per_hist).astype(np.float32)
        board.log(
            **{"hist/stream": Histogram(samples, num_bins=bins, precision="exact")}
        )
    elapsed = time.perf_counter() - start
    objects = histograms
    values = histograms * values_per_hist
    return PhaseResult("histogram_only", objects, values, elapsed)


def random_image(width: int, height: int) -> np.ndarray:
    return np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)


def run_media_only(board: Board, images: int, width: int, height: int) -> PhaseResult:
    start = time.perf_counter()
    for idx in range(images):
        img = random_image(width, height)
        board.log(
            **{f"media/random": Media(img, media_type="image", caption=f"img {idx}")}
        )
    elapsed = time.perf_counter() - start
    objects = images
    values = images * width * height * 3
    return PhaseResult("media_only", objects, values, elapsed)


def run_tensor_only(
    board: Board, logs: int, tensor_size: int, name: str
) -> PhaseResult:
    start = time.perf_counter()
    for idx in range(logs):
        tensor = np.random.randn(tensor_size).astype(np.float32)
        board.log_tensor(name, tensor, metadata={"index": idx})
    elapsed = time.perf_counter() - start
    objects = logs
    values = logs * tensor_size
    return PhaseResult("tensor_only", objects, values, elapsed)


def run_kde_only(board: Board, logs: int, sample_count: int, name: str) -> PhaseResult:
    start = time.perf_counter()
    for idx in range(logs):
        samples = np.random.randn(sample_count).astype(np.float32)
        board.log_kernel_density(
            name,
            values=samples,
            num_points=256,
            percentile_clip=(1.0, 99.0),
            metadata={"index": idx},
        )
    elapsed = time.perf_counter() - start
    objects = logs
    values = logs * sample_count
    return PhaseResult("kde_only", objects, values, elapsed)


def run_scalar_tensor_mix(
    board: Board,
    steps: int,
    scalars_per_step: int,
    tensor_size: int,
    interval: int,
    tensor_name: str,
) -> PhaseResult:
    metric_names = [f"mix/scalar_{i:02d}" for i in range(scalars_per_step)]
    tensor_payload = np.random.randn(tensor_size).astype(np.float32)
    tensors_emitted = math.ceil(steps / interval)
    start = time.perf_counter()
    for step in range(steps):
        payload = {name: np.random.random() for name in metric_names}
        board.log(**payload)
        if (step + 1) % interval == 0:
            board.log_tensor(tensor_name, tensor_payload, metadata={"step": step + 1})
    elapsed = time.perf_counter() - start
    objects = steps + tensors_emitted
    values = steps * scalars_per_step + tensors_emitted * tensor_size
    return PhaseResult("scalar+tensor", objects, values, elapsed)


def run_scalar_tensor_kde_mix(
    board: Board,
    steps: int,
    scalars_per_step: int,
    tensor_size: int,
    tensor_interval: int,
    kde_samples: int,
    kde_interval: int,
) -> PhaseResult:
    metric_names = [f"mix2/scalar_{i:02d}" for i in range(scalars_per_step)]
    tensor_payload = np.random.randn(tensor_size).astype(np.float32)
    tensors_emitted = math.ceil(steps / tensor_interval)
    kdes_emitted = math.ceil(steps / kde_interval)
    start = time.perf_counter()
    for step in range(steps):
        payload = {name: np.random.random() for name in metric_names}
        board.log(**payload)
        step_index = step + 1
        if step_index % tensor_interval == 0:
            board.log_tensor(
                "mix2/tensor", tensor_payload, metadata={"step": step_index}
            )
        if step_index % kde_interval == 0:
            samples = np.random.randn(kde_samples).astype(np.float32)
            board.log_kernel_density(
                "mix2/kde",
                values=samples,
                num_points=192,
                percentile_clip=(2.0, 98.0),
                metadata={"step": step_index},
            )
    elapsed = time.perf_counter() - start
    objects = steps + tensors_emitted + kdes_emitted
    values = (
        steps * scalars_per_step
        + tensors_emitted * tensor_size
        + kdes_emitted * kde_samples
    )
    return PhaseResult("scalar+tensor+kde", objects, values, elapsed)


# ---------------------- CLI + runner ----------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KohakuBoard non-blocking benchmark suite"
    )
    parser.add_argument("--base-dir", type=Path, default=Path.cwd() / "kohakuboard")
    parser.add_argument("--board-id", type=str, default="non_blocking_bench")
    parser.add_argument(
        "--clean-start",
        action="store_true",
        help="Remove board directory before running benchmarks",
    )
    parser.add_argument(
        "--clean", dest="clean_start", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--purge-after-phase",
        action="store_true",
        help="Delete board directory after each phase",
    )
    parser.add_argument("--scalar-steps", type=int, default=4000)
    parser.add_argument("--scalars-per-step", type=int, default=32)
    parser.add_argument("--histograms", type=int, default=400)
    parser.add_argument("--hist-values", type=int, default=10000)
    parser.add_argument("--hist-bins", type=int, default=128)
    parser.add_argument("--media-count", type=int, default=200)
    parser.add_argument("--media-width", type=int, default=224)
    parser.add_argument("--media-height", type=int, default=224)
    parser.add_argument("--tensor-logs", type=int, default=200)
    parser.add_argument("--tensor-size", type=int, default=131072)
    parser.add_argument("--kde-logs", type=int, default=200)
    parser.add_argument("--kde-samples", type=int, default=4096)
    parser.add_argument("--mix-steps", type=int, default=2000)
    parser.add_argument("--mix-tensor-interval", type=int, default=20)
    parser.add_argument("--mix2-steps", type=int, default=2000)
    parser.add_argument("--mix2-tensor-interval", type=int, default=15)
    parser.add_argument("--mix2-kde-interval", type=int, default=25)
    # Skip flags
    parser.add_argument("--skip-scalar", action="store_true")
    parser.add_argument("--skip-hist", action="store_true")
    parser.add_argument("--skip-media", action="store_true")
    parser.add_argument("--skip-tensor", action="store_true")
    parser.add_argument("--skip-kde", action="store_true")
    parser.add_argument("--skip-mix-scalar-tensor", action="store_true")
    parser.add_argument("--skip-mix-full", action="store_true")
    return parser.parse_args()


def run_phase(
    args: argparse.Namespace,
    title: str,
    description: str,
    runner,
    *runner_args,
) -> PhaseResult:
    board_dir = args.base_dir / args.board_id
    print(f"\n{title}\n  {description}")

    board = build_board(args)
    try:
        result = runner(board, *runner_args)
    finally:
        board.finish()

    if args.purge_after_phase:
        remove_board_dir(board_dir)

    return result


def print_results(results: list[PhaseResult]) -> None:
    if not results:
        print("\nNo phases were executed.")
        return
    header = f"{'Phase':<22}{'Objects':>12}{'Values':>15}{'Time(s)':>10}{'Values/s':>15}{'Objects/s':>15}"
    print("\n" + header)
    print("-" * len(header))
    for res in results:
        print(
            f"{res.name:<22}"
            f"{res.objects:>12,}"
            f"{res.values:>15,}"
            f"{res.elapsed:>10.2f}"
            f"{res.values_per_sec:>15,.0f}"
            f"{res.objects_per_sec:>15,.0f}"
        )

    total_values = sum(r.values for r in results)
    total_objects = sum(r.objects for r in results)
    total_time = sum(r.elapsed for r in results)
    print("-" * len(header))
    total_value_rate = total_values / total_time if total_time > 0 else 0.0
    total_object_rate = total_objects / total_time if total_time > 0 else 0.0
    print(
        f"{'TOTAL':<22}"
        f"{total_objects:>12,}"
        f"{total_values:>15,}"
        f"{total_time:>10.2f}"
        f"{total_value_rate:>15,.0f}"
        f"{total_object_rate:>15,.0f}"
    )


def describe_setup(args: argparse.Namespace) -> None:
    print("\nBenchmark configuration:")
    print(
        f"  Scalars        : {args.scalar_steps} steps × {args.scalars_per_step} values"
    )
    print(
        f"  Histograms     : {args.histograms} hist × {args.hist_values} values (bins={args.hist_bins})"
    )
    print(
        f"  Media          : {args.media_count} images @ {args.media_width}x{args.media_height}"
    )
    print(f"  Tensors        : {args.tensor_logs} tensors × {args.tensor_size} values")
    print(f"  KDE            : {args.kde_logs} logs × {args.kde_samples} samples")
    print(
        f"  Mix A          : {args.mix_steps} scalar steps, tensor every {args.mix_tensor_interval}"
    )
    print(
        f"  Mix B          : {args.mix2_steps} scalar steps, tensor every {args.mix2_tensor_interval}, "
        f"KDE every {args.mix2_kde_interval}"
    )


def main() -> None:
    args = parse_args()
    args.base_dir = args.base_dir.resolve()
    if args.clean_start:
        remove_board_dir(args.base_dir / args.board_id)
    describe_setup(args)

    results: list[PhaseResult] = []

    phases: list[tuple[str, str, bool, callable, tuple]] = [
        (
            "[1/7] Scalar Only",
            f"{args.scalar_steps} steps × {args.scalars_per_step} metrics",
            args.skip_scalar,
            run_scalar_only,
            (
                args.scalar_steps,
                args.scalars_per_step,
            ),
        ),
        (
            "[2/7] Histogram Only",
            f"{args.histograms} histograms × {args.hist_values} values (bins={args.hist_bins})",
            args.skip_hist,
            run_hist_only,
            (
                args.histograms,
                args.hist_values,
                args.hist_bins,
            ),
        ),
        (
            "[3/7] Media Only",
            f"{args.media_count} images @ {args.media_width}×{args.media_height}",
            args.skip_media,
            run_media_only,
            (
                args.media_count,
                args.media_width,
                args.media_height,
            ),
        ),
        (
            "[4/7] Tensor Only",
            f"{args.tensor_logs} tensors × {args.tensor_size} values",
            args.skip_tensor,
            run_tensor_only,
            (
                args.tensor_logs,
                args.tensor_size,
                "tensor/only",
            ),
        ),
        (
            "[5/7] KDE Only",
            f"{args.kde_logs} KDEs × {args.kde_samples} samples",
            args.skip_kde,
            run_kde_only,
            (
                args.kde_logs,
                args.kde_samples,
                "kde/only",
            ),
        ),
        (
            "[6/7] Scalar + Tensor",
            f"{args.mix_steps} steps, tensor every {args.mix_tensor_interval}",
            args.skip_mix_scalar_tensor,
            run_scalar_tensor_mix,
            (
                args.mix_steps,
                args.scalars_per_step,
                args.tensor_size,
                args.mix_tensor_interval,
                "mix/tensor",
            ),
        ),
        (
            "[7/7] Scalar + Tensor + KDE",
            f"{args.mix2_steps} steps, tensor every {args.mix2_tensor_interval}, KDE every {args.mix2_kde_interval}",
            args.skip_mix_full,
            run_scalar_tensor_kde_mix,
            (
                args.mix2_steps,
                args.scalars_per_step,
                args.tensor_size,
                args.mix2_tensor_interval,
                args.kde_samples,
                args.mix2_kde_interval,
            ),
        ),
    ]

    for title, desc, skip, fn, fn_args in phases:
        if skip:
            print(f"\n{title}\n  (skipped)")
            continue
        res = run_phase(args, title, desc, fn, *fn_args)
        print(
            f"  Result: {res.values:,} values ({res.objects:,} objects) in "
            f"{res.elapsed:.2f}s -> {res.values_per_sec:,.0f} values/s, "
            f"{res.objects_per_sec:,.0f} objects/s"
        )
        results.append(res)

    print_results(results)
    print("\nFinished. Pass --keep-data to inspect the generated boards afterwards.")


if __name__ == "__main__":
    main()
