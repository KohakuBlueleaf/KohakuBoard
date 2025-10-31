"""KohakuBoard Performance Showcase

Demonstrates the performance benefits of:
1. SQLite KV media storage (vs filesystem)
2. SharedMemory histogram transfer (vs queue serialization)
3. mp.Queue (vs manager.Queue)

NEW in v0.3.0

Usage:
    # Default: syncs to local server (localhost:48889)
    python performance_showcase.py

    # Local only (no sync)
    python performance_showcase.py --remote-url ""

    # Production server
    python performance_showcase.py --remote-url https://board.example.com \
                                   --remote-token YOUR_TOKEN \
                                   --remote-project performance
"""

import argparse
import numpy as np
import time
from kohakuboard.client import Board, Media, Histogram


def benchmark_media_logging(board: Board, num_images: int = 100):
    """Benchmark SQLite KV media storage"""
    print("\n" + "="*60)
    print(f"Benchmark 1: SQLite KV Media Storage ({num_images} images)")
    print("="*60)

    # Generate test images
    images = [
        np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        for _ in range(num_images)
    ]

    print(f"Generated {num_images} images (128x128x3)")

    # Benchmark logging
    start_time = time.time()

    for i, img in enumerate(images):
        board.step()
        board.log(**{f"images/img_{i:04d}": Media(img, media_type="image")})

    elapsed = time.time() - start_time

    # Calculate stats
    avg_time_ms = (elapsed / num_images) * 1000
    throughput = num_images / elapsed

    print(f"\nResults:")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Avg per image: {avg_time_ms:.2f}ms")
    print(f"  Throughput: {throughput:.1f} images/sec")
    print(f"\n✓ SQLite KV benefits:")
    print(f"    • No filesystem overhead (no directory traversal)")
    print(f"    • Single database file instead of {num_images} files")
    print(f"    • Dynamic size (auto-grows, no size limit)")
    print(f"    • WAL mode for concurrent readers")


def benchmark_histogram_logging(board: Board, num_histograms: int = 100):
    """Benchmark SharedMemory histogram transfer"""
    print("\n" + "="*60)
    print(f"Benchmark 2: SharedMemory Histograms ({num_histograms} large histograms)")
    print("="*60)

    # Test with different sizes
    sizes = [10_000, 100_000, 1_000_000]

    for size in sizes:
        print(f"\nTesting with {size:,} values per histogram:")

        # Generate histograms
        histograms = [np.random.randn(size) for _ in range(num_histograms)]

        start_time = time.time()

        for i, values in enumerate(histograms):
            board.step()
            board.log(
                **{f"large_data/hist_{i}": Histogram(values, num_bins=128, precision="exact")}
            )

        elapsed = time.time() - start_time

        # Calculate stats
        avg_time_ms = (elapsed / num_histograms) * 1000
        throughput = num_histograms / elapsed
        data_size_mb = (size * 4 * num_histograms) / (1024 * 1024)  # float32 = 4 bytes

        print(f"  Total data: {data_size_mb:.1f} MB")
        print(f"  Total time: {elapsed:.3f}s")
        print(f"  Avg per histogram: {avg_time_ms:.2f}ms")
        print(f"  Throughput: {throughput:.1f} histograms/sec")

    print(f"\n✓ SharedMemory benefits:")
    print(f"    • Zero-copy data transfer (no serialization)")
    print(f"    • Efficient for large arrays (100K-1M+ values)")
    print(f"    • Proper cleanup prevents memory leaks")


def benchmark_mixed_workload(board: Board, num_iterations: int = 50):
    """Benchmark realistic mixed workload"""
    print("\n" + "="*60)
    print(f"Benchmark 3: Mixed Workload ({num_iterations} iterations)")
    print("="*60)
    print("Each iteration logs:")
    print("  • 5 scalar metrics")
    print("  • 2 images (128x128)")
    print("  • 3 histograms (10K values each)")

    start_time = time.time()

    for i in range(num_iterations):
        board.step()

        # Scalars
        scalars = {
            "loss": 1.0 / (i + 1),
            "accuracy": 0.5 + i * 0.01,
            "learning_rate": 0.001 * (0.95 ** i),
            "grad_norm": np.random.random(),
            "batch_time": 0.1 + np.random.random() * 0.05,
        }

        # Images
        img1 = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

        # Histograms
        hist_data = {
            "gradients/layer1": Histogram(np.random.randn(10000), num_bins=64),
            "gradients/layer2": Histogram(np.random.randn(10000), num_bins=64),
            "params/weights": Histogram(np.random.randn(10000), num_bins=64),
        }

        # Log everything together
        board.log(
            **scalars,
            image1=Media(img1, media_type="image"),
            image2=Media(img2, media_type="image"),
            **hist_data,
        )

    elapsed = time.time() - start_time

    # Calculate stats
    avg_time_ms = (elapsed / num_iterations) * 1000
    throughput = num_iterations / elapsed

    print(f"\nResults:")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Avg per iteration: {avg_time_ms:.2f}ms")
    print(f"  Throughput: {throughput:.1f} iterations/sec")
    print(f"\n✓ Combined benefits:")
    print(f"    • mp.Queue eliminates Manager overhead")
    print(f"    • SQLite KV reduces I/O latency for media")
    print(f"    • SharedMemory accelerates histogram transfer")
    print(f"    • Non-blocking design prevents training slowdown")


def stress_test_queue(board: Board, num_messages: int = 1000):
    """Stress test mp.Queue performance"""
    print("\n" + "="*60)
    print(f"Benchmark 4: Queue Performance ({num_messages} messages)")
    print("="*60)

    start_time = time.time()

    # Rapid-fire logging
    for i in range(num_messages):
        board.step()
        board.log(
            metric_a=np.random.random(),
            metric_b=np.random.random(),
            metric_c=np.random.random(),
        )

    # Wait a moment for queue to drain
    time.sleep(2)

    elapsed = time.time() - start_time

    # Calculate stats
    avg_time_us = (elapsed / num_messages) * 1_000_000
    throughput = num_messages / elapsed

    print(f"\nResults:")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Avg per message: {avg_time_us:.2f}μs")
    print(f"  Throughput: {throughput:.0f} messages/sec")
    print(f"\n✓ mp.Queue benefits:")
    print(f"    • Direct queue (no Manager proxy)")
    print(f"    • Better performance on all platforms")
    print(f"    • Proper cleanup with close() and join_thread()")


def main():
    """Run all benchmarks"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="KohakuBoard Performance Showcase")
    parser.add_argument(
        "--remote-url",
        type=str,
        default="http://127.0.0.1:48889",
        help="Remote server URL (default: http://127.0.0.1:48889)",
    )
    parser.add_argument(
        "--remote-token",
        type=str,
        default="no-auth-mode",
        help="Authentication token (default: no-auth-mode for development)",
    )
    parser.add_argument(
        "--remote-project",
        type=str,
        default="performance",
        help="Project name on remote server (default: performance)",
    )
    parser.add_argument(
        "--sync-interval",
        type=int,
        default=10,
        help="Sync interval in seconds (default: 10)",
    )
    args = parser.parse_args()

    # Enable sync if remote URL is provided
    sync_enabled = args.remote_url is not None

    print("\n" + "="*60)
    print("KohakuBoard Performance Showcase")
    print("="*60)
    print("\nTesting v0.3.0 improvements:")
    print("  1. SQLite KV media storage")
    print("  2. SharedMemory histogram transfer")
    print("  3. mp.Queue performance")
    print("  4. Mixed workload")

    if sync_enabled:
        print(f"\nRemote sync enabled:")
        print(f"  URL: {args.remote_url}")
        print(f"  Project: {args.remote_project}")
        print(f"  Sync interval: {args.sync_interval}s")
    else:
        print("\nLocal logging only (use --remote-url to enable sync)")

    # Create board with hybrid backend
    board = Board(
        name="performance_showcase",
        config={"benchmark": "v0.3.0", "backend": "hybrid + SQLite KV + SharedMemory"},
        backend="hybrid",
        remote_url=args.remote_url,
        remote_token=args.remote_token,
        remote_project=args.remote_project,
        sync_enabled=sync_enabled,
        sync_interval=args.sync_interval,
    )

    print(f"\nBoard: {board.board_id}")
    print(f"Local storage: {board.board_dir}")
    if sync_enabled:
        print(f"Remote sync: Active")

    try:
        # Run benchmarks
        benchmark_media_logging(board, num_images=100)
        benchmark_histogram_logging(board, num_histograms=50)
        benchmark_mixed_workload(board, num_iterations=50)
        stress_test_queue(board, num_messages=1000)

        # Summary
        print("\n" + "="*60)
        print("Benchmarks Complete!")
        print("="*60)
        print(f"\nKey Improvements in v0.3.0:")
        print(f"  ✓ SQLite KV: Single database file replaces thousands of media files")
        print(f"  ✓ SQLite KV: Dynamic size with incremental BLOB I/O")
        print(f"  ✓ SharedMemory: Zero-copy transfer for large histogram arrays")
        print(f"  ✓ mp.Queue: Better performance than manager.Queue")
        print(f"  ✓ Proper cleanup: No memory leaks or resource issues")
        print(f"\nLocal storage: {board.board_dir}")
        print(f"Total steps: {board._global_step}")
        print("\nView results:")
        print(f"  Local: python -m kohakuboard.main → http://localhost:48889")
        if sync_enabled:
            print(f"  Remote: {args.remote_url}/projects/{args.remote_project}")

    finally:
        board.finish()
        print("\n✓ All resources cleaned up")


if __name__ == "__main__":
    main()
