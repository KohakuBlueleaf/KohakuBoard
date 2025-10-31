"""KohakuBoard All Features Demo

This example demonstrates all logging capabilities:
- Scalars (metrics)
- Media (images, videos, audio)
- Tables (with and without embedded media)
- Histograms (raw values and precomputed)
- Unified logging API

NEW in v0.3.0:
- SQLite KV media storage (no filesystem overhead, dynamic size)
- SharedMemory for histogram data transfer (zero-copy)
- mp.Queue for better performance

Usage:
    # Default: syncs to local server (localhost:48889)
    python all_features_demo.py

    # Local only (no sync)
    python all_features_demo.py --remote-url ""

    # Production server
    python all_features_demo.py --remote-url https://board.example.com \
                                --remote-token YOUR_TOKEN \
                                --remote-project all_features
"""

import argparse
import numpy as np
import time
from pathlib import Path

from kohakuboard.client import Board, Media, Table, Histogram


def generate_sample_image(step: int, size=(64, 64, 3)):
    """Generate a colorful sample image"""
    # Create gradient pattern
    height, width, channels = size
    img = np.zeros(size, dtype=np.uint8)

    # Red gradient (horizontal)
    img[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)
    # Green gradient (vertical)
    img[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1)
    # Blue based on step
    img[:, :, 2] = (step * 10) % 256

    return img


def generate_sample_audio(duration_sec=1.0, sample_rate=44100):
    """Generate a simple sine wave audio"""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec))
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t)
    # Convert to 16-bit PCM
    audio = (audio * 32767).astype(np.int16)
    return audio, sample_rate


def demo_scalars(board: Board):
    """Demo 1: Basic scalar logging"""
    print("\n" + "="*60)
    print("DEMO 1: Scalar Logging")
    print("="*60)

    for i in range(10):
        board.step()  # Increment global step

        # Log multiple scalars at once
        loss = 1.0 / (i + 1)  # Decreasing loss
        accuracy = 1.0 - loss  # Increasing accuracy

        board.log(
            loss=loss,
            accuracy=accuracy,
            learning_rate=0.001 * (0.95 ** i),  # Decaying LR
        )

        print(f"Step {board._global_step}: loss={loss:.4f}, acc={accuracy:.4f}")
        time.sleep(0.05)

    print("✓ Logged 10 steps with 3 scalar metrics each")


def demo_namespace_scalars(board: Board):
    """Demo 2: Namespace-based organization"""
    print("\n" + "="*60)
    print("DEMO 2: Namespace-based Scalars (train/ and val/ tabs)")
    print("="*60)

    for i in range(5):
        board.step()

        # Training metrics (train/ tab)
        train_loss = np.random.random() * 0.5
        train_acc = 0.6 + np.random.random() * 0.3

        # Validation metrics (val/ tab)
        val_loss = train_loss * 0.8
        val_acc = train_acc * 1.1

        # Log all at once - they'll be grouped by namespace prefix
        board.log(**{
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
        })

        print(f"Step {board._global_step}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        time.sleep(0.05)

    print("✓ Logged metrics with namespace organization (train/, val/)")


def demo_media_images(board: Board):
    """Demo 3: Image logging"""
    print("\n" + "="*60)
    print("DEMO 3: Image Logging (SQLite KV storage)")
    print("="*60)

    for i in range(3):
        board.step()

        # Generate sample image
        img = generate_sample_image(i, size=(128, 128, 3))

        # Log single image
        board.log(
            sample_image=Media(img, media_type="image", caption=f"Step {i} visualization")
        )

        print(f"Step {board._global_step}: Logged image to SQLite KV (128x128x3)")
        time.sleep(0.05)

    print("✓ Logged 3 images (stored in SQLite KV, not filesystem)")


def demo_media_multiple(board: Board):
    """Demo 4: Multiple images in one log call"""
    print("\n" + "="*60)
    print("DEMO 4: Multiple Images (Grid/Gallery)")
    print("="*60)

    board.step()

    # Generate a batch of images
    images = [generate_sample_image(i, size=(64, 64, 3)) for i in range(8)]

    # Log multiple images at once
    for idx, img in enumerate(images):
        board.log(**{
            f"gallery/image_{idx}": Media(img, media_type="image")
        })
        time.sleep(0.05)

    print(f"Step {board._global_step}: Logged 8 images in gallery/ namespace")
    print("✓ All images share the same step (no step inflation)")


def demo_tables_basic(board: Board):
    """Demo 5: Basic table logging"""
    print("\n" + "="*60)
    print("DEMO 5: Basic Table Logging")
    print("="*60)

    board.step()

    # Create table data
    experiment_results = [
        {"model": "ResNet-18", "params": "11M", "accuracy": 0.892, "time": 45.2},
        {"model": "ResNet-34", "params": "21M", "accuracy": 0.913, "time": 72.1},
        {"model": "ResNet-50", "params": "25M", "accuracy": 0.925, "time": 89.3},
        {"model": "ConvNeXt-T", "params": "28M", "accuracy": 0.941, "time": 95.7},
    ]

    # Log table
    board.log(
        model_comparison=Table(experiment_results)
    )

    print(f"Step {board._global_step}: Logged table with {len(experiment_results)} rows")
    print("✓ Table columns: model, params, accuracy, time")
    time.sleep(0.05)


def demo_tables_with_media(board: Board):
    """Demo 6: Tables with embedded media"""
    print("\n" + "="*60)
    print("DEMO 6: Tables with Embedded Media")
    print("="*60)

    board.step()

    # Create table with images
    predictions_table = []
    for i in range(4):
        img = generate_sample_image(i, size=(32, 32, 3))
        predictions_table.append({
            "sample_id": i,
            "image": Media(img, media_type="image"),
            "ground_truth": f"class_{i % 3}",
            "prediction": f"class_{(i + 1) % 3}",
            "confidence": 0.8 + np.random.random() * 0.2,
            "correct": "✓" if i % 2 == 0 else "✗",
        })

    # Log table with embedded media
    board.log_table("predictions", Table(predictions_table))

    print(f"Step {board._global_step}: Logged table with {len(predictions_table)} embedded images")
    print("✓ Media stored in SQLite KV, referenced by ID in table")
    time.sleep(0.05)


def demo_histograms_raw(board: Board):
    """Demo 7: Histogram logging (raw values)"""
    print("\n" + "="*60)
    print("DEMO 7: Histogram Logging - Raw Values (SharedMemory)")
    print("="*60)

    for i in range(3):
        board.step()

        # Generate random data
        values = np.random.randn(10000) * (i + 1)  # Increasing variance

        # Log histogram - will be computed on writer side
        # Data transferred via SharedMemory (zero-copy)
        board.log(
            **{"weights/distribution": Histogram(values, num_bins=64, precision="exact")}
        )

        print(f"Step {board._global_step}: Logged histogram (10k values via SharedMemory)")

    print("✓ Raw values sent via SharedMemory, computed in writer process")


def demo_histograms_precomputed(board: Board):
    """Demo 8: Precomputed histograms"""
    print("\n" + "="*60)
    print("DEMO 8: Precomputed Histograms (SharedMemory)")
    print("="*60)

    for i in range(3):
        board.step()

        # Generate random data
        values = np.random.randn(50000)

        # Precompute histogram bins/counts
        hist = Histogram(values, num_bins=128, precision="compact")
        hist.compute_bins()  # Compute now, before sending

        # Log precomputed histogram
        # Bins and counts arrays transferred via SharedMemory
        board.log(
            **{"gradients/layer1": hist}
        )

        print(f"Step {board._global_step}: Logged precomputed histogram (bins+counts via SharedMemory)")

    print("✓ Precomputed bins/counts sent via SharedMemory")


def demo_histograms_multi(board: Board):
    """Demo 9: Multiple histograms in one call"""
    print("\n" + "="*60)
    print("DEMO 9: Multiple Histograms (Batch Logging)")
    print("="*60)

    board.step()

    # Simulate neural network layers
    layers = ["layer1", "layer2", "layer3", "fc"]
    histogram_data = {}

    for layer in layers:
        # Generate random gradient-like values
        values = np.random.randn(5000) * 0.01

        # Create histogram (both raw and precomputed work)
        histogram_data[f"gradients/{layer}"] = Histogram(
            values, num_bins=64, precision="compact"
        ).compute_bins()

    # Log all histograms at once - single step, single queue message!
    board.log(**histogram_data)

    print(f"Step {board._global_step}: Logged {len(layers)} histograms in one call")
    print("✓ All histograms share same step (prevents step inflation)")


def demo_unified_logging(board: Board):
    """Demo 10: Unified API - Mix all data types"""
    print("\n" + "="*60)
    print("DEMO 10: Unified Logging - Mix All Data Types")
    print("="*60)

    for i in range(3):
        board.step()

        # Generate all data types
        img = generate_sample_image(i, size=(64, 64, 3))
        values = np.random.randn(1000)

        metrics_table = [
            {"metric": "precision", "value": 0.85 + i * 0.05},
            {"metric": "recall", "value": 0.82 + i * 0.05},
            {"metric": "f1", "value": 0.83 + i * 0.05},
        ]

        # Log EVERYTHING in one call - all share the same step!
        board.log(
            # Scalars
            loss=1.0 / (i + 1),
            accuracy=0.7 + i * 0.1,

            # Media
            visualization=Media(img, media_type="image"),

            # Histogram
            **{"weights/dist": Histogram(values, num_bins=32)},

            # Table
            metrics_summary=Table(metrics_table),
        )

        print(f"Step {board._global_step}: Logged scalars + image + histogram + table")

    print("✓ All data types logged together at same step")


def demo_performance_large_histograms(board: Board):
    """Demo 11: Performance test with large histograms"""
    print("\n" + "="*60)
    print("DEMO 11: Performance - Large Histograms (1M values)")
    print("="*60)

    for i in range(3):
        board.step()

        # Generate LARGE histogram (1 million values)
        values = np.random.randn(1_000_000)

        start_time = time.time()

        # Log large histogram - SharedMemory makes this fast!
        board.log(
            **{"large_data/histogram": Histogram(values, num_bins=256, precision="exact")}
        )

        elapsed = time.time() - start_time

        print(f"Step {board._global_step}: Logged 1M value histogram in {elapsed*1000:.2f}ms (SharedMemory)")

    print("✓ SharedMemory enables zero-copy transfer of large arrays")


def main():
    """Run all demos"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="KohakuBoard All Features Demo")
    parser.add_argument(
        "--remote-url",
        type=str,
        default="http://127.0.0.1:48889",
        help="Remote server URL (e.g., https://board.example.com)",
    )
    parser.add_argument(
        "--remote-token",
        type=str,
        default="no-auth",
        help="Authentication token for remote server",
    )
    parser.add_argument(
        "--remote-project",
        type=str,
        default="all_features",
        help="Project name on remote server (default: all_features)",
    )
    parser.add_argument(
        "--sync-interval",
        type=int,
        default=1,
        help="Sync interval in seconds (default: 10)",
    )
    args = parser.parse_args()

    # Enable sync if remote URL is provided
    sync_enabled = args.remote_url is not None

    print("\n" + "="*60)
    print("KohakuBoard All Features Demo")
    print("="*60)
    print("\nThis demo showcases:")
    print("  • SQLite KV media storage (replaces filesystem)")
    print("  • SharedMemory histogram transfer (zero-copy)")
    print("  • mp.Queue (better performance than manager.Queue)")
    print("  • Unified logging API (all data types together)")

    if sync_enabled:
        print(f"\nRemote sync enabled:")
        print(f"  URL: {args.remote_url}")
        print(f"  Project: {args.remote_project}")
        print(f"  Sync interval: {args.sync_interval}s")
    else:
        print("\nLocal logging only (use --remote-url to enable sync)")

    # Create board
    board = Board(
        name="all_features_demo",
        config={
            "demo": "comprehensive",
            "features": ["scalars", "media", "tables", "histograms"],
            "storage": "hybrid (Lance + SQLite + SQLite KV)",
        },
        backend="hybrid",
        remote_url=args.remote_url,
        remote_token=args.remote_token,
        remote_project=args.remote_project,
        sync_enabled=sync_enabled,
        sync_interval=args.sync_interval,
    )

    print(f"\nBoard created: {board.board_id}")
    print(f"Local storage: {board.board_dir}")
    if sync_enabled:
        print(f"Remote sync: Active")

    # Run all demos
    demo_scalars(board)
    demo_namespace_scalars(board)
    demo_media_images(board)
    demo_media_multiple(board)
    demo_tables_basic(board)
    demo_tables_with_media(board)
    demo_histograms_raw(board)
    demo_histograms_precomputed(board)
    demo_histograms_multi(board)
    demo_unified_logging(board)
    demo_performance_large_histograms(board)

    # Summary
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print(f"\nLocal storage: {board.board_dir}")
    print(f"Total steps: {board._global_step}")
    print("\nLogged data:")
    print("  • Scalars: loss, accuracy, learning_rate, etc.")
    print("  • Namespaces: train/, val/, gallery/, gradients/, weights/")
    print("  • Media: Images stored in SQLite KV (media/blobs.db)")
    print("  • Tables: With and without embedded media")
    print("  • Histograms: Raw and precomputed (via SharedMemory)")
    print("\nView results:")
    print(f"  Local: python -m kohakuboard.main → http://localhost:48889")
    if sync_enabled:
        print(f"  Remote: {args.remote_url}/projects/{args.remote_project}")


if __name__ == "__main__":
    main()
