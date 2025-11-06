"""KohakuBoard Quick Start Example

A minimal example showing the most common use cases.
For comprehensive examples, see all_features_demo.py

Highlights:
- Scalars, media, tables, histograms
- Tensor snapshots saved to KohakuVault
- Kernel density coefficients computed automatically from raw samples

Usage:
    # Default: syncs to local server (localhost:48889)
    python quick_start.py

    # Local only (no sync)
    python quick_start.py --remote-url ""

    # Production server
    python quick_start.py --remote-url https://board.example.com \
                          --remote-token YOUR_TOKEN \
                          --remote-project my_project
"""

import argparse
import numpy as np
from kohakuboard.client import Board, Media, Table, Histogram


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="KohakuBoard Quick Start Demo")
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
        default="quick_start",
        help="Project name on remote server (default: quick_start)",
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

    if sync_enabled:
        print(f"Remote sync enabled:")
        print(f"  URL: {args.remote_url}")
        print(f"  Project: {args.remote_project}")
        print(f"  Sync interval: {args.sync_interval}s")
    else:
        print("Local logging only (use --remote-url to enable sync)")

    # 1. Create a board
    board = Board(
        name="quick_start_demo",
        config={"model": "ResNet", "lr": 0.001, "batch_size": 32},
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

    # 2. Training loop simulation
    for epoch in range(5):
        print(f"\nEpoch {epoch + 1}/5")

        # Simulate training
        for step in range(10):
            board.step()  # Increment global step

            # Log training metrics (scalars)
            loss = 2.0 / (board._global_step + 1)  # Decreasing loss
            accuracy = 1.0 - loss / 2.0  # Increasing accuracy

            board.log(
                loss=loss,
                accuracy=accuracy,
                learning_rate=0.001 * (0.95**epoch),
            )

        # End of epoch: log validation metrics and extra data
        board.step()

        # Validation metrics (with namespace)
        board.log(
            **{
                "val/loss": loss * 0.8,
                "val/accuracy": accuracy * 1.1,
            }
        )

        # Log a sample image (SQLite KV storage)
        sample_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        board.log(sample_visualization=Media(sample_img, media_type="image"))

        # Log parameter histogram (SharedMemory transfer)
        weights = np.random.randn(10000) * 0.01
        board.log(
            **{"params/layer1": Histogram(weights, num_bins=64, precision="exact")}
        )

        # Log tensor snapshot (stored in KohakuVault tensor store)
        weights_matrix = weights.reshape(100, 100).astype(np.float32)
        board.log_tensor(
            "params/layer1_snapshot",
            weights_matrix,
            metadata={"epoch": epoch + 1, "shape": list(weights_matrix.shape)},
        )

        # Log kernel density coefficients directly from raw values
        board.log_kernel_density(
            "params/layer1_kde",
            values=weights,
            num_points=128,
            percentile_clip=(1.0, 99.0),
        )

        # Log metrics table
        metrics_data = [
            {"metric": "precision", "value": 0.85 + epoch * 0.03},
            {"metric": "recall", "value": 0.82 + epoch * 0.03},
            {"metric": "f1_score", "value": 0.835 + epoch * 0.03},
        ]
        board.log_table("epoch_metrics", Table(metrics_data))

        print(f"  loss: {loss:.4f}, val_loss: {loss * 0.8:.4f}")

    # 3. Finish (optional - auto-called on exit)
    board.finish()

    print(f"\n✓ Training complete!")
    print(f"\nView results:")
    print(f"  Local: python -m kohakuboard.main → http://localhost:48889")
    if sync_enabled:
        print(f"  Remote: {args.remote_url}/projects/{args.remote_project}")


if __name__ == "__main__":
    main()
