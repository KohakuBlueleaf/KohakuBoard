#!/usr/bin/env python3
"""Format KohakuBoard code (frontend and backend).

This script:
1. Formats frontend code
2. Builds frontend (generates auto-imports/routes)
3. Formats backend Python code
"""

import platform
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path | None = None, description: str = ""):
    """Run a command and exit if it fails."""
    if description:
        print(f"\n{'=' * 60}")
        print(f"  {description}")
        print(f"{'=' * 60}")

    print(f"Running: {' '.join(cmd)}")
    if cwd:
        print(f"  in: {cwd}")

    # Use shell=True on Windows (npm is npm.cmd), False on Unix
    use_shell = platform.system() == "Windows"

    result = subprocess.run(cmd, cwd=cwd, shell=use_shell)

    if result.returncode != 0:
        print(f"\n❌ Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"✓ {description or 'Command'} completed")


def main():
    """Main function."""
    root_dir = Path(__file__).parent.parent
    ui_dir = root_dir / "src" / "kohaku-board-ui"
    backend_dir = root_dir / "src" / "kohakuboard"

    print("\nKohakuBoard Code Formatter")
    print("=" * 60)

    # Step 1: Format UI
    run_command(
        ["npm", "run", "format"],
        cwd=ui_dir,
        description="Formatting frontend code",
    )

    # Step 2: Build UI (generates auto-imports/routes)
    run_command(
        ["npm", "run", "build"],
        cwd=ui_dir,
        description="Building frontend (generating auto-imports/routes)",
    )

    # Step 3: Format Python backend
    run_command(
        ["black", str(backend_dir)],
        cwd=root_dir,
        description="Formatting Python code",
    )

    print("\n" + "=" * 60)
    print("✅ All KohakuBoard code formatted successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
