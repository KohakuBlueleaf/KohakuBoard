#!/usr/bin/env python3
"""Generate PNG favicons from SVG logo"""

import subprocess
import sys
from pathlib import Path


def generate_png_with_inkscape(svg_path, output_path, size):
    """Generate PNG using Inkscape CLI (if available)"""
    try:
        subprocess.run(
            [
                "inkscape",
                str(svg_path),
                "--export-filename",
                str(output_path),
                "--export-width",
                str(size),
                "--export-height",
                str(size),
            ],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def generate_png_with_cairosvg(svg_path, output_path, size):
    """Generate PNG using cairosvg library"""
    try:
        import cairosvg

        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(output_path),
            output_width=size,
            output_height=size,
        )
        return True
    except ImportError:
        return False


def main():
    # Paths
    script_dir = Path(__file__).parent
    svg_path = script_dir / "logo-square.svg"

    # Favicon sizes to generate
    sizes = [
        (16, "favicon-16x16.png"),
        (32, "favicon-32x32.png"),
        (180, "apple-touch-icon.png"),
        (192, "android-chrome-192x192.png"),
        (512, "android-chrome-512x512.png"),
    ]

    # Output directories
    ui_public = script_dir.parent / "src" / "kohaku-hub-ui" / "public"
    admin_public = script_dir.parent / "src" / "kohaku-hub-admin" / "public"

    print(f"Generating PNG favicons from {svg_path}...")
    print()

    # Try cairosvg first, then inkscape
    method = None
    if generate_png_with_cairosvg(svg_path, script_dir / "test.png", 16):
        method = "cairosvg"
        (script_dir / "test.png").unlink()
        print("✓ Using cairosvg")
    elif generate_png_with_inkscape(svg_path, script_dir / "test.png", 16):
        method = "inkscape"
        (script_dir / "test.png").unlink()
        print("✓ Using Inkscape")
    else:
        print("❌ Error: Neither cairosvg nor Inkscape is available")
        print()
        print("Please install one of the following:")
        print("  1. pip install cairosvg")
        print("  2. Install Inkscape: https://inkscape.org/")
        print()
        print("Alternatively, open images/generate-png-favicons.html in a browser")
        return 1

    print()

    # Generate favicons for both directories
    for output_dir in [ui_public, admin_public]:
        print(f"Generating favicons for {output_dir.name}...")

        for size, filename in sizes:
            output_path = output_dir / filename

            if method == "cairosvg":
                success = generate_png_with_cairosvg(svg_path, output_path, size)
            else:
                success = generate_png_with_inkscape(svg_path, output_path, size)

            if success:
                print(f"  ✓ {filename} ({size}x{size})")
            else:
                print(f"  ❌ Failed to generate {filename}")

        print()

    print("✓ All favicons generated successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
