"""KohakuBoard CLI - Command line interface for board management

Provides two main commands:
- kobo open: Open local board directory with web server
- kobo sync: Upload local board to remote server
"""

import json
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

import click
import uvicorn


@click.group()
def cli():
    """KohakuBoard CLI - ML experiment tracking

    Manage your ML training boards locally or sync to remote server.

    Examples:
        kobo open ./kohakuboard
        kobo sync ./kohakuboard/20250115_103045_a1b2c3d4 -r https://board.example.com -p my-project
    """
    pass


@cli.command()
@click.argument("folder", type=click.Path(exists=True))
@click.option("--port", default=48889, help="Server port (default: 48889)")
@click.option("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
@click.option("--browser", is_flag=True, help="Open browser automatically")
def open(folder, port, host, reload, browser):
    """Open local board folder in browser

    Starts a local web server to browse boards in the specified folder.
    The server runs in local mode with no authentication required.

    Examples:
        kobo open ./kohakuboard
        kobo open /path/to/experiments --port 8080
        kobo open ./boards --reload --browser
    """
    folder_path = Path(folder).resolve()

    # Set environment for local mode
    os.environ["KOHAKU_BOARD_MODE"] = "local"
    os.environ["KOHAKU_BOARD_DATA_DIR"] = str(folder_path)
    os.environ["KOHAKU_BOARD_PORT"] = str(port)
    os.environ["KOHAKU_BOARD_HOST"] = host

    click.echo("🚀 Starting KohakuBoard server (local mode)")
    click.echo(f"📁 Board directory: {folder_path}")
    click.echo(f"🌐 Server URL: http://localhost:{port}")
    if reload:
        click.echo(f"🔄 Auto-reload: Enabled")
    click.echo()

    # Open browser after delay
    if browser:

        def open_browser():
            time.sleep(1.5)  # Wait for server to start
            click.echo(f"🔗 Opening browser at http://localhost:{port}")
            try:
                webbrowser.open(f"http://localhost:{port}")
            except Exception as e:
                click.echo(f"⚠️  Could not open browser: {e}", err=True)

        thread = threading.Thread(target=open_browser, daemon=True)
        thread.start()

    # Run uvicorn
    try:
        uvicorn.run(
            "kohakuboard.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    except KeyboardInterrupt:
        click.echo("\n👋 Server stopped")
        sys.exit(0)


@cli.command()
@click.argument("folder", type=click.Path(exists=True))
@click.option(
    "--remote",
    "-r",
    required=True,
    help="Remote server URL (e.g., https://board.example.com)",
)
@click.option(
    "--token",
    "-t",
    envvar="KOBO_TOKEN",
    help="Authentication token (or set KOBO_TOKEN env var)",
)
@click.option(
    "--project",
    "-p",
    required=True,
    help="Project name on remote server",
)
@click.option(
    "--private/--public",
    default=True,
    help="Board visibility (default: private)",
)
def sync(folder, remote, token, project, private):
    """Sync local board to remote server

    Uploads DuckDB file and media files to remote KohakuBoard server.
    Requires authentication token (use --token or set KOBO_TOKEN env var).

    Examples:
        export KOBO_TOKEN=your_token_here
        kobo sync ./kohakuboard/20250115_103045_a1b2c3d4 \\
            --remote https://board.example.com \\
            --project resnet-training \\
            --private

        kobo sync ./boards/my_run -r https://board.example.com -p my-project --public
    """
    # Check token
    if not token:
        click.echo("❌ Error: No authentication token provided", err=True)
        click.echo("   Use --token or set KOBO_TOKEN environment variable", err=True)
        sys.exit(1)

    board_dir = Path(folder).resolve()

    # Check board directory
    if not board_dir.exists():
        click.echo(f"❌ Error: Board directory not found: {folder}", err=True)
        sys.exit(1)

    # Read metadata
    metadata_file = board_dir / "metadata.json"
    if not metadata_file.exists():
        click.echo(f"❌ Error: metadata.json not found in {folder}", err=True)
        click.echo("   This does not appear to be a valid board directory", err=True)
        sys.exit(1)

    with open(metadata_file) as f:
        metadata = json.load(f)

    run_id = metadata.get("board_id") or board_dir.name
    name = metadata.get("name", run_id)

    # Display info
    click.echo("📤 Syncing board to remote server")
    click.echo(f"   Name: {name}")
    click.echo(f"   Run ID: {run_id}")
    click.echo(f"   Project: {project}")
    click.echo(f"   Remote: {remote}")
    click.echo(f"   Visibility: {'private' if private else 'public'}")
    click.echo()

    # Import sync client
    from kohakuboard.client.sync_client import SyncClient

    client = SyncClient(remote, token)

    # Upload with progress
    try:
        with click.progressbar(
            length=100,
            label="Uploading",
            show_percent=True,
            show_eta=True,
        ) as bar:

            def update_progress(progress):
                bar.update(progress - bar.pos)

            result = client.sync_board(
                board_dir=board_dir,
                project=project,
                private=private,
                progress_callback=update_progress,
            )

        # Success
        click.echo()
        click.echo("✅ Sync completed successfully!")
        click.echo(f"   URL: {remote}{result['url']}")
        click.echo(f"   Total size: {result['total_size'] / 1024 / 1024:.2f} MB")

    except FileNotFoundError as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        sys.exit(1)

    except Exception as e:
        click.echo(f"\n❌ Sync failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
