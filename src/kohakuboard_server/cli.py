"""KohakuBoard Server CLI - kobo-serve command

Provides the full-featured server with authentication and database.
"""

import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

import click
import uvicorn


@click.command()
@click.option("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
@click.option("--port", default=48889, help="Server port (default: 48889)")
@click.option(
    "--data-dir",
    default="./kohakuboard",
    help="Board data directory (default: ./kohakuboard)",
)
@click.option(
    "--db",
    default="sqlite:///kohakuboard.db",
    help="Database URL (default: sqlite:///kohakuboard.db)",
)
@click.option(
    "--db-backend",
    type=click.Choice(["sqlite", "postgres"]),
    default="sqlite",
    help="Database backend (default: sqlite)",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload for development",
)
@click.option(
    "--workers",
    default=1,
    help="Number of worker processes (default: 1, use 4+ for production)",
)
@click.option(
    "--session-secret",
    help="Session secret for authentication (required in production)",
)
@click.option(
    "--browser",
    is_flag=True,
    help="Open browser automatically",
)
@click.option(
    "--no-auth",
    is_flag=True,
    help="Disable authentication (for testing only, NOT for production)",
)
def cli(
    host,
    port,
    data_dir,
    db,
    db_backend,
    reload,
    workers,
    session_secret,
    browser,
    no_auth,
):
    """KohakuBoard Server - Full-featured ML experiment tracking server

    Run with authentication, database, and multi-user support.

    Examples:
        # Development with auto-reload (SQLite)
        kobo-serve --reload

        # Production with PostgreSQL
        kobo-serve --db postgresql://user:pass@localhost/kohakuboard \\
                   --db-backend postgres \\
                   --workers 4 \\
                   --session-secret $(openssl rand -hex 32)

        # Custom configuration
        kobo-serve --port 8080 \\
                   --data-dir /var/kohakuboard \\
                   --db sqlite:///data/board.db \\
                   --workers 2
    """
    data_dir_path = Path(data_dir).resolve()
    data_dir_path.mkdir(parents=True, exist_ok=True)

    # Set environment for server
    os.environ["KOHAKU_BOARD_HOST"] = host
    os.environ["KOHAKU_BOARD_PORT"] = str(port)
    os.environ["KOHAKU_BOARD_DATA_DIR"] = str(data_dir_path)
    os.environ["KOHAKU_BOARD_DB_BACKEND"] = db_backend
    os.environ["KOHAKU_BOARD_DATABASE_URL"] = db
    os.environ["KOHAKU_BOARD_BASE_URL"] = f"http://localhost:{port}"
    os.environ["KOHAKU_BOARD_NO_AUTH"] = "true" if no_auth else "false"

    # Session secret (required for production)
    if session_secret:
        os.environ["KOHAKU_BOARD_AUTH_SESSION_SECRET"] = session_secret
    elif workers > 1 and not reload:
        click.echo(
            "⚠️  Warning: Using default session secret in multi-worker mode", err=True
        )
        click.echo("   Generate one with: openssl rand -hex 32", err=True)
        click.echo()

    # Auth config defaults
    os.environ.setdefault("KOHAKU_BOARD_AUTH_REQUIRE_EMAIL_VERIFICATION", "false")
    os.environ.setdefault("KOHAKU_BOARD_AUTH_INVITATION_ONLY", "false")

    click.echo("🚀 Starting KohakuBoard Server")
    click.echo(f"📁 Data directory: {data_dir_path}")
    click.echo(f"💾 Database: {db}")
    click.echo(f"🌐 Server URL: http://localhost:{port}")
    if no_auth:
        click.echo(f"👥 Authentication: DISABLED (test mode)")
        click.echo(f"⚠️  WARNING: No-auth mode is for TESTING ONLY!")
    else:
        click.echo(f"👥 Authentication: Enabled")
    if reload:
        click.echo(f"🔄 Auto-reload: Enabled (development)")
    if workers > 1:
        click.echo(f"⚡ Workers: {workers}")
    click.echo()

    # Open browser after delay
    if browser:

        def open_browser():
            time.sleep(2)
            click.echo(f"🔗 Opening browser at http://localhost:{port}")
            try:
                webbrowser.open(f"http://localhost:{port}")
            except Exception as e:
                click.echo(f"⚠️  Could not open browser: {e}", err=True)

        thread = threading.Thread(target=open_browser, daemon=True)
        thread.start()

    # Run uvicorn
    try:
        if reload or workers == 1:
            uvicorn.run(
                "kohakuboard_server.main:app",
                host=host,
                port=port,
                reload=reload,
                log_level="info",
            )
        else:
            uvicorn.run(
                "kohakuboard_server.main:app",
                host=host,
                port=port,
                workers=workers,
                log_level="info",
            )
    except KeyboardInterrupt:
        click.echo("\n👋 Server stopped")
        sys.exit(0)


if __name__ == "__main__":
    cli()
