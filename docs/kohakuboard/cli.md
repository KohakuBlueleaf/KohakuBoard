---
title: KohakuBoard CLI
description: Command-line interface for managing boards
icon: i-carbon-terminal
---

# KohakuBoard CLI

Command-line tools for managing ML experiment boards locally and remotely.

---

## Installation

```bash
# From KohakuHub repository
cd /path/to/KohakuHub
pip install -e src/kohakuboard/

# CLI is available as 'kobo' command
kobo --help
```

---

## Commands

### `kobo open` - Browse Local Boards

Open a local board directory with a web server (no authentication required).

**Usage:**

```bash
kobo open <folder> [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `folder` | Path to board directory (e.g., `./kohakuboard`) |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--port` | `48889` | Server port |
| `--host` | `0.0.0.0` | Server host (use `127.0.0.1` for localhost only) |
| `--reload` | `False` | Enable auto-reload for development |
| `--browser` | `False` | Open browser automatically |

**Examples:**

```bash
# Basic usage
kobo open ./kohakuboard

# Custom port
kobo open /path/to/experiments --port 8080

# Development mode with auto-reload
kobo open ./boards --reload --browser

# Localhost only (more secure)
kobo open ./kohakuboard --host 127.0.0.1
```

**What It Does:**

1. Starts a FastAPI server in **local mode** (no authentication)
2. Reads boards from the specified directory
3. Serves the web UI at `http://localhost:48889`
4. Auto-discovers all boards in subdirectories

**Directory Structure:**

```
kohakuboard/                    # Root directory
├── 20250129_150423_abc123/     # Board 1
│   ├── metadata.json
│   ├── data/
│   └── media/
├── 20250129_154512_def456/     # Board 2
│   ├── metadata.json
│   ├── data/
│   └── media/
└── ...
```

**Features:**

- ✅ **No authentication** - Direct file access
- ✅ **Auto-discovery** - Finds all boards automatically
- ✅ **Live reload** - See changes immediately (with `--reload`)
- ✅ **Multi-board** - Browse all experiments in one place

---

### `kobo-serve` - Start Remote Server

Start the authenticated KohakuBoard server (separate `kohakuboard_server` package).

⚠️ **Status:** Still stabilizing. Manual file copy is the supported sync flow today.

**Usage:**

```bash
kobo-serve [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `0.0.0.0` | Server host |
| `--port` | `48889` | Server port |
| `--data-dir` | `./kohakuboard` | Directory containing board folders (same as local mode) |
| `--db` | `sqlite:///kohakuboard.db` | Auth database URL |
| `--db-backend` | `sqlite` | Auth DB backend (`sqlite`, `postgres`) |
| `--reload` | `False` | Enable auto-reload (dev only) |
| `--workers` | `1` | Uvicorn workers |
| `--session-secret` | - | Required secret for production |
| `--browser` | `False` | Open browser automatically |
| `--no-auth` | `False` | Disable auth (testing only) |

**Examples:**

```bash
# Development with auto-reload (SQLite auth DB)
kobo-serve --reload

# Production with PostgreSQL auth DB
kobo-serve \\
    --db postgresql://user:pass@localhost/kohakuboard \\
    --db-backend postgres \\
    --data-dir /var/kohakuboard \\
    --workers 4 \\
    --session-secret $(openssl rand -hex 32)
```

**What It Does:**

1. Starts FastAPI server with authentication (when not using `--no-auth`)
2. Reads the same board folders produced by the training client (`--data-dir`)
3. Serves the Vue frontend + REST APIs over HTTPS-friendly endpoints
4. Relies on you copying/rsyncing runs into the data directory (for now)

**Features:**

- ✅ **Authentication** - User accounts and login
- ✅ **Projects** - Organize runs into projects
- ✅ **Collaboration** - Share experiments with team
- ✅ **PostgreSQL support** - Production-ready database
- ⚠️ **WIP**: Sync/upload endpoints still target the old DuckDB exporter; copy folders manually

**Security Note:**

Always use `--session-secret` in production:

Generate a session secret with `openssl rand -hex 32` and pass it via `--session-secret`.

---

### `kobo sync` - Legacy Upload Command

`kobo sync` still expects a legacy `board.duckdb` export and will fail on modern hybrid (KohakuVault + SQLite) boards.

⚠️ **Status:** Legacy. Use manual copy/rsync until the refreshed sync API lands.

**Recommended workflow today:**

```bash
# Copy run folder into the server's data-dir
rsync -a ./kohakuboard/default/20250201_120301_xyz \\
      server:/var/kohakuboard/default/
```

After copying, restart or reload `kobo-serve` (or refresh `kobo open`) and the run is immediately available.
| `--private/--public` | No | Board visibility (default: `--private`) |

**Examples:**

```bash
# Using environment variable for token
export KOBO_TOKEN=your_token_here
kobo sync ./kohakuboard/20250115_103045_a1b2c3d4 \\
    --remote https://board.example.com \\
    --project resnet-training \\
    --private

# Using --token flag
kobo sync ./boards/my_run \\
    -r https://board.example.com \\
    -t your_token \\
    -p my-project \\
    --public
```

**What It Does:**

1. Reads board metadata and data files
2. Uploads database files to remote server
3. Uploads media files (images, videos, audio)
4. Links run to specified project
5. Sets visibility (private/public)

**Progress:**

Shows upload progress with:
- Percentage complete
- Estimated time remaining
- Total upload size

**Requirements:**

- Valid board directory with `metadata.json`
- Authentication token from remote server
- Network access to remote server

**Future Features (WIP):**

- ⏳ Incremental sync (only upload changes)
- ⏳ Resume interrupted uploads
- ⏳ Compression for faster uploads
- ⏳ Selective upload (skip media files)

---

## Environment Variables

| Variable | Description | Used By |
|----------|-------------|---------|
| `KOBO_TOKEN` | Authentication token for remote server | `kobo sync` |
| `KOHAKU_BOARD_MODE` | Server mode (`local` or `remote`) | Auto-set by CLI |
| `KOHAKU_BOARD_DATA_DIR` | Board data directory | `kobo open`, `kobo serve` |
| `KOHAKU_BOARD_PORT` | Server port | Auto-set by CLI |
| `KOHAKU_BOARD_HOST` | Server host | Auto-set by CLI |

---

## Quick Start Examples

### Local Development

```bash
# 1. Train your model with KohakuBoard logging
python train.py  # Creates ./kohakuboard/{board_id}/

# 2. View results locally
kobo open ./kohakuboard --browser
```

### Team Collaboration (WIP)

```bash
# 1. Start remote server (once, on shared server)
kobo serve \\
    --db postgresql://user:pass@localhost/kohakuboard \\
    --workers 4 \\
    --session-secret $(openssl rand -hex 32)

# 2. Train locally
python train.py

# 3. Sync to remote server
kobo sync ./kohakuboard/20250129_150423_abc123 \\
    -r https://board.example.com \\
    -p my-project
```

---

## Comparison: Local vs Remote Mode

| Feature | Local Mode (`kobo open`) | Remote Mode (`kobo serve`) |
|---------|---------------------------|----------------------------|
| **Authentication** | ❌ None | ✅ User accounts |
| **Database** | ❌ Not needed | ✅ SQLite or PostgreSQL |
| **Multi-user** | ❌ Single user | ✅ Multiple users |
| **Projects** | ❌ No projects | ✅ Organize into projects |
| **Upload** | ❌ Direct file access | ✅ Upload via API |
| **Deployment** | ✅ Simple (just run) | ⚠️ Requires setup |
| **Use Case** | Personal experiments | Team collaboration |
| **Status** | ✅ Fully working | ⚠️ WIP |

---

## Troubleshooting

### Port Already in Use

```
ERROR: [Errno 48] Address already in use
```

**Fix:**

```bash
# Use different port
kobo open ./kohakuboard --port 8080
```

### Permission Denied

```
PermissionError: [Errno 13] Permission denied: './kohakuboard'
```

**Fix:**

```bash
# Check folder permissions
ls -la kohakuboard/

# Run with proper permissions
sudo kobo open ./kohakuboard  # (not recommended)

# Or change owner
chown -R $USER ./kohakuboard
```

### Board Not Found

```
Error: metadata.json not found in ./kohakuboard/my-run
```

**Fix:**

```bash
# Verify board directory structure
ls -la ./kohakuboard/my-run/

# Should contain:
# - metadata.json
# - data/
# - media/ (optional)
```

---

## Advanced Usage

### Reverse Proxy Setup

For production deployment behind Nginx:

```nginx
server {
    listen 80;
    server_name board.example.com;

    location / {
        proxy_pass http://127.0.0.1:48889;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Then run:

```bash
kobo serve --host 127.0.0.1 --workers 4
```

### Systemd Service

Create `/etc/systemd/system/kohakuboard.service`:

```ini
[Unit]
Description=KohakuBoard Server
After=network.target

[Service]
Type=simple
User=kohakuboard
WorkingDirectory=/var/kohakuboard
ExecStart=/usr/local/bin/kobo serve \\
    --data-dir /var/kohakuboard/data \\
    --db postgresql://user:pass@localhost/kohakuboard \\
    --db-backend postgres \\
    --workers 4 \\
    --session-secret <your_secret>
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable kohakuboard
sudo systemctl start kohakuboard
sudo systemctl status kohakuboard
```

---

## See Also

- [Getting Started](/docs/kohakuboard/getting-started) - Quick start guide
- [Server Setup](/docs/kohakuboard/server) - Remote server configuration
- [Configuration](/docs/kohakuboard/configuration) - Advanced configuration options
