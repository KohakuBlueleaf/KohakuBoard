"""Configuration for KohakuBoard (Local Mode)

Local mode configuration - no auth, no database.
"""

import os

from pydantic import BaseModel


class AppConfig(BaseModel):
    """Application configuration"""

    host: str = "0.0.0.0"
    port: int = 48889
    api_base: str = "/api"
    cors_origins: list[str] = ["http://localhost:5175", "http://localhost:28081"]
    board_data_dir: str = "./kohakuboard"


class SyncConfig(BaseModel):
    """Sync configuration for client-side remote sync"""

    enabled: bool = False
    remote_url: str = ""
    remote_token: str = ""
    remote_project: str = "local"
    sync_interval: int = 10  # seconds


class Config(BaseModel):
    """Main configuration"""

    app: AppConfig
    sync: SyncConfig = SyncConfig()

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        return cls(
            app=AppConfig(
                host=os.getenv("KOHAKU_BOARD_HOST", "0.0.0.0"),
                port=int(os.getenv("KOHAKU_BOARD_PORT", "48889")),
                api_base=os.getenv("KOHAKU_BOARD_API_BASE", "/api"),
                board_data_dir=os.getenv("KOHAKU_BOARD_DATA_DIR", "./kohakuboard"),
            ),
            sync=SyncConfig(
                enabled=os.getenv("KOHAKU_SYNC_ENABLED", "false").lower() == "true",
                remote_url=os.getenv("KOHAKU_SYNC_REMOTE_URL", ""),
                remote_token=os.getenv("KOHAKU_SYNC_REMOTE_TOKEN", ""),
                remote_project=os.getenv("KOHAKU_SYNC_REMOTE_PROJECT", "local"),
                sync_interval=int(os.getenv("KOHAKU_SYNC_INTERVAL", "10")),
            ),
        )


cfg = Config.from_env()
