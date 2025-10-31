"""Configuration for KohakuBoard"""

import os

from pydantic import BaseModel


class AppConfig(BaseModel):
    """Application configuration"""

    host: str = "0.0.0.0"
    port: int = 48889
    api_base: str = "/api"
    base_url: str = "http://localhost:28081"
    cors_origins: list[str] = ["http://localhost:5175", "http://localhost:28081"]
    board_data_dir: str = "./kohakuboard"
    mode: str = "local"  # "local" or "remote"
    db_backend: str = "sqlite"  # "sqlite" or "postgres"
    database_url: str = "sqlite:///kohakuboard.db"


class MockDataConfig(BaseModel):
    """Mock data generation configuration"""

    default_steps: int = 1000
    default_metrics_count: int = 4
    default_noise_level: float = 0.1
    max_steps: int = 100000
    max_metrics: int = 50


class AuthConfig(BaseModel):
    """Authentication configuration"""

    require_email_verification: bool = False
    invitation_only: bool = False
    session_secret: str = "change-me-in-production"
    session_expire_hours: int = 168  # 7 days
    token_expire_days: int = 365


class SMTPConfig(BaseModel):
    """SMTP configuration for email"""

    enabled: bool = False
    host: str = "localhost"
    port: int = 587
    username: str = ""
    password: str = ""
    from_email: str = "noreply@localhost"
    use_tls: bool = True


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
    mock: MockDataConfig
    auth: AuthConfig = AuthConfig()
    smtp: SMTPConfig = SMTPConfig()
    sync: SyncConfig = SyncConfig()

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        mode = os.getenv("KOHAKU_BOARD_MODE", "local")

        return cls(
            app=AppConfig(
                host=os.getenv("KOHAKU_BOARD_HOST", "0.0.0.0"),
                port=int(os.getenv("KOHAKU_BOARD_PORT", "48889")),
                api_base=os.getenv("KOHAKU_BOARD_API_BASE", "/api"),
                base_url=os.getenv("KOHAKU_BOARD_BASE_URL", "http://localhost:28081"),
                board_data_dir=os.getenv("KOHAKU_BOARD_DATA_DIR", "./kohakuboard"),
                mode=mode,
                db_backend=os.getenv("KOHAKU_BOARD_DB_BACKEND", "sqlite"),
                database_url=os.getenv(
                    "KOHAKU_BOARD_DATABASE_URL", "sqlite:///kohakuboard.db"
                ),
            ),
            mock=MockDataConfig(
                default_steps=int(os.getenv("KOHAKU_BOARD_DEFAULT_STEPS", "1000")),
                default_metrics_count=int(
                    os.getenv("KOHAKU_BOARD_DEFAULT_METRICS", "4")
                ),
                default_noise_level=float(os.getenv("KOHAKU_BOARD_NOISE_LEVEL", "0.1")),
            ),
            auth=AuthConfig(
                require_email_verification=os.getenv(
                    "KOHAKU_BOARD_AUTH_REQUIRE_EMAIL_VERIFICATION", "false"
                ).lower()
                == "true",
                invitation_only=os.getenv(
                    "KOHAKU_BOARD_AUTH_INVITATION_ONLY", "false"
                ).lower()
                == "true",
                session_secret=os.getenv(
                    "KOHAKU_BOARD_AUTH_SESSION_SECRET", "change-me-in-production"
                ),
                session_expire_hours=int(
                    os.getenv("KOHAKU_BOARD_AUTH_SESSION_EXPIRE_HOURS", "168")
                ),
                token_expire_days=int(
                    os.getenv("KOHAKU_BOARD_AUTH_TOKEN_EXPIRE_DAYS", "365")
                ),
            ),
            smtp=SMTPConfig(
                enabled=os.getenv("KOHAKU_BOARD_SMTP_ENABLED", "false").lower()
                == "true",
                host=os.getenv("KOHAKU_BOARD_SMTP_HOST", "localhost"),
                port=int(os.getenv("KOHAKU_BOARD_SMTP_PORT", "587")),
                username=os.getenv("KOHAKU_BOARD_SMTP_USERNAME", ""),
                password=os.getenv("KOHAKU_BOARD_SMTP_PASSWORD", ""),
                from_email=os.getenv("KOHAKU_BOARD_SMTP_FROM", "noreply@localhost"),
                use_tls=os.getenv("KOHAKU_BOARD_SMTP_TLS", "true").lower() == "true",
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
