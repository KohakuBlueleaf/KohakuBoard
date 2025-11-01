"""Configuration for KohakuBoard Server

Server-specific configuration including auth, database, and server settings.
"""

import os

from pydantic import BaseModel


class AppConfig(BaseModel):
    """Application configuration"""

    host: str = "0.0.0.0"
    port: int = 48889
    api_base: str = "/api"
    base_url: str = "http://localhost:48889"
    cors_origins: list[str] = ["http://localhost:5175", "http://localhost:28081"]
    board_data_dir: str = "./kohakuboard"
    mode: str = "remote"  # Always remote for server
    db_backend: str = "sqlite"  # "sqlite" or "postgres"
    database_url: str = "sqlite:///kohakuboard.db"
    no_auth: bool = False  # Disable authentication (for testing only)


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


class Config(BaseModel):
    """Server configuration"""

    app: AppConfig
    auth: AuthConfig = AuthConfig()
    smtp: SMTPConfig = SMTPConfig()

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        return cls(
            app=AppConfig(
                host=os.getenv("KOHAKU_BOARD_HOST", "0.0.0.0"),
                port=int(os.getenv("KOHAKU_BOARD_PORT", "48889")),
                api_base=os.getenv("KOHAKU_BOARD_API_BASE", "/api"),
                base_url=os.getenv("KOHAKU_BOARD_BASE_URL", "http://localhost:48889"),
                board_data_dir=os.getenv("KOHAKU_BOARD_DATA_DIR", "./kohakuboard"),
                db_backend=os.getenv("KOHAKU_BOARD_DB_BACKEND", "sqlite"),
                database_url=os.getenv(
                    "KOHAKU_BOARD_DATABASE_URL", "sqlite:///kohakuboard.db"
                ),
                no_auth=os.getenv("KOHAKU_BOARD_NO_AUTH", "false").lower() == "true",
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
        )


cfg = Config.from_env()
