"""Database models for KohakuBoard.

Uses EXACTLY the same schema as KohakuHub for User, Session, Token, UserOrganization.
This ensures future integration with KohakuHub by sharing the same database.
"""

from datetime import datetime, timezone
from functools import partial

from peewee import (
    AutoField,
    BigIntegerField,
    BlobField,
    BooleanField,
    CharField,
    DatabaseProxy,
    DateTimeField,
    ForeignKeyField,
    Model,
    PostgresqlDatabase,
    SqliteDatabase,
    TextField,
)

# Database proxy - will be initialized later
db = DatabaseProxy()


def init_db(backend: str, database_url: str):
    """Initialize database connection"""
    if backend == "postgres":
        # Parse PostgreSQL URL
        url = database_url.replace("postgresql://", "")
        user_pass, host_db = url.split("@")
        user, password = user_pass.split(":")
        host_port, dbname = host_db.split("/")
        if ":" in host_port:
            host, port = host_port.split(":")
        else:
            host, port = host_port, 5432

        real_db = PostgresqlDatabase(
            dbname,
            user=user,
            password=password,
            host=host,
            port=int(port),
        )
    else:
        # SQLite
        db_path = database_url.replace("sqlite:///", "")
        real_db = SqliteDatabase(db_path, pragmas={"foreign_keys": 1})

    # Initialize the proxy with the real database
    db.initialize(real_db)

    # Create tables
    with db:
        db.create_tables(
            [User, EmailVerification, Session, Token, UserOrganization, Board],
            safe=True,
        )

    return db


class BaseModel(Model):
    """Base model for all database models"""

    class Meta:
        database = db


class User(BaseModel):
    """Unified User/Organization model - EXACTLY same as KohakuHub

    When is_org=False: Regular user with email/password
    When is_org=True: Organization (no email/password required)
    """

    id = AutoField()
    username = CharField(unique=True, index=True)
    normalized_name = CharField(unique=True, index=True)
    is_org = BooleanField(default=False, index=True)

    # User-specific fields (nullable for orgs)
    email = CharField(unique=True, index=True, null=True)
    password_hash = CharField(null=True)
    email_verified = BooleanField(default=False)
    is_active = BooleanField(default=True)

    # Separate quotas for private and public repositories
    private_quota_bytes = BigIntegerField(null=True)
    public_quota_bytes = BigIntegerField(null=True)
    private_used_bytes = BigIntegerField(default=0)
    public_used_bytes = BigIntegerField(default=0)

    # Profile fields
    full_name = CharField(null=True)
    bio = TextField(null=True)
    description = TextField(null=True)
    website = CharField(null=True)
    social_media = TextField(null=True)

    # Avatar
    avatar = BlobField(null=True)
    avatar_updated_at = DateTimeField(null=True)
    created_at = DateTimeField(default=partial(datetime.now, tz=timezone.utc))

    class Meta:
        table_name = "user"


class Session(BaseModel):
    """User session - EXACTLY same as KohakuHub"""

    id = AutoField()
    session_id = CharField(unique=True, index=True)
    user = ForeignKeyField(User, backref="sessions", on_delete="CASCADE", index=True)
    secret = CharField()
    expires_at = DateTimeField()
    created_at = DateTimeField(default=partial(datetime.now, tz=timezone.utc))

    class Meta:
        table_name = "session"


class Token(BaseModel):
    """API token - EXACTLY same as KohakuHub"""

    id = AutoField()
    user = ForeignKeyField(User, backref="tokens", on_delete="CASCADE", index=True)
    token_hash = CharField(unique=True, index=True)
    name = CharField()
    last_used = DateTimeField(null=True)
    created_at = DateTimeField(default=partial(datetime.now, tz=timezone.utc))

    class Meta:
        table_name = "token"


class EmailVerification(BaseModel):
    """Email verification tokens - EXACTLY same as KohakuHub"""

    id = AutoField()
    user = ForeignKeyField(
        User, backref="email_verifications", on_delete="CASCADE", index=True
    )
    token = CharField(unique=True, index=True)
    expires_at = DateTimeField()
    created_at = DateTimeField(default=partial(datetime.now, tz=timezone.utc))

    class Meta:
        table_name = "emailverification"


class UserOrganization(BaseModel):
    """User-Organization membership - EXACTLY same as KohakuHub

    Links regular users (is_org=False) to organizations (is_org=True).
    """

    id = AutoField()
    user = ForeignKeyField(
        User, backref="org_memberships", on_delete="CASCADE", index=True
    )
    organization = ForeignKeyField(
        User, backref="members", on_delete="CASCADE", index=True
    )
    role = CharField(default="member")  # visitor, member, admin, super-admin
    created_at = DateTimeField(default=partial(datetime.now, tz=timezone.utc))

    class Meta:
        table_name = "userorganization"
        indexes = ((("user", "organization"), True),)


class Board(BaseModel):
    """Board metadata for KohakuBoard

    Stores information about training runs/boards.
    Can be owned by users OR organizations (owner is User with is_org flag).
    """

    id = AutoField()
    run_id = CharField(unique=True, index=True)  # Globally unique identifier
    name = CharField()
    project_name = CharField(index=True)  # User-defined project grouping
    owner = ForeignKeyField(User, backref="boards", on_delete="CASCADE", index=True)
    private = BooleanField(default=True)

    # Storage metadata
    config = TextField(null=True)  # JSON string
    backend = CharField(default="duckdb")
    storage_path = CharField()  # Relative: users/{username}/{project}/{run_id}
    total_size_bytes = BigIntegerField(default=0)

    # Timestamps
    created_at = DateTimeField(default=partial(datetime.now, tz=timezone.utc))
    updated_at = DateTimeField(default=partial(datetime.now, tz=timezone.utc))
    last_synced_at = DateTimeField(null=True)

    class Meta:
        table_name = "board"
        indexes = ((("owner", "project_name"), False),)
