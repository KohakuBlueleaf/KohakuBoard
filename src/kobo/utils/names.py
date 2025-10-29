"""Name normalization utilities."""


def normalize_name(name: str) -> str:
    """Normalize name for conflict detection.

    Converts to lowercase and replaces hyphens/underscores with empty string.
    This prevents username squatting with similar names.

    Examples:
        "My-Name" -> "myname"
        "my_name" -> "myname"
        "MyName" -> "myname"
    """
    return name.lower().replace("-", "").replace("_", "")
