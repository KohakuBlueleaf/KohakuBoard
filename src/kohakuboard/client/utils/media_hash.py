"""
Media Hash Utility

Provides content-addressable storage for media files using SHA3-224 hash
with XOR compression to 112 bits, encoded as 22-character base-36 string.
"""

import hashlib
import numpy as np


def generate_media_hash(content_bytes: bytes) -> str:
    """
    Generate 22-character base-36 hash from media content.

    Algorithm:
    1. Compute SHA3-224 hash → 224 bits (28 bytes)
    2. XOR first 112 bits with last 112 bits → 112 bits (14 bytes)
    3. Convert to integer → base-36 string (0-9, a-z) → zero-pad to 22 chars

    Properties:
    - Deterministic: same content always produces same hash
    - Collision-resistant: 2^112 address space (~5.2 × 10^33 unique hashes)
    - Compact: 22 characters vs 64 for full SHA256
    - URL-safe: only lowercase alphanumeric characters

    Args:
        content_bytes: Raw bytes of media file content

    Returns:
        22-character lowercase alphanumeric string

    Example:
        >>> content = b"hello world"
        >>> hash1 = generate_media_hash(content)
        >>> hash2 = generate_media_hash(content)
        >>> hash1 == hash2
        True
        >>> len(hash1)
        22
        >>> hash1
        '0a1b2c3d4e5f6g7h8i9j0k'  # Example output
    """
    # Step 1: SHA3-224 hash (224 bits = 28 bytes)
    sha3 = hashlib.sha3_224(content_bytes).digest()

    # Step 2: XOR first 112 bits (14 bytes) with last 112 bits (14 bytes)
    first_half = int.from_bytes(sha3[:14], byteorder="big")
    last_half = int.from_bytes(sha3[14:], byteorder="big")
    xor_result = first_half ^ last_half

    # Step 3: Convert to base-36 string and zero-pad to 22 characters
    # Base-36 uses digits 0-9 and letters a-z (case-insensitive, we use lowercase)
    # 2^112 > 36^22 > 2^113, so 22 characters are sufficient
    base36_str = np.base_repr(xor_result, 36).lower()

    # Zero-pad to ensure consistent 22-character length
    return base36_str.zfill(22)


def verify_hash_format(hash_str: str) -> bool:
    """
    Verify that a string is a valid media hash.

    Args:
        hash_str: String to verify

    Returns:
        True if valid media hash format, False otherwise

    Example:
        >>> verify_hash_format("0a1b2c3d4e5f6g7h8i9j0k")
        True
        >>> verify_hash_format("invalid-hash")
        False
        >>> verify_hash_format("tooshort")
        False
    """
    import re

    if not isinstance(hash_str, str):
        return False

    # Must be exactly 22 lowercase alphanumeric characters (0-9, a-z)
    pattern = r"^[0-9a-z]{22}$"
    return bool(re.match(pattern, hash_str))
