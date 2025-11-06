"""Tensor storage backed by KohakuVault KVault."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from kohakuvault import KVault

from kohakuboard.logger import get_logger


class TensorKVStorage:
    """Store tensor payloads grouped by namespace in KVault databases.

    Naming convention:
        - Tensor log name: namespace/sub_name (e.g. params/linear1.weight)
        - Namespace file: {namespace}.db  → params.db
        - KVault key: "{sub_name}:{step}" → "linear1.weight:42"
    """

    def __init__(self, base_dir: Path, logger=None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logger or get_logger("TENSOR_STORAGE")
        self.tensors_dir = self.base_dir / "tensors"
        self.tensors_dir.mkdir(exist_ok=True)

        # Cached KVault handles keyed by namespace
        self._vaults: Dict[str, KVault] = {}

    @staticmethod
    def _sanitize(name: str) -> str:
        return name.replace("/", "__")

    def _split_name(self, name: str) -> tuple[str, str]:
        if "/" in name:
            namespace, sub = name.split("/", 1)
        else:
            namespace, sub = name, ""
        return namespace, (sub.replace("/", "__") if sub else "value")

    def _get_vault(self, namespace: str) -> Tuple[KVault, str]:
        sanitized = self._sanitize(namespace)
        if sanitized not in self._vaults:
            db_path = self.tensors_dir / f"{sanitized}.db"
            kv = KVault(str(db_path))
            self._vaults[sanitized] = kv
            self.logger.debug(f"Opened tensor KVault: {db_path.name}")
        return self._vaults[sanitized], sanitized

    def store_tensor(
        self, name: str, step: int, payload: bytes
    ) -> tuple[str, str, int]:
        """Store tensor payload and return (namespace_file, key, size)."""
        namespace, sub = self._split_name(name)
        kv, sanitized = self._get_vault(namespace)
        kv_key = f"{sub}:{step}"

        with kv.cache(64 * 1024 * 1024):
            kv[kv_key] = payload

        self.logger.debug(
            f"Stored tensor payload {name} (step={step}) -> {sanitized}.db[{kv_key}]"
        )

        return sanitized, kv_key, len(payload)

    def load_tensor(self, namespace_file: str, key: str) -> bytes:
        """Retrieve tensor payload bytes."""
        kv = self._vaults.get(namespace_file)
        if kv is None:
            kv = KVault(str(self.tensors_dir / f"{namespace_file}.db"))
            self._vaults[namespace_file] = kv
        return kv.get(key)

    def close(self):
        for kv in self._vaults.values():
            try:
                kv.close()
            except Exception:
                pass
        self._vaults.clear()
