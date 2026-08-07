"""Result cache, keyed by ``(config, case, image)``.

Re-running a four-hour sweep because you fixed a typo in a normaliser is the
fastest way to stop running sweeps. Inference results are cached; **scoring is
never cached**, so changing a comparison rule re-scores instantly against the
stored model outputs.

That split is the point: the expensive half is the model call, and the half you
iterate on is the scoring.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DIR = Path(".docvlm-cache")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS inference (
    key         TEXT PRIMARY KEY,
    config_hash TEXT NOT NULL,
    case_id     TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  REAL NOT NULL DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS idx_inference_config ON inference(config_hash);
"""


@dataclass
class CachedInference:
    data: dict[str, Any] | None
    raw: str
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    error: str
    meta: dict[str, Any]


class ResultCache:
    """SQLite-backed cache. Thread-safe, single file, easy to delete."""

    def __init__(self, directory: str | Path = DEFAULT_DIR, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = Path(directory) / "inference.sqlite3"
        self.disabled_reason = ""
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        if not enabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except (sqlite3.Error, OSError) as exc:
            # SQLite needs byte-range locks, which some network and FUSE mounts
            # do not provide. Losing the cache is a slowdown; aborting the run
            # over it would be worse.
            self.enabled = False
            self._conn = None
            self.disabled_reason = (
                f"cache disabled: {type(exc).__name__} at {self.path} ({exc}). "
                "Point --cache-dir at a local disk to re-enable it."
            )

    @staticmethod
    def key(config_hash: str, case_id: str, image_hash: str, schema_hash: str) -> str:
        """The schema hash is in the key because it is sent to the model as the
        decoding constraint — changing it changes the output."""
        raw = f"{config_hash}|{case_id}|{image_hash}|{schema_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> CachedInference | None:
        if not self.enabled or self._conn is None:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM inference WHERE key = ?", (key,)
            ).fetchone()
        if not row:
            return None
        payload = json.loads(row[0])
        return CachedInference(
            data=payload.get("data"),
            raw=payload.get("raw", ""),
            latency_ms=payload.get("latency_ms", 0.0),
            tokens_in=payload.get("tokens_in", 0),
            tokens_out=payload.get("tokens_out", 0),
            cost_usd=payload.get("cost_usd", 0.0),
            error=payload.get("error", ""),
            meta=payload.get("meta", {}),
        )

    def put(self, key: str, config_hash: str, case_id: str, value: CachedInference) -> None:
        if not self.enabled or self._conn is None:
            return
        payload = json.dumps(
            {
                "data": value.data,
                "raw": value.raw,
                "latency_ms": value.latency_ms,
                "tokens_in": value.tokens_in,
                "tokens_out": value.tokens_out,
                "cost_usd": value.cost_usd,
                "error": value.error,
                "meta": value.meta,
            },
            ensure_ascii=False,
            default=str,
        )
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO inference (key, config_hash, case_id, payload) "
                "VALUES (?, ?, ?, ?)",
                (key, config_hash, case_id, payload),
            )
            self._conn.commit()

    def clear(self, config_hash: str | None = None) -> int:
        """Drop cached inferences, optionally for a single config."""
        if not self.enabled or self._conn is None:
            return 0
        with self._lock:
            if config_hash:
                cur = self._conn.execute(
                    "DELETE FROM inference WHERE config_hash = ?", (config_hash,)
                )
            else:
                cur = self._conn.execute("DELETE FROM inference")
            self._conn.commit()
            return cur.rowcount

    def stats(self) -> dict[str, Any]:
        if not self.enabled or self._conn is None:
            return {"enabled": False, "rows": 0}
        with self._lock:
            rows = self._conn.execute("SELECT COUNT(*) FROM inference").fetchone()[0]
        return {
            "enabled": True,
            "rows": rows,
            "path": str(self.path),
            "size_bytes": self.path.stat().st_size if self.path.exists() else 0,
        }

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
