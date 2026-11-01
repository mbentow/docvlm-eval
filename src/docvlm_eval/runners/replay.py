"""Replay runner — score a pipeline that does more than one model call.

A production extractor is rarely *image → one VLM call → JSON*. It reads, looks
candidates up in a catalogue, asks a judge, merges a second pass. None of that
fits behind ``extract()``, and re-implementing it here would measure the copy,
not the system.

So the pipeline runs where it lives and writes what it decided, one JSON line
per document::

    {"_meta": {"pipeline": "intake", "version": "v1.4.2", "concurrency": 2}}
    {"id": "0001", "predicted": {...}, "latency_ms": 8120, "confidence": 0.91,
     "started_at": 1767225600.0, "finished_at": 1767225608.1}

Only ``id`` and ``predicted`` are required. ``started_at``/``finished_at`` are
epoch seconds or ISO-8601; with them the report can state real throughput, not
just per-document latency. ``error`` marks a document the pipeline gave up on,
scored ``refused`` like any backend failure.

What separates this from ``mock`` with ``predictions:``:

- telemetry is the pipeline's, never invented;
- a document missing from the file is ``refused``, not silently filled with
  the ground truth — a replay that skips the hard cases must not look perfect;
- the file's SHA-256 and the ``_meta`` line go into provenance, so two runs of
  the same file are recognisably the same and a re-export is not;
- it is not cached: the file *is* the cache.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from docvlm_eval.runners.base import Runner, RunnerOutput


class ReplayRunner(Runner):
    """Serves recorded pipeline output, document by document."""

    name = "replay"
    cacheable = False

    def __init__(
        self,
        model: str = "pipeline",
        params: dict[str, Any] | None = None,
        *,
        predictions: str | Path | None = None,
    ) -> None:
        super().__init__(model, params)
        source = self.params.get("predictions", predictions)
        if not source:
            raise ValueError("replay runner needs params.predictions (a .jsonl file)")
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"replay predictions not found: {path}")

        raw = path.read_bytes()
        self.file_name = path.name
        self.file_sha256 = hashlib.sha256(raw).hexdigest()[:16]
        self.meta: dict[str, Any] = {}
        self._records: dict[str, dict[str, Any]] = {}
        for n, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "_meta" in record:
                self.meta.update(record["_meta"] or {})
                continue
            if "id" not in record:
                raise ValueError(f"{path}:{n}: record has no 'id'")
            case_id = str(record["id"])
            if case_id in self._records:
                # Two answers for one document means the export joined wrong.
                # Picking either would score a decision the pipeline never made.
                raise ValueError(f"{path}:{n}: duplicate id {case_id!r}")
            self._records[case_id] = record
        self._served: set[str] = set()

    async def extract(
        self,
        image: bytes,
        prompt: str,
        json_schema: dict[str, Any],
        case_id: str = "",
    ) -> RunnerOutput:
        record = self._records.get(case_id)
        if record is None:
            return RunnerOutput(data=None, error="no prediction recorded for this case")
        self._served.add(case_id)

        error = str(record.get("error") or "")
        data = record.get("predicted")
        if not isinstance(data, dict):
            data = None
            error = error or "recorded prediction is not an object"
        return RunnerOutput(
            data=None if error else data,
            raw=json.dumps(data, ensure_ascii=False, default=str) if data is not None else "",
            latency_ms=float(record.get("latency_ms") or 0.0),
            tokens_in=int(record.get("tokens_in") or 0),
            tokens_out=int(record.get("tokens_out") or 0),
            cost_usd=float(record.get("cost_usd") or 0.0),
            error=error,
            confidence=_unit(record.get("confidence")),
            started_at=_epoch(record.get("started_at")),
            finished_at=_epoch(record.get("finished_at")),
        )

    async def describe(self) -> dict[str, Any]:
        return {
            "runner": self.name,
            "model": self.model,
            "predictions_file": self.file_name,
            "predictions_sha256": self.file_sha256,
            "n_recorded": len(self._records),
            # Recorded but never asked for: the export and the corpus disagree
            # about which documents exist. Worth seeing, not worth failing.
            "n_not_in_corpus": len(set(self._records) - self._served),
            "pipeline": self.meta,
        }


def _unit(value: Any) -> float | None:
    """Confidence in ``[0, 1]``; a 0–100 score is accepted and rescaled."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number > 1:
        number /= 100
    return max(0.0, min(1.0, number))


def _epoch(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0
