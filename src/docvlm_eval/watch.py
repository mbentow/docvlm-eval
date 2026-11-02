"""Score a predictions file while the pipeline is still writing it.

``docvlm-eval watch`` tails the replay ``.jsonl`` a long benchmark appends to,
scores each new document against the corpus, and runs the sequential test from
:mod:`docvlm_eval.sequential` at every planned look. When a look decides, it
writes the decision to a status file and exits, and the benchmark can be
stopped. Hours of a one-document-at-a-time backend are the cost this avoids.

The tail tolerates a half-written last line (the writer may be mid-flush) and
only ever reads forward, so it is safe to point at a file that grows.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docvlm_eval.compare import score_case
from docvlm_eval.corpus import Corpus
from docvlm_eval.runners.replay import _epoch, _unit
from docvlm_eval.types import CaseResult

BAR_WIDTH = 20


@dataclass
class PredictionTail:
    """Reads complete JSON lines appended to a file since the last call."""

    path: Path
    offset: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    seen: set[str] = field(default_factory=set)
    duplicates: int = 0

    def read_new(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("rb") as fh:
            fh.seek(self.offset)
            chunk = fh.read()
        # Only up to the last newline: anything after it is a line still being
        # written, and parsing it now would either fail or score half a record.
        end = chunk.rfind(b"\n")
        if end < 0:
            return []
        self.offset += end + 1
        out = []
        for line in chunk[:end].decode("utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if "_meta" in record:
                self.meta.update(record["_meta"] or {})
                continue
            case_id = str(record.get("id", ""))
            if not case_id:
                continue
            if case_id in self.seen:
                # The first answer is the one the order was planned around; a
                # retry appended later must not replace it mid-test.
                self.duplicates += 1
                continue
            self.seen.add(case_id)
            out.append(record)
        return out


def score_record(record: dict[str, Any], corpus: Corpus) -> CaseResult | None:
    """Score one replay record. ``None`` when the id is not in the corpus."""
    case = corpus.by_id(str(record["id"]))
    if case is None:
        return None
    error = str(record.get("error") or "")
    data = record.get("predicted")
    if not isinstance(data, dict):
        data, error = None, error or "recorded prediction is not an object"
    result = score_case(
        case.id,
        case.tags,
        case.truth,
        None if error else data,
        corpus.schema,
        error=error,
        latency_ms=float(record.get("latency_ms") or 0.0),
        tokens_in=int(record.get("tokens_in") or 0),
        tokens_out=int(record.get("tokens_out") or 0),
        cost_usd=float(record.get("cost_usd") or 0.0),
    )
    result.confidence = _unit(record.get("confidence"))
    result.started_at = _epoch(record.get("started_at"))
    result.finished_at = _epoch(record.get("finished_at"))
    return result


def progress_line(done: int, total: int, elapsed_s: float, note: str = "") -> str:
    """``[########------------] 40% · 120/300 · 6m12s elapsed · ETA ~9m · note``"""
    frac = done / total if total else 0.0
    filled = round(frac * BAR_WIDTH)
    bar = "#" * filled + "-" * (BAR_WIDTH - filled)
    eta = ""
    if done and done < total:
        eta = f" · ETA ~{_dur(elapsed_s / done * (total - done))}"
    tail = f" · {note}" if note else ""
    return f"[{bar}] {frac:4.0%} · {done}/{total} · {_dur(elapsed_s)} elapsed{eta}{tail}"


def _dur(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def elapsed_since(cases: list[CaseResult], fallback_start: float) -> float:
    """Wall time of the benchmark itself when it recorded timestamps, else of the watch."""
    starts = [c.started_at for c in cases if c.started_at > 0]
    start = min(starts) if starts else fallback_start
    return max(0.0, time.time() - start)
