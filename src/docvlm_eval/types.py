"""Result types shared across the engine.

Everything here is a plain dataclass so that a run can be serialised to JSON and
compared later without importing the schema module that produced it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Outcome(StrEnum):
    """What happened to a single field on a single document.

    The split between ``MISSING`` and ``HALLUCINATED`` is the reason this tool
    exists. An empty field is an inconvenience; an invented field is a risk.
    Averaged accuracy hides the difference.
    """

    CORRECT = "correct"
    """Prediction matches the ground truth (including both being empty)."""

    MISSING = "missing"
    """Ground truth has a value, the model returned nothing."""

    HALLUCINATED = "hallucinated"
    """Ground truth is empty, the model returned a value."""

    WRONG = "wrong"
    """Both have values and they disagree."""

    MALFORMED = "malformed"
    """The model returned something that does not fit the declared type."""

    REFUSED = "refused"
    """The backend errored, timed out, or the model declined to answer."""

    @property
    def is_correct(self) -> bool:
        return self is Outcome.CORRECT


@dataclass
class FieldResult:
    """Scoring of one field of one case."""

    field: str
    outcome: Outcome
    truth: Any = None
    predicted: Any = None
    score: float = 0.0
    """1.0 for a match, otherwise the similarity that was measured (0..1)."""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FieldResult:
        return cls(
            field=d["field"],
            outcome=Outcome(d["outcome"]),
            truth=d.get("truth"),
            predicted=d.get("predicted"),
            score=d.get("score", 0.0),
            detail=d.get("detail", ""),
        )


@dataclass
class CaseResult:
    """Scoring of one document."""

    case_id: str
    tags: list[str] = field(default_factory=list)
    fields: list[FieldResult] = field(default_factory=list)
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error: str = ""
    raw_output: str = ""
    cached: bool = False

    @property
    def all_fields_correct(self) -> bool:
        """The business metric: the whole document usable without a human."""
        return bool(self.fields) and all(f.outcome.is_correct for f in self.fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "tags": self.tags,
            "fields": [f.to_dict() for f in self.fields],
            "latency_ms": self.latency_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
            "error": self.error,
            "raw_output": self.raw_output,
            "cached": self.cached,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CaseResult:
        return cls(
            case_id=d["case_id"],
            tags=d.get("tags", []),
            fields=[FieldResult.from_dict(f) for f in d.get("fields", [])],
            latency_ms=d.get("latency_ms", 0.0),
            tokens_in=d.get("tokens_in", 0),
            tokens_out=d.get("tokens_out", 0),
            cost_usd=d.get("cost_usd", 0.0),
            error=d.get("error", ""),
            raw_output=d.get("raw_output", ""),
            cached=d.get("cached", False),
        )


@dataclass
class RunResult:
    """One config evaluated over one corpus. Immutable once written to disk.

    ``provenance`` is what makes a comparison meaningful three weeks later:
    corpus hash, prompt hash, model digest, library version, temperature, seed.
    Without it you are comparing two numbers of unknown origin.
    """

    name: str
    corpus_name: str
    corpus_hash: str
    config_name: str
    config_hash: str
    field_names: list[str] = field(default_factory=list)
    cases: list[CaseResult] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    """Per-field macro weight in force for this run, resolved from the schema
    and the config. Stored so that ``report`` and ``diff`` reproduce exactly the
    numbers ``run`` printed, instead of silently falling back to defaults."""
    critical_fields: list[str] = field(default_factory=list)
    """Fields where a hallucination is a safety issue, resolved the same way."""
    provenance: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "corpus_name": self.corpus_name,
                "corpus_hash": self.corpus_hash,
                "config_name": self.config_name,
                "config_hash": self.config_hash,
                "field_names": self.field_names,
                "weights": self.weights,
                "critical_fields": self.critical_fields,
                "provenance": self.provenance,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "cases": [c.to_dict() for c in self.cases],
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    @classmethod
    def from_json(cls, text: str) -> RunResult:
        d = json.loads(text)
        return cls(
            name=d["name"],
            corpus_name=d["corpus_name"],
            corpus_hash=d["corpus_hash"],
            config_name=d["config_name"],
            config_hash=d["config_hash"],
            field_names=d.get("field_names", []),
            cases=[CaseResult.from_dict(c) for c in d.get("cases", [])],
            weights={str(k): float(v) for k, v in (d.get("weights") or {}).items()},
            critical_fields=list(d.get("critical_fields") or []),
            provenance=d.get("provenance", {}),
            started_at=d.get("started_at", ""),
            finished_at=d.get("finished_at", ""),
        )
