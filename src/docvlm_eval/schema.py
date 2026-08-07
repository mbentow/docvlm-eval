"""Extraction schemas.

A schema is a Pydantic v2 model. It does two jobs at once:

1. it becomes the JSON Schema handed to the backend (Ollama ``format``,
   OpenAI ``response_format``), so the model is constrained instead of asked
   politely;
2. it carries, per field, **how that field should be compared** — which is
   where most home-made evaluators go wrong.

Keep schemas shallow. Deep or recursive schemas measurably degrade constrained
decoding on current vision models.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import Field as PydanticField
from pydantic.fields import FieldInfo

from docvlm_eval.normalize import DEFAULT_TEXT_CHAIN


class Compare(StrEnum):
    """How a field is compared against its ground truth."""

    EXACT = "exact"
    """Byte-for-byte equality after ``str()``. For codes you control."""

    TEXT = "text"
    """Normalised equality, then fuzzy similarity above ``fuzzy_threshold``."""

    DIGITS = "digits"
    """Reduce both sides to digits. For licence numbers, IDs, phone numbers."""

    NUMBER = "number"
    """Parse as number, compare within ``tolerance`` (absolute)."""

    DATE = "date"
    """Parse as date, compare within ``tolerance_days``."""

    BOOL = "bool"
    """Parse as boolean. Model wording (`sim`, `yes`, `1`) is handled."""

    SET_TEXT = "set_text"
    """Unordered list of strings. Scored as an exact-set match, with per-item
    precision/recall exposed in the detail string. This is the honest way to
    score a field like *procedures requested*: reading three exams on a
    single-exam request is an error, because the model failed to stop."""


@dataclass(frozen=True)
class FieldSpec:
    """Comparison policy for one field."""

    compare: str = Compare.TEXT
    normalize: tuple[str, ...] = DEFAULT_TEXT_CHAIN
    fuzzy_threshold: float = 1.0
    """Below 1.0, a normalised-similarity above this counts as correct.
    Leave at 1.0 for anything an operator would have to retype."""
    tolerance: float = 0.0
    """Absolute tolerance for ``NUMBER``."""
    tolerance_days: int = 0
    """Tolerance for ``DATE``."""
    dayfirst: bool = True
    """Parse ``03/04`` as 3 April (pt-BR/EU) rather than 4 March."""
    weight: float = 1.0
    """Relative weight in the macro average. 0 excludes the field from
    aggregates while still reporting it — useful for diagnostics fields."""
    critical: bool = False
    """Marks a field where a hallucination is a safety issue, not a nuisance.
    Reports flag it; ``--fail-under`` can gate on it separately."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "compare": self.compare,
            "normalize": list(self.normalize),
            "fuzzy_threshold": self.fuzzy_threshold,
            "tolerance": self.tolerance,
            "tolerance_days": self.tolerance_days,
            "dayfirst": self.dayfirst,
            "weight": self.weight,
            "critical": self.critical,
        }


def field(
    default: Any = ...,
    *,
    description: str | None = None,
    compare: str = Compare.TEXT,
    normalize: tuple[str, ...] | list[str] | None = None,
    fuzzy_threshold: float = 1.0,
    tolerance: float = 0.0,
    tolerance_days: int = 0,
    dayfirst: bool = True,
    weight: float = 1.0,
    critical: bool = False,
    **kwargs: Any,
) -> Any:
    """Declare a schema field together with its comparison policy.

    Example::

        class Request(ExtractionSchema):
            patient_name: str | None = field(
                None,
                description="Full name of the patient",
                compare=Compare.TEXT,
                fuzzy_threshold=0.92,
            )
            crm: str | None = field(None, compare=Compare.DIGITS, critical=True)
    """
    try:
        compare = Compare(compare)
    except ValueError as exc:
        raise ValueError(
            f"unknown compare mode {compare!r}; expected one of {[c.value for c in Compare]}"
        ) from exc
    if normalize is None:
        normalize = DEFAULT_TEXT_CHAIN if compare in (Compare.TEXT, Compare.SET_TEXT) else ()
    spec = FieldSpec(
        compare=compare,
        normalize=tuple(normalize),
        fuzzy_threshold=fuzzy_threshold,
        tolerance=tolerance,
        tolerance_days=tolerance_days,
        dayfirst=dayfirst,
        weight=weight,
        critical=critical,
    )
    extra = dict(kwargs.pop("json_schema_extra", {}) or {})
    return PydanticField(
        default,
        description=description,
        json_schema_extra={**extra, "docvlm": spec.to_dict()},
        **kwargs,
    )


def spec_of(info: FieldInfo) -> FieldSpec:
    """Read the :class:`FieldSpec` off a Pydantic field, with sane fallbacks.

    A plain ``str | None = None`` field still works — it is compared as text.
    """
    extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
    raw = extra.get("docvlm")
    if not isinstance(raw, dict):
        return FieldSpec()
    return FieldSpec(
        compare=raw.get("compare", Compare.TEXT),
        normalize=tuple(raw.get("normalize", DEFAULT_TEXT_CHAIN)),
        fuzzy_threshold=float(raw.get("fuzzy_threshold", 1.0)),
        tolerance=float(raw.get("tolerance", 0.0)),
        tolerance_days=int(raw.get("tolerance_days", 0)),
        dayfirst=bool(raw.get("dayfirst", True)),
        weight=float(raw.get("weight", 1.0)),
        critical=bool(raw.get("critical", False)),
    )


class ExtractionSchema(BaseModel):
    """Base class for extraction schemas."""

    model_config = {"extra": "ignore", "populate_by_name": True}

    @classmethod
    def specs(cls) -> dict[str, FieldSpec]:
        """Comparison policy for every field, in declaration order."""
        return {name: spec_of(info) for name, info in cls.model_fields.items()}

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """JSON Schema for constrained decoding.

        The ``docvlm`` block is stripped: it is evaluation metadata and would
        only add noise to the model's context.
        """
        return _strip_docvlm(cls.model_json_schema())

    @classmethod
    def schema_hash(cls) -> str:
        payload = json.dumps(cls.json_schema(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @classmethod
    def field_description_block(cls) -> str:
        """A compact ``name: description`` block for prompt templates."""
        lines = []
        for name, info in cls.model_fields.items():
            desc = (info.description or "").strip()
            lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        return "\n".join(lines)


def _strip_docvlm(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip_docvlm(v) for k, v in node.items() if k != "docvlm"}
    if isinstance(node, list):
        return [_strip_docvlm(v) for v in node]
    return node


_SCHEMA_MODULE_SEQ = 0


def load_schema(path: str | Path) -> type[ExtractionSchema]:
    """Import ``schema.py`` from a corpus directory and return its schema class.

    The file must define exactly one :class:`ExtractionSchema` subclass, or set
    ``SCHEMA`` to the one to use.
    """
    global _SCHEMA_MODULE_SEQ
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"schema file not found: {path}")

    _SCHEMA_MODULE_SEQ += 1
    mod_name = f"docvlm_schema_{_SCHEMA_MODULE_SEQ}_{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot import schema from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

    declared = getattr(module, "SCHEMA", None)
    if isinstance(declared, type) and issubclass(declared, ExtractionSchema):
        return declared

    found = [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type)
        and issubclass(obj, ExtractionSchema)
        and obj is not ExtractionSchema
        and obj.__module__ == mod_name
    ]
    if len(found) == 1:
        return found[0]
    if not found:
        raise ValueError(f"{path} defines no ExtractionSchema subclass")
    raise ValueError(
        f"{path} defines {len(found)} schemas ({', '.join(c.__name__ for c in found)}); "
        "set SCHEMA = <TheOne> to disambiguate"
    )
