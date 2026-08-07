"""Runner interface.

A runner turns *(image bytes, prompt, JSON Schema)* into *(parsed dict, telemetry)*.
Everything backend-specific lives behind this boundary so that adding MLX or a
hosted API never touches the scoring code.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class RunnerOutput:
    """One backend call."""

    data: dict[str, Any] | None
    """Parsed object, or ``None`` if the call failed or the output was unusable."""
    raw: str = ""
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class Runner(ABC):
    """Base class for backends.

    Implementations must not raise on a failed document — a single bad document
    should not abort a 200-document sweep. Return ``RunnerOutput(data=None,
    error=...)`` and let the scorer record it as ``refused``.
    """

    name: str = "base"

    def __init__(self, model: str, params: dict[str, Any] | None = None) -> None:
        self.model = model
        self.params = dict(params or {})

    @abstractmethod
    async def extract(
        self,
        image: bytes,
        prompt: str,
        json_schema: dict[str, Any],
        case_id: str = "",
    ) -> RunnerOutput:
        """Run one document.

        ``case_id`` is passed for logging and for replay backends. A real
        backend must ignore it — it is an identifier, not a hint.
        """

    async def describe(self) -> dict[str, Any]:
        """Backend facts worth recording in the run provenance (model digest,
        server version, quantisation). Best-effort; never raises."""
        return {"runner": self.name, "model": self.model}

    async def aclose(self) -> None:  # noqa: B027 - optional hook, not every backend has one
        """Release connections. Safe to call twice."""

    # -- shared helpers ---------------------------------------------------- #

    @staticmethod
    def parse_json(text: str) -> dict[str, Any] | None:
        """Recover an object from model output.

        With constrained decoding the raw text is already valid JSON. This
        fallback exists for backends that ignore the schema, and for the honest
        reason that it lets you *measure* how often that happens instead of
        assuming it never does.
        """
        if not text:
            return None
        candidates = [text.strip()]
        fenced = _FENCE.search(text)
        if fenced:
            candidates.insert(0, fenced.group(1).strip())
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]
        return None
