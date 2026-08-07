"""Mock runner — no backend required.

Its job is not to be a toy. It lets the test suite exercise the whole pipeline
deterministically, and it lets a new user run ``docvlm-eval`` end to end in
thirty seconds without pulling a 19 GB model. It can also replay a recorded
``predictions.jsonl`` to score outputs produced somewhere else entirely.

The ``noise`` and ``hallucinate`` knobs deliberately produce every failure mode,
so the shape of the report — including the hallucination column the whole tool
is built around — is visible before any GPU is involved.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

from docvlm_eval.runners.base import Runner, RunnerOutput

INVENTED_TEXT = "VALOR INVENTADO"
INVENTED_DIGITS = "99999999"


class MockRunner(Runner):
    """Replays fixed predictions, or perturbs the ground truth by a set amount."""

    name = "mock"

    def __init__(
        self,
        model: str = "mock",
        params: dict[str, Any] | None = None,
        *,
        predictions: str | Path | None = None,
        truths: dict[str, dict[str, Any]] | None = None,
        noise: float = 0.0,
        hallucinate: float = 0.0,
        seed: int = 7,
        latency_ms: float = 12.0,
    ) -> None:
        super().__init__(model, params)
        self.noise = float(self.params.get("noise", noise))
        self.hallucinate = float(self.params.get("hallucinate", hallucinate))
        self.latency_ms = float(self.params.get("latency_ms", latency_ms))
        self.seed = int(self.params.get("seed", seed))
        self._truths = truths or {}
        self._predictions: dict[str, dict[str, Any]] = {}

        source = self.params.get("predictions", predictions)
        if source:
            for line in Path(source).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self._predictions[str(record["id"])] = record.get("predicted", {})

    async def extract(
        self,
        image: bytes,
        prompt: str,
        json_schema: dict[str, Any],
        case_id: str = "",
    ) -> RunnerOutput:
        if self.latency_ms:
            await asyncio.sleep(min(self.latency_ms, 50) / 1000)
        if case_id in self._predictions:
            data = self._predictions[case_id]
        else:
            data = self._perturb(case_id, self._truths.get(case_id, {}), json_schema)
        return RunnerOutput(
            data=data,
            raw=json.dumps(data, ensure_ascii=False, default=str),
            latency_ms=self.latency_ms,
            tokens_in=1024,
            tokens_out=64,
        )

    def _perturb(
        self, case_id: str, truth: dict[str, Any], json_schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Seed per case, never from a shared generator.

        A shared RNG would make the output depend on completion order, so the
        same run would score differently at ``-j 1`` and ``-j 4``, and differently
        again on a partial cache hit. That is exactly the kind of irreproducible
        benchmark this tool exists to argue against.
        """
        rng = random.Random(f"{self.seed}|{case_id}")
        out: dict[str, Any] = {}
        for key, prop in json_schema.get("properties", {}).items():
            value = truth.get(key)
            is_empty = value is None or value == "" or value == []

            # Hallucination: the truth is empty and the model fills it in anyway.
            # This is the failure the whole report is built around, so it gets
            # its own knob rather than depending on the corpus happening to
            # contain enough empty fields.
            if is_empty and rng.random() < self.hallucinate:
                out[key] = _invented(key, prop)
                continue

            if rng.random() >= self.noise:
                out[key] = value
                continue

            mode = rng.choice(["drop", "typo", "typo"])
            if mode == "drop":
                out[key] = None
            elif isinstance(value, str) and len(value) > 3:
                idx = rng.randrange(len(value))
                out[key] = value[:idx] + "X" + value[idx + 1 :]
            elif isinstance(value, bool):
                out[key] = not value
            elif isinstance(value, list) and value:
                out[key] = value[:-1] if len(value) > 1 else [*value, "EXAME EXTRA"]
            else:
                out[key] = value
        return out

    async def describe(self) -> dict[str, Any]:
        return {
            "runner": self.name,
            "model": self.model,
            "noise": self.noise,
            "hallucinate": self.hallucinate,
            "seed": self.seed,
            "replayed_predictions": len(self._predictions),
        }


def _invented(key: str, prop: dict[str, Any]) -> Any:
    """Something type-plausible, so the outcome is `hallucinated`, not `malformed`."""
    types = prop.get("type")
    if isinstance(types, list):
        types = next((t for t in types if t != "null"), "string")
    if types == "boolean":
        return True
    if types in ("number", "integer"):
        return 42
    if types == "array":
        return ["ITEM INVENTADO"]
    return INVENTED_DIGITS if "id" in key or "crm" in key else INVENTED_TEXT
