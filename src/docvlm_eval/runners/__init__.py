"""Backend adapters."""

from __future__ import annotations

from typing import Any

from docvlm_eval.runners.base import Runner, RunnerOutput
from docvlm_eval.runners.mock import MockRunner
from docvlm_eval.runners.ollama import OllamaRunner

REGISTRY: dict[str, type[Runner]] = {
    "ollama": OllamaRunner,
    "mock": MockRunner,
}


def build_runner(kind: str, model: str, params: dict[str, Any] | None = None, **kwargs) -> Runner:
    """Instantiate a runner by name, as written in a config file."""
    try:
        cls = REGISTRY[kind]
    except KeyError as exc:
        raise ValueError(f"unknown runner {kind!r}; available: {sorted(REGISTRY)}") from exc
    return cls(model=model, params=params, **kwargs)


__all__ = ["MockRunner", "OllamaRunner", "REGISTRY", "Runner", "RunnerOutput", "build_runner"]
