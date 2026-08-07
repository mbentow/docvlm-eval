"""Run configurations.

A config is a named combination of runner + model + parameters + prompt. It is
hashed, and the hash goes into the run and into the cache key, so that editing a
prompt invalidates exactly the results it should invalidate and nothing else.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


@dataclass
class Config:
    """One evaluated configuration."""

    name: str
    runner: str = "ollama"
    model: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    """Inline prompt text, or the text loaded from ``prompt_file``."""
    prompt_file: str = ""
    host: str | list[str] = ""
    """Backend endpoint(s). ``${OLLAMA_HOST}`` style references are expanded."""
    preprocess: str = "none"
    """Reserved for the image-preparation stage (see the docprep tool). Recorded
    in provenance so a run made with preprocessing is never silently compared to
    one made without."""
    weights: dict[str, float] = field(default_factory=dict)
    """Per-field weight override for the macro average. 0 = report but exclude."""
    critical_fields: list[str] = field(default_factory=list)
    """Fields where a hallucination is a safety issue rather than a nuisance."""
    source_path: str = ""

    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(self.prompt.encode()).hexdigest()[:12]

    @property
    def hash(self) -> str:
        payload = json.dumps(
            {
                "runner": self.runner,
                "model": self.model,
                "params": self.params,
                "prompt": self.prompt,
                "preprocess": self.preprocess,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["prompt_hash"] = self.prompt_hash
        d["config_hash"] = self.hash
        # The prompt itself stays private: only its hash goes into provenance.
        d.pop("prompt", None)
        # Absolute paths leak the machine a run was produced on into every
        # published report. The file name is the part that identifies it.
        if d.get("source_path"):
            d["source_path"] = Path(d["source_path"]).name
        return d

    def hosts(self) -> list[str]:
        if isinstance(self.host, str):
            return [h.strip() for h in self.host.split(",") if h.strip()]
        return [str(h) for h in self.host if h]


def _expand_env(value: Any) -> Any:
    """Expand ``${VAR}`` and ``${VAR:-default}`` inside strings.

    Endpoints differ between laptops, CI and the inference box; hard-coding them
    into a committed config is how configs stop being shareable.

    ``:-`` matches the shell: an unset **or empty** variable falls back to the
    default. An exported-but-empty ``OLLAMA_HOST`` is the common case, and
    treating it as "set" produces an empty host and a confusing error later.
    """
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            return os.environ.get(match.group(1)) or (match.group(2) or "")

        return _ENV_REF.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: str | Path) -> Config:
    """Read a YAML config. ``prompt_file`` is resolved relative to the config."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: config must be a YAML mapping")
    raw = _expand_env(raw)

    prompt = str(raw.get("prompt_text", "") or "")
    prompt_file = str(raw.get("prompt", "") or raw.get("prompt_file", "") or "")
    if prompt_file:
        candidate = (path.parent / prompt_file).resolve()
        if not candidate.exists():
            candidate = Path(prompt_file).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"{path}: prompt file not found: {prompt_file}")
        prompt = candidate.read_text(encoding="utf-8")

    return Config(
        name=str(raw.get("name") or path.stem),
        runner=str(raw.get("runner", "ollama")),
        model=str(raw.get("model", "")),
        params=dict(raw.get("params", {}) or {}),
        prompt=prompt,
        prompt_file=prompt_file,
        host=raw.get("host", "") or "",
        preprocess=str(raw.get("preprocess", "none")),
        weights={str(k): float(v) for k, v in (raw.get("weights", {}) or {}).items()},
        critical_fields=[str(f) for f in (raw.get("critical_fields", []) or [])],
        source_path=str(path),
    )


DEFAULT_PROMPT = """You are reading a scanned document.

Extract the fields described by the JSON schema. Rules:
1. Copy values exactly as they appear on the document. Do not correct, expand or
   translate them.
2. If a field is not present or you cannot read it, return null. Do not guess.
3. Return only the JSON object.
"""
