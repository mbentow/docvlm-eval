"""Run persistence.

Runs are immutable JSON files under ``runs/``. Writing a run twice under the
same name is refused unless ``--overwrite`` is given: a benchmark you can
silently edit is not a benchmark.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from docvlm_eval.types import RunResult

DEFAULT_RUNS_DIR = Path("runs")
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str) -> str:
    return _SAFE.sub("-", name).strip("-") or "run"


@dataclass
class RunStore:
    directory: Path = DEFAULT_RUNS_DIR

    def path_for(self, name: str) -> Path:
        return Path(self.directory) / f"{safe_name(name)}.json"

    def save(self, run: RunResult, *, overwrite: bool = False) -> Path:
        path = self.path_for(run.name)
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"run {run.name!r} already exists at {path}. "
                "Use --overwrite, or pass --name to store it under a new name."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(run.to_json(), encoding="utf-8")
        return path

    def load(self, name_or_path: str) -> RunResult:
        candidate = Path(name_or_path)
        if not candidate.exists():
            candidate = self.path_for(name_or_path)
        if not candidate.exists():
            available = ", ".join(self.list_names()) or "none"
            raise FileNotFoundError(
                f"run {name_or_path!r} not found in {self.directory}. Available: {available}"
            )
        return RunResult.from_json(candidate.read_text(encoding="utf-8"))

    def list_names(self) -> list[str]:
        directory = Path(self.directory)
        if not directory.is_dir():
            return []
        return sorted(p.stem for p in directory.glob("*.json"))

    def list_runs(self) -> list[tuple[str, str, int]]:
        """``(name, finished_at, n_cases)`` for every stored run."""
        out = []
        for name in self.list_names():
            try:
                run = self.load(name)
            except (ValueError, KeyError):
                continue
            out.append((name, run.finished_at, len(run.cases)))
        return sorted(out, key=lambda t: t[1], reverse=True)
