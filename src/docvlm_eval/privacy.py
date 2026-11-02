"""Keep personal data out of everything a run writes.

A field declared ``pii=True`` (a patient's name, a document number) is scored
on its real value in memory, then stored only as a keyed hash. Every number in
a report survives — outcomes, scores, agreement between runs, since equal
values still hash equal — but a run file, a report or a diff pasted into a
ticket no longer carries the value. ``raw_output`` is dropped for the same
reason: it is the model's whole answer, PII included.

The hash is keyed with the corpus hash, so a value cannot be looked up in a
table built for another corpus. It is not encryption: anyone who holds the
corpus can recompute it, which is also what makes agreement across runs work.

The second guard is where things get written. A run of a corpus with PII fields
is refused inside a git work tree unless git ignores the path: one ``git add .``
is how private evaluation data ends up in a public repository.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from docvlm_eval.types import CaseResult

PREFIX = "pii:"


class TrackedOutputError(RuntimeError):
    pass


def pseudonymise(value: Any, key: str) -> Any:
    if value is None or value == "" or value == []:
        return value
    if isinstance(value, str) and value.startswith(PREFIX):
        return value
    canon = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return PREFIX + hashlib.sha256(f"{key}|{canon}".encode()).hexdigest()[:16]


def redact_case(case: CaseResult, pii_fields: set[str], key: str) -> CaseResult:
    if not pii_fields:
        return case
    for f in case.fields:
        if f.field in pii_fields:
            f.truth = pseudonymise(f.truth, key)
            f.predicted = pseudonymise(f.predicted, key)
    case.raw_output = ""
    return case


def guard_untracked(path: str | Path, *, allow: bool = False) -> None:
    """Refuse output directory ``path`` if it sits in a git work tree and git
    would track what is written there."""
    if allow:
        return
    target = Path(path).resolve()
    probe = target if target.is_dir() else target.parent
    while not probe.exists():
        probe = probe.parent
    try:
        inside = subprocess.run(
            ["git", "-C", str(probe), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return  # no git, nothing to protect against
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return
    # Ask about a file *inside* the directory: ``runs/`` in .gitignore matches
    # only a directory, and a directory not created yet looks like a file to git.
    ignored = subprocess.run(
        ["git", "-C", str(probe), "check-ignore", "-q", str(target / "run.json")],
        capture_output=True,
        timeout=10,
    )
    if ignored.returncode != 0:
        raise TrackedOutputError(
            f"{target} is inside a git work tree and not ignored, and this corpus has PII "
            "fields. Write runs outside the repository, add the path to .gitignore, or pass "
            "--allow-tracked if you are certain."
        )
