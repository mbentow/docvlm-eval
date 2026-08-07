"""Corpus loading and validation.

The format is deliberately boring: a directory, a ``manifest.jsonl``, a
``schema.py``, and the images. A clever format is the reason nobody uses your
evaluator, including you, six months from now.

::

    corpora/my-corpus/
    ├─ manifest.jsonl
    ├─ schema.py
    └─ images/
       ├─ 0001.jpg
       └─ 0002.jpg

Each manifest line is one case::

    {"id": "0001", "image": "images/0001.jpg",
     "truth": {"patient_name": "Maria Silva", "urgent": false},
     "tags": ["handwritten", "phone_photo", "low_light"]}

``tags`` are what turn the tool from useful into valuable. "89% accuracy" is not
actionable; "94% on printed, 58% on handwritten in low light" is.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docvlm_eval.schema import ExtractionSchema, load_schema

MANIFEST_NAME = "manifest.jsonl"
SCHEMA_NAME = "schema.py"

#: Tags worth using consistently. Not enforced — just a shared vocabulary.
SUGGESTED_TAGS = (
    "handwritten",
    "printed",
    "phone_photo",
    "scanned",
    "rotated",
    "low_light",
    "partial",
    "multi_page",
    "stamped",
    "faded",
    "skewed",
)


@dataclass
class Case:
    """One document plus its ground truth."""

    id: str
    image_path: Path
    truth: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def image_bytes(self) -> bytes:
        return self.image_path.read_bytes()

    def image_hash(self) -> str:
        return hashlib.sha256(self.image_bytes()).hexdigest()[:16]


@dataclass
class Corpus:
    """A versioned set of cases."""

    name: str
    root: Path
    cases: list[Case]
    schema: type[ExtractionSchema]
    hash: str

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def filter(
        self,
        *,
        tags: list[str] | None = None,
        exclude_tags: list[str] | None = None,
        limit: int | None = None,
    ) -> Corpus:
        """Return a narrowed view. The hash follows the selection, so a run on a
        subset never silently compares against a run on the whole thing."""
        cases = list(self.cases)
        if tags:
            wanted = set(tags)
            cases = [c for c in cases if wanted & set(c.tags)]
        if exclude_tags:
            unwanted = set(exclude_tags)
            cases = [c for c in cases if not (unwanted & set(c.tags))]
        if limit is not None:
            cases = cases[:limit]
        return Corpus(
            name=self.name,
            root=self.root,
            cases=cases,
            schema=self.schema,
            hash=_hash_cases(cases, self.schema),
        )

    def tag_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for case in self.cases:
            for tag in case.tags:
                counts[tag] = counts.get(tag, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


class CorpusError(ValueError):
    """The corpus is malformed. Raised with every problem found, not just the first."""


def load_corpus(path: str | Path, *, strict: bool = True) -> Corpus:
    """Load and validate a corpus directory.

    ``strict`` also checks that every ``truth`` key exists in the schema — a
    typo in a ground-truth key would otherwise be scored as a permanent
    ``missing`` and quietly cap your accuracy.
    """
    root = Path(path).resolve()
    if not root.is_dir():
        raise CorpusError(f"corpus directory not found: {root}")

    manifest = root / MANIFEST_NAME
    if not manifest.exists():
        raise CorpusError(f"missing {MANIFEST_NAME} in {root}")

    schema_path = root / SCHEMA_NAME
    if not schema_path.exists():
        raise CorpusError(f"missing {SCHEMA_NAME} in {root}")
    schema = load_schema(schema_path)
    known_fields = set(schema.model_fields)

    cases: list[Case] = []
    problems: list[str] = []
    seen_ids: set[str] = set()

    for lineno, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"{MANIFEST_NAME}:{lineno}: invalid JSON ({exc.msg})")
            continue

        case_id = str(record.get("id") or "").strip()
        if not case_id:
            problems.append(f"{MANIFEST_NAME}:{lineno}: missing 'id'")
            continue
        if case_id in seen_ids:
            problems.append(f"{MANIFEST_NAME}:{lineno}: duplicate id {case_id!r}")
            continue
        seen_ids.add(case_id)

        rel_image = record.get("image")
        if not rel_image:
            problems.append(f"{MANIFEST_NAME}:{lineno}: missing 'image'")
            continue
        image_path = (root / rel_image).resolve()
        if not image_path.exists():
            problems.append(f"{MANIFEST_NAME}:{lineno}: image not found: {rel_image}")
            continue

        truth = record.get("truth")
        if not isinstance(truth, dict):
            problems.append(f"{MANIFEST_NAME}:{lineno}: 'truth' must be an object")
            continue
        if strict:
            unknown = set(truth) - known_fields
            if unknown:
                problems.append(
                    f"{MANIFEST_NAME}:{lineno}: truth has keys not in schema: {sorted(unknown)}"
                )

        cases.append(
            Case(
                id=case_id,
                image_path=image_path,
                truth=truth,
                tags=[str(t) for t in record.get("tags", [])],
                notes=str(record.get("notes", "")),
            )
        )

    if problems:
        raise CorpusError(
            f"{len(problems)} problem(s) in corpus {root.name}:\n  - " + "\n  - ".join(problems)
        )
    if not cases:
        raise CorpusError(f"corpus {root.name} has no cases")

    return Corpus(
        name=root.name,
        root=root,
        cases=cases,
        schema=schema,
        hash=_hash_cases(cases, schema),
    )


def _hash_cases(cases: list[Case], schema: type[ExtractionSchema]) -> str:
    """Content hash over case ids, image bytes, truth and the schema.

    Changing a single ground-truth value changes the hash, so two runs can never
    be compared across an edit to the corpus without it showing up.
    """
    digest = hashlib.sha256()
    digest.update(schema.schema_hash().encode())
    for case in sorted(cases, key=lambda c: c.id):
        digest.update(case.id.encode())
        digest.update(case.image_hash().encode())
        digest.update(
            json.dumps(case.truth, sort_keys=True, ensure_ascii=False, default=str).encode()
        )
    return digest.hexdigest()[:12]
