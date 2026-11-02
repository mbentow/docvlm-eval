"""Self-agreement: how much a configuration disagrees with itself.

Run the same config twice. Every document whose correctness flips between the
two runs is noise the backend produces on its own — sampling, batching, a
runtime that reorders a reduction. A candidate whose difference from the
baseline is of the same size as that noise has not been shown to be different.

The numbers that matter:

- **value agreement** per field: the two runs returned the same value;
- **correctness flips**: one run right, the other wrong. This is the part of
  the noise that moves a score;
- **noise delta**: the paired difference between the two runs, with its
  interval. If it excludes zero, the "same" config is not reproducible and no
  diff made with it can be trusted until that is fixed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from docvlm_eval.sequential import _interval, doc_score
from docvlm_eval.types import RunResult


@dataclass
class FieldAgreement:
    name: str
    n: int
    value_agreement: float
    correctness_flips: float


@dataclass
class AgreementReport:
    n_paired: int
    fields: list[FieldAgreement] = field(default_factory=list)
    document_flips: float = 0.0
    """Documents whose all-fields-correct status differs between the runs."""
    noise_delta: float = 0.0
    noise_low: float = 0.0
    noise_high: float = 0.0
    same_config: bool = True

    @property
    def reproducible(self) -> bool:
        return self.noise_low <= 0 <= self.noise_high

    def detectable_delta(self, n: int | None = None) -> float:
        """Smallest macro difference distinguishable from this noise at ``n``
        documents (two-sided 95%). Below it, a diff is a coin toss."""
        n = n or self.n_paired
        if not n:
            return 0.0
        return 1.96 * math.sqrt(max(self.document_flips, 1 / n) / n)


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def agreement(
    a: RunResult, b: RunResult, *, weights: dict[str, float] | None = None
) -> AgreementReport:
    weights = a.weights if weights is None else weights
    b_by_id = {c.case_id: c for c in b.cases}
    pairs = [(c, b_by_id[c.case_id]) for c in a.cases if c.case_id in b_by_id]
    report = AgreementReport(n_paired=len(pairs), same_config=a.config_hash == b.config_hash)
    if not pairs:
        return report

    names = a.field_names or sorted({f.field for c, _ in pairs for f in c.fields})
    for name in names:
        same = flips = n = 0
        for ca, cb in pairs:
            fa = next((f for f in ca.fields if f.field == name), None)
            fb = next((f for f in cb.fields if f.field == name), None)
            if fa is None or fb is None:
                continue
            n += 1
            same += _canon(fa.predicted) == _canon(fb.predicted)
            flips += fa.outcome.is_correct != fb.outcome.is_correct
        if n:
            report.fields.append(FieldAgreement(name, n, same / n, flips / n))

    report.document_flips = sum(
        ca.all_fields_correct != cb.all_fields_correct for ca, cb in pairs
    ) / len(pairs)
    diffs = [doc_score(cb, "macro", weights) - doc_score(ca, "macro", weights) for ca, cb in pairs]
    if len(diffs) > 1:
        report.noise_delta, report.noise_low, report.noise_high = _interval(diffs, 0.05)
    return report


def render(report: AgreementReport) -> str:
    lines = [f"SELF-AGREEMENT over {report.n_paired} paired documents", ""]
    if not report.same_config:
        lines += [
            "! the two runs have different config hashes — this measures a difference, not noise",
            "",
        ]
    lines.append(f"{'FIELD':<20} {'SAME VALUE':>10} {'FLIPS':>8} {'n':>6}")
    for f in report.fields:
        lines.append(
            f"{f.name:<20} {f.value_agreement:>10.1%} {f.correctness_flips:>8.1%} {f.n:>6}"
        )
    lines += [
        "",
        f"documents flipping all-fields-correct: {report.document_flips:.1%}",
        f"noise delta (macro): {report.noise_delta:+.4f} "
        f"[{report.noise_low:+.4f}, {report.noise_high:+.4f}]",
        f"smallest delta distinguishable from this noise at n={report.n_paired}: "
        f"±{report.detectable_delta():.4f}",
    ]
    if not report.reproducible:
        lines.append(
            "VERDICT: NOT REPRODUCIBLE — the same config scores differently on its own; "
            "fix determinism (temperature 0, fixed runtime) before trusting any diff"
        )
    else:
        lines.append("VERDICT: reproducible within noise")
    return "\n".join(lines)
