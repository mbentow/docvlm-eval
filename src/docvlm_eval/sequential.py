"""Sequential A/B: stop a long evaluation as soon as the answer is known.

A candidate that is clearly better, clearly worse, or clearly the same rarely
needs the whole corpus to say so. On a backend that serves one document at a
time, the documents you do not run are the only speed-up there is.

Looking at a running result and stopping "when it looks significant" is the
classic way to manufacture a finding: check a null difference often enough and
the interval will exclude zero on some look. So the looks are planned, and each
gets a slice of the error budget:

- ``alpha`` is spent with the O'Brien–Fleming-type function of Lan & DeMets,
  ``α(t) = 2 − 2Φ(z₁₋α/₂ / √t)``. Early looks get almost nothing — stopping at
  10% of the corpus needs an overwhelming effect — and the last look gets
  nearly the full ``alpha``, so running to the end costs little versus a single
  fixed test.
- Look ``k`` tests at ``α(t_k) − α(t_{k−1})``. The increments sum to ``alpha``,
  so by the union bound the chance of *any* false stop over all looks is at
  most ``alpha``, whatever the correlation between looks. Conservative, and
  correct without simulation.
- Half of each increment goes to the quality question, half to the
  critical-hallucination question, so the two together stay within ``alpha``.

The statistic is the per-document paired difference, ``d_i = cand_i − base_i``,
on documents both runs have, with a normal interval. Pairing removes the
document difficulty that dominates variance; the normal interval gives any
quantile cheaply, which a bootstrap at ``α = 10⁻⁴`` would not.

Validity requires that the order in which documents arrive does not depend on
their outcome. Feed the pipeline in the order from ``docvlm-eval order``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from statistics import NormalDist
from typing import Any

from docvlm_eval.types import CaseResult, Outcome

_Z = NormalDist()

#: Below this many paired documents no look is taken: the normal interval is not
#: trustworthy, and a pipeline's first documents are often its warm-up.
MIN_N = 30

#: Added to the sum of squared deviations. With identical outputs on the first
#: documents the sample variance is 0 and the interval collapses to a point,
#: which would declare equivalence on 30 documents. Half a discordant pair is
#: enough to keep the interval honest without moving it on real data.
VARIANCE_FLOOR = 0.5

DECISIONS = ("better", "worse", "harm", "equivalent", "inconclusive")


def spent_alpha(t: float, alpha: float) -> float:
    """Cumulative alpha spent at information fraction ``t`` (O'Brien–Fleming type)."""
    if t <= 0:
        return 0.0
    if t >= 1:
        return alpha
    return 2 - 2 * _Z.cdf(_Z.inv_cdf(1 - alpha / 2) / math.sqrt(t))


@dataclass
class SequentialPlan:
    n_total: int
    """Documents the full run would have. Fixes the information fractions."""
    alpha: float = 0.05
    looks: int = 10
    margin: float = 0.01
    """Equivalence margin: a difference smaller than this is not worth having.
    Stopping for equivalence means the whole interval sits inside ±margin."""
    metric: str = "macro"
    """``macro`` (weighted share of correct fields per document) or ``afc``
    (document entirely correct)."""
    min_n: int = MIN_N

    def look_sizes(self) -> list[int]:
        sizes = []
        for k in range(1, self.looks + 1):
            n = math.ceil(self.n_total * k / self.looks)
            if n >= min(self.min_n, self.n_total) and (not sizes or n > sizes[-1]):
                sizes.append(n)
        return sizes

    def alpha_at(self, look_index: int) -> float:
        """Alpha increment for look ``look_index`` (0-based) of ``look_sizes``."""
        sizes = self.look_sizes()
        t = sizes[look_index] / self.n_total
        t_prev = sizes[look_index - 1] / self.n_total if look_index else 0.0
        return spent_alpha(t, self.alpha) - spent_alpha(t_prev, self.alpha)


@dataclass
class Look:
    n: int
    alpha: float
    delta: float
    low: float
    high: float
    harm_delta: float
    harm_low: float
    decision: str = "continue"
    final: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SequentialState:
    plan: SequentialPlan
    weights: dict[str, float] = field(default_factory=dict)
    critical: set[str] = field(default_factory=set)
    looks: list[Look] = field(default_factory=list)

    @property
    def decision(self) -> str:
        return self.looks[-1].decision if self.looks else "continue"

    @property
    def stopped(self) -> bool:
        return self.decision != "continue"


def doc_score(case: CaseResult, metric: str, weights: dict[str, float]) -> float:
    if metric == "afc":
        return 1.0 if case.all_fields_correct else 0.0
    total = hits = 0.0
    for f in case.fields:
        w = weights.get(f.field, 1.0)
        if w <= 0:
            continue
        total += w
        hits += w if f.outcome.is_correct else 0.0
    return hits / total if total else 0.0


def doc_harm(case: CaseResult, critical: set[str]) -> float:
    return float(
        sum(1 for f in case.fields if f.field in critical and f.outcome is Outcome.HALLUCINATED)
    )


def _interval(diffs: list[float], alpha: float, sided: int = 2) -> tuple[float, float, float]:
    n = len(diffs)
    mean = sum(diffs) / n
    ss = sum((d - mean) ** 2 for d in diffs) + VARIANCE_FLOOR
    se = math.sqrt(ss / (n - 1) / n) if n > 1 else float("inf")
    z = _Z.inv_cdf(1 - alpha / sided)
    return mean, mean - z * se, mean + z * se


def evaluate(
    state: SequentialState,
    baseline: dict[str, CaseResult],
    candidate_in_order: list[CaseResult],
) -> list[Look]:
    """Take every planned look the candidate has reached and not yet taken.

    ``candidate_in_order`` is the candidate's documents in arrival order. Only
    documents present in the baseline count. Returns the new looks; stops
    adding once a decision is reached.
    """
    paired = [(baseline[c.case_id], c) for c in candidate_in_order if c.case_id in baseline]
    sizes = state.plan.look_sizes()
    new: list[Look] = []
    for i, size in enumerate(sizes):
        if i < len(state.looks):
            continue
        if state.stopped or len(paired) < size:
            break
        window = paired[:size]
        a = state.plan.alpha_at(i)
        quality = [
            doc_score(c, state.plan.metric, state.weights)
            - doc_score(b, state.plan.metric, state.weights)
            for b, c in window
        ]
        harm = [doc_harm(c, state.critical) - doc_harm(b, state.critical) for b, c in window]
        delta, low, high = _interval(quality, a / 2)
        harm_delta, harm_low, _ = _interval(harm, a / 2, sided=1)

        if state.critical and harm_low > 0:
            decision = "harm"
        elif low > 0:
            decision = "better"
        elif high < 0:
            decision = "worse"
        elif -state.plan.margin < low and high < state.plan.margin:
            decision = "equivalent"
        elif i == len(sizes) - 1:
            decision = "inconclusive"
        else:
            decision = "continue"
        look = Look(
            n=size,
            alpha=a,
            delta=delta,
            low=low,
            high=high,
            harm_delta=harm_delta,
            harm_low=harm_low,
            decision=decision,
            final=i == len(sizes) - 1,
        )
        state.looks.append(look)
        new.append(look)
    return new


def verdict(state: SequentialState) -> str:
    """One sentence. ``inconclusive`` is a result too: it says how big the
    difference is *not*."""
    if not state.looks:
        return f"no look yet (needs {state.plan.look_sizes()[:1] or ['?']} paired documents)"
    look = state.looks[-1]
    saved = state.plan.n_total - look.n
    where = f"at {look.n}/{state.plan.n_total} documents"
    ci = f"Δ {look.delta:+.4f} [{look.low:+.4f}, {look.high:+.4f}]"
    if look.decision == "harm":
        return (
            f"STOP — critical hallucinations up {look.harm_delta:+.3f}/doc {where}; "
            f"{saved} documents not needed"
        )
    if look.decision == "better":
        return f"STOP — better, {ci} {where}; {saved} documents not needed"
    if look.decision == "worse":
        return f"STOP — worse, {ci} {where}; {saved} documents not needed"
    if look.decision == "equivalent":
        return (
            f"STOP — equivalent within ±{state.plan.margin:.3f}, {ci} {where}; "
            f"{saved} documents not needed"
        )
    if look.decision == "inconclusive":
        return f"inconclusive over the whole corpus, {ci}"
    return f"continue — {ci} {where}"
