"""Aggregation, confidence intervals and run comparison.

Two ideas do most of the work here.

**Resample cases, not fields.** Fields within a document are correlated — a
blurry photo hurts every field at once. Bootstrapping over fields would produce
intervals far too narrow. Every interval in this module comes from resampling
whole documents.

**Report the interval.** On 180 documents, a 2-point difference is inside the
noise. A benchmark that prints ``0.919`` without a band next to it is asking you
to over-read it.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from docvlm_eval.types import CaseResult, Outcome, RunResult

DEFAULT_BOOTSTRAP = 2000
DEFAULT_SEED = 20260807

MIN_CASES_FOR_CI = 5
MIN_BOOTSTRAP = 200
"""Below this, the percentile estimate is coarser than the label "95% CI"
implies — at 20 iterations it degenerates to min/max. Better to print no
interval than a dishonest one."""


def _percentile_bounds(samples: list[float], alpha: float) -> tuple[float, float]:
    """Symmetric percentile bounds of a sorted bootstrap distribution.

    Both tails are indexed from their own end. Taking ``int((1-alpha/2)*B)``
    from the left leaves one fewer draw above the upper bound than below the
    lower one, which shifts every interval slightly upward.
    """
    b = len(samples)
    tail = int((alpha / 2) * b)
    return samples[tail], samples[b - 1 - tail]


@dataclass
class Interval:
    """A point estimate with a percentile bootstrap interval."""

    value: float
    low: float
    high: float

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2

    def __str__(self) -> str:
        return f"{self.value:.3f} [{self.low:.3f}, {self.high:.3f}]"

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class FieldMetrics:
    """Everything measured for one field, over one run."""

    name: str
    n: int
    accuracy: float
    accuracy_ci: Interval | None
    missing_rate: float
    hallucination_rate: float
    wrong_rate: float
    malformed_rate: float
    refused_rate: float
    mean_score: float
    critical: bool = False

    def counts(self) -> dict[str, float]:
        return {
            "missing": self.missing_rate,
            "hallucinated": self.hallucination_rate,
            "wrong": self.wrong_rate,
            "malformed": self.malformed_rate,
            "refused": self.refused_rate,
        }

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["accuracy_ci"] = self.accuracy_ci.to_dict() if self.accuracy_ci else None
        return d


@dataclass
class TagMetrics:
    tag: str
    n: int
    accuracy: float
    all_fields_correct: float


@dataclass
class RunMetrics:
    """The full picture for one run."""

    run_name: str
    config_name: str
    corpus_name: str
    corpus_hash: str
    n_cases: int
    fields: list[FieldMetrics] = field(default_factory=list)
    macro_accuracy: float = 0.0
    macro_accuracy_ci: Interval | None = None
    all_fields_correct: float = 0.0
    all_fields_correct_ci: Interval | None = None
    hallucination_rate: float = 0.0
    critical_hallucination_rate: float = 0.0
    by_tag: list[TagMetrics] = field(default_factory=list)
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_mean: float = 0.0
    tokens_in_mean: float = 0.0
    tokens_out_mean: float = 0.0
    cost_total: float = 0.0
    n_refused: int = 0
    n_cached: int = 0
    """How many cases were served from cache. Their latency is a real
    measurement from the original call, but it was not measured *now* — on a
    fully cached re-run, treat the latency numbers as historical."""
    provenance: dict[str, Any] = field(default_factory=dict)

    def field_by_name(self, name: str) -> FieldMetrics | None:
        return next((f for f in self.fields if f.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "config_name": self.config_name,
            "corpus_name": self.corpus_name,
            "corpus_hash": self.corpus_hash,
            "n_cases": self.n_cases,
            "fields": [f.to_dict() for f in self.fields],
            "macro_accuracy": self.macro_accuracy,
            "macro_accuracy_ci": self.macro_accuracy_ci.to_dict()
            if self.macro_accuracy_ci
            else None,
            "all_fields_correct": self.all_fields_correct,
            "all_fields_correct_ci": self.all_fields_correct_ci.to_dict()
            if self.all_fields_correct_ci
            else None,
            "hallucination_rate": self.hallucination_rate,
            "critical_hallucination_rate": self.critical_hallucination_rate,
            "by_tag": [asdict(t) for t in self.by_tag],
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "latency_mean": self.latency_mean,
            "tokens_in_mean": self.tokens_in_mean,
            "tokens_out_mean": self.tokens_out_mean,
            "cost_total": self.cost_total,
            "n_refused": self.n_refused,
            "n_cached": self.n_cached,
            "provenance": self.provenance,
        }


# --------------------------------------------------------------------------- #
# Core aggregation
# --------------------------------------------------------------------------- #


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Returns 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[k]


def _rate(cases: list[CaseResult], name: str, outcome: Outcome) -> float:
    total = sum(1 for c in cases for f in c.fields if f.field == name)
    if not total:
        return 0.0
    hits = sum(1 for c in cases for f in c.fields if f.field == name and f.outcome is outcome)
    return hits / total


def _field_accuracy(cases: list[CaseResult], name: str) -> float:
    total = sum(1 for c in cases for f in c.fields if f.field == name)
    if not total:
        return 0.0
    hits = sum(1 for c in cases for f in c.fields if f.field == name and f.outcome.is_correct)
    return hits / total


def _macro_accuracy(cases: list[CaseResult], names: list[str], weights: dict[str, float]) -> float:
    """Weighted mean of per-field accuracies.

    Macro, not micro: a field that appears on every document should not
    dominate simply because it is always present. Fields with weight 0 are
    reported but excluded here.
    """
    total_w = sum(weights.get(n, 1.0) for n in names)
    if not total_w:
        return 0.0
    return sum(_field_accuracy(cases, n) * weights.get(n, 1.0) for n in names) / total_w


def _all_fields_correct(cases: list[CaseResult]) -> float:
    if not cases:
        return 0.0
    return sum(1 for c in cases if c.all_fields_correct) / len(cases)


def bootstrap_ci(
    cases: list[CaseResult],
    statistic,
    *,
    iterations: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> Interval | None:
    """Percentile bootstrap over resampled **cases**.

    Returns ``None`` when the sample or the number of iterations is too small
    for the interval to mean what its label says.
    """
    if len(cases) < MIN_CASES_FOR_CI or iterations < MIN_BOOTSTRAP:
        return None
    point = statistic(cases)
    rng = random.Random(seed)
    n = len(cases)
    samples = []
    for _ in range(iterations):
        resample = [cases[rng.randrange(n)] for _ in range(n)]
        samples.append(statistic(resample))
    samples.sort()
    lo, hi = _percentile_bounds(samples, alpha)
    return Interval(point, lo, hi)


def compute_metrics(
    run: RunResult,
    *,
    weights: dict[str, float] | None = None,
    critical: set[str] | None = None,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> RunMetrics:
    """Turn a raw run into every number the reports print.

    ``weights`` and ``critical`` default to the policy recorded in the run, so
    ``report`` on a stored run reproduces exactly what ``run`` printed. Passing
    them explicitly is a deliberate override.
    """
    cases = run.cases
    names = run.field_names or _infer_field_names(cases)
    weights = run.weights if weights is None else weights
    critical = set(run.critical_fields) if critical is None else critical
    scoring_names = [n for n in names if weights.get(n, 1.0) > 0]

    fields: list[FieldMetrics] = []
    for name in names:
        n_obs = sum(1 for c in cases for f in c.fields if f.field == name)
        scores = [f.score for c in cases for f in c.fields if f.field == name]
        fields.append(
            FieldMetrics(
                name=name,
                n=n_obs,
                accuracy=_field_accuracy(cases, name),
                accuracy_ci=bootstrap_ci(
                    cases,
                    lambda cs, nm=name: _field_accuracy(cs, nm),
                    iterations=bootstrap,
                    seed=seed,
                )
                if bootstrap
                else None,
                missing_rate=_rate(cases, name, Outcome.MISSING),
                hallucination_rate=_rate(cases, name, Outcome.HALLUCINATED),
                wrong_rate=_rate(cases, name, Outcome.WRONG),
                malformed_rate=_rate(cases, name, Outcome.MALFORMED),
                refused_rate=_rate(cases, name, Outcome.REFUSED),
                mean_score=statistics.fmean(scores) if scores else 0.0,
                critical=name in critical,
            )
        )

    # Cached cases keep the latency measured when the call was actually made —
    # discarding it would make a fully cached re-run report 0 ms, which reads as
    # "instant" rather than "not measured now". ``n_cached`` says which it is.
    latencies = [c.latency_ms for c in cases if c.latency_ms > 0]
    all_field_results = [f for c in cases for f in c.fields]
    n_hall = sum(1 for f in all_field_results if f.outcome is Outcome.HALLUCINATED)
    crit_results = [f for f in all_field_results if f.field in critical]
    n_crit_hall = sum(1 for f in crit_results if f.outcome is Outcome.HALLUCINATED)

    by_tag = []
    for tag in sorted({t for c in cases for t in c.tags}):
        tagged = [c for c in cases if tag in c.tags]
        by_tag.append(
            TagMetrics(
                tag=tag,
                n=len(tagged),
                accuracy=_macro_accuracy(tagged, scoring_names, weights),
                all_fields_correct=_all_fields_correct(tagged),
            )
        )
    by_tag.sort(key=lambda t: t.accuracy)

    return RunMetrics(
        run_name=run.name,
        config_name=run.config_name,
        corpus_name=run.corpus_name,
        corpus_hash=run.corpus_hash,
        n_cases=len(cases),
        fields=fields,
        macro_accuracy=_macro_accuracy(cases, scoring_names, weights),
        macro_accuracy_ci=bootstrap_ci(
            cases,
            lambda cs: _macro_accuracy(cs, scoring_names, weights),
            iterations=bootstrap,
            seed=seed,
        )
        if bootstrap
        else None,
        all_fields_correct=_all_fields_correct(cases),
        all_fields_correct_ci=bootstrap_ci(
            cases, _all_fields_correct, iterations=bootstrap, seed=seed
        )
        if bootstrap
        else None,
        hallucination_rate=n_hall / len(all_field_results) if all_field_results else 0.0,
        critical_hallucination_rate=n_crit_hall / len(crit_results) if crit_results else 0.0,
        by_tag=by_tag,
        latency_p50=percentile(latencies, 50),
        latency_p95=percentile(latencies, 95),
        latency_mean=statistics.fmean(latencies) if latencies else 0.0,
        tokens_in_mean=statistics.fmean([c.tokens_in for c in cases]) if cases else 0.0,
        tokens_out_mean=statistics.fmean([c.tokens_out for c in cases]) if cases else 0.0,
        cost_total=sum(c.cost_usd for c in cases),
        n_refused=sum(
            1 for c in cases if c.fields and all(f.outcome is Outcome.REFUSED for f in c.fields)
        ),
        n_cached=sum(1 for c in cases if c.cached),
        provenance=run.provenance,
    )


def _infer_field_names(cases: list[CaseResult]) -> list[str]:
    names: list[str] = []
    for case in cases:
        for f in case.fields:
            if f.field not in names:
                names.append(f.field)
    return names


# --------------------------------------------------------------------------- #
# Comparing two runs
# --------------------------------------------------------------------------- #


@dataclass
class FieldDelta:
    name: str
    baseline: float
    candidate: float
    delta: float
    delta_ci: Interval | None
    significant: bool
    hallucination_delta: float
    missing_delta: float


@dataclass
class RunDiff:
    baseline: RunMetrics
    candidate: RunMetrics
    fields: list[FieldDelta]
    n_paired: int
    """Documents present in both runs. Every delta and interval is over these."""
    macro_delta: float
    macro_delta_ci: Interval | None
    afc_delta: float
    afc_delta_ci: Interval | None
    latency_delta_p50: float
    latency_ratio_p50: float
    comparable: bool
    incomparable_reason: str = ""

    @property
    def regressions(self) -> list[FieldDelta]:
        """Fields that got significantly worse. The reason ``diff`` exists:
        an improvement on one field routinely breaks another."""
        return [f for f in self.fields if f.significant and f.delta < 0]

    @property
    def improvements(self) -> list[FieldDelta]:
        return [f for f in self.fields if f.significant and f.delta > 0]


def _paired(baseline: RunResult, candidate: RunResult) -> tuple[list[CaseResult], list[CaseResult]]:
    """Align the two runs on the case ids they share, in the same order.

    Paired resampling is what makes the delta interval tight enough to be
    useful: it removes the variance of "which documents are hard".
    """
    b_by_id = {c.case_id: c for c in baseline.cases}
    c_by_id = {c.case_id: c for c in candidate.cases}
    shared = [cid for cid in b_by_id if cid in c_by_id]
    shared.sort()
    return [b_by_id[i] for i in shared], [c_by_id[i] for i in shared]


def _paired_bootstrap(
    b_cases: list[CaseResult],
    c_cases: list[CaseResult],
    statistic,
    *,
    iterations: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> Interval | None:
    """Bootstrap the *difference*, resampling document indices once per draw."""
    n = len(b_cases)
    if n < MIN_CASES_FOR_CI or iterations < MIN_BOOTSTRAP:
        return None
    point = statistic(c_cases) - statistic(b_cases)
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        samples.append(statistic([c_cases[i] for i in idx]) - statistic([b_cases[i] for i in idx]))
    samples.sort()
    lo, hi = _percentile_bounds(samples, alpha)
    return Interval(point, lo, hi)


def diff_runs(
    baseline: RunResult,
    candidate: RunResult,
    *,
    weights: dict[str, float] | None = None,
    critical: set[str] | None = None,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> RunDiff:
    """Compare two runs on the cases they have in common.

    Point estimates and intervals are both computed on the **paired subset**, so
    the interval is always centred on the delta printed beside it. Comparing a
    filtered run against a full one would otherwise put a confidence band around
    a number it was not computed from.
    """
    weights = baseline.weights if weights is None else weights
    critical = set(baseline.critical_fields) if critical is None else critical
    b_metrics = compute_metrics(
        baseline, weights=weights, critical=critical, bootstrap=bootstrap, seed=seed
    )
    c_metrics = compute_metrics(
        candidate, weights=weights, critical=critical, bootstrap=bootstrap, seed=seed
    )

    comparable = baseline.corpus_hash == candidate.corpus_hash
    reason = (
        ""
        if comparable
        else (
            f"corpus hash differs ({baseline.corpus_hash} vs {candidate.corpus_hash}); "
            "the two runs did not see the same documents or the same ground truth"
        )
    )

    b_cases, c_cases = _paired(baseline, candidate)
    names = baseline.field_names or _infer_field_names(baseline.cases)
    scoring_names = [n for n in names if weights.get(n, 1.0) > 0]

    field_deltas: list[FieldDelta] = []
    for name in names:
        b_acc = _field_accuracy(b_cases, name)
        c_acc = _field_accuracy(c_cases, name)
        ci = (
            _paired_bootstrap(
                b_cases,
                c_cases,
                lambda cs, nm=name: _field_accuracy(cs, nm),
                iterations=bootstrap,
                seed=seed,
            )
            if bootstrap
            else None
        )
        field_deltas.append(
            FieldDelta(
                name=name,
                baseline=b_acc,
                candidate=c_acc,
                delta=c_acc - b_acc,
                delta_ci=ci,
                significant=bool(ci and (ci.low > 0 or ci.high < 0)),
                hallucination_delta=_rate(c_cases, name, Outcome.HALLUCINATED)
                - _rate(b_cases, name, Outcome.HALLUCINATED),
                missing_delta=_rate(c_cases, name, Outcome.MISSING)
                - _rate(b_cases, name, Outcome.MISSING),
            )
        )

    macro_ci = (
        _paired_bootstrap(
            b_cases,
            c_cases,
            lambda cs: _macro_accuracy(cs, scoring_names, weights),
            iterations=bootstrap,
            seed=seed,
        )
        if bootstrap
        else None
    )
    afc_ci = (
        _paired_bootstrap(b_cases, c_cases, _all_fields_correct, iterations=bootstrap, seed=seed)
        if bootstrap
        else None
    )

    # Deltas come from the paired subset, matching the interval beside them.
    macro_delta = _macro_accuracy(c_cases, scoring_names, weights) - _macro_accuracy(
        b_cases, scoring_names, weights
    )
    afc_delta = _all_fields_correct(c_cases) - _all_fields_correct(b_cases)

    b_p50, c_p50 = b_metrics.latency_p50, c_metrics.latency_p50
    return RunDiff(
        baseline=b_metrics,
        candidate=c_metrics,
        fields=field_deltas,
        n_paired=len(b_cases),
        macro_delta=macro_delta,
        macro_delta_ci=macro_ci,
        afc_delta=afc_delta,
        afc_delta_ci=afc_ci,
        latency_delta_p50=c_p50 - b_p50,
        latency_ratio_p50=(c_p50 / b_p50) if b_p50 else 0.0,
        comparable=comparable,
        incomparable_reason=reason,
    )
