"""Selective prediction: the coverage/accuracy trade-off.

Accuracy alone does not tell you whether a system is deployable. The question a
document pipeline actually has to answer is:

    *If I send the least-confident 20% of documents to a human, what accuracy
    remains on the 80% that go through automatically?*

That curve — not the headline number — is what decides the product. A model at
0.75 accuracy that knows which quarter it got wrong is worth far more than a
model at 0.80 that is uniformly, confidently mediocre: the first one can run at
75% coverage with 0.95 accuracy, the second cannot run unattended at all.

Two things are measured here:

**Risk–coverage curve.** Sort documents by confidence, accept the top ``c``
fraction, and report accuracy on the accepted set. ``AURC`` (area under the
risk–coverage curve, lower is better) summarises the whole curve in one number.

**How much of that is real.** A curve is only meaningful if the confidence
signal actually ranks. ``AURC`` is therefore reported next to the AURC of a
random ordering — if they are close, your confidence signal is noise and the
"send the worst 20% to a human" plan will not work, whatever the shape of the
plot suggests.

Confidence source, in order of preference:

1. a field named ``confidence`` (or ``--confidence-field``) that the model fills
   in as part of the schema;
2. otherwise, the **mean field score** of the document, which is free: it
   already blends fuzzy similarity and exact matches;
3. abstention count is exposed separately, because a model that returns nulls
   when unsure is doing selective prediction implicitly, and that deserves to be
   measured rather than penalised.
"""

from __future__ import annotations

import hashlib
import random
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from docvlm_eval.normalize import parse_number
from docvlm_eval.types import CaseResult, Outcome

DEFAULT_CONFIDENCE_FIELD = "confidence"
#: Coverage levels reported as named operating points.
OPERATING_POINTS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
#: Shuffles averaged to estimate the AURC a useless signal would produce.
RANDOM_TRIALS = 25
#: A policy that automates less than this is not a policy. Without the floor, a
#: useless confidence signal "meets" a high target by accepting a single lucky
#: document, which reads as success and is not one.
MIN_VIABLE_COVERAGE = 0.05


@dataclass
class Point:
    """One operating point on the curve."""

    coverage: float
    """Fraction of documents processed automatically."""
    n_accepted: int
    accuracy: float
    """All-fields-correct among the accepted documents."""
    risk: float
    """1 - accuracy. The error rate a reviewer never sees."""
    hallucination_rate: float
    """Hallucinated fields among accepted documents — the errors that hurt."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SelectiveReport:
    curve: list[Point] = field(default_factory=list)
    """Fine-grained curve, one point per document, for plotting."""
    operating_points: list[Point] = field(default_factory=list)
    aurc: float = 0.0
    """Area under the risk-coverage curve. Lower is better."""
    aurc_random: float = 0.0
    """Same, with documents shuffled. The floor a useless signal produces."""
    confidence_source: str = ""
    n_cases: int = 0

    @property
    def ranking_gain(self) -> float:
        """How much of the curve comes from the signal rather than the base rate.

        ``0`` means the confidence signal ranks no better than shuffling, and no
        review policy built on it will work. Above ~0.15 the signal is carrying
        real information.
        """
        if self.aurc_random <= 0:
            return 0.0
        return max(0.0, (self.aurc_random - self.aurc) / self.aurc_random)

    def accuracy_at(self, coverage: float) -> float | None:
        """Accuracy if you automate this fraction and review the rest."""
        best = None
        for point in self.curve:
            if point.coverage <= coverage + 1e-9:
                best = point
        return best.accuracy if best else None

    def coverage_for(self, target_accuracy: float) -> float:
        """Largest fraction you can automate and still hit an accuracy target.

        This is the number to take into a conversation with the business: *"at
        94% accuracy we can automate 62% of documents."*
        """
        best = 0.0
        for point in self.curve:
            if point.accuracy >= target_accuracy:
                best = max(best, point.coverage)
        return best

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence_source": self.confidence_source,
            "n_cases": self.n_cases,
            "aurc": self.aurc,
            "aurc_random": self.aurc_random,
            "ranking_gain": self.ranking_gain,
            "operating_points": [p.to_dict() for p in self.operating_points],
        }


def case_confidence(case: CaseResult, confidence_field: str | None = None) -> float:
    """Confidence in ``[0, 1]`` for one document.

    An explicit field wins. Otherwise the mean field score is used — it is not a
    calibrated probability and is not treated as one; it only has to *rank*.
    """
    name = confidence_field or DEFAULT_CONFIDENCE_FIELD
    for result in case.fields:
        if result.field == name:
            value = parse_number(result.predicted)
            if value is not None:
                return max(0.0, min(1.0, value if value <= 1 else value / 100))
    scores = [f.score for f in case.fields if f.field != name]
    return statistics.fmean(scores) if scores else 0.0


def _accuracy(cases: list[CaseResult]) -> float:
    return sum(1 for c in cases if c.all_fields_correct) / len(cases) if cases else 0.0


def _hallucination(cases: list[CaseResult]) -> float:
    results = [f for c in cases for f in c.fields]
    if not results:
        return 0.0
    return sum(1 for f in results if f.outcome is Outcome.HALLUCINATED) / len(results)


def _aurc(ordered: list[CaseResult]) -> float:
    """Mean risk across every prefix of the ordering."""
    if not ordered:
        return 0.0
    correct = 0
    total_risk = 0.0
    for i, case in enumerate(ordered, start=1):
        correct += 1 if case.all_fields_correct else 0
        total_risk += 1 - correct / i
    return total_risk / len(ordered)


def _rank(cases: list[CaseResult], confidence_field: str | None, seed: int) -> list[CaseResult]:
    """Most confident first.

    Ties are broken by a seeded hash, never by input order: ``sorted`` is
    stable, so a degenerate confidence signal would otherwise inherit whatever
    order the corpus happens to have and draw a curve that ranks nothing.
    """

    def key(case: CaseResult) -> tuple[float, str]:
        jitter = hashlib.sha1(f"{seed}|{case.case_id}".encode()).hexdigest()
        return (-case_confidence(case, confidence_field), jitter)

    return sorted(cases, key=key)


def selective_report(
    cases: list[CaseResult],
    *,
    confidence_field: str | None = None,
    seed: int = 20260807,
) -> SelectiveReport:
    """Build the risk–coverage curve for one run."""
    usable = [c for c in cases if c.fields]
    if not usable:
        return SelectiveReport(confidence_source="none", n_cases=0)

    explicit = any(
        f.field == (confidence_field or DEFAULT_CONFIDENCE_FIELD) for c in usable for f in c.fields
    )
    source = (
        f"schema field '{confidence_field or DEFAULT_CONFIDENCE_FIELD}'"
        if explicit
        else "mean field score (no confidence field in the schema)"
    )

    ordered = _rank(usable, confidence_field, seed)

    curve: list[Point] = []
    for i in range(1, len(ordered) + 1):
        accepted = ordered[:i]
        accuracy = _accuracy(accepted)
        curve.append(
            Point(
                coverage=i / len(ordered),
                n_accepted=i,
                accuracy=accuracy,
                risk=1 - accuracy,
                hallucination_rate=_hallucination(accepted),
            )
        )

    # Average several shuffles: a single one is noisy enough to move
    # `ranking_gain` by several points on a small corpus.
    rng = random.Random(seed)
    shuffles = []
    for _ in range(RANDOM_TRIALS):
        shuffled = list(usable)
        rng.shuffle(shuffled)
        shuffles.append(_aurc(shuffled))

    report = SelectiveReport(
        curve=curve,
        aurc=_aurc(ordered),
        aurc_random=statistics.fmean(shuffles),
        confidence_source=source,
        n_cases=len(ordered),
    )
    for target in OPERATING_POINTS:
        idx = max(0, min(len(curve) - 1, round(target * len(curve)) - 1))
        report.operating_points.append(curve[idx])
    return report


def render(report: SelectiveReport, targets: tuple[float, ...] = (0.90, 0.95, 0.99)) -> str:
    """Markdown block, ready to paste into a report."""
    if not report.n_cases:
        return "_No cases to build a coverage curve from._\n"

    lines = [
        "### Coverage vs accuracy",
        "",
        f"Confidence source: {report.confidence_source}  ",
        f"AURC **{report.aurc:.4f}** vs random ordering {report.aurc_random:.4f} "
        f"— ranking gain **{report.ranking_gain * 100:.0f}%**",
        "",
    ]
    if report.ranking_gain < 0.05:
        lines += [
            "> ⚠️ The confidence signal ranks no better than shuffling the documents. "
            "A *send the worst N% to review* policy built on it would pick documents "
            "at random. Add an explicit confidence field to the schema before "
            "designing around this curve.",
            "",
        ]
    lines += [
        "| automate | n | accuracy | error escaping review | hallucinated |",
        "|---:|---:|---:|---:|---:|",
    ]
    for point in report.operating_points:
        lines.append(
            f"| {point.coverage * 100:.0f}% | {point.n_accepted} | {point.accuracy:.3f} "
            f"| {point.risk * 100:.1f}% | {point.hallucination_rate * 100:.2f}% |"
        )
    lines += ["", "**How much can be automated at a given quality bar:**", ""]
    for target in targets:
        coverage = report.coverage_for(target)
        lines.append(
            f"- accuracy ≥ {target:.2f} → automate **{coverage * 100:.0f}%** of documents"
            + ("" if coverage else "  _(not reachable at any coverage)_")
        )
    return "\n".join(lines) + "\n"


@dataclass
class HoldoutResult:
    """Does the review policy still work on documents it was not tuned on?"""

    target_accuracy: float
    n_splits: int
    coverage: float
    """Mean realised coverage on the held-out half."""
    accuracy: float
    """Mean realised accuracy on the automated slice of the held-out half."""
    accuracy_low: float
    accuracy_high: float
    baseline_accuracy: float
    """Accuracy with no policy at all — automate everything."""
    hit_rate: float
    """Fraction of splits where the target was actually met out of sample."""

    @property
    def lift(self) -> float:
        return self.accuracy - self.baseline_accuracy

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["lift"] = self.lift
        return d


def validate_holdout(
    cases: list[CaseResult],
    target_accuracy: float = 0.95,
    *,
    confidence_field: str | None = None,
    n_splits: int = 200,
    seed: int = 20260807,
) -> HoldoutResult:
    """Pick the confidence threshold on one half, measure it on the other.

    Reading ``accuracy_at(0.5)`` off the curve and calling it a result is
    circular: the coverage level was chosen by looking at the same documents it
    is then evaluated on. On a small corpus that overstates the policy badly.

    Here the threshold that reaches ``target_accuracy`` is fitted on a random
    half and applied, unchanged, to the other half. Repeated over ``n_splits``
    random partitions, the mean is what the policy would actually deliver — and
    ``hit_rate`` says how often the target survives the transfer.
    """
    usable = [c for c in cases if c.fields]
    if len(usable) < 8:
        return HoldoutResult(target_accuracy, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    rng = random.Random(seed)
    covs, accs, hits = [], [], 0
    baseline = _accuracy(usable)

    for _ in range(n_splits):
        shuffled = list(usable)
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        fit, test = shuffled[:half], shuffled[half:]

        # What transfers is the COVERAGE FRACTION, not a raw confidence
        # threshold. A threshold admits the whole block of documents sharing
        # that confidence value, so with a degenerate signal ">= 0.5" accepts
        # everything and the policy silently evaluates to "no policy at all".
        fit_sorted = _rank(fit, confidence_field, seed)
        best_cov, correct = 0.0, 0
        for i, case in enumerate(fit_sorted, start=1):
            correct += 1 if case.all_fields_correct else 0
            if correct / i >= target_accuracy:
                best_cov = i / len(fit_sorted)
        if best_cov < MIN_VIABLE_COVERAGE:
            covs.append(0.0)
            continue

        test_sorted = _rank(test, confidence_field, seed)
        accepted = test_sorted[: max(1, round(best_cov * len(test_sorted)))]
        covs.append(len(accepted) / len(test))
        if accepted:
            realised = _accuracy(accepted)
            accs.append(realised)
            hits += 1 if realised >= target_accuracy else 0

    if not accs:
        return HoldoutResult(target_accuracy, n_splits, 0.0, 0.0, 0.0, 0.0, baseline, 0.0)

    accs.sort()
    return HoldoutResult(
        target_accuracy=target_accuracy,
        n_splits=n_splits,
        coverage=statistics.fmean(covs),
        accuracy=statistics.fmean(accs),
        accuracy_low=accs[int(0.025 * len(accs))],
        accuracy_high=accs[min(len(accs) - 1, int(0.975 * len(accs)))],
        baseline_accuracy=baseline,
        hit_rate=hits / len(accs),
    )
