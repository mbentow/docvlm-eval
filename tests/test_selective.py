"""Selective prediction — the coverage/accuracy trade-off."""

from __future__ import annotations

from docvlm_eval.selective import (
    case_confidence,
    render,
    selective_report,
    validate_holdout,
)
from docvlm_eval.types import CaseResult, FieldResult, Outcome


def case(cid: str, correct: bool, score: float, conf: float | None = None):
    fields = [
        FieldResult("a", Outcome.CORRECT if correct else Outcome.WRONG, score=score),
        FieldResult("b", Outcome.CORRECT, score=score),
    ]
    if conf is not None:
        fields.append(FieldResult("confidence", Outcome.CORRECT, predicted=conf, score=1.0))
    return CaseResult(case_id=cid, fields=fields)


def test_perfect_confidence_puts_every_error_last():
    """The ideal signal: automating 50% of documents leaves zero errors."""
    cases = [case(f"g{i}", True, 0.9, conf=0.99) for i in range(10)]
    cases += [case(f"b{i}", False, 0.1, conf=0.10) for i in range(10)]
    rep = selective_report(cases)
    assert rep.accuracy_at(0.5) == 1.0
    assert rep.accuracy_at(1.0) == 0.5
    assert rep.aurc < rep.aurc_random


def test_useless_confidence_is_called_out():
    """Constant confidence cannot rank, so AURC must land on the random floor
    and `ranking_gain` must be ~0. Silently drawing a pretty curve here would
    invite a review policy that picks documents at random."""
    cases = [case(f"g{i}", i % 2 == 0, 0.5, conf=0.5) for i in range(40)]
    rep = selective_report(cases)
    assert rep.ranking_gain < 0.05
    assert "no better than shuffling" in render(rep)


def test_explicit_confidence_field_wins_over_the_score_proxy():
    c = case("1", True, 0.20, conf=0.95)
    assert case_confidence(c) == 0.95
    assert case_confidence(case("2", True, 0.20)) == 0.20


def test_confidence_on_a_0_100_scale_is_normalised():
    assert case_confidence(case("1", True, 0.5, conf=87)) == 0.87


def test_falls_back_to_mean_field_score_without_a_confidence_field():
    rep = selective_report(
        [case(f"g{i}", True, 0.9) for i in range(5)] + [case(f"b{i}", False, 0.2) for i in range(5)]
    )
    assert "mean field score" in rep.confidence_source
    assert rep.accuracy_at(0.5) == 1.0


def test_coverage_for_answers_the_business_question():
    """ "At 90% accuracy, how much can we automate?" — 8 good, 2 bad, ranked."""
    cases = [case(f"g{i}", True, 0.9, conf=0.9) for i in range(8)]
    cases += [case(f"b{i}", False, 0.1, conf=0.1) for i in range(2)]
    rep = selective_report(cases)
    assert rep.coverage_for(0.90) == 0.8
    assert rep.coverage_for(1.01) == 0.0  # unreachable target returns 0, not a lie


def test_hallucination_is_tracked_along_the_curve():
    """Coverage that keeps accuracy but concentrates hallucinations is not a win."""
    good = [case(f"g{i}", True, 0.9, conf=0.9) for i in range(5)]
    bad = []
    for i in range(5):
        c = case(f"h{i}", False, 0.1, conf=0.1)
        c.fields[0].outcome = Outcome.HALLUCINATED
        bad.append(c)
    rep = selective_report(good + bad)
    assert rep.operating_points[0].hallucination_rate == 0.0
    assert rep.operating_points[-1].hallucination_rate > 0.0


def test_curve_is_monotone_in_coverage_count():
    cases = [case(str(i), i % 3 != 0, 0.9 - i * 0.01) for i in range(30)]
    rep = selective_report(cases)
    assert [p.n_accepted for p in rep.curve] == list(range(1, 31))
    assert rep.curve[-1].coverage == 1.0


def test_empty_input_does_not_explode():
    rep = selective_report([])
    assert rep.n_cases == 0
    assert "No cases" in render(rep)


# --------------------------------------------------------------------------- #
# Held-out validation
# --------------------------------------------------------------------------- #


def test_holdout_confirms_a_policy_that_really_works():
    """Confidence perfectly separates good from bad: the threshold transfers."""
    cases = [case(f"g{i}", True, 0.9, conf=0.95) for i in range(40)]
    cases += [case(f"b{i}", False, 0.1, conf=0.05) for i in range(40)]
    r = validate_holdout(cases, 0.95, n_splits=60)
    assert r.lift > 0.30, "a separating signal must beat automating everything by a lot"
    assert r.hit_rate > 0.40
    assert 0.4 < r.coverage < 0.65, "should land near the true good/bad boundary"
    # Note the target is NOT met on every split: "largest coverage meeting the
    # target" is optimistic by construction, and out-of-sample transfer is
    # exactly what this function exists to expose.
    assert r.hit_rate < 1.0


def test_holdout_refuses_to_endorse_a_useless_signal():
    """Confidence carries no information: the threshold fitted on one half
    delivers only the base rate on the other. Reading accuracy_at(0.5) off the
    curve would have looked far better than this."""
    cases = [case(str(i), i % 2 == 0, 0.5, conf=0.5) for i in range(80)]
    r = validate_holdout(cases, 0.95, n_splits=60)
    assert r.hit_rate < 0.2, "a signal that cannot rank must not appear to hit the target"
    assert r.accuracy < 0.95
    assert r.coverage < 0.2, "no viable amount of work can be automated on this signal"


def test_holdout_needs_enough_cases():
    assert validate_holdout([case("1", True, 0.9)], 0.9).n_splits == 0
