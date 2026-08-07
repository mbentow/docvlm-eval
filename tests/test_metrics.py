"""Aggregation, intervals and diffs."""

from __future__ import annotations

from docvlm_eval.metrics import compute_metrics, diff_runs, percentile
from docvlm_eval.types import CaseResult, FieldResult, Outcome, RunResult


def make_case(case_id: str, outcomes: dict[str, Outcome], tags=None, latency=100.0):
    return CaseResult(
        case_id=case_id,
        tags=tags or [],
        fields=[
            FieldResult(name, o, score=1.0 if o.is_correct else 0.0) for name, o in outcomes.items()
        ],
        latency_ms=latency,
    )


def make_run(name: str, cases: list[CaseResult], fields: list[str], corpus_hash="abc"):
    return RunResult(
        name=name,
        corpus_name="c",
        corpus_hash=corpus_hash,
        config_name=name,
        config_hash="cfg",
        field_names=fields,
        cases=cases,
    )


def test_percentile_nearest_rank():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 95) == 5
    assert percentile([], 50) == 0.0


def test_per_field_accuracy_is_not_the_average():
    """The whole point: one field at 1.00 and one at 0.20 average to 0.60, and
    the average is the number that would ship a broken field."""
    cases = [make_case(str(i), {"good": Outcome.CORRECT, "bad": Outcome.WRONG}) for i in range(8)]
    cases += [make_case("8", {"good": Outcome.CORRECT, "bad": Outcome.CORRECT})]
    metrics = compute_metrics(make_run("r", cases, ["good", "bad"]), bootstrap=0)
    good = metrics.field_by_name("good")
    bad = metrics.field_by_name("bad")
    assert good.accuracy == 1.0
    assert bad.accuracy < 0.2
    assert bad.accuracy < metrics.macro_accuracy < good.accuracy


def test_failure_modes_are_reported_separately():
    cases = [
        make_case("1", {"f": Outcome.MISSING}),
        make_case("2", {"f": Outcome.HALLUCINATED}),
        make_case("3", {"f": Outcome.WRONG}),
        make_case("4", {"f": Outcome.MALFORMED}),
        make_case("5", {"f": Outcome.CORRECT}),
    ]
    m = compute_metrics(make_run("r", cases, ["f"]), bootstrap=0).field_by_name("f")
    assert m.missing_rate == m.hallucination_rate == m.wrong_rate == m.malformed_rate == 0.2
    assert m.accuracy == 0.2


def test_all_fields_correct_is_stricter_than_macro():
    """Two documents, each with one different broken field: macro is 0.50,
    all-fields-correct is 0.00. Only the second one describes the workload."""
    cases = [
        make_case("1", {"a": Outcome.CORRECT, "b": Outcome.WRONG}),
        make_case("2", {"a": Outcome.WRONG, "b": Outcome.CORRECT}),
    ]
    metrics = compute_metrics(make_run("r", cases, ["a", "b"]), bootstrap=0)
    assert metrics.macro_accuracy == 0.5
    assert metrics.all_fields_correct == 0.0


def test_zero_weight_field_is_reported_but_excluded_from_macro():
    cases = [make_case(str(i), {"a": Outcome.CORRECT, "diag": Outcome.WRONG}) for i in range(6)]
    run = make_run("r", cases, ["a", "diag"])
    metrics = compute_metrics(run, weights={"diag": 0.0}, bootstrap=0)
    assert metrics.macro_accuracy == 1.0
    assert metrics.field_by_name("diag").accuracy == 0.0


def test_critical_hallucination_rate_only_counts_critical_fields():
    cases = [
        make_case("1", {"crm": Outcome.HALLUCINATED, "note": Outcome.HALLUCINATED}),
        make_case("2", {"crm": Outcome.CORRECT, "note": Outcome.HALLUCINATED}),
    ]
    metrics = compute_metrics(make_run("r", cases, ["crm", "note"]), critical={"crm"}, bootstrap=0)
    assert metrics.hallucination_rate == 0.75
    assert metrics.critical_hallucination_rate == 0.5


def test_by_tag_reveals_the_condition_that_fails():
    cases = [make_case(f"p{i}", {"f": Outcome.CORRECT}, tags=["printed"]) for i in range(10)]
    cases += [make_case(f"h{i}", {"f": Outcome.WRONG}, tags=["handwritten"]) for i in range(10)]
    metrics = compute_metrics(make_run("r", cases, ["f"]), bootstrap=0)
    by_tag = {t.tag: t.accuracy for t in metrics.by_tag}
    assert by_tag["printed"] == 1.0
    assert by_tag["handwritten"] == 0.0
    assert metrics.by_tag[0].tag == "handwritten"  # worst first


def test_bootstrap_interval_brackets_the_point_estimate():
    cases = [make_case(str(i), {"f": Outcome.CORRECT}) for i in range(30)]
    cases += [make_case(f"w{i}", {"f": Outcome.WRONG}) for i in range(10)]
    metrics = compute_metrics(make_run("r", cases, ["f"]), bootstrap=400)
    ci = metrics.macro_accuracy_ci
    assert ci is not None
    assert ci.low <= metrics.macro_accuracy <= ci.high
    assert 0 < ci.half_width < 0.3


def test_no_interval_when_the_sample_is_too_small_to_mean_anything():
    cases = [make_case(str(i), {"f": Outcome.CORRECT}) for i in range(3)]
    metrics = compute_metrics(make_run("r", cases, ["f"]), bootstrap=400)
    assert metrics.macro_accuracy_ci is None


def test_cached_cases_keep_their_measured_latency_but_are_counted():
    """A fully cached re-run must not report 0 ms — that reads as "instant"
    rather than "not measured now". n_cached is how the reader tells."""
    fresh = make_case("1", {"f": Outcome.CORRECT}, latency=5000)
    cached = make_case("2", {"f": Outcome.CORRECT}, latency=4800)
    cached.cached = True
    metrics = compute_metrics(make_run("r", [fresh, cached], ["f"]), bootstrap=0)
    assert metrics.latency_p50 > 0
    assert metrics.n_cached == 1


# --------------------------------------------------------------------------- #
# Diff
# --------------------------------------------------------------------------- #


def test_diff_flags_a_real_improvement_as_significant():
    base = make_run("base", [make_case(str(i), {"f": Outcome.WRONG}) for i in range(40)], ["f"])
    cand = make_run("cand", [make_case(str(i), {"f": Outcome.CORRECT}) for i in range(40)], ["f"])
    result = diff_runs(base, cand, bootstrap=400)
    assert result.macro_delta == 1.0
    assert result.improvements and not result.regressions


def test_diff_calls_a_tiny_difference_noise():
    """One document out of forty is not a result — and saying so is the whole
    reason the paired bootstrap is here."""
    base_cases = [make_case(str(i), {"f": Outcome.CORRECT}) for i in range(39)]
    base_cases.append(make_case("39", {"f": Outcome.WRONG}))
    cand_cases = [make_case(str(i), {"f": Outcome.CORRECT}) for i in range(40)]
    result = diff_runs(
        make_run("b", base_cases, ["f"]), make_run("c", cand_cases, ["f"]), bootstrap=600
    )
    assert result.macro_delta > 0
    assert not result.fields[0].significant


def test_diff_detects_a_per_field_regression_hidden_by_a_flat_average():
    """The scenario the tool is built for: the average did not move, but one
    field got better and another got worse."""
    n = 60
    base_cases = [make_case(str(i), {"a": Outcome.CORRECT, "b": Outcome.WRONG}) for i in range(n)]
    cand_cases = [make_case(str(i), {"a": Outcome.WRONG, "b": Outcome.CORRECT}) for i in range(n)]
    result = diff_runs(
        make_run("b", base_cases, ["a", "b"]),
        make_run("c", cand_cases, ["a", "b"]),
        bootstrap=400,
    )
    assert abs(result.macro_delta) < 1e-9
    assert len(result.regressions) == 1 and result.regressions[0].name == "a"
    assert len(result.improvements) == 1 and result.improvements[0].name == "b"


def test_diff_refuses_to_pretend_two_corpora_are_the_same():
    base = make_run("b", [make_case("1", {"f": Outcome.CORRECT})], ["f"], corpus_hash="aaa")
    cand = make_run("c", [make_case("1", {"f": Outcome.CORRECT})], ["f"], corpus_hash="bbb")
    result = diff_runs(base, cand, bootstrap=0)
    assert not result.comparable
    assert "corpus hash differs" in result.incomparable_reason


def test_diff_pairs_on_shared_case_ids_only():
    base = make_run("b", [make_case(str(i), {"f": Outcome.CORRECT}) for i in range(10)], ["f"])
    cand = make_run("c", [make_case(str(i), {"f": Outcome.CORRECT}) for i in range(5)], ["f"])
    result = diff_runs(base, cand, bootstrap=0)
    assert result.fields[0].baseline == result.fields[0].candidate == 1.0
    assert result.n_paired == 5


def test_diff_delta_is_computed_on_the_same_population_as_its_interval():
    """The candidate is a filtered subset. If the delta came from the full runs
    while the interval came from the pairs, the band would sit around a number
    it was not computed from."""
    base_cases = [make_case(str(i), {"f": Outcome.CORRECT}) for i in range(10)]
    base_cases += [make_case(f"x{i}", {"f": Outcome.WRONG}) for i in range(10)]
    cand_cases = [make_case(str(i), {"f": Outcome.CORRECT}) for i in range(10)]
    result = diff_runs(
        make_run("b", base_cases, ["f"]), make_run("c", cand_cases, ["f"]), bootstrap=300
    )
    assert result.n_paired == 10
    assert result.macro_delta == 0.0  # paired subset: both perfect on those ten
    assert result.macro_delta_ci.low <= result.macro_delta <= result.macro_delta_ci.high


# --------------------------------------------------------------------------- #
# Scoring policy travels with the run
# --------------------------------------------------------------------------- #


def test_metrics_default_to_the_policy_stored_in_the_run():
    """`report` on a stored run must reproduce what `run` printed. Recomputing
    with default weights would silently give a different macro."""
    cases = [make_case(str(i), {"a": Outcome.CORRECT, "diag": Outcome.WRONG}) for i in range(6)]
    run = make_run("r", cases, ["a", "diag"])
    run.weights = {"a": 1.0, "diag": 0.0}
    run.critical_fields = ["a"]
    metrics = compute_metrics(run, bootstrap=0)
    assert metrics.macro_accuracy == 1.0
    assert metrics.field_by_name("a").critical


def test_explicit_arguments_still_override_the_stored_policy():
    cases = [make_case(str(i), {"a": Outcome.CORRECT, "diag": Outcome.WRONG}) for i in range(6)]
    run = make_run("r", cases, ["a", "diag"])
    run.weights = {"a": 1.0, "diag": 0.0}
    metrics = compute_metrics(run, weights={}, bootstrap=0)
    assert metrics.macro_accuracy == 0.5


# --------------------------------------------------------------------------- #
# Bootstrap honesty
# --------------------------------------------------------------------------- #


def test_percentile_bounds_are_symmetric():
    """Indexing the upper tail from the left leaves one fewer draw above the
    bound than below it, shifting every interval upward."""
    from docvlm_eval.metrics import _percentile_bounds

    samples = list(range(1000))
    lo, hi = _percentile_bounds(samples, 0.05)
    assert lo == 25
    assert hi == 974
    assert (lo - 0) == (999 - hi)


def test_no_interval_when_there_are_too_few_bootstrap_draws():
    """At 20 iterations the "95% CI" would be the min and max of the draws."""
    cases = [make_case(str(i), {"f": Outcome.CORRECT}) for i in range(40)]
    assert compute_metrics(make_run("r", cases, ["f"]), bootstrap=20).macro_accuracy_ci is None
    assert compute_metrics(make_run("r", cases, ["f"]), bootstrap=400).macro_accuracy_ci
