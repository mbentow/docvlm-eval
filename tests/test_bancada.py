"""Early stopping, run order, self-agreement and PII handling."""

from __future__ import annotations

import asyncio
import json
import random
import subprocess

import pytest
from typer.testing import CliRunner

from docvlm_eval.agree import agreement
from docvlm_eval.cli import app
from docvlm_eval.config import Config
from docvlm_eval.corpus import load_corpus
from docvlm_eval.engine import run_config
from docvlm_eval.order import stratified_order
from docvlm_eval.privacy import PREFIX, TrackedOutputError, guard_untracked
from docvlm_eval.sequential import SequentialPlan, SequentialState, evaluate, spent_alpha
from docvlm_eval.store import RunStore
from docvlm_eval.types import CaseResult, FieldResult, Outcome, RunResult
from docvlm_eval.watch import PredictionTail, progress_line
from tests.test_end_to_end import build_corpus


def case(cid: str, ok: bool, hall: bool = False) -> CaseResult:
    outcome = Outcome.HALLUCINATED if hall else (Outcome.CORRECT if ok else Outcome.WRONG)
    return CaseResult(case_id=cid, fields=[FieldResult("crm", outcome)])


def simulate(n, p_base, p_cand, rng, *, looks=10, margin=0.01, shared=0.6):
    """Paired documents: a shared difficulty makes the runs correlated, as real ones are."""
    base, cand = {}, []
    for i in range(n):
        cid = f"{i:05d}"
        hard = rng.random() < shared
        u = rng.random()
        base[cid] = case(cid, (u if hard else rng.random()) < p_base)
        cand.append(case(cid, (u if hard else rng.random()) < p_cand))
    state = SequentialState(SequentialPlan(n, looks=looks, margin=margin), critical={"crm"})
    evaluate(state, base, cand)
    return state


# -- sequential ---------------------------------------------------------------


def test_alpha_spending_is_monotone_and_complete():
    values = [spent_alpha(t / 10, 0.05) for t in range(11)]
    assert values[0] == 0 and values[-1] == 0.05
    assert all(a <= b for a, b in zip(values, values[1:], strict=False))
    assert spent_alpha(0.1, 0.05) < 1e-6  # stopping at 10% needs an overwhelming effect


def test_false_stop_rate_under_the_null_stays_below_alpha():
    rng = random.Random(11)
    trials = 400
    false = sum(simulate(300, 0.8, 0.8, rng).decision in ("better", "worse") for _ in range(trials))
    # alpha = 0.05 split in half for the quality question; allow sampling slack.
    assert false / trials <= 0.05


def test_a_large_improvement_stops_early():
    state = simulate(600, 0.60, 0.85, random.Random(3))
    assert state.decision == "better"
    assert state.looks[-1].n < 600


def test_a_large_regression_stops_early():
    state = simulate(600, 0.85, 0.60, random.Random(4))
    assert state.decision == "worse"
    assert state.looks[-1].n < 600


def test_identical_outputs_stop_as_equivalent_before_the_end():
    base = {f"{i}": case(f"{i}", i % 3 != 0) for i in range(400)}
    cand = [case(cid, c.all_fields_correct) for cid, c in base.items()]
    state = SequentialState(SequentialPlan(400, margin=0.02))
    evaluate(state, base, cand)
    assert state.decision == "equivalent"
    assert state.looks[-1].n < 400


def test_thirty_identical_documents_are_not_enough_for_equivalence():
    base = {f"{i}": case(f"{i}", True) for i in range(300)}
    cand = [case(cid, True) for cid in list(base)[:30]]
    state = SequentialState(SequentialPlan(300, looks=10, margin=0.01))
    evaluate(state, base, cand)
    assert state.decision == "continue"


def test_more_critical_hallucinations_stop_as_harm():
    base = {f"{i}": case(f"{i}", True) for i in range(300)}
    cand = [case(cid, True, hall=int(cid) % 4 == 0) for cid in base]
    state = SequentialState(SequentialPlan(300), critical={"crm"})
    evaluate(state, base, cand)
    assert state.decision == "harm"


def test_looks_are_not_retaken_after_a_decision():
    state = simulate(600, 0.60, 0.85, random.Random(3))
    n_looks = len(state.looks)
    evaluate(state, {}, [])
    assert len(state.looks) == n_looks


# -- order --------------------------------------------------------------------


def tagged_corpus(tmp_path, sizes):
    root = build_corpus(tmp_path, n=sum(sizes.values()))
    lines = [json.loads(x) for x in (root / "manifest.jsonl").read_text().splitlines()]
    strata = [name for name, k in sizes.items() for _ in range(k)]
    for line, name in zip(lines, strata, strict=True):
        line["tags"] = [name, "extra"]
    (root / "manifest.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
    return load_corpus(root)


def test_every_prefix_keeps_the_strata_proportional(tmp_path):
    corpus = tagged_corpus(tmp_path, {"easy": 60, "hard": 30, "rare": 10})
    order = stratified_order(corpus, seed=1)
    assert sorted(order) == sorted(c.id for c in corpus)
    stratum = {c.id: c.tags[0] for c in corpus}
    for k in range(1, len(order) + 1):
        seen = [stratum[i] for i in order[:k]]
        for name, share in (("easy", 0.6), ("hard", 0.3), ("rare", 0.1)):
            assert abs(seen.count(name) - share * k) <= 1.0


def test_order_is_a_pure_function_of_corpus_and_seed(tmp_path):
    corpus = tagged_corpus(tmp_path, {"a": 7, "b": 5})
    assert stratified_order(corpus, seed=5) == stratified_order(corpus, seed=5)
    assert stratified_order(corpus, seed=5) != stratified_order(corpus, seed=6)


# -- agreement ----------------------------------------------------------------


def run_of(cases, name="r"):
    return RunResult(name, "c", "h", name, "cfg", field_names=["crm"], cases=cases)


def test_same_outputs_are_reproducible():
    cases = [case(f"{i}", i % 2 == 0) for i in range(50)]
    rep = agreement(run_of(cases), run_of([case(c.case_id, c.all_fields_correct) for c in cases]))
    assert rep.reproducible
    assert rep.document_flips == 0


def test_a_config_that_drifts_on_its_own_is_flagged():
    a = [case(f"{i}", True) for i in range(200)]
    b = [case(f"{i}", i % 5 != 0) for i in range(200)]
    rep = agreement(run_of(a), run_of(b))
    assert rep.document_flips == pytest.approx(0.2)
    assert not rep.reproducible


# -- privacy ------------------------------------------------------------------

PII_SCHEMA = """
from docvlm_eval import Compare, ExtractionSchema, field

class Doc(ExtractionSchema):
    name: str | None = field(None, compare=Compare.TEXT, pii=True)
    crm: str | None = field(None, compare=Compare.DIGITS, critical=True)

SCHEMA = Doc
"""


def test_pii_fields_are_scored_then_hashed(tmp_path):
    root = build_corpus(tmp_path, n=4, schema_src=PII_SCHEMA)
    corpus = load_corpus(root)
    cfg = Config(name="m", runner="mock", model="mock", params={"seed": 1})
    result = asyncio.run(run_config(corpus, cfg))
    text = result.to_json()
    assert "Patient 000" not in text
    name_fields = [f for c in result.cases for f in c.fields if f.field == "name"]
    assert all(f.outcome is Outcome.CORRECT for f in name_fields)
    assert all(str(f.predicted).startswith(PREFIX) for f in name_fields)
    assert all(f.predicted == f.truth for f in name_fields)  # equal values hash equal
    assert all(c.raw_output == "" for c in result.cases)


def test_schema_without_pii_keeps_its_hash():
    """Adding the flag must not move the hash of every existing corpus."""
    corpus = load_corpus("corpora/synthetic-forms-hard")
    stored = json.loads(open("examples/runs/hard-qwen3vl-8b.json").read())
    assert corpus.hash == stored["corpus_hash"]


def git_repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


def test_tracked_output_is_refused(tmp_path):
    repo = git_repo(tmp_path)
    with pytest.raises(TrackedOutputError):
        guard_untracked(repo / "runs")


def test_ignored_output_is_allowed(tmp_path):
    repo = git_repo(tmp_path)
    (repo / ".gitignore").write_text("runs/\n")
    guard_untracked(repo / "runs")
    guard_untracked(repo / "elsewhere", allow=True)


def test_outside_git_is_allowed(tmp_path):
    guard_untracked(tmp_path / "runs")


# -- watch --------------------------------------------------------------------


def test_tail_skips_a_half_written_line_and_duplicates(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text('{"_meta": {"v": 1}}\n{"id": "a", "predicted": {}}\n{"id": "b", "pre')
    tail = PredictionTail(path)
    assert [r["id"] for r in tail.read_new()] == ["a"]
    with path.open("a") as fh:
        fh.write('dicted": {}}\n{"id": "a", "predicted": {}}\n')
    assert [r["id"] for r in tail.read_new()] == ["b"]
    assert tail.duplicates == 1
    assert tail.meta == {"v": 1}


def test_progress_line_format():
    line = progress_line(120, 300, 372, "Δ +0.01")
    assert line.startswith("[########------------]  40% · 120/300 · 6m12s elapsed · ETA ~9m18s")
    assert line.endswith("· Δ +0.01")


def write_baseline(tmp_path, root, n):
    corpus = load_corpus(root)
    cases = []
    for c in corpus:
        crm_ok = int(c.id) % 3 != 0  # the baseline misses a third of the licences
        cases.append(
            CaseResult(
                case_id=c.id,
                fields=[
                    FieldResult("name", Outcome.CORRECT, score=1.0),
                    FieldResult("crm", Outcome.CORRECT if crm_ok else Outcome.WRONG),
                ],
            )
        )
    run = RunResult(
        "base",
        corpus.name,
        corpus.hash,
        "base",
        "x",
        ["name", "crm"],
        cases,
        critical_fields=["crm"],
    )
    RunStore(tmp_path / "runs").save(run)


def preds(root, n, *, crm_wrong=False):
    lines = []
    for i in range(n):
        cid = f"{i:03d}"
        crm = "1" if crm_wrong else f"{10000 + i}"
        lines.append(json.dumps({"id": cid, "predicted": {"name": f"Patient {cid}", "crm": crm}}))
    return "\n".join(lines) + "\n"


def test_watch_stops_a_clear_win_and_saves_the_partial_run(tmp_path):
    root = build_corpus(tmp_path, n=300)
    write_baseline(tmp_path, root, 300)
    p = tmp_path / "p.jsonl"
    p.write_text(preds(root, 300))
    status = tmp_path / "status.json"
    res = CliRunner().invoke(
        app,
        [
            "watch", "--corpus", str(root), "-b", "base", "-p", str(p),
            "--runs-dir", str(tmp_path / "runs"), "--once",
            "--status-file", str(status), "--save", "cand",
        ],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    decision = json.loads(status.read_text())
    assert decision["decision"] == "better" and decision["stop"]
    assert "STOP — better" in res.output
    saved = RunStore(tmp_path / "runs").load("cand")
    assert saved.provenance["sequential"]["decision"] == "better"


def test_watch_exits_2_on_a_regression(tmp_path):
    root = build_corpus(tmp_path, n=300)
    write_baseline(tmp_path, root, 300)
    p = tmp_path / "p.jsonl"
    p.write_text(preds(root, 300, crm_wrong=True))
    res = CliRunner().invoke(
        app,
        ["watch", "--corpus", str(root), "-b", "base", "-p", str(p),
         "--runs-dir", str(tmp_path / "runs"), "--once"],
    )  # fmt: skip
    assert res.exit_code == 2, res.output


def test_order_command_writes_every_id(tmp_path):
    root = build_corpus(tmp_path, n=9)
    out = tmp_path / "order.txt"
    res = CliRunner().invoke(app, ["order", "--corpus", str(root), "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert sorted(out.read_text().split()) == [f"{i:03d}" for i in range(9)]
