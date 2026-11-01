"""Replay runner: scoring a pipeline's recorded decisions, with its own telemetry."""

from __future__ import annotations

import asyncio
import json

import pytest

from docvlm_eval.cache import ResultCache
from docvlm_eval.config import Config
from docvlm_eval.corpus import load_corpus
from docvlm_eval.engine import run_config
from docvlm_eval.metrics import compute_metrics, diff_runs, throughput_stats
from docvlm_eval.runners.replay import ReplayRunner
from docvlm_eval.selective import selective_report
from docvlm_eval.types import CaseResult
from tests.test_end_to_end import build_corpus


def write_predictions(path, rows, meta=None):
    lines = [json.dumps({"_meta": meta})] if meta else []
    lines += [json.dumps(r) for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def replay_config(path, name="pipe"):
    return Config(name=name, runner="replay", model="pipeline", params={"predictions": str(path)})


def rows_for(n, *, step=10.0, latency=8.0, wrong=(), conf=None):
    """One document every ``step`` seconds, each taking ``latency``."""
    rows = []
    for i in range(n):
        cid = f"{i:03d}"
        row = {
            "id": cid,
            "predicted": {"name": f"Patient {cid}", "crm": f"{10000 + i}"},
            "latency_ms": latency * 1000,
            "started_at": 1_000.0 + i * step,
            "finished_at": 1_000.0 + i * step + latency,
        }
        if i in wrong:
            row["predicted"]["crm"] = "1"
        if conf is not None:
            row["confidence"] = conf(i)
        rows.append(row)
    return rows


def run(corpus_root, cfg, cache=None):
    corpus = load_corpus(corpus_root)
    return asyncio.run(run_config(corpus, cfg, concurrency=4, cache=cache))


def test_scores_recorded_decisions_with_their_telemetry(tmp_path):
    root = build_corpus(tmp_path, n=6)
    preds = write_predictions(
        tmp_path / "p.jsonl",
        rows_for(6, wrong={2}),
        meta={"pipeline": "intake", "version": "v9.9.9"},
    )
    result = run(root, replay_config(preds))
    m = compute_metrics(result, bootstrap=0)

    crm = m.field_by_name("crm")
    assert crm.accuracy == pytest.approx(5 / 6)
    # The pipeline's latency, not a mock constant and not the replay's own speed.
    assert m.latency_p50 == pytest.approx(8000)
    backend = result.provenance["backend"]
    assert backend["pipeline"]["version"] == "v9.9.9"
    assert len(backend["predictions_sha256"]) == 16


def test_missing_document_is_refused_not_filled_from_truth(tmp_path):
    root = build_corpus(tmp_path, n=4)
    preds = write_predictions(tmp_path / "p.jsonl", rows_for(4)[:3])
    m = compute_metrics(run(root, replay_config(preds)), bootstrap=0)
    assert m.n_refused == 1
    assert m.all_fields_correct == pytest.approx(3 / 4)


def test_duplicate_id_is_rejected(tmp_path):
    rows = rows_for(2)
    preds = write_predictions(tmp_path / "p.jsonl", [*rows, rows[0]])
    with pytest.raises(ValueError, match="duplicate id"):
        ReplayRunner(params={"predictions": str(preds)})


def test_replay_is_never_cached(tmp_path):
    """Re-exporting the file must change the score even with a cache present."""
    root = build_corpus(tmp_path, n=3)
    cache = ResultCache(tmp_path / "cache")
    preds = write_predictions(tmp_path / "p.jsonl", rows_for(3))
    assert compute_metrics(run(root, replay_config(preds), cache), bootstrap=0).n_cached == 0

    write_predictions(preds, rows_for(3, wrong={0, 1, 2}))
    m = compute_metrics(run(root, replay_config(preds), cache), bootstrap=0)
    assert m.field_by_name("crm").accuracy == 0.0


def test_throughput_from_recorded_timestamps(tmp_path):
    root = build_corpus(tmp_path, n=7)
    # 7 documents, one every 10 s, 8 s each: 68 s of wall clock, 56 s busy.
    preds = write_predictions(tmp_path / "p.jsonl", rows_for(7))
    m = compute_metrics(run(root, replay_config(preds)), bootstrap=0)
    assert m.n_timed == 7
    assert m.throughput_per_min == pytest.approx(7 / 68 * 60)
    assert m.mean_in_flight == pytest.approx(56 / 68)


def test_iso_timestamps_are_accepted(tmp_path):
    root = build_corpus(tmp_path, n=2)
    rows = rows_for(2)
    rows[0].update(started_at="2026-11-01T10:00:00Z", finished_at="2026-11-01T10:00:30Z")
    rows[1].update(started_at="2026-11-01T10:00:30Z", finished_at="2026-11-01T10:01:00Z")
    preds = write_predictions(tmp_path / "p.jsonl", rows)
    m = compute_metrics(run(root, replay_config(preds)), bootstrap=0)
    assert m.throughput_per_min == pytest.approx(2.0)
    assert m.mean_in_flight == pytest.approx(1.0)


def test_cache_hits_are_not_counted_as_instant():
    cases = [CaseResult(case_id="a", started_at=10, finished_at=20), CaseResult(case_id="b")]
    assert throughput_stats(cases) == (0.0, 0.0, 1)


def test_diff_reports_throughput_ratio(tmp_path):
    root = build_corpus(tmp_path, n=5)
    slow = write_predictions(tmp_path / "slow.jsonl", rows_for(5, step=20, latency=18))
    fast = write_predictions(tmp_path / "fast.jsonl", rows_for(5, step=10, latency=8))
    d = diff_runs(run(root, replay_config(slow, "slow")), run(root, replay_config(fast, "fast")))
    # 5 docs over 98 s vs over 48 s.
    assert d.throughput_ratio == pytest.approx(98 / 48)


def test_recorded_confidence_ranks_the_curve(tmp_path):
    """The two wrong documents carry the lowest pipeline score, so automating
    the confident 80% leaves nothing wrong behind."""
    root = build_corpus(tmp_path, n=10)
    rows = rows_for(10, wrong={3, 7}, conf=lambda i: 0.2 if i in (3, 7) else 0.9)
    preds = write_predictions(tmp_path / "p.jsonl", rows)
    rep = selective_report(run(root, replay_config(preds)).cases)
    assert rep.confidence_source.startswith("confidence recorded by the pipeline")
    assert rep.accuracy_at(0.8) == 1.0
    assert rep.ranking_gain > 0.5


def test_unscored_documents_go_to_review_first(tmp_path):
    """Without a score a document must not borrow the ground-truth fallback."""
    root = build_corpus(tmp_path, n=4)
    rows = rows_for(4, wrong={0}, conf=lambda i: 0.9)
    del rows[0]["confidence"]
    preds = write_predictions(tmp_path / "p.jsonl", rows)
    rep = selective_report(run(root, replay_config(preds)).cases)
    assert "1 case(s) without one ranked last" in rep.confidence_source
    assert rep.accuracy_at(0.75) == 1.0


def test_confidence_on_a_0_100_scale_is_rescaled(tmp_path):
    root = build_corpus(tmp_path, n=1)
    preds = write_predictions(tmp_path / "p.jsonl", rows_for(1, conf=lambda i: 87))
    assert run(root, replay_config(preds)).cases[0].confidence == pytest.approx(0.87)
