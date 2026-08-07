"""End-to-end: corpus -> run -> store -> report -> diff, with no backend."""

from __future__ import annotations

import asyncio
import json

from typer.testing import CliRunner

from docvlm_eval.cache import ResultCache
from docvlm_eval.cli import app
from docvlm_eval.config import Config
from docvlm_eval.corpus import load_corpus
from docvlm_eval.engine import render_prompt, run_config
from docvlm_eval.metrics import compute_metrics
from docvlm_eval.report import html_run, markdown_run
from docvlm_eval.store import RunStore
from docvlm_eval.types import RunResult

SCHEMA_SRC = """
from docvlm_eval import Compare, ExtractionSchema, field

class Doc(ExtractionSchema):
    name: str | None = field(None, description="patient name", compare=Compare.TEXT)
    crm: str | None = field(None, description="licence", compare=Compare.DIGITS, critical=True)

SCHEMA = Doc
"""


def build_corpus(tmp_path, n=12, schema_src=SCHEMA_SRC, empty_crm=False):
    root = tmp_path / "corp"
    (root / "images").mkdir(parents=True)
    (root / "schema.py").write_text(schema_src)
    lines = []
    for i in range(n):
        cid = f"{i:03d}"
        (root / "images" / f"{cid}.jpg").write_bytes(b"\xff\xd8" + cid.encode())
        lines.append(
            json.dumps(
                {
                    "id": cid,
                    "image": f"images/{cid}.jpg",
                    "truth": {
                        "name": f"Patient {cid}",
                        "crm": None if empty_crm else f"{10000 + i}",
                    },
                    "tags": ["printed"] if i % 2 else ["handwritten"],
                }
            )
        )
    (root / "manifest.jsonl").write_text("\n".join(lines))
    return root


def mock_config(name="mock", noise=0.0):
    return Config(name=name, runner="mock", model="mock", params={"noise": noise, "seed": 3})


def test_perfect_mock_scores_perfectly(tmp_path):
    corpus = load_corpus(build_corpus(tmp_path))
    run = asyncio.run(run_config(corpus, mock_config(), concurrency=3))
    metrics = compute_metrics(run, bootstrap=0)
    assert metrics.macro_accuracy == 1.0
    assert metrics.all_fields_correct == 1.0
    assert metrics.n_cases == 12


def test_noisy_mock_produces_every_failure_mode(tmp_path):
    corpus = load_corpus(build_corpus(tmp_path, n=40))
    run = asyncio.run(run_config(corpus, mock_config("noisy", noise=0.5), concurrency=4))
    metrics = compute_metrics(run, critical={"crm"}, bootstrap=200)
    assert 0 < metrics.macro_accuracy < 1
    seen = {mode for f in metrics.fields for mode, rate in f.counts().items() if rate > 0}
    assert "missing" in seen and "wrong" in seen


def test_provenance_records_what_makes_a_run_reproducible(tmp_path):
    corpus = load_corpus(build_corpus(tmp_path))
    run = asyncio.run(run_config(corpus, mock_config(), concurrency=2))
    prov = run.provenance
    for key in (
        "docvlm_eval_version",
        "python",
        "config",
        "prompt_hash",
        "schema_hash",
        "corpus_hash",
        "backend",
    ):
        assert key in prov, key
    assert prov["corpus_hash"] == corpus.hash


def test_cache_avoids_the_second_call(tmp_path):
    corpus = load_corpus(build_corpus(tmp_path))
    cache = ResultCache(tmp_path / "cache")
    cfg = mock_config()
    first = asyncio.run(run_config(corpus, cfg, concurrency=2, cache=cache))
    second = asyncio.run(run_config(corpus, cfg, concurrency=2, cache=cache))
    cache.close()
    assert not any(c.cached for c in first.cases)
    assert all(c.cached for c in second.cases)


def test_changing_the_prompt_invalidates_the_cache(tmp_path):
    """Otherwise you A/B two prompts and silently compare a prompt against
    itself. This is the failure mode that quietly wastes an afternoon."""
    corpus = load_corpus(build_corpus(tmp_path))
    cache = ResultCache(tmp_path / "cache")
    asyncio.run(run_config(corpus, mock_config(), concurrency=2, cache=cache))
    other = mock_config()
    other.prompt = "a completely different prompt"
    second = asyncio.run(run_config(corpus, other, concurrency=2, cache=cache))
    cache.close()
    assert not any(c.cached for c in second.cases)


def test_prompt_template_gets_the_field_list(tmp_path):
    corpus = load_corpus(build_corpus(tmp_path))
    cfg = mock_config()
    cfg.prompt = "Extract:\n{fields}\n"
    rendered = render_prompt(cfg, corpus)
    assert "- name: patient name" in rendered
    assert "- crm: licence" in rendered


def test_store_refuses_to_silently_overwrite_a_run(tmp_path):
    corpus = load_corpus(build_corpus(tmp_path))
    run = asyncio.run(run_config(corpus, mock_config(), concurrency=2))
    store = RunStore(tmp_path / "runs")
    store.save(run)
    try:
        store.save(run)
    except FileExistsError as exc:
        assert "--overwrite" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected FileExistsError")
    store.save(run, overwrite=True)


def test_run_survives_a_json_round_trip(tmp_path):
    corpus = load_corpus(build_corpus(tmp_path))
    run = asyncio.run(run_config(corpus, mock_config("noisy", noise=0.4), concurrency=2))
    restored = RunResult.from_json(run.to_json())
    before = compute_metrics(run, bootstrap=0)
    after = compute_metrics(restored, bootstrap=0)
    assert before.macro_accuracy == after.macro_accuracy
    assert before.all_fields_correct == after.all_fields_correct


def test_reports_render(tmp_path):
    corpus = load_corpus(build_corpus(tmp_path))
    run = asyncio.run(run_config(corpus, mock_config("noisy", noise=0.3), concurrency=2))
    metrics = compute_metrics(run, critical={"crm"}, bootstrap=100)
    md = markdown_run(metrics)
    assert "ALL-FIELDS-CORRECT" in md and "| `crm` *" in md
    doc = html_run(metrics)
    assert doc.startswith("<!doctype html>") and "all fields correct" in doc
    assert "http" not in doc.split("<style>")[0].replace("http-equiv", "")  # no CDN


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_validate_run_report_diff(tmp_path):
    runner = CliRunner()
    corpus = str(build_corpus(tmp_path))
    runs_dir = str(tmp_path / "runs")
    cache_dir = str(tmp_path / "cache")

    cfg_a = tmp_path / "a.yaml"
    cfg_a.write_text("name: a\nrunner: mock\nmodel: mock\nparams:\n  noise: 0.0\n")
    cfg_b = tmp_path / "b.yaml"
    cfg_b.write_text("name: b\nrunner: mock\nmodel: mock\nparams:\n  noise: 0.6\n  seed: 9\n")

    out = runner.invoke(app, ["validate", "--corpus", corpus])
    assert out.exit_code == 0, out.output
    assert "OK" in out.output

    for cfg in (cfg_a, cfg_b):
        res = runner.invoke(
            app,
            [
                "run",
                "--corpus",
                corpus,
                "--config",
                str(cfg),
                "--runs-dir",
                runs_dir,
                "--cache-dir",
                cache_dir,
                "--bootstrap",
                "100",
            ],
        )
        assert res.exit_code == 0, res.output

    res = runner.invoke(app, ["list", "--runs-dir", runs_dir])
    assert "a" in res.output and "b" in res.output

    res = runner.invoke(
        app,
        [
            "diff",
            "--baseline",
            "a",
            "--candidate",
            "b",
            "--runs-dir",
            runs_dir,
            "--bootstrap",
            "200",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "VERDICT" in res.output


def test_cli_fail_under_exits_nonzero(tmp_path):
    """The CI gate. Without a non-zero exit, nothing stops a regression."""
    runner = CliRunner()
    corpus = str(build_corpus(tmp_path))
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("name: bad\nrunner: mock\nmodel: mock\nparams:\n  noise: 0.9\n  seed: 5\n")
    res = runner.invoke(
        app,
        [
            "run",
            "--corpus",
            corpus,
            "--config",
            str(cfg),
            "--runs-dir",
            str(tmp_path / "r"),
            "--cache-dir",
            str(tmp_path / "c"),
            "--bootstrap",
            "0",
            "--fail-under",
            "0.99",
        ],
    )
    assert res.exit_code == 2


def test_schema_declared_critical_reaches_the_report_without_any_yaml(tmp_path):
    """`critical=True` sits next to the field it describes; if it only worked
    when repeated in YAML it would be decoration."""
    corpus = load_corpus(build_corpus(tmp_path))
    cfg = mock_config("h", noise=0.0)
    cfg.params["hallucinate"] = 1.0
    run = asyncio.run(run_config(corpus, cfg, concurrency=2))
    assert run.critical_fields == ["crm"]  # from schema.py, no config entry
    assert compute_metrics(run, bootstrap=0).field_by_name("crm").critical


def test_schema_weight_zero_excludes_a_field_from_the_macro(tmp_path):
    schema = SCHEMA_SRC.replace(
        "compare=Compare.DIGITS, critical=True", "compare=Compare.DIGITS, weight=0.0"
    )
    corpus = load_corpus(build_corpus(tmp_path, schema_src=schema))
    cfg = mock_config()
    run = asyncio.run(run_config(corpus, cfg, concurrency=2))
    assert run.weights["crm"] == 0.0


def test_mock_actually_produces_hallucinations(tmp_path):
    """The 30-second demo has to be able to show the tool's headline metric."""
    corpus = load_corpus(build_corpus(tmp_path, empty_crm=True))
    cfg = mock_config("h", noise=0.0)
    cfg.params["hallucinate"] = 1.0
    run = asyncio.run(run_config(corpus, cfg, concurrency=2))
    metrics = compute_metrics(run, bootstrap=0)
    assert metrics.hallucination_rate > 0
    assert metrics.critical_hallucination_rate > 0


def test_mock_result_does_not_depend_on_concurrency(tmp_path):
    """A shared RNG made the output depend on completion order, so the same run
    scored differently at -j 1 and -j 4 — the exact irreproducibility this tool
    exists to argue against."""
    corpus = load_corpus(build_corpus(tmp_path, n=20))
    cfg = mock_config("n", noise=0.5)
    serial = asyncio.run(run_config(corpus, cfg, concurrency=1))
    parallel = asyncio.run(run_config(corpus, cfg, concurrency=8))
    assert [c.raw_output for c in serial.cases] == [c.raw_output for c in parallel.cases]


def test_cli_diff_refuses_two_different_corpora(tmp_path):
    """Case ids repeat across corpora, so pairing them compares unrelated
    documents and reports a confident, significant, meaningless delta."""
    runner = CliRunner()
    runs_dir = tmp_path / "runs"
    cfg = mock_config()
    corpora = (("a", build_corpus(tmp_path / "one")), ("b", build_corpus(tmp_path / "two", n=13)))
    for name, root in corpora:
        run = asyncio.run(run_config(load_corpus(root), cfg, run_name=name))
        RunStore(runs_dir).save(run, overwrite=True)

    res = runner.invoke(app, ["diff", "-b", "a", "-k", "b", "--runs-dir", str(runs_dir)])
    assert res.exit_code == 3
    assert "NOT COMPARABLE" in res.output

    forced = runner.invoke(
        app,
        ["diff", "-b", "a", "-k", "b", "--runs-dir", str(runs_dir), "--force", "--bootstrap", "0"],
    )
    assert forced.exit_code == 0


def test_config_never_serialises_an_absolute_path(tmp_path):
    """Provenance should not record which machine produced a run."""
    from docvlm_eval.config import load_config

    path = tmp_path / "deep" / "nested" / "c.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("name: x\nmodel: m\n")
    assert load_config(path).to_dict()["source_path"] == "c.yaml"


def test_cli_reports_a_bad_corpus_without_a_traceback(tmp_path):
    runner = CliRunner()
    (tmp_path / "empty").mkdir()
    res = runner.invoke(app, ["validate", "--corpus", str(tmp_path / "empty")])
    assert res.exit_code == 3
    assert "Traceback" not in res.output
