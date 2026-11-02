"""Command line interface.

::

    docvlm-eval validate --corpus corpora/synthetic-forms
    docvlm-eval run      --corpus corpora/synthetic-forms --config configs/mock.yaml
    docvlm-eval report   --run mock-baseline --format html --out out/
    docvlm-eval diff     --baseline mock-baseline --candidate qwen3vl-30b
    docvlm-eval sweep    --corpus corpora/synthetic-forms --configs "configs/*.yaml"
    docvlm-eval order    --corpus corpora/synthetic-forms --out order.txt
    docvlm-eval watch    --corpus corpora/synthetic-forms -b baseline -p preds.jsonl
    docvlm-eval agree    -a run-1 -b run-2

``--fail-under`` on ``run`` and ``diff`` makes the tool usable as a CI gate,
which is what turns an evaluator from a personal script into something a team
adopts.
"""

from __future__ import annotations

import asyncio
import glob as globlib
import json
import sys
import time
from pathlib import Path

import typer
from rich.console import Console

from docvlm_eval import __version__
from docvlm_eval.agree import agreement
from docvlm_eval.agree import render as render_agreement
from docvlm_eval.cache import ResultCache
from docvlm_eval.config import load_config
from docvlm_eval.corpus import CorpusError, load_corpus
from docvlm_eval.engine import pii_fields, run_config
from docvlm_eval.leakage import leakage_report
from docvlm_eval.leakage import render as render_leakage
from docvlm_eval.metrics import compute_metrics, diff_runs
from docvlm_eval.order import stratified_order, stratum_of
from docvlm_eval.privacy import TrackedOutputError, guard_untracked, redact_case
from docvlm_eval.report import (
    html_run,
    markdown_diff,
    markdown_run,
    markdown_selective,
    print_diff,
    print_run,
    print_sweep,
    sweep_table,
    write_reports,
)
from docvlm_eval.sequential import SequentialPlan, SequentialState, evaluate
from docvlm_eval.sequential import verdict as sequential_verdict
from docvlm_eval.store import RunStore
from docvlm_eval.types import RunResult
from docvlm_eval.watch import PredictionTail, elapsed_since, progress_line, score_record

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Field-level evaluation for document extraction with vision models.",
)
console = Console()
err = Console(stderr=True)

EXIT_GATE_FAILED = 2
EXIT_BAD_INPUT = 3


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"docvlm-eval {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version."
    ),
) -> None:
    """docvlm-eval"""


# --------------------------------------------------------------------------- #


@app.command()
def validate(
    corpus: Path = typer.Option(..., "--corpus", "-c", help="Corpus directory."),
    strict: bool = typer.Option(True, help="Fail on truth keys absent from the schema."),
) -> None:
    """Check a corpus before spending GPU hours on it.

    Catches the boring failures that otherwise show up as a mysterious accuracy
    ceiling: a typo in a ground-truth key, a missing image, a duplicate id.
    """
    try:
        loaded = load_corpus(corpus, strict=strict)
    except CorpusError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_BAD_INPUT) from exc

    console.print(f"[green]OK[/green] {loaded.name} @ {loaded.hash} — {len(loaded)} cases")
    console.print(f"schema: {loaded.schema.__name__} @ {loaded.schema.schema_hash()}")
    specs = loaded.schema.specs()
    for name, spec in specs.items():
        flags = []
        if spec.critical:
            flags.append("critical")
        if spec.fuzzy_threshold < 1.0:
            flags.append(f"fuzzy≥{spec.fuzzy_threshold}")
        if spec.weight != 1.0:
            flags.append(f"weight={spec.weight}")
        suffix = f"  [dim]({', '.join(flags)})[/dim]" if flags else ""
        console.print(f"  · {name}: [cyan]{spec.compare}[/cyan]{suffix}")
    counts = loaded.tag_counts()
    if counts:
        console.print("tags: " + ", ".join(f"{t}={n}" for t, n in counts.items()))
    else:
        console.print(
            "[yellow]no tags — you will not be able to tell which condition is failing[/yellow]"
        )


@app.command()
def run(
    corpus: Path = typer.Option(..., "--corpus", "-c"),
    config: Path = typer.Option(..., "--config", "-f"),
    name: str = typer.Option("", "--name", "-n", help="Run name (default: config name)."),
    limit: int = typer.Option(0, help="Evaluate only the first N cases."),
    tags: str = typer.Option("", help="Comma-separated: only cases with these tags."),
    exclude_tags: str = typer.Option("", help="Comma-separated: skip cases with these tags."),
    concurrency: int = typer.Option(2, "--concurrency", "-j", help="Parallel documents."),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Ignore and do not write the result cache."
    ),
    cache_dir: Path = typer.Option(Path(".docvlm-cache"), help="Cache directory."),
    runs_dir: Path = typer.Option(Path("runs"), help="Where runs are stored."),
    out: Path = typer.Option(None, help="Also write md/html/json reports here."),
    bootstrap: int = typer.Option(2000, help="Bootstrap iterations (<200 disables CIs)."),
    overwrite: bool = typer.Option(False, help="Replace an existing run of the same name."),
    fail_under: float = typer.Option(
        None, "--fail-under", help="Exit non-zero if macro accuracy is below this."
    ),
    fail_hallucination_over: float = typer.Option(
        None, help="Exit non-zero if the hallucination rate exceeds this."
    ),
    allow_tracked: bool = typer.Option(
        False, help="Allow writing a PII corpus's runs/cache inside a git work tree."
    ),
) -> None:
    """Evaluate one config over one corpus."""
    try:
        loaded = load_corpus(corpus)
    except CorpusError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_BAD_INPUT) from exc

    loaded = loaded.filter(
        tags=_split(tags),
        exclude_tags=_split(exclude_tags),
        limit=limit or None,
    )
    if not len(loaded):
        err.print("[red]no cases left after filtering[/red]")
        raise typer.Exit(EXIT_BAD_INPUT)

    if pii_fields(loaded):
        try:
            guard_untracked(runs_dir, allow=allow_tracked)
            if not no_cache:
                guard_untracked(cache_dir, allow=allow_tracked)
        except TrackedOutputError as exc:
            err.print(f"[red]{exc}[/red]")
            raise typer.Exit(EXIT_BAD_INPUT) from exc

    cfg = load_config(config)
    cache = ResultCache(cache_dir, enabled=not no_cache)
    if cache.disabled_reason:
        err.print(f"[yellow]{cache.disabled_reason}[/yellow]")
    store = RunStore(runs_dir)

    done = {"n": 0}

    def progress(_case) -> None:
        done["n"] += 1
        console.print(f"  [{done['n']}/{len(loaded)}] {_case.case_id}", end="\r", highlight=False)

    console.print(
        f"Running [bold]{cfg.name}[/bold] over {loaded.name} @ {loaded.hash} "
        f"({len(loaded)} cases, j={concurrency})"
    )
    result = asyncio.run(
        run_config(
            loaded,
            cfg,
            run_name=name or cfg.name,
            concurrency=concurrency,
            cache=cache,
            progress=progress,
        )
    )
    cache.close()
    console.print(" " * 60, end="\r")

    path = store.save(result, overwrite=overwrite)
    # No weights/critical passed: the run already carries the resolved policy,
    # so `run`, `report` and `diff` cannot print different numbers for it.
    metrics = compute_metrics(result, bootstrap=bootstrap)
    print_run(metrics, console, cases=result.cases)
    console.print(f"[dim]run saved to {path}[/dim]")

    if out:
        paths = write_reports(metrics, out, cases=result.cases)
        console.print(f"[dim]reports: {', '.join(str(p) for p in paths.values())}[/dim]")

    _gate(metrics, fail_under, fail_hallucination_over)


@app.command()
def report(
    run_name: str = typer.Option(..., "--run", "-r", help="Run name or path to a run JSON."),
    fmt: str = typer.Option("term", "--format", help="term | md | html | json"),
    out: Path = typer.Option(
        None, help="Write md + html + json here (all three; --format is ignored)."
    ),
    runs_dir: Path = typer.Option(Path("runs"), help="Where runs are stored."),
    bootstrap: int = typer.Option(2000, help="Bootstrap iterations (<200 disables CIs)."),
) -> None:
    """Re-render a stored run.

    Scoring is not cached, so this recomputes every metric from the stored model
    outputs — change a normaliser and the numbers update without touching a GPU.
    """
    store = RunStore(runs_dir)
    try:
        result = store.load(run_name)
    except FileNotFoundError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_BAD_INPUT) from exc

    metrics = compute_metrics(result, bootstrap=bootstrap)
    if out:
        paths = write_reports(metrics, out, cases=result.cases)
        console.print("\n".join(str(p) for p in paths.values()))
        return
    if fmt == "term":
        print_run(metrics, console, cases=result.cases)
    elif fmt == "md":
        print(markdown_run(metrics))
        print(markdown_selective(result.cases))
    elif fmt == "html":
        print(html_run(metrics))
    elif fmt == "json":
        import json

        print(json.dumps(metrics.to_dict(), indent=2, default=str))
    else:
        err.print(f"[red]unknown format {fmt!r}[/red]")
        raise typer.Exit(EXIT_BAD_INPUT)


@app.command()
def echo(
    run_name: str = typer.Option(..., "--run", "-r", help="Run name or path to a run JSON."),
    config: Path = typer.Option(..., "--config", help="Config whose prompt produced the run."),
    runs_dir: Path = typer.Option(Path("runs"), help="Where runs are stored."),
) -> None:
    """Check whether answers came from the page or from the prompt.

    Needs the config because a run stores only the prompt *hash* — the prompt
    text stays out of run files on purpose, so results can be published without
    leaking the prompt. Pass the config that produced the run and the hashes are
    checked for you.

    Exits non-zero when a literal is both over-produced and less accurate, so it
    can gate a release the same way ``diff`` does.
    """
    store = RunStore(runs_dir)
    try:
        result = store.load(run_name)
        cfg = load_config(config)
    except (FileNotFoundError, OSError) as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_BAD_INPUT) from exc

    stored = str(result.provenance.get("prompt_hash") or "")
    if stored and stored != cfg.prompt_hash:
        # Scoring the wrong prompt against a run is worse than not scoring it:
        # every literal comes back clean and the check looks like it passed.
        err.print(
            f"[red]This config's prompt ({cfg.prompt_hash}) is not the one that "
            f"produced '{result.name}' ({stored}).[/red]"
        )
        raise typer.Exit(EXIT_BAD_INPUT)

    report_ = leakage_report(result.cases, cfg.prompt)
    console.print(render_leakage(report_))
    if not report_.clean:
        raise typer.Exit(EXIT_GATE_FAILED)


@app.command()
def diff(
    baseline: str = typer.Option(..., "--baseline", "-b"),
    candidate: str = typer.Option(..., "--candidate", "-k"),
    runs_dir: Path = typer.Option(Path("runs"), help="Where runs are stored."),
    fmt: str = typer.Option("term", "--format", help="term | md"),
    bootstrap: int = typer.Option(2000, help="Bootstrap iterations (<200 disables CIs)."),
    force: bool = typer.Option(
        False, help="Compare anyway when the two runs used different corpora."
    ),
    fail_on_regression: bool = typer.Option(
        False, help="Exit non-zero if any field regressed significantly."
    ),
    fail_under_delta: float = typer.Option(
        None, help="Exit non-zero if the macro delta is below this (e.g. -0.005)."
    ),
) -> None:
    """Compare two runs on the documents they have in common.

    The delta comes with a paired bootstrap interval. If that interval contains
    zero, the difference is not a result — and this is where most model-swap
    blog posts go wrong.
    """
    store = RunStore(runs_dir)
    try:
        base = store.load(baseline)
        cand = store.load(candidate)
    except FileNotFoundError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_BAD_INPUT) from exc

    if base.corpus_hash != cand.corpus_hash and not force:
        # Case ids are reused across corpora (0001, 0002, …), so pairing two
        # different corpora silently compares unrelated documents and produces a
        # confident, significant, meaningless delta. Refuse by default.
        err.print(
            f"[red]NOT COMPARABLE[/red] corpus hash differs "
            f"({base.corpus_hash} vs {cand.corpus_hash}): the two runs did not see the same "
            "documents or the same ground truth.\n"
            "Case ids are reused across corpora, so pairing them would compare unrelated "
            "documents. Re-run both configs on one corpus, or pass --force if you know "
            "what you are doing."
        )
        raise typer.Exit(EXIT_BAD_INPUT)

    result = diff_runs(base, cand, bootstrap=bootstrap)
    if fmt == "md":
        print(markdown_diff(result))
    else:
        print_diff(result, console)

    failed = False
    if fail_on_regression and result.regressions:
        err.print(
            f"[red]FAIL[/red] {len(result.regressions)} field(s) regressed significantly: "
            + ", ".join(f.name for f in result.regressions)
        )
        failed = True
    if fail_under_delta is not None and result.macro_delta < fail_under_delta:
        err.print(
            f"[red]FAIL[/red] macro delta {result.macro_delta:+.4f} < {fail_under_delta:+.4f}"
        )
        failed = True
    if failed:
        raise typer.Exit(EXIT_GATE_FAILED)


@app.command()
def sweep(
    corpus: Path = typer.Option(..., "--corpus", "-c", help="Corpus directory."),
    configs: str = typer.Option(..., "--configs", help='Glob, e.g. "configs/*.yaml".'),
    out: Path = typer.Option(Path("out"), help="Report directory."),
    runs_dir: Path = typer.Option(Path("runs"), help="Where runs are stored."),
    concurrency: int = typer.Option(2, "--concurrency", "-j", help="Parallel documents."),
    limit: int = typer.Option(0, help="Evaluate only the first N cases."),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Ignore and do not write the result cache."
    ),
    cache_dir: Path = typer.Option(Path(".docvlm-cache"), help="Cache directory."),
    bootstrap: int = typer.Option(2000, help="Bootstrap iterations (<200 disables CIs)."),
    baseline: str = typer.Option("", help="Config name to diff every other run against."),
) -> None:
    """Run several configs over one corpus and rank them.

    The cache is shared across configs, so re-running a sweep after adding one
    config only pays for that config.
    """
    paths = sorted(Path(p) for p in globlib.glob(configs))
    if not paths:
        err.print(f"[red]no config matched {configs!r}[/red]")
        raise typer.Exit(EXIT_BAD_INPUT)

    try:
        loaded = load_corpus(corpus)
    except CorpusError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_BAD_INPUT) from exc
    loaded = loaded.filter(limit=limit or None)

    cache = ResultCache(cache_dir, enabled=not no_cache)
    store = RunStore(runs_dir)
    all_metrics = []
    runs = {}

    for path in paths:
        cfg = load_config(path)
        console.print(f"→ [bold]{cfg.name}[/bold] ({path.name})")
        result = asyncio.run(run_config(loaded, cfg, concurrency=concurrency, cache=cache))
        store.save(result, overwrite=True)
        runs[cfg.name] = result
        metrics = compute_metrics(result, bootstrap=bootstrap)
        all_metrics.append(metrics)
        write_reports(metrics, out, cases=result.cases)
    cache.close()

    print_sweep(all_metrics, console)
    out.mkdir(parents=True, exist_ok=True)
    summary = [
        f"# Sweep — corpus `{loaded.name}` @ `{loaded.hash}` ({len(loaded)} cases)",
        "",
        sweep_table(all_metrics),
        "",
    ]
    if baseline and baseline in runs:
        for cfg_name, result in runs.items():
            if cfg_name == baseline:
                continue
            summary.append(markdown_diff(diff_runs(runs[baseline], result, bootstrap=bootstrap)))
    (out / "sweep.md").write_text("\n".join(summary), encoding="utf-8")
    console.print(f"[dim]sweep summary: {out / 'sweep.md'}[/dim]")


@app.command(name="list")
def list_runs(runs_dir: Path = typer.Option(Path("runs"))) -> None:
    """List stored runs."""
    store = RunStore(runs_dir)
    rows = store.list_runs()
    if not rows:
        console.print(f"[yellow]no runs in {runs_dir}[/yellow]")
        return
    for name, finished, n in rows:
        console.print(f"{name:<40} {finished:<28} {n} cases")


@app.command()
def cache(
    clear: bool = typer.Option(False, help="Delete cached inferences."),
    cache_dir: Path = typer.Option(Path(".docvlm-cache")),
) -> None:
    """Inspect or clear the inference cache."""
    store = ResultCache(cache_dir)
    if clear:
        n = store.clear()
        console.print(f"cleared {n} cached inference(s)")
    else:
        stats = store.stats()
        console.print(
            f"{stats['rows']} cached inference(s) · "
            f"{stats.get('size_bytes', 0) / 1024:.1f} KB · {stats.get('path')}"
        )
    store.close()


@app.command()
def order(
    corpus: Path = typer.Option(..., "--corpus", "-c"),
    out: Path = typer.Option(..., "--out", "-o", help="One case id per line."),
    by: str = typer.Option(
        "first", help="Stratum: 'first' tag, 'all' tags, or a tag prefix such as 'stratum_'."
    ),
    seed: int = typer.Option(7),
) -> None:
    """Write a run order where every prefix is a fair sample of the corpus.

    Feed the pipeline in this order and ``watch`` can stop it early without
    measuring only the documents that happened to come first.
    """
    try:
        loaded = load_corpus(corpus)
    except CorpusError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_BAD_INPUT) from exc
    ids = stratified_order(loaded, by=by, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ids) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for case in loaded:
        key = stratum_of(case.tags, by)
        counts[key] = counts.get(key, 0) + 1
    console.print(f"{len(ids)} cases in {len(counts)} strata -> {out}")
    for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        console.print(f"  {key:<24} {n:>5}  {n / len(ids):5.1%}")


@app.command()
def watch(
    corpus: Path = typer.Option(..., "--corpus", "-c"),
    baseline: str = typer.Option(..., "--baseline", "-b"),
    predictions: Path = typer.Option(
        ..., "--predictions", "-p", help="Replay .jsonl being written."
    ),
    runs_dir: Path = typer.Option(Path("runs")),
    metric: str = typer.Option("macro", help="macro | afc (all fields correct)."),
    margin: float = typer.Option(0.01, help="Equivalence margin on the metric."),
    looks: int = typer.Option(10, help="Planned looks over the whole corpus."),
    alpha: float = typer.Option(0.05, help="Total false-stop budget over all looks."),
    follow: bool = typer.Option(True, "--follow/--once", help="Keep tailing, or score once."),
    interval: float = typer.Option(15.0, help="Seconds between reads while following."),
    idle_timeout: float = typer.Option(
        0.0, help="Give up after this many seconds without a new document (0 = never)."
    ),
    status_file: Path = typer.Option(None, help="JSON decision, rewritten on every read."),
    save: str = typer.Option("", help="Also store the candidate as a run under this name."),
    overwrite: bool = typer.Option(False),
    allow_tracked: bool = typer.Option(False),
) -> None:
    """Score a growing predictions file and stop as soon as the A/B is decided.

    Exit 0 when better, equivalent, inconclusive or still running (--once);
    exit 2 when worse or when critical hallucinations went up — so it gates.
    """
    try:
        loaded = load_corpus(corpus)
        base = RunStore(runs_dir).load(baseline)
    except (CorpusError, FileNotFoundError) as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_BAD_INPUT) from exc
    if base.corpus_hash != loaded.hash:
        err.print(f"[red]NOT COMPARABLE[/red] baseline corpus {base.corpus_hash} != {loaded.hash}")
        raise typer.Exit(EXIT_BAD_INPUT)
    if metric not in ("macro", "afc"):
        err.print("[red]--metric must be macro or afc[/red]")
        raise typer.Exit(EXIT_BAD_INPUT)
    pii = pii_fields(loaded)
    if save and pii:
        try:
            guard_untracked(runs_dir, allow=allow_tracked)
        except TrackedOutputError as exc:
            err.print(f"[red]{exc}[/red]")
            raise typer.Exit(EXIT_BAD_INPUT) from exc

    base_by_id = {c.case_id: c for c in base.cases if loaded.by_id(c.case_id)}
    plan = SequentialPlan(
        n_total=len(base_by_id), alpha=alpha, looks=looks, margin=margin, metric=metric
    )
    state = SequentialState(plan, weights=base.weights, critical=set(base.critical_fields))
    tail = PredictionTail(predictions)
    cases = []
    started = time.time()
    last_new = started
    console.print(
        f"watching {predictions} vs [bold]{base.name}[/bold] · {plan.n_total} paired documents · "
        f"looks at {plan.look_sizes()} · alpha {alpha} · margin ±{margin}"
    )

    while True:
        fresh = [r for r in (score_record(rec, loaded) for rec in tail.read_new()) if r]
        if fresh:
            last_new = time.time()
            cases.extend(fresh)
        for look in evaluate(state, base_by_id, cases):
            console.print(
                f"  look n={look.n:<5} α={look.alpha:.2e}  Δ {look.delta:+.4f} "
                f"[{look.low:+.4f}, {look.high:+.4f}]  -> {look.decision}"
            )
        paired = sum(1 for c in cases if c.case_id in base_by_id)
        note = (
            f"Δ {state.looks[-1].delta:+.4f} · {state.decision}"
            if state.looks
            else f"first look at {plan.look_sizes()[0] if plan.look_sizes() else '?'}"
        )
        console.print(
            progress_line(paired, plan.n_total, elapsed_since(cases, started), note),
            highlight=False,
        )
        if status_file:
            _write_status(status_file, state, paired, tail.meta)
        done = state.stopped or paired >= plan.n_total
        idle = idle_timeout and time.time() - last_new > idle_timeout
        if done or not follow or idle:
            break
        time.sleep(interval)

    if tail.duplicates:
        err.print(f"[yellow]{tail.duplicates} duplicate id(s) ignored (first answer kept)[/yellow]")
    console.print(f"[bold]VERDICT:[/bold] {sequential_verdict(state)}")

    if save:
        for case in cases:
            redact_case(case, pii, loaded.hash)
        run_result = RunResult(
            name=save,
            corpus_name=loaded.name,
            corpus_hash=loaded.hash,
            config_name=save,
            config_hash="",
            field_names=list(loaded.schema.model_fields),
            cases=cases,
            weights=base.weights,
            critical_fields=base.critical_fields,
            provenance={
                "docvlm_eval_version": __version__,
                "backend": {"runner": "watch", "pipeline": tail.meta},
                "sequential": {
                    "decision": state.decision,
                    "looks": [lk.to_dict() for lk in state.looks],
                    "n_total": plan.n_total,
                },
                "partial": paired < plan.n_total,
            },
        )
        path = RunStore(runs_dir).save(run_result, overwrite=overwrite)
        console.print(f"[dim]run saved to {path}[/dim]")

    if state.decision in ("worse", "harm"):
        raise typer.Exit(EXIT_GATE_FAILED)


def _write_status(path: Path, state: SequentialState, paired: int, meta: dict) -> None:
    payload = {
        "decision": state.decision,
        "stop": state.stopped,
        "verdict": sequential_verdict(state),
        "n": paired,
        "n_total": state.plan.n_total,
        "looks": [lk.to_dict() for lk in state.looks],
        "pipeline": meta,
        "updated_at": time.time(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic: a reader never sees half a decision


@app.command()
def agree(
    a: str = typer.Option(..., "-a", help="First run of the config."),
    b: str = typer.Option(..., "-b", help="Second run of the same config."),
    runs_dir: Path = typer.Option(Path("runs")),
) -> None:
    """How much a config disagrees with itself: the noise floor under every diff."""
    store = RunStore(runs_dir)
    try:
        ra, rb = store.load(a), store.load(b)
    except FileNotFoundError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_BAD_INPUT) from exc
    if ra.corpus_hash != rb.corpus_hash:
        err.print("[red]NOT COMPARABLE[/red] the two runs used different corpora")
        raise typer.Exit(EXIT_BAD_INPUT)
    report = agreement(ra, rb)
    print(render_agreement(report))
    if not report.reproducible:
        raise typer.Exit(EXIT_GATE_FAILED)


def _split(value: str) -> list[str] | None:
    items = [v.strip() for v in value.split(",") if v.strip()]
    return items or None


def _gate(metrics, fail_under: float | None, fail_hallucination_over: float | None) -> None:
    failed = False
    if fail_under is not None and metrics.macro_accuracy < fail_under:
        err.print(f"[red]FAIL[/red] macro accuracy {metrics.macro_accuracy:.4f} < {fail_under:.4f}")
        failed = True
    if fail_hallucination_over is not None and metrics.hallucination_rate > fail_hallucination_over:
        err.print(
            f"[red]FAIL[/red] hallucination rate {metrics.hallucination_rate:.4f} > "
            f"{fail_hallucination_over:.4f}"
        )
        failed = True
    if failed:
        sys.exit(EXIT_GATE_FAILED)


if __name__ == "__main__":  # pragma: no cover
    app()
