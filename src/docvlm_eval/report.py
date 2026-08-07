"""Reports: terminal, Markdown, HTML.

Layout choices are opinionated, because a report's job is to make the *right*
number hard to miss:

* the per-field table comes first — a macro average hides that one field is at
  0.61 while another is at 0.97;
* failure modes are separate columns, with hallucinations flagged;
* ``ALL-FIELDS-CORRECT`` is on its own line, because that is the one that maps
  to "this document can skip human review";
* the tag breakdown is sorted worst-first — that is where the work is.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from docvlm_eval.metrics import RunDiff, RunMetrics

WARN_HALLUCINATION = 0.01
WARN_ACCURACY_GAP = 0.10


# --------------------------------------------------------------------------- #
# Terminal
# --------------------------------------------------------------------------- #


def print_run(metrics: RunMetrics, console: Console | None = None) -> None:
    console = console or Console()
    console.print()
    console.print(
        f"[bold]CONFIG[/bold] {metrics.config_name}    "
        f"[bold]CORPUS[/bold] {metrics.corpus_name} @ {metrics.corpus_hash} "
        f"({metrics.n_cases} cases)"
    )

    table = Table(show_edge=False, header_style="bold", pad_edge=False)
    table.add_column("FIELD", no_wrap=True)
    table.add_column("ACC", justify="right")
    table.add_column("95% CI", justify="center")
    table.add_column("MISS", justify="right")
    table.add_column("HALL", justify="right")
    table.add_column("WRONG", justify="right")
    table.add_column("MALF", justify="right")
    table.add_column("REF", justify="right")
    table.add_column("", width=2)

    for f in metrics.fields:
        table.add_row(
            f.name + (" *" if f.critical else ""),
            f"{f.accuracy:.3f}",
            f"{f.accuracy_ci.low:.2f}–{f.accuracy_ci.high:.2f}" if f.accuracy_ci else "—",
            _pct(f.missing_rate),
            _pct(f.hallucination_rate, warn=f.hallucination_rate > WARN_HALLUCINATION),
            _pct(f.wrong_rate),
            _pct(f.malformed_rate),
            _pct(f.refused_rate),
            "[red]![/red]" if f.hallucination_rate > WARN_HALLUCINATION else "",
        )
    console.print(table)

    macro = metrics.macro_accuracy_ci
    afc = metrics.all_fields_correct_ci
    console.print(
        f"[bold]MACRO[/bold]              {metrics.macro_accuracy:.3f}"
        + (f"  [dim][{macro.low:.3f}, {macro.high:.3f}][/dim]" if macro else "")
    )
    console.print(
        f"[bold]ALL-FIELDS-CORRECT[/bold] {metrics.all_fields_correct:.3f}"
        + (f"  [dim][{afc.low:.3f}, {afc.high:.3f}][/dim]" if afc else "")
        + "   [dim]<- the business metric[/dim]"
    )
    if metrics.critical_hallucination_rate:
        console.print(
            f"[bold red]CRITICAL HALLUCINATION[/bold red] "
            f"{metrics.critical_hallucination_rate * 100:.2f}%"
        )

    if metrics.by_tag:
        console.print()
        tag_table = Table(show_edge=False, header_style="bold", pad_edge=False)
        tag_table.add_column("BY TAG", no_wrap=True)
        tag_table.add_column("ACC", justify="right")
        tag_table.add_column("ALL-OK", justify="right")
        tag_table.add_column("n", justify="right")
        tag_table.add_column("", width=2)
        best = max(t.accuracy for t in metrics.by_tag)
        for t in metrics.by_tag:
            gap = best - t.accuracy
            tag_table.add_row(
                t.tag,
                f"{t.accuracy:.3f}",
                f"{t.all_fields_correct:.3f}",
                str(t.n),
                "[yellow]![/yellow]" if gap > WARN_ACCURACY_GAP else "",
            )
        console.print(tag_table)

    console.print()
    console.print(
        f"LATENCY p50 {metrics.latency_p50:,.0f}ms   p95 {metrics.latency_p95:,.0f}ms   "
        f"tokens in/out {metrics.tokens_in_mean:,.0f}/{metrics.tokens_out_mean:,.0f}"
        + (f"   cost ${metrics.cost_total:.4f}" if metrics.cost_total else "")
    )
    if metrics.n_cached:
        console.print(
            f"[dim]{metrics.n_cached}/{metrics.n_cases} served from cache — "
            "latency above is from the original call, not measured now[/dim]"
        )
    if metrics.n_refused:
        console.print(f"[yellow]{metrics.n_refused} case(s) produced no usable output[/yellow]")
    console.print()


def print_diff(diff: RunDiff, console: Console | None = None) -> None:
    console = console or Console()
    console.print()
    console.print(
        f"[bold]CONFIG[/bold] {diff.candidate.config_name}    "
        f"[bold]vs BASELINE[/bold] {diff.baseline.config_name}"
    )
    console.print(
        f"Corpus: {diff.baseline.corpus_name} @ {diff.baseline.corpus_hash} "
        f"({diff.n_paired} paired cases)"
    )
    if not diff.comparable:
        console.print(f"[bold red]NOT COMPARABLE[/bold red] {diff.incomparable_reason}")

    table = Table(show_edge=False, header_style="bold", pad_edge=False)
    table.add_column("FIELD", no_wrap=True)
    table.add_column("BASE", justify="right")
    table.add_column("CAND", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("Δ 95% CI", justify="center")
    table.add_column("Δ HALL", justify="right")
    table.add_column("", width=3)

    for f in diff.fields:
        if f.significant:
            marker = "[green]▲[/green]" if f.delta > 0 else "[red]▼[/red]"
        else:
            marker = "[dim]~[/dim]"
        table.add_row(
            f.name,
            f"{f.baseline:.3f}",
            f"{f.candidate:.3f}",
            _signed(f.delta),
            f"{f.delta_ci.low:+.3f}, {f.delta_ci.high:+.3f}" if f.delta_ci else "—",
            _signed(f.hallucination_delta, pct=True),
            marker,
        )
    console.print(table)

    console.print(
        f"[bold]MACRO[/bold]              {diff.candidate.macro_accuracy:.3f}  "
        f"{_signed(diff.macro_delta)}"
        + (
            f"  [dim][{diff.macro_delta_ci.low:+.3f}, {diff.macro_delta_ci.high:+.3f}][/dim]"
            if diff.macro_delta_ci
            else ""
        )
    )
    console.print(
        f"[bold]ALL-FIELDS-CORRECT[/bold] {diff.candidate.all_fields_correct:.3f}  "
        f"{_signed(diff.afc_delta)}"
        + (
            f"  [dim][{diff.afc_delta_ci.low:+.3f}, {diff.afc_delta_ci.high:+.3f}][/dim]"
            if diff.afc_delta_ci
            else ""
        )
    )
    console.print()
    console.print(f"[bold]VERDICT:[/bold] {verdict(diff)}")
    console.print()


def verdict(diff: RunDiff) -> str:
    """One sentence, stating explicitly when a difference is not significant.

    "Quality up" when the interval straddles zero is the most common way an
    honest benchmark still misleads.
    """
    parts: list[str] = []
    ci = diff.macro_delta_ci
    if ci and ci.low > 0:
        parts.append(f"quality up {diff.macro_delta:+.3f} (significant)")
    elif ci and ci.high < 0:
        parts.append(f"quality down {diff.macro_delta:+.3f} (significant)")
    else:
        parts.append(
            f"quality change {diff.macro_delta:+.3f} is inside the noise "
            f"(n={diff.n_paired}) — not a result"
        )

    regressions = diff.regressions
    if regressions:
        worst = min(regressions, key=lambda f: f.delta)
        parts.append(
            f"{len(regressions)} field(s) regressed, worst {worst.name} {worst.delta:+.3f}"
        )

    hall = diff.candidate.hallucination_rate - diff.baseline.hallucination_rate
    if hall > 0.005:
        parts.append(f"hallucination rate up {hall * 100:+.2f}pp — check before shipping")

    if diff.latency_ratio_p50 and abs(diff.latency_ratio_p50 - 1) > 0.15:
        parts.append(
            f"latency p50 {diff.latency_ratio_p50:.2f}x "
            f"({diff.candidate.latency_p50:,.0f}ms) — check the queue SLA"
        )
    return "; ".join(parts)


def _pct(value: float, warn: bool = False) -> str:
    text = f"{value * 100:.1f}%"
    return f"[red]{text}[/red]" if warn else text


def _signed(value: float, pct: bool = False) -> str:
    if pct:
        return f"{value * 100:+.1f}pp"
    return f"{value:+.3f}"


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


def markdown_run(metrics: RunMetrics) -> str:
    lines = [
        f"# Run report — `{metrics.run_name}`",
        "",
        f"- **Config**: `{metrics.config_name}`",
        f"- **Corpus**: `{metrics.corpus_name}` @ `{metrics.corpus_hash}` "
        f"({metrics.n_cases} cases)",
    ]
    backend = metrics.provenance.get("backend", {})
    if backend:
        lines.append(
            f"- **Backend**: `{backend.get('runner')}` · `{backend.get('model')}`"
            + (f" · digest `{backend['model_digest']}`" if backend.get("model_digest") else "")
            + (f" · {backend['quantization']}" if backend.get("quantization") else "")
        )
    lines += [
        f"- **Prompt hash**: `{metrics.provenance.get('prompt_hash', '—')}`",
        f"- **Schema hash**: `{metrics.provenance.get('schema_hash', '—')}`",
        "",
        "## Per field",
        "",
        "| field | acc | 95% CI | missing | hallucinated | wrong | malformed | refused |",
        "|---|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for f in metrics.fields:
        ci = f"{f.accuracy_ci.low:.3f}–{f.accuracy_ci.high:.3f}" if f.accuracy_ci else "—"
        flag = " ⚠️" if f.hallucination_rate > WARN_HALLUCINATION else ""
        lines.append(
            f"| `{f.name}`{' *' if f.critical else ''} | {f.accuracy:.3f} | {ci} | "
            f"{f.missing_rate * 100:.1f}% | {f.hallucination_rate * 100:.1f}%{flag} | "
            f"{f.wrong_rate * 100:.1f}% | {f.malformed_rate * 100:.1f}% | "
            f"{f.refused_rate * 100:.1f}% |"
        )

    macro_ci = (
        f" ({metrics.macro_accuracy_ci.low:.3f}–{metrics.macro_accuracy_ci.high:.3f})"
        if metrics.macro_accuracy_ci
        else ""
    )
    afc_ci = (
        f" ({metrics.all_fields_correct_ci.low:.3f}–{metrics.all_fields_correct_ci.high:.3f})"
        if metrics.all_fields_correct_ci
        else ""
    )
    lines += [
        "",
        f"**MACRO** {metrics.macro_accuracy:.3f}{macro_ci}  ",
        f"**ALL-FIELDS-CORRECT** {metrics.all_fields_correct:.3f}{afc_ci} "
        "— the metric that maps to *this document skips human review*",
        "",
    ]

    if metrics.by_tag:
        lines += [
            "## By tag",
            "",
            "| tag | acc | all-fields-correct | n |",
            "|---|---:|---:|---:|",
        ]
        for t in metrics.by_tag:
            lines.append(f"| `{t.tag}` | {t.accuracy:.3f} | {t.all_fields_correct:.3f} | {t.n} |")
        lines.append("")

    lines += [
        "## Operational",
        "",
        f"- latency p50 **{metrics.latency_p50:,.0f} ms**, p95 **{metrics.latency_p95:,.0f} ms**",
        f"- tokens in/out (mean) {metrics.tokens_in_mean:,.0f} / {metrics.tokens_out_mean:,.0f}",
        f"- cost total ${metrics.cost_total:.4f}",
        f"- cases with no usable output: {metrics.n_refused}",
    ]
    if metrics.n_cached:
        lines.append(
            f"- {metrics.n_cached}/{metrics.n_cases} served from cache; the latency above is "
            "from the original call"
        )
    lines.append("")
    return "\n".join(lines)


def markdown_diff(diff: RunDiff) -> str:
    lines = [
        f"# Diff — `{diff.candidate.config_name}` vs `{diff.baseline.config_name}`",
        "",
        f"Corpus `{diff.baseline.corpus_name}` @ `{diff.baseline.corpus_hash}` "
        f"({diff.n_paired} paired cases)",
        "",
    ]
    if not diff.comparable:
        lines += [f"> ⛔ **NOT COMPARABLE** — {diff.incomparable_reason}", ""]
    lines += [
        "| field | baseline | candidate | Δ | Δ 95% CI | Δ hallucinated | Δ missing | sig |",
        "|---|---:|---:|---:|:---:|---:|---:|:---:|",
    ]
    for f in diff.fields:
        ci = f"{f.delta_ci.low:+.3f}, {f.delta_ci.high:+.3f}" if f.delta_ci else "—"
        mark = ("▲" if f.delta > 0 else "▼") if f.significant else "~"
        lines.append(
            f"| `{f.name}` | {f.baseline:.3f} | {f.candidate:.3f} | {f.delta:+.3f} | {ci} | "
            f"{f.hallucination_delta * 100:+.1f}pp | {f.missing_delta * 100:+.1f}pp | {mark} |"
        )
    lines += [
        "",
        f"**MACRO** {diff.candidate.macro_accuracy:.3f} ({diff.macro_delta:+.3f})  ",
        f"**ALL-FIELDS-CORRECT** {diff.candidate.all_fields_correct:.3f} ({diff.afc_delta:+.3f})",
        "",
        "| | baseline | candidate |",
        "|---|---:|---:|",
        f"| hallucination rate | {diff.baseline.hallucination_rate * 100:.2f}% "
        f"| {diff.candidate.hallucination_rate * 100:.2f}% |",
        f"| critical hallucination rate | "
        f"{diff.baseline.critical_hallucination_rate * 100:.2f}% "
        f"| {diff.candidate.critical_hallucination_rate * 100:.2f}% |",
        f"| latency p50 | {diff.baseline.latency_p50:,.0f} ms "
        f"| {diff.candidate.latency_p50:,.0f} ms |",
        "",
        f"**Verdict:** {_strip_markup(verdict(diff))}",
        "",
    ]
    return "\n".join(lines)


def _strip_markup(text: str) -> str:
    import re

    return re.sub(r"\[/?[a-z ]+\]", "", text)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

_HTML_SHELL = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ font: 15px/1.55 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif;
        max-width: 60rem; margin: 2rem auto; padding: 0 1rem; }}
 h1 {{ font-size: 1.4rem; margin-bottom: .2rem; }}
 .meta {{ color: #777; font-size: .85rem; margin-bottom: 1.4rem; }}
 table {{ border-collapse: collapse; width: 100%; margin: .8rem 0 1.6rem; font-size: .9rem; }}
 th, td {{ padding: .38rem .55rem; border-bottom: 1px solid #8883; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; font-family: ui-monospace,monospace; }}
 th {{ text-align: right; font-weight: 600; border-bottom: 2px solid #8886; }}
 .warn {{ color: #c0392b; font-weight: 600; }}
 .bar {{ background: linear-gradient(90deg,#4a90d9 var(--w),transparent var(--w));
         border-radius: 2px; }}
 .headline {{ display: flex; gap: 2.5rem; margin: 1.2rem 0 2rem; flex-wrap: wrap; }}
 .headline div {{ }}
 .headline .n {{ font-size: 1.9rem; font-weight: 650; }}
 .headline .l {{ font-size: .75rem; text-transform: uppercase; letter-spacing: .06em;
                 color: #777; }}
 .note {{ background: #4a90d915; border-left: 3px solid #4a90d9; padding: .7rem 1rem;
          font-size: .87rem; }}
 details {{ margin-top: 2rem; font-size: .8rem; }}
 pre {{ overflow-x: auto; background: #8881; padding: .8rem; border-radius: 4px; }}
</style></head><body>
{body}
</body></html>
"""


def html_run(metrics: RunMetrics) -> str:
    """Self-contained HTML. No CDN, no build step — it has to survive being
    emailed to somebody."""
    e = html.escape
    rows = []
    for f in metrics.fields:
        warn = ' class="warn"' if f.hallucination_rate > WARN_HALLUCINATION else ""
        ci = f"{f.accuracy_ci.low:.3f}–{f.accuracy_ci.high:.3f}" if f.accuracy_ci else "—"
        rows.append(
            f"<tr><td>{e(f.name)}{' ★' if f.critical else ''}</td>"
            f'<td class="bar" style="--w:{f.accuracy * 100:.0f}%">{f.accuracy:.3f}</td>'
            f"<td>{ci}</td>"
            f"<td>{f.missing_rate * 100:.1f}%</td>"
            f"<td{warn}>{f.hallucination_rate * 100:.1f}%</td>"
            f"<td>{f.wrong_rate * 100:.1f}%</td>"
            f"<td>{f.malformed_rate * 100:.1f}%</td>"
            f"<td>{f.refused_rate * 100:.1f}%</td></tr>"
        )

    tag_rows = "".join(
        f"<tr><td>{e(t.tag)}</td>"
        f'<td class="bar" style="--w:{t.accuracy * 100:.0f}%">{t.accuracy:.3f}</td>'
        f"<td>{t.all_fields_correct:.3f}</td><td>{t.n}</td></tr>"
        for t in metrics.by_tag
    )

    macro_ci = (
        f"[{metrics.macro_accuracy_ci.low:.3f}, {metrics.macro_accuracy_ci.high:.3f}]"
        if metrics.macro_accuracy_ci
        else ""
    )
    afc_ci = (
        f"[{metrics.all_fields_correct_ci.low:.3f}, {metrics.all_fields_correct_ci.high:.3f}]"
        if metrics.all_fields_correct_ci
        else ""
    )

    body = f"""
<h1>{e(metrics.run_name)}</h1>
<div class="meta">config <code>{e(metrics.config_name)}</code> ·
 corpus <code>{e(metrics.corpus_name)}</code> @ <code>{e(metrics.corpus_hash)}</code> ·
 {metrics.n_cases} cases</div>

<div class="headline">
  <div><div class="n">{metrics.macro_accuracy:.3f}</div>
       <div class="l">macro accuracy <small>{macro_ci}</small></div></div>
  <div><div class="n">{metrics.all_fields_correct:.3f}</div>
       <div class="l">all fields correct <small>{afc_ci}</small></div></div>
  <div><div class="n">{metrics.hallucination_rate * 100:.2f}%</div>
       <div class="l">hallucination rate</div></div>
  <div><div class="n">{metrics.latency_p50:,.0f}<small>ms</small></div>
       <div class="l">latency p50 · p95 {metrics.latency_p95:,.0f}ms</div></div>
</div>

<div class="note"><b>all fields correct</b> is the number that maps to the
business: a document with one wrong field still needs a human.</div>

<h2>Per field</h2>
<table><thead><tr><th>field</th><th>acc</th><th>95% CI</th><th>missing</th>
<th>hallucinated</th><th>wrong</th><th>malformed</th><th>refused</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>

{
        "<h2>By tag</h2><table><thead><tr><th>tag</th><th>acc</th><th>all-ok</th><th>n</th>"
        "</tr></thead><tbody>" + tag_rows + "</tbody></table>"
        if tag_rows
        else ""
    }

<details><summary>Provenance</summary>
<pre>{e(json.dumps(metrics.provenance, indent=2, default=str))}</pre></details>
"""
    return _HTML_SHELL.format(title=e(metrics.run_name), body=body)


def write_reports(
    metrics: RunMetrics, out_dir: str | Path, *, stem: str | None = None
) -> dict[str, Path]:
    """Write ``.md``, ``.html`` and ``.json`` for one run."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = stem or metrics.run_name
    paths = {
        "markdown": out / f"{stem}.md",
        "html": out / f"{stem}.html",
        "json": out / f"{stem}.metrics.json",
    }
    paths["markdown"].write_text(markdown_run(metrics), encoding="utf-8")
    paths["html"].write_text(html_run(metrics), encoding="utf-8")
    paths["json"].write_text(json.dumps(metrics.to_dict(), indent=2, default=str), encoding="utf-8")
    return paths


def sweep_table(all_metrics: list[RunMetrics]) -> str:
    """Leaderboard across a sweep, best macro accuracy first."""
    lines = [
        "| config | macro | 95% CI | all-fields-correct | hallucination | p50 ms | n |",
        "|---|---:|:---:|---:|---:|---:|---:|",
    ]
    for m in sorted(all_metrics, key=lambda m: -m.macro_accuracy):
        ci = (
            f"{m.macro_accuracy_ci.low:.3f}–{m.macro_accuracy_ci.high:.3f}"
            if m.macro_accuracy_ci
            else "—"
        )
        lines.append(
            f"| `{m.config_name}` | {m.macro_accuracy:.3f} | {ci} | "
            f"{m.all_fields_correct:.3f} | {m.hallucination_rate * 100:.2f}% | "
            f"{m.latency_p50:,.0f} | {m.n_cases} |"
        )
    return "\n".join(lines)


def print_sweep(all_metrics: list[RunMetrics], console: Console | None = None) -> None:
    console = console or Console()
    table = Table(show_edge=False, header_style="bold", pad_edge=False)
    table.add_column("CONFIG", no_wrap=True)
    table.add_column("MACRO", justify="right")
    table.add_column("95% CI", justify="center")
    table.add_column("ALL-OK", justify="right")
    table.add_column("HALL", justify="right")
    table.add_column("p50 ms", justify="right")
    for m in sorted(all_metrics, key=lambda m: -m.macro_accuracy):
        table.add_row(
            m.config_name,
            f"{m.macro_accuracy:.3f}",
            f"{m.macro_accuracy_ci.low:.3f}–{m.macro_accuracy_ci.high:.3f}"
            if m.macro_accuracy_ci
            else "—",
            f"{m.all_fields_correct:.3f}",
            _pct(m.hallucination_rate, warn=m.hallucination_rate > WARN_HALLUCINATION),
            f"{m.latency_p50:,.0f}",
        )
    console.print()
    console.print(table)
    console.print()
