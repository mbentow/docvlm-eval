# Changelog

## 0.2.0 — 2026-08-07

First public release.

**Engine**

- Corpus format (`manifest.jsonl` + `schema.py` + images) with content hashing over ids,
  image bytes, ground truth and schema.
- Pydantic v2 extraction schemas carrying a per-field comparison policy; the same class
  produces the JSON Schema used for constrained decoding.
- Comparison modes: `EXACT`, `TEXT`, `DIGITS`, `NUMBER`, `DATE`, `BOOL`, `SET_TEXT`.
- Separated failure modes: `correct`, `missing`, `hallucinated`, `wrong`, `malformed`,
  `refused`, plus a distinct hallucination rate for fields marked `critical`.
- Macro accuracy and `ALL-FIELDS-CORRECT`, both with document-level bootstrap intervals.
- Tag slicing, sorted worst-first.
- `diff` with a paired bootstrap on shared case ids; per-field significance.
- Inference cache keyed by `(config, case, image, schema)`; scoring always recomputed.
- Runners: Ollama (constrained decoding, multi-host round robin, model digest in provenance)
  and a deterministic mock.
- Reports: terminal, Markdown, self-contained HTML, JSON.
- `--fail-under`, `--fail-hallucination-over`, `--fail-on-regression` for CI.

**Corpus**

- Public synthetic corpus generator with fabricated data and per-tag degradations.
- `--hard` variant: the clean corpus saturates at ~1.000 on current vision models.

**Bugs caught before release, each now covered by a test**

Every one of these would have changed a printed number without raising anything.

- ISO ground-truth dates were reinterpreted by the `dayfirst` setting, shifting every date in
  a corpus by weeks.
- pt-BR month names (`14 de março de 2026`) scored as `malformed` — an evaluator bug that read
  as a model failure.
- The bootstrap percentile bounds were indexed asymmetrically: the upper tail kept one fewer
  draw than the lower one, shifting every interval upward. The iteration count also had no
  floor, so `--bootstrap 20` printed min/max labelled "95% CI".
- `diff` warned about mismatched corpora but compared them anyway. Case ids repeat across
  corpora, so it paired unrelated documents and reported a confident, significant, meaningless
  delta. It now exits non-zero unless `--force`.
- `diff` took point estimates from the full runs and intervals from the paired subset, so a
  filtered comparison put a confidence band around a number it was not computed from.
- `critical` and `weight` declared in the schema were inert — only the YAML equivalents were
  read — and `report`/`diff` ignored both, so they could print different numbers than `run`
  had for the same run. Policy is now resolved once and stored in the run.
- `DIGITS` charged an unparseable *ground truth* to the model instead of flagging the corpus.
- The mock backend used one shared RNG across concurrent documents, so the same run scored
  differently at `-j 1` and `-j 4`. Seeding is now per case.
- A fully cached re-run reported 0 ms latency, which read as "instant" rather than "not
  measured now"; cached cases keep their measured latency and are counted separately.
- Config provenance embedded the absolute path of the machine that produced the run.
