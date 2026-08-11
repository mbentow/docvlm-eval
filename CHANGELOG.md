# Changelog

## 0.4.0 — 2026-08-11

**Prompt echo detection (`leakage.py`, `docvlm-eval echo`).** Answers a question accuracy
cannot: did the model read the document, or repeat the instructions? Every worked example in a
prompt is a string the model can emit without looking at the image, and it passes for a normal
answer because it is one.

- A literal is flagged only when it is **over-produced relative to the ground truth *and* less
  accurate than the model's own baseline**. Both conditions, never either.
- That second condition is the whole design. Counting how often the model says your example
  proves nothing — examples are chosen *because* they are the common case. Measured on a 200k
  deployment: the inline example appeared in 4.4% of predictions against a 7.7% base rate in
  the truth, and those predictions were 90.2% correct versus 80.6% elsewhere. Frequency raised
  the question; accuracy answered it; the check correctly stays silent.
- Lift is `None`, not infinity, when the truth never contains the literal — on a small corpus
  that is usually a fact about the corpus, not about the model.
- Literals come from quoted spans and runs of two or more ALL-CAPS tokens. Single caps words
  are excluded: `JSON`, `ONLY`, `NOT` are how prompts shout, not answers a model would copy,
  and including them buries the signal.
- Fewer than 3 echoing cases is reported but never flagged — accuracy on a subset that small is
  noise.
- `echo` refuses to run when the config's `prompt_hash` does not match the run's. Scoring the
  wrong prompt is worse than not scoring: everything comes back clean and the check looks like
  it passed.
- Exits 2 on a flagged literal, so it gates a release like `diff` does.

## 0.3.0 — 2026-08-09

**Selective prediction (`selective.py`).** Risk–coverage curve, AURC, and the two questions a
document pipeline actually has to answer: *what accuracy remains if I automate X%?* and
*how much can I automate at a given quality bar?* Rendered in the terminal, in Markdown and
in the run JSON.

- Guard against a persuasive but meaningless curve: AURC is compared to the mean of 25 random
  orderings, and the report says plainly when the confidence signal ranks no better than
  shuffling.
- `validate_holdout()` fits the coverage level on a random half and measures it on the other,
  because reading an operating point off the curve that produced it is circular. On the
  bundled 30b run the curve promises 1.000 at 50% coverage; held out, the same policy delivers
  0.917 at 54% and meets its target on 60% of splits. Both numbers are true; only the second
  is a forecast.
- Coverage is transferred as a **fraction**, not as a raw confidence threshold: a threshold
  admits the whole block of documents sharing that value, so a degenerate signal accepts
  everything and the policy silently evaluates to no policy at all.
- Policies automating less than 5% are rejected — "hits 95% by accepting one lucky document"
  reads as success and is not one.
- Confidence comes from a declared `confidence` field, or falls back to the mean field score.
- Hallucination rate is tracked along the curve — coverage that preserves accuracy while
  concentrating invented values is not a win.

**Image preprocessing profiles (`preprocess.py`).** Named, deterministic, part of the config
hash so they reach the cache key and the run provenance. Shipped with a negative result:
on the bundled hard corpus, upscaling regressed accuracy and doubled latency.

**Bug found by the new tests:** ties in the confidence ranking were broken by input order,
because `sorted` is stable. A constant confidence signal therefore inherited whatever order
the corpus happened to have and produced a curve that looked informative while ranking
nothing. Ties are now broken by a seeded hash.

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
