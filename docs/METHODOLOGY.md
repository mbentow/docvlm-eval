# Methodology

What this tool measures, how, and where it can mislead you. Read the limitations section
before quoting any number from it.

## 1. The unit of measurement is a field, not a document

Every field of every document receives exactly one outcome:

| outcome | condition |
|---|---|
| `correct` | prediction matches truth, **or** both are empty |
| `missing` | truth non-empty, prediction empty |
| `hallucinated` | truth empty, prediction non-empty |
| `wrong` | both non-empty, they disagree |
| `malformed` | prediction does not fit the declared type |
| `refused` | backend error, timeout, or model declined |

`accuracy = correct / total`.

### Why `missing` and `hallucinated` are separate

They cost the same point and have completely different consequences. A blank field is queued
for a human. An invented field enters a downstream system that will treat it as fact. A model
at 92% accuracy with 3% hallucination can be strictly worse than one at 88% with 0% — and no
averaged metric shows that.

Fields can be marked `critical` in the schema, which produces a separate
`critical_hallucination_rate`. A hallucinated free-text note and a hallucinated licence number
are not the same event.

### Empty is a real answer

Correctly returning nothing when the truth is nothing scores as `correct`. Treating it as
"no data" would reward a model that fills in every field, which is the failure mode most worth
avoiding.

Models express emptiness inconsistently — `null`, `"N/A"`, `"-"`, `"não informado"`, or by
omitting the key. All of these count as empty. Treating `"N/A"` as a value would inflate the
hallucination rate with what is in fact an abstention.

## 2. Normalisation is declared per field

There is no global normalisation, on purpose.

| mode | behaviour |
|---|---|
| `EXACT` | string equality, no normalisation |
| `TEXT` | normalise (whitespace, case, accents, punctuation), then optional fuzzy match |
| `DIGITS` | reduce both sides to digits; no digits in the prediction → `malformed` |
| `NUMBER` | parse (pt-BR and en formats), compare within absolute `tolerance` |
| `DATE` | parse, compare within `tolerance_days` |
| `BOOL` | parse `sim/yes/1/urgente`, `nao/no/0/rotina` |
| `SET_TEXT` | unordered set; exact-set match required |

**Fuzzy matching is opt-in per field.** `fuzzy_threshold` defaults to 1.0. Enable it only
where a human reviewer would genuinely accept the near miss. For a name, yes; for a licence
number, never — one wrong digit points at a different person, and the downstream system will
match it with full confidence.

**`SET_TEXT` is strict on purpose.** Returning three exams on a single-exam form is scored as
an error, because it means the model did not know when to stop. Per-item precision and recall
are reported in the detail string for diagnosis, but the field only counts as correct on an
exact set match. When the prediction shares nothing with the truth *and* is longer, the
outcome is `hallucinated` rather than `wrong` — that is the "transcribed the whole pre-printed
menu" failure, not a near miss.

**Dates:** ISO `YYYY-MM-DD` is always parsed as ISO, regardless of `dayfirst`. Ground truth is
normally written that way, and letting `dayfirst` reinterpret it shifts every date in the
corpus. Predictions go through `dayfirst` (default `True`, matching pt-BR and EU documents)
plus a pt-BR month-name pass, because dateutil only knows English and would otherwise score
`14 de março de 2026` as malformed — an evaluator bug that looks like a model failure.

## 3. Aggregation

**`MACRO`** — weighted mean of per-field accuracies. Macro rather than micro so a field
present on every document does not dominate. `weight: 0` reports a field but excludes it from
the aggregate.

**`ALL-FIELDS-CORRECT`** — the fraction of documents with no error in any field. This is the
number that corresponds to "can skip human review", and it is always lower than macro. Quote
this one in a business conversation.

**By tag** — macro accuracy and all-fields-correct restricted to cases carrying each tag,
sorted worst first. Tags are free-form; a shared vocabulary is suggested in `corpus.py`
(`handwritten`, `printed`, `phone_photo`, `scanned`, `rotated`, `low_light`, `partial`,
`stamped`, `faded`, `skewed`).

## 4. Confidence intervals

**Percentile bootstrap over resampled documents**, 2000 iterations by default, seeded.

Documents are the resampling unit, never fields. Fields within a document are strongly
correlated — one bad photograph hurts all of them — so resampling fields would produce
intervals far too narrow and make every difference look significant.

No interval is reported below 5 cases, where it would be theatre rather than information.

### Diffs use a paired bootstrap

`diff` aligns the two runs on their shared case ids and resamples **document indices once per
draw**, applying the same indices to both runs. This removes the variance of "which documents
are hard" and leaves the variance of "which model is better", which is the quantity of
interest. A per-field delta is called *significant* only when its 95% interval excludes zero.

`--fail-on-regression` fires only on a significant drop, so sampling noise does not block a
pipeline.

### Sample size, concretely

At n=60, the 95% half-width on a proportion near 0.75 is roughly ±0.11. Differences smaller
than that are not measurable with that corpus, whatever the point estimate says. If you need
to detect a 2-point difference, you need a corpus in the high hundreds. The tool will tell
you this in the verdict line rather than letting you over-read the number.

## 5. Reproducibility

Every run records:

- `docvlm_eval` version, Python version, platform
- runner, model tag, **model digest**, quantisation, parameter size, server version
- corpus content hash (ids + image bytes + ground truth + schema)
- prompt hash, schema hash, temperature, concurrency, preprocessing name

Model **tags are mutable** — `qwen3-vl:30b` today is not necessarily the blob you benchmarked
last month — so the digest is recorded alongside it.

The corpus hash covers the ground truth, so editing one truth value changes the hash and
`diff` marks the two runs as not comparable. Filtering to a subset also changes the hash: a
run on 20 documents can never be silently compared to a run on 60.

The prompt *text* is never serialised into a run — only its hash. A report can be published
without leaking a tuned prompt.

## 6. Caching

Inference results are cached by `(config_hash, case_id, image_hash, schema_hash)`. Scoring is
never cached.

That split is deliberate: the expensive half is the model call, and the half you iterate on is
the scoring. Change a normaliser and `docvlm-eval report` recomputes every number from stored
outputs in milliseconds. Change the prompt or the schema and the key changes, so the cache
invalidates exactly what it should — you cannot accidentally A/B a prompt against itself.

If the cache directory is on a filesystem without byte-range locking (some network and FUSE
mounts), the cache disables itself with a warning rather than aborting the run.

## 7. Limitations

**The corpus is the whole thing.** This tool measures a model against your ground truth. If
the ground truth is wrong, careless, or unrepresentative, it will report that with great
precision. Building an honest evaluation set is tedious and is the part that determines
whether any of this is worth anything.

**The bundled synthetic corpus saturates.** Current vision models score at or near 1.000 on
`corpora/synthetic-forms`. Rendered fonts, even jittered, are much easier than photographed
handwriting. `--hard` degrades effective resolution, adds motion blur, glare and an occluding
stamp, which produces a usable spread — but it is still synthetic. Do not generalise from it
to your documents.

**Latency is measured end to end from the client**, including queueing on the server. If the
backend is shared with other work — as it was for the numbers in the README, which ran against
a box also serving a production workload — latency reflects contention, not model speed. Use
a dedicated box before quoting latency. Accuracy is unaffected.

**Cached cases keep the latency measured when the call was made.** A fully cached re-run
reports historical latency, and says so (`n_cached`). It is not measuring anything now.

**Costs are recorded as 0 for local inference.** Energy is not yet measured.

**Set-valued fields conflate two errors.** A missing item and a spurious item both make the
field `wrong`. Precision and recall are in the detail string, but not yet aggregated as
separate metrics.

**Ground truth may not be reachable.** In some real workflows the "correct" answer is not on
the paper at all — a clinic that adds exams by protocol, a form listing a package of options
without saying which applies. Those cap recall permanently and are not model failures. Tag
them and exclude them, or your ceiling will look like a model problem.

## 8. What this tool is not

- not a production extraction pipeline
- not an annotation or labelling tool
- not a training or fine-tuning harness
- not general-purpose: the scope is documents → structured JSON
