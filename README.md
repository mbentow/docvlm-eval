# docvlm-eval

**Field-level evaluation for document extraction with vision language models.**

Answers one question with a number: *did this change make my documents read better, or worse?*

You give it a schema, a set of documents and their ground truth. It runs N configurations —
models, quantisations, prompts, preprocessing — and reports accuracy **per field**,
hallucination rate, missing-field rate, latency and cost, with bootstrap confidence intervals
so you can tell a result from noise.

[![CI](https://github.com/mbentow/docvlm-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/mbentow/docvlm-eval/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

---

## Why average accuracy lies

Verbatim output from this tool, comparing `qwen3-vl:8b` against `qwen3-vl:30b-a3b` on 60
synthetic medical request forms, both running locally on Apple Silicon:

```
CONFIG qwen3vl-8b    vs BASELINE qwen3vl-30b
Corpus: synthetic-forms-hard @ d45cda24b564 (60 paired cases)
FIELD        ┃  BASE ┃  CAND ┃      Δ ┃    Δ 95% CI    ┃ Δ HALL ┃
━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━
patient_name │ 0.800 │ 0.833 │ +0.033 │ -0.033, +0.100 │ +0.0pp │ ~
doctor_crm   │ 0.750 │ 0.767 │ +0.017 │ -0.050, +0.100 │ +0.0pp │ ~
insurer      │ 0.733 │ 0.733 │ +0.000 │ -0.083, +0.083 │ +0.0pp │ ~
member_id    │ 0.717 │ 0.767 │ +0.050 │ -0.033, +0.133 │ -3.3pp │ ~
exams        │ 0.633 │ 0.700 │ +0.067 │ -0.033, +0.167 │ +0.0pp │ ~
request_date │ 0.717 │ 0.700 │ -0.017 │ -0.083, +0.033 │ +0.0pp │ ~
urgent       │ 0.917 │ 0.967 │ +0.050 │ -0.033, +0.133 │ +0.0pp │ ~
MACRO              0.781  +0.029  [-0.005, +0.064]
ALL-FIELDS-CORRECT 0.583  +0.050  [-0.050, +0.150]

VERDICT: quality change +0.029 is inside the noise (n=60) — not a result; latency p50 1.42x
(12,744ms) — check the queue SLA
```

Three things that table says, and a leaderboard would not:

1. **The 4× larger model did not win.** Every per-field delta favours the smaller model except
   one, and the model with ~31B parameters is the *baseline* here. If you had assumed bigger
   was better and shipped it, nothing in an average would have told you otherwise.

2. **But that is not a result either.** The +0.029 confidence interval crosses zero. With 60
   documents the honest statement is "no measurable difference", and the tool says so in the
   verdict line rather than letting you read a ranking into noise. Most model-swap comparisons
   published online never check this.

3. **The two models fail in opposite ways.** Almost identical accuracy, completely different
   risk profile:

   | | missing fields | hallucinated fields | critical hallucination |
   |---|---:|---:|---:|
   | qwen3-vl:8b | **5.71%** | **0.00%** | **0.00%** |
   | qwen3-vl:30b-a3b | 0.95% | 0.48% | **1.67%** |

   The 8B leaves a field blank when it cannot read it. The 30B fills it in. On `member_id` —
   an insurance card number — the 8B returned nothing 10% of the time and invented nothing;
   the 30B never left it blank and invented a value on 3.3% of documents where the card had
   no number at all.

   In a clinical or financial workflow that decides the choice. A blank field is an
   inconvenience: an operator retypes it. An invented insurance number is wrong data entering
   a system that will trust it. **No averaged metric can show you this**, because both
   failures cost exactly one point.

That distinction — `missing` versus `hallucinated` — is the single most important thing this
tool measures.

---

## Install

```bash
pip install -e ".[dev]"      # or: uv pip install -e ".[dev]"
```

## 30 seconds, no GPU

The repository ships a synthetic corpus and a mock backend, so the whole pipeline runs with
nothing installed but Python:

```bash
docvlm-eval validate --corpus corpora/synthetic-forms
docvlm-eval run --corpus corpora/synthetic-forms --config configs/mock-baseline.yaml
```

The mock perturbs the ground truth by a fixed amount, producing every failure mode including
hallucinations, so the shape of the report is visible before you pull a 19 GB model.

## Against a real model

```bash
export OLLAMA_HOST=http://localhost:11434

docvlm-eval run  --corpus corpora/synthetic-forms-hard --config configs/qwen3vl-30b.yaml -j 3
docvlm-eval run  --corpus corpora/synthetic-forms-hard --config configs/qwen3vl-8b.yaml  -j 3
docvlm-eval diff --baseline qwen3vl-30b --candidate qwen3vl-8b
```

`diff` refuses to compare runs made on different corpora — case ids repeat, so pairing them
would silently compare unrelated documents and report a confident, significant, meaningless
delta.

Or sweep everything at once, with a shared cache:

```bash
docvlm-eval sweep --corpus corpora/synthetic-forms-hard --configs "configs/qwen*.yaml" \
                  --baseline qwen3vl-30b --out out/
```

The four runs behind this README are committed, so every number here is reproducible with no
backend at all:

```bash
docvlm-eval report --run hard-qwen3vl-8b --runs-dir examples/runs
docvlm-eval diff -b hard-qwen3vl-30b -k hard-qwen3vl-8b --runs-dir examples/runs
```

---

## What it measures

**Per field, then aggregated.** For every field of every document, exactly one outcome:

| outcome | meaning |
|---|---|
| `correct` | matches, *including* correctly returning nothing when the truth is empty |
| `missing` | the truth has a value, the model returned nothing |
| `hallucinated` | the truth is empty, the model invented a value [^1] |
| `wrong` | both have values and they disagree |
| `malformed` | the value does not fit the declared type |
| `refused` | the backend errored or the model declined |

[^1]: One deliberate extension: a `SET_TEXT` field where the prediction shares *nothing* with
the truth and is longer also counts as `hallucinated`. That is the "transcribed the whole
pre-printed list of options" failure, not a near miss. See `docs/METHODOLOGY.md` §2.

Backend failures are `refused`, never folded into `wrong` — otherwise an unreliable server
looks like an inaccurate model. Unparseable *ground truth* is `malformed` too, so a corpus bug
shows up as a corpus bug instead of capping a field's accuracy for reasons no prompt can fix.

**Two headline numbers, not one.**

- `MACRO` — weighted mean of per-field accuracy.
- `ALL-FIELDS-CORRECT` — the fraction of documents with **no** error anywhere. This is the
  one that maps to the business: a document with one wrong field still needs a human. On the
  run above it is 0.583 while macro is 0.781 — the gap between "most fields are fine" and
  "this document can go through untouched".

**Confidence intervals, from resampling documents.** Fields within a document are
correlated — a blurry photo hurts all of them at once — so the bootstrap resamples whole
documents, never fields. Diffs use a *paired* bootstrap on the shared document ids, which is
what makes the delta interval tight enough to be useful.

**Breakdown by tag.** Every case carries tags, and the report slices accuracy by them,
worst first:

```
BY TAG      ┃   ACC ┃ ALL-OK ┃  n ┃
━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━╇━━━
motion_blur │ 0.361 │  0.000 │ 17 │ !
low_light   │ 0.610 │  0.182 │ 11 │ !
stamped     │ 0.643 │  0.500 │  2 │ !
phone_photo │ 0.758 │  0.538 │ 39 │ !
handwritten │ 0.777 │  0.564 │ 39 │ !
printed     │ 0.789 │  0.619 │ 21 │ !
skewed      │ 0.816 │  0.571 │ 14 │ !
scanned     │ 0.823 │  0.667 │ 21 │ !
partial     │ 0.857 │  0.800 │  5 │ !
glare       │ 0.941 │  0.824 │ 17 │
faded       │ 1.000 │  1.000 │  2 │
rotated     │ 1.000 │  1.000 │  1 │
```

"78% accuracy" is not actionable. "36% under motion blur, 79% on printed — and not one
motion-blurred document came out clean" tells you what to build next. Note the `n` column:
`rotated` at 1.000 is one document, which is a hint, not a finding. Tag your corpus
generously, and read the counts.

**Operational cost.** Latency p50/p95, tokens in/out, cost per document (0 for local).

**Coverage vs accuracy — the number that decides the product.** Accuracy alone does not tell
you whether a system is deployable. This does, for `qwen3-vl:30b-a3b` on the hard corpus:

```
AUTOMATE ┃   ACC ┃ ERR ESCAPING ┃  HALL
━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━
     50% │ 1.000 │         0.0% │ 0.00%
     60% │ 0.889 │        11.1% │ 0.00%
     70% │ 0.762 │        23.8% │ 0.00%
     80% │ 0.667 │        33.3% │ 0.00%
    100% │ 0.533 │        46.7% │ 0.48%
AURC 0.1353 vs random 0.4421 — ranking gain 69%
```

Read as a whole the run scores 0.533. But **its most-confident half is 1.000, with nothing
escaping review** — the model does know which documents it got wrong. A model at 0.75 that
knows which quarter it failed beats a model at 0.80 that is uniformly, confidently mediocre.

**But do not read a coverage level off that table and call it a result.** The level was chosen
by looking at the same documents it is then evaluated on. `validate_holdout` fits the coverage
on a random half and measures it on the other, 200 times:

```
target 0.95 → automate 54% → real accuracy 0.917 [0.571, 1.000] vs 0.533 with no policy
              lift +0.384, target actually met on 60% of splits
```

The policy is worth having — a **+0.38 lift** on the automated slice is not noise. But the
target survives the transfer only 60% of the time, and the interval reaches down to 0.571.
That gap between the curve (1.000 at 50%) and the honest out-of-sample number (0.917 at 54%)
is what selecting an operating point on your evaluation set costs you, and it is the number
to take into a production decision.

Two guards make the section trustworthy: AURC is compared against the mean of 25 random
orderings, so a confidence signal that does not rank is reported as such instead of getting a
persuasive curve; and a policy that would automate less than 5% is rejected rather than
counted as a success, because "meets 95% by accepting one lucky document" is not a policy.
Confidence comes from a `confidence` field in your schema if you declare one, and from the
mean field score otherwise.

---

## Comparison is declared per field, not globally

This is where home-made evaluators usually go wrong. A patient name should ignore case and
accents. A licence number should be reduced to digits. A date must be parsed and compared as
a date. One global rule for all three quietly reports the wrong number.

```python
from docvlm_eval import Compare, ExtractionSchema, field


class MedicalRequest(ExtractionSchema):
    patient_name: str | None = field(
        None,
        description="Full name of the patient",
        compare=Compare.TEXT,
        fuzzy_threshold=0.90,  # an operator would accept a near miss
    )
    doctor_crm: str | None = field(
        None,
        description="Licence number of the requesting doctor",
        compare=Compare.DIGITS,
        critical=True,  # one wrong digit = a different doctor
    )
    exams: list[str] = field(
        default_factory=list,
        description="Every exam explicitly requested",
        compare=Compare.SET_TEXT,  # strict: reading 3 on a 1-exam form is an error
    )
    request_date: str | None = field(None, compare=Compare.DATE, dayfirst=True)
    urgent: bool | None = field(None, compare=Compare.BOOL)
```

Available modes: `EXACT`, `TEXT`, `DIGITS`, `NUMBER`, `DATE`, `BOOL`, `SET_TEXT`. Fields marked
`critical` get their own hallucination rate, because a hallucinated licence number is a
different class of problem from a hallucinated free-text note.

The same class becomes the JSON Schema sent to the backend for constrained decoding, so the
prompt, the constraint and the scoring can never drift apart.

## Corpus format

Deliberately boring. A clever format is why nobody uses your evaluator, including you, six
months from now.

```
corpora/my-corpus/
├─ manifest.jsonl
├─ schema.py
└─ images/
```

```jsonl
{"id":"0001","image":"images/0001.jpg","truth":{"patient_name":"Marina Ferrazzo","doctor_crm":"20154-RS","insurer":null,"exams":["Holter 24h"],"urgent":false},"tags":["handwritten","phone_photo","low_light"]}
```

`docvlm-eval validate` checks it before you spend GPU hours: missing images, duplicate ids,
and — the one that actually bites — **ground-truth keys that are not in the schema**, which
would otherwise show up as a field permanently stuck at 0% while you go looking for the cause
in the model.

---

## Details that separate this from a script

**Reproducibility.** Every run records the model digest (not just the mutable tag), server
version, quantisation, corpus content hash, prompt hash, schema hash, library version,
temperature and concurrency. Without that, a comparison three weeks later is two numbers of
unknown origin.

**The corpus hash covers the ground truth.** Edit one truth value and the hash changes, so
`diff` exits non-zero rather than comparing runs from before and after the edit (`--force` to
override). Filtering to a subset also changes the hash. Deltas and their intervals are both
computed on the paired subset, so the band always sits around the number printed beside it.

**Scoring policy travels with the run.** `critical` and `weight` are declared in the schema,
optionally overridden per config, resolved once, and stored in the run. `run`, `report` and
`diff` therefore cannot print three different macro accuracies for the same run.

**Cache the inference, never the scoring.** Model outputs are cached by
`(config, case, image, schema)`; scoring is always recomputed. Change a normaliser and every
number updates instantly, without touching a GPU. Change the prompt and the cache invalidates
itself, so you cannot accidentally A/B a prompt against itself.

**Prompts stay out of the output.** Provenance records the prompt *hash*, never the text, so a
run report can be published without leaking a tuned prompt.

**CI gate.**

```bash
docvlm-eval run  --corpus corpora/my-corpus --config configs/prod.yaml --fail-under 0.85 \
                 --fail-hallucination-over 0.01
docvlm-eval diff -b prod -k candidate --fail-on-regression
```

Exits non-zero on regression. `--fail-on-regression` only fires on a *statistically
significant* per-field drop, so ordinary sampling noise does not block your pipeline.

---

## The synthetic corpus, and an honest note about it

`corpora/synthetic-forms` is generated by `tools/generate_synthetic_corpus.py` from a fixed
seed and a list of invented names. Every value is fabricated. It exists so that anyone can run
this tool without having documents of their own.

**It saturates, and that is worth reporting.** Both `qwen3-vl:8b` and `qwen3-vl:30b` score at
or near 1.000 on the clean version:

| corpus | model | macro | all-fields-correct | hallucination |
|---|---|---:|---:|---:|
| `synthetic-forms` | qwen3-vl:8b | 1.000 | 1.000 | 0.00% |
| `synthetic-forms` | qwen3-vl:30b-a3b | 0.981 | 0.867 | 0.24% |
| `synthetic-forms-hard` | qwen3-vl:8b | 0.781 | 0.583 | 0.00% |
| `synthetic-forms-hard` | qwen3-vl:30b-a3b | 0.752 | 0.533 | 0.48% |

A benchmark everybody passes measures nothing. Rendered fonts — even jittered and slanted —
are far easier than photographed handwriting. `--hard` reproduces what actually breaks
extraction in the field: low effective resolution, motion blur, glare, a stamp across the
licence number. That is what produced the numbers at the top of this README.

Treat the synthetic corpus as a **smoke test and a demo**, not as evidence about your
documents. The corpus is the part that matters, and it is the part only you can build.

> Reproduce: `python tools/generate_synthetic_corpus.py --out corpora/synthetic-forms-hard -n 60 --hard --seed 424242`
>
> Patients, doctors, clinics and insurers are all invented — the insurer names are not real
> trade names. Every image is watermarked *SYNTHETIC DOCUMENT — Not a real request*.

Full methodology and limitations: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## Backends

| runner | status |
|---|---|
| `ollama` | supported — constrained decoding via `format`, multi-host round robin, model digest recorded |
| `mock` | supported — replays a `predictions.jsonl` (`params.predictions`), or perturbs the truth deterministically, seeded per case |
| `mlx` | planned |
| `openai` (any compatible endpoint) | planned |

The `Runner` interface is 20 lines; adding a backend does not touch the scoring code.

## Roadmap

- [x] corpus, schema, Ollama runner, per-field scoring, markdown/HTML reports
- [x] separated failure modes, tag slicing, `diff` with paired bootstrap
- [x] result cache, `--fail-under` / `--fail-on-regression` for CI
- [ ] MLX and OpenAI-compatible runners
- [ ] calibration: ECE and the coverage-vs-accuracy curve — *"if I send the least confident
      20% to human review, what accuracy remains in the automated 80%?"*
- [ ] energy per document on Apple Silicon

## Development

```bash
uv pip install -e ".[dev]"
pytest          # 94 tests, concentrated on normalisation and scoring
ruff check .
```

Scoring is where a silent bug is most expensive — it would not crash, it would just change
every number. That module has the densest tests, and they earned it: an ISO ground-truth date
being reinterpreted by the `dayfirst` setting, pt-BR month names scored as malformed, a
bootstrap interval indexed asymmetrically, and a shared RNG in the mock backend that made
results depend on `-j` were all caught by tests rather than by a reader.

See `CHANGELOG.md` for the full list.

## Scope

Deliberately **not** in scope: production extraction, annotation tooling, fine-tuning, and
anything beyond documents → structured JSON.

## License

Apache-2.0. See [LICENSE](LICENSE).

---

> This repository contains the generic tooling. Domain schemas, prompts and tuned thresholds
> from production deployments are not included.
