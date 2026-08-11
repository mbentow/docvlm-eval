<h1 align="center">docvlm-eval</h1>

<p align="center">
  <strong>Did that change make your documents read better, or worse?</strong><br>
  Field-level evaluation for document extraction with vision language models.
</p>

<p align="center">
  <a href="https://github.com/mbentow/docvlm-eval/actions/workflows/ci.yml"><img src="https://github.com/mbentow/docvlm-eval/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/GPU-not%20required-brightgreen" alt="No GPU required">
</p>

<p align="center">
  <a href="#who-this-is-for">Who it's for</a> ·
  <a href="#try-it-in-30-seconds">Quickstart</a> ·
  <a href="#what-it-measures">What it measures</a> ·
  <a href="docs/CORPUS.md">Corpus</a> ·
  <a href="docs/DESIGN.md">Design notes</a> ·
  <a href="docs/METHODOLOGY.md">Methodology</a>
</p>

---

Two vision models read the same 60 medical request forms.

Their accuracy is **identical, within noise**. One of them is safe to put in a clinic. The
other is not.

No averaged metric will tell you which one. This tool tells you in a single table — and that
gap between "same score" and "opposite risk" is the whole reason it exists.

<br>

## The 30-second version

Verbatim output, comparing `qwen3-vl:8b` against `qwen3-vl:30b-a3b`, both on a local Ollama at
`Q4_K_M`:

```
FIELD        ┃  BASE ┃  CAND ┃      Δ ┃    Δ 95% CI    ┃ Δ HALL ┃
━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━
patient_name │ 0.800 │ 0.833 │ +0.033 │ -0.033, +0.100 │ +0.0pp │ ~
doctor_crm   │ 0.750 │ 0.767 │ +0.017 │ -0.050, +0.100 │ +0.0pp │ ~
member_id    │ 0.717 │ 0.767 │ +0.050 │ -0.033, +0.133 │ -3.3pp │ ~
exams        │ 0.633 │ 0.700 │ +0.067 │ -0.033, +0.167 │ +0.0pp │ ~
MACRO              0.781  +0.029  [-0.005, +0.064]
ALL-FIELDS-CORRECT 0.583  +0.050  [-0.050, +0.150]

VERDICT: quality change +0.029 is inside the noise (n=60) — not a result
```

Three things that table says, and a leaderboard would not:

**1 · The 4× larger model did not win.** The 31B model is the *baseline* here. Ship it on the
assumption that bigger is better and no average would have argued with you.

**2 · But that is not a result either.** The interval crosses zero. With 60 documents the
honest statement is "no measurable difference", and the tool prints that in the verdict line
instead of letting you read a ranking into noise.

**3 · The two models fail in opposite ways.** Here is the number that decides the product:

|                    | missing fields | hallucinated | **critical hallucination** |
| ------------------ | -------------: | -----------: | -------------------------: |
| `qwen3-vl:8b`      |          5.71% |    **0.00%** |                  **0.00%** |
| `qwen3-vl:30b-a3b` |          0.95% |        0.48% |                  **1.67%** |

The 8B leaves a field blank when it cannot read it. The 30B fills it in. On `member_id` — an
insurance card number — the 8B returned nothing 10% of the time and invented nothing; the 30B
never left it blank and **invented a value on 3.3% of documents where the card had no number
at all**.

> [!IMPORTANT]
> A blank field is an operator retyping it. An invented insurance number is wrong data
> entering a system that will trust it. Both cost exactly one point in any averaged metric.
>
> That distinction — `missing` versus `hallucinated` — is the single most important thing this
> tool measures.

> [!NOTE]
> CI asserts the numbers on this page against the committed run files, on every push. Move
> macro accuracy off 0.781 and the build fails. This document cannot quietly go stale.

<br>

## Who this is for

You have documents, a schema, and a decision to make about a model. You are one of these
three people:

| If you are… | you reach for this when… | and you get back… |
| --- | --- | --- |
| **Choosing a model or a quantisation** | "is the 30B worth 4× the memory over the 8B?" | a per-field delta with a confidence interval, and a verdict line that says *not a result* when the interval crosses zero |
| **Tuning a prompt or preprocessing** | "did that rewrite help, or did I just get lucky on 20 documents?" | a paired bootstrap over the same documents, so the change is separated from the sampling |
| **Deciding what to automate** | "what can go through untouched, and what must a human see?" | a coverage-vs-accuracy curve, validated out of sample, with the escaping-error rate at each level |

### What you can do with it

- **Compare two models, prompts, or preprocessing profiles** on your own documents, field by field
- **Separate a blank field from an invented one** — the distinction averaged metrics destroy
- **Find where accuracy actually breaks** by slicing on tags: motion blur, glare, handwriting, stamps
- **Decide an automation threshold** and see what it really delivers on documents it was not fitted on
- **Gate a pipeline in CI** with `--fail-under`, `--fail-hallucination-over`, `--fail-on-regression`
- **Reproduce any number later** — every run records model digest, prompt hash, corpus hash, versions

### When *not* to use it

- **You want production extraction.** This measures extraction; it does not do it.
- **You have no ground truth.** Sixty labelled documents of your own beat any bundled corpus.
  Without them this tool has nothing to compare against, and no evaluator can invent it for you.
- **You want a leaderboard number.** It will keep telling you your sample is too small to rank
  anything, which is correct and unsatisfying.
- **Your task is not documents → structured JSON.** Free-form summarisation, RAG answers and
  chat quality need a different instrument.

<br>

## Try it in 30 seconds

No GPU. No model. The repository ships a synthetic corpus and a mock backend:

```bash
pip install -e ".[dev]"

docvlm-eval validate --corpus corpora/synthetic-forms
docvlm-eval run --corpus corpora/synthetic-forms --config configs/mock-baseline.yaml
```

The mock perturbs the ground truth by a fixed amount, producing every failure mode including
hallucinations — so you see the shape of the report before pulling a 19 GB model.

Every number in this README is reproducible with no backend at all, because the runs behind it
are committed:

```bash
docvlm-eval report --run hard-qwen3vl-8b --runs-dir examples/runs
docvlm-eval diff -b hard-qwen3vl-30b -k hard-qwen3vl-8b --runs-dir examples/runs
```

## Run it against a real model

```bash
export OLLAMA_HOST=http://localhost:11434

docvlm-eval run  --corpus corpora/synthetic-forms-hard --config configs/qwen3vl-30b.yaml -j 3
docvlm-eval run  --corpus corpora/synthetic-forms-hard --config configs/qwen3vl-8b.yaml  -j 3
docvlm-eval diff --baseline qwen3vl-30b --candidate qwen3vl-8b
```

Or sweep every config at once against a shared cache:

```bash
docvlm-eval sweep --corpus corpora/synthetic-forms-hard --configs "configs/qwen*.yaml" \
                  --baseline qwen3vl-30b --out out/
```

> [!WARNING]
> `diff` refuses to compare runs made on different corpora. Case ids repeat across corpora, so
> pairing them would silently compare unrelated documents and report a confident, significant,
> meaningless delta.

<br>

## What it measures

**Six outcomes per field, never one score.**

| outcome | meaning |
|---|---|
| `correct` | matches — *including* correctly returning nothing when the truth is empty |
| `missing` | the truth has a value, the model returned nothing |
| `hallucinated` | the truth is empty, the model invented a value |
| `wrong` | both have values and they disagree |
| `malformed` | the value does not fit the declared type |
| `refused` | the backend errored or the model declined |

Backend failures are `refused`, never folded into `wrong` — otherwise an unreliable server
looks like an inaccurate model. Unparseable *ground truth* is `malformed` too, so a corpus bug
shows up as a corpus bug instead of capping a field's accuracy for reasons no prompt can fix.

**Two headline numbers, because one is a lie.**

- `MACRO` — weighted mean of per-field accuracy → **0.781**
- `ALL-FIELDS-CORRECT` — documents with *no* error anywhere → **0.583**

The gap between them is the gap between "most fields are fine" and "this document can go
through untouched". Only the second maps to a business decision: one wrong field still needs a
human.

**Confidence intervals that resample documents, not fields.** A blurry photo hurts every field
at once, so fields within a document are correlated. Diffs use a *paired* bootstrap on shared
document ids, which is what makes the interval tight enough to be useful.

**Accuracy sliced by tag — worst first.** This is where the actionable finding lives:

```
BY TAG      ┃   ACC ┃ ALL-OK ┃  n ┃
━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━╇━━━
motion_blur │ 0.361 │  0.000 │ 17 │ !
low_light   │ 0.610 │  0.182 │ 11 │ !
phone_photo │ 0.758 │  0.538 │ 39 │ !
handwritten │ 0.777 │  0.564 │ 39 │ !
printed     │ 0.789 │  0.619 │ 21 │ !
scanned     │ 0.823 │  0.667 │ 21 │ !
glare       │ 0.941 │  0.824 │ 17 │
rotated     │ 1.000 │  1.000 │  1 │
```

"78% accuracy" is not actionable. "36% under motion blur, and not one motion-blurred document
came out clean" tells you what to build next.

Note the `n` column: `rotated` at 1.000 is **one document**. That is a hint, not a finding —
and the report never lets you forget which is which.

**Operational cost.** Latency p50/p95, tokens in/out, cost per document.

**Whether the answer came from the page or from your prompt.** Every worked example you put in
the instructions is a string the model can emit without looking at the image, and it will pass
for a normal answer — because it *is* one.

```console
$ docvlm-eval echo --run 30b-p2-padrao --config configs/30b-p2-padrao.yaml
```

The obvious version of this check does more harm than good. Counting how often the model says
your example proves nothing: examples get chosen *because* they are the common case, so they
*should* be frequent. Measured on a 200k-document deployment, the string used as an inline
example appeared in 4.4% of predictions — alarming until you check the ground truth, where the
same value is 7.7% of all documents. The model was saying it *less* than reality.

So a literal is flagged only when it is **over-produced relative to the ground truth *and*
less accurate than the model's own baseline**. Frequency raises the question; accuracy answers
it. On that same deployment, predictions containing the example were 90.2% correct against
80.6% for everything else — higher, not lower. The check clears it and leaves the prompt alone,
which is the whole point.

<br>

## Can this actually be deployed?

Accuracy alone does not answer that. This does — `qwen3-vl:30b-a3b`, hard corpus:

```
AUTOMATE ┃   ACC ┃ ERR ESCAPING ┃  HALL
━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━
     50% │ 1.000 │         0.0% │ 0.00%
     70% │ 0.762 │        23.8% │ 0.00%
    100% │ 0.533 │        46.7% │ 0.48%
AURC 0.1353 vs random 0.4421 — ranking gain 69%
```

Read as a whole the run scores 0.533. But **its most-confident half is 1.000, with nothing
escaping review** — the model knows which documents it got wrong. A model at 0.75 that knows
which quarter it failed beats a model at 0.80 that is uniformly, confidently mediocre.

**Now the part almost nobody publishes.** That operating point was chosen by looking at the
same documents it is then evaluated on. `validate_holdout` fits the coverage on a random half
and measures it on the other, 200 times:

```
target 0.95 → automate 54% → real accuracy 0.917 [0.571, 1.000] vs 0.533 with no policy
              lift +0.384, target actually met on 60% of splits
```

The policy is worth having: **+0.38 lift** on the automated slice is not noise. But the target
survives the transfer only 60% of the time. The gap between the curve (1.000 at 50%) and the
honest out-of-sample number (0.917 at 54%) is exactly what selecting an operating point on
your evaluation set costs you — and it is the number to take into a production decision.

Two guards keep the section trustworthy:

- AURC is compared against the mean of 25 random orderings. A confidence signal that ranks no
  better than shuffling is reported as such, instead of getting a persuasive curve.
- A policy that would automate less than 5% is rejected, because "meets 95% by accepting one
  lucky document" is not a policy.

<br>

## Declaring how a field is compared

A patient name should ignore case and accents. A licence number should be reduced to digits. A
date must be parsed as a date. **One global rule for all three quietly reports the wrong
number.**

```python
class MedicalRequest(ExtractionSchema):
    patient_name: str | None = field(
        None,
        compare=Compare.TEXT,
        fuzzy_threshold=0.90,  # an operator would accept a near miss
    )
    doctor_crm: str | None = field(
        None,
        compare=Compare.DIGITS,
        critical=True,  # one wrong digit = a different doctor
    )
    exams: list[str] = field(
        default_factory=list,
        compare=Compare.SET_TEXT,  # reading 3 exams on a 1-exam form is an error
    )
    request_date: str | None = field(None, compare=Compare.DATE, dayfirst=True)
    urgent: bool | None = field(None, compare=Compare.BOOL)
```

Modes: `EXACT` · `TEXT` · `DIGITS` · `NUMBER` · `DATE` · `BOOL` · `SET_TEXT`.

Fields marked `critical` get their own hallucination rate — a hallucinated licence number is a
different class of problem from a hallucinated free-text note.

The same class becomes the JSON Schema sent to the backend for constrained decoding, so **the
prompt, the constraint and the scoring can never drift apart.**

<br>

## Why this is not a 200-line script

- **Reproducibility.** Every run records model digest (not the mutable tag), server version,
  quantisation, corpus hash, prompt hash, schema hash, library version, temperature,
  concurrency. Without it, a comparison three weeks later is two numbers of unknown origin.
- **The corpus hash covers the ground truth.** Edit one truth value and `diff` exits non-zero
  rather than comparing across the edit.
- **Cache the inference, never the scoring.** Change a normaliser and every number updates
  instantly without touching a GPU. Change the prompt and the cache invalidates itself, so you
  cannot accidentally A/B a prompt against itself.
- **Prompts stay out of the output.** Provenance records the prompt *hash*, never the text —
  publish a report without leaking a tuned prompt.
- **CI gate.** `--fail-under`, `--fail-hallucination-over`, and a `--fail-on-regression` that
  only fires on a *statistically significant* per-field drop, so sampling noise does not block
  your pipeline.

<br>

## Documentation

| | |
|---|---|
| [**Corpus format**](docs/CORPUS.md) | Build your own in three files — and the honest limits of the bundled synthetic one, including the fact that its clean half saturates |
| [**Design notes**](docs/DESIGN.md) | Every decision, and the bug that caused it |
| [**Methodology**](docs/METHODOLOGY.md) | Scoring rules, the statistics, and what this does not measure |

<br>

## Backends

| runner | status |
|---|---|
| `ollama` | supported — constrained decoding via `format`, multi-host round robin, digest recorded |
| `mock` | supported — replays a `predictions.jsonl`, or perturbs the truth deterministically |
| `mlx` | planned |
| `openai`-compatible | planned — covers vLLM, SGLang, TGI |

The `Runner` interface is 20 lines. Adding a backend does not touch scoring code.

## Roadmap

- [x] per-field scoring, separated failure modes, tag slicing, markdown/HTML reports
- [x] `diff` with paired bootstrap, result cache, CI gates
- [x] selective prediction: risk–coverage, AURC vs random ordering, out-of-sample validation
- [ ] MLX and OpenAI-compatible runners
- [ ] calibration proper: expected calibration error and reliability diagrams
- [ ] energy per document

## License

Apache-2.0. See [LICENSE](LICENSE).

---

> This repository contains the generic tooling. Domain schemas, prompts and tuned thresholds
> from production deployments are not included.
