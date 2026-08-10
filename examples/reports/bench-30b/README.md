# Benchmark: one model, four prompts, three image profiles

**Model held fixed at `qwen3-vl:30b-a3b-instruct`** (digest `c871fc73fabc5516`, Q4_K_M,
Ollama 0.31.2), `temperature: 0`, `format: schema`. Corpus `synthetic-forms-hard`
@ `d45cda24b564`, 60 documents. Every run is committed under `examples/runs/`.

Changing one variable at a time is the whole point: the six configs below differ from the
baseline in exactly one thing each.

## The matrix

| config | macro | 95% CI | all-fields-correct | hallucination | critical hall. | missing | p50 | tokens in |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|
| `30b-p1-padrao` *(baseline)* | 0.748 | 0.660–0.829 | **0.550** | 0.95% | 3.33% | 1.67% | 11,221 ms | 1,696 |
| `30b-p2-padrao` | 0.755 | 0.669–0.831 | 0.533 | 0.24% | 0.83% | 4.05% | 11,437 ms | 1,604 |
| `30b-p3-padrao` | 0.752 | 0.669–0.831 | 0.533 | **0.00%** | **0.00%** | 5.24% | 11,922 ms | 1,796 |
| `30b-p4-padrao` | **0.781** | 0.705–0.850 | 0.517 | 0.24% | 0.83% | **0.00%** | 11,566 ms | 1,802 |
| `30b-p1-alta` | 0.726 | 0.631–0.814 | 0.533 | 1.19% | 4.17% | 0.48% | 22,924 ms | 3,152 |
| `30b-p1-alta-pb` | 0.733 | 0.643–0.819 | 0.517 | 0.95% | 3.33% | 0.24% | 23,043 ms | 3,152 |

Prompts: `p1` = explicit rules · `p2` = terse · `p3` = aggressive abstention ·
`p4` = field-by-field location, licence number read digit by digit.
Profiles: `padrao` 1440px · `alta` 2048px + contrast + sharpen · `alta_pb` same in greyscale.

## Three findings

### 1. Telling the model *where* each field lives is a real gain — but it moves the business metric the wrong way

`p4` vs `p1`: **macro +0.033 [+0.012, +0.055] — significant.** Two fields improve
significantly, `insurer` and `member_id`, both +0.067. Missing fields drop to zero across the
board and `member_id` hallucination falls 5 points.

And yet **all-fields-correct goes from 0.550 to 0.517**. `exams` got slightly worse
(0.667 → 0.650), and `exams` is the field that most often decides whether a whole document is
clean. Macro says ship it; the metric that maps to human review says wait.

This is the case the tool exists for. An average would have reported "+3.3 points" and
nothing else.

### 2. Abstention is free — and no accuracy metric shows it

`p3` vs `p1`: macro +0.005, **inside the noise. Not a result.**

But hallucination goes 0.95% → **0.00%**, and on critical fields 3.33% → **0.00%**. What
changed is not how much the model gets right, it is *how it fails*: invented values became
blanks. `insurer` missing rose 10.0% → 21.7%, `member_id` 1.7% → 13.3%.

Same accuracy, zero invented values, more fields for a human to fill in. In a clinical or
financial pipeline that is a strictly better trade — and the accuracy column is flat, so
nothing but the failure-mode split would have told you.

### 3. Escalating image resolution did not help, and cost 2×

`alta` vs `padrao`: macro −0.021 (interval −0.048 to +0.002, so leaning negative but not
conclusive), **latency 2.04×**, tokens 1.86×, and critical hallucination *up* 3.33% → 4.17%.
`alta_pb` lands in the same place.

The source documents are already low-resolution. Upscaling adds no information — but it does
add image tokens, and the model becomes more willing to commit to a value it cannot actually
read. Escalation is worth having as a fallback for a specific failure; it is not worth
applying by default.

The ladder was not run to `maxima` (2560px lossless): two consecutive steps regressing was
enough to stop.

## Where the gain lands

`p4` vs `p1`, macro accuracy by tag — the improvement is concentrated exactly where the
baseline was worst:

| tag | n | p1 | p4 | Δ |
|---|---:|---:|---:|---:|
| `motion_blur` | 17 | 0.252 | 0.353 | **+0.101** |
| `low_light` | 11 | 0.584 | 0.649 | +0.065 |
| `stamped` | 2 | 0.643 | 0.786 | +0.143 |
| `phone_photo` | 39 | 0.711 | 0.747 | +0.037 |
| `printed` | 21 | 0.735 | 0.776 | +0.041 |
| `handwritten` | 39 | 0.755 | 0.784 | +0.029 |
| `scanned` | 21 | 0.816 | 0.844 | +0.027 |
| `glare` | 17 | 0.941 | 0.941 | +0.000 |

Documents that were already fine stayed fine; the prompt bought its gain on the hard ones.
`stamped` at n=2 is a hint, not a finding.

## Caveats

- **60 documents.** Half-width on a proportion near 0.75 is about ±0.11, so only the `p4`
  macro gain and its two per-field improvements clear the bar. Everything else is directional.
- **Synthetic corpus.** Rendered fonts are not photographed handwriting. Treat the *shape* of
  these results as transferable and the *numbers* as not.
- **Latency is contended.** These runs shared a box with a production workload, so p50 is
  inflated. The 2.04× ratio between profiles is still meaningful; the absolute milliseconds
  are not.

## Reproduce

```bash
export OLLAMA_HOST=http://localhost:11434
docvlm-eval sweep --corpus corpora/synthetic-forms-hard \
                  --configs "configs/30b-*.yaml" \
                  --baseline 30b-p1-padrao -j 3 --out out/
```

Or re-score the committed runs with no backend at all:

```bash
docvlm-eval diff -b 30b-p1-padrao -k 30b-p4-padrao --runs-dir examples/runs
```
