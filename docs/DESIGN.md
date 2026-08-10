# Design notes

← [back to README](../README.md)

The decisions that separate this from a 200-line script, and — where there was one — the bug
that caused each decision.

---

## Reproducibility is recorded, not assumed

Every run records the model **digest**, not just the mutable tag; server version; quantisation;
corpus content hash; prompt hash; schema hash; library version; temperature; concurrency.

Without that, a comparison made three weeks later is two numbers of unknown origin. `qwen3-vl:8b`
today and `qwen3-vl:8b` next month are not necessarily the same weights.

## The corpus hash covers the ground truth

Edit one truth value and the hash changes, so `diff` exits non-zero rather than silently
comparing runs from before and after the edit. `--force` overrides it when you mean to.

Filtering to a subset also changes the hash. Deltas and their intervals are both computed on
the paired subset, so the band always sits around the number printed beside it.

## Scoring policy travels with the run

`critical` and `weight` are declared in the schema, optionally overridden per config, resolved
once, and stored in the run. `run`, `report` and `diff` therefore cannot print three different
macro accuracies for the same run.

## Cache the inference, never the scoring

Model outputs are cached by `(config, case, image, schema)`. Scoring is always recomputed.

- Change a normaliser → every number updates instantly, without touching a GPU.
- Change the prompt → the cache invalidates itself, so you cannot accidentally A/B a prompt
  against itself.

## Prompts stay out of the output

Provenance records the prompt *hash*, never the text. A run report can be published without
leaking a tuned prompt.

## CI gates that do not fire on noise

```bash
docvlm-eval run  --corpus corpora/my-corpus --config configs/prod.yaml \
                 --fail-under 0.85 --fail-hallucination-over 0.01
docvlm-eval diff -b prod -k candidate --fail-on-regression
```

`--fail-on-regression` only fires on a *statistically significant* per-field drop. A gate that
blocks the pipeline on ordinary sampling noise gets disabled within a week, and then you have
no gate at all.

---

## Where the bugs were

Scoring is where a silent bug is most expensive: it does not crash, it just changes every
number. That module has the densest tests, and they earned it.

| bug | what it would have done |
|---|---|
| ISO ground-truth date reinterpreted by the `dayfirst` setting | silently wrong date accuracy |
| pt-BR month names scored as `malformed` | a field capped below its real accuracy |
| bootstrap interval indexed asymmetrically | intervals subtly wrong in one direction |
| shared RNG in the mock backend | results depended on `-j`, so reruns disagreed |
| ties in the confidence ranking broken by input order | `sorted()` is stable, so a *constant* confidence signal inherited corpus order and drew a curve that looked informative while ranking nothing — caught by the test that asserts a useless signal is reported as useless |

Each of those was caught by a test rather than by a reader. The last one is the reason the
risk–coverage section compares AURC against random orderings at all.

## Development

```bash
uv pip install -e ".[dev]"
pytest
ruff check .
```

`./release.sh` runs every gate CI runs, including re-deriving the README's headline numbers
from the committed run files.

See [`CHANGELOG.md`](../CHANGELOG.md) for the full history.
