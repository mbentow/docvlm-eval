# Contributing

## Setup

```bash
uv pip install -e ".[dev]"
pytest
ruff check .
```

## Where to be careful

`src/docvlm_eval/compare.py` and `metrics.py` decide every number the tool prints. A bug there
does not crash — it silently changes results. Any change to those modules needs a test that
states the behaviour in user terms, not implementation terms.

Good: `test_set_over_reading_the_whole_menu_is_flagged_as_hallucination`
Less good: `test_cmp_set_text_branch_3`

## Adding a backend

Subclass `Runner` and implement `extract`. Two rules:

1. **Never raise on a single document.** One bad document must not abort a 200-document sweep.
   Return `RunnerOutput(data=None, error="...")` and let the scorer record it as `refused`.
2. **Implement `describe()`** with whatever identifies the model exactly — digest, revision,
   quantisation, server version. Tags are mutable; a run that only records the tag is not
   reproducible.

Register it in `runners/__init__.py::REGISTRY`.

## Adding a comparison mode

Add the constant to `Compare`, the handler to `compare.py::_HANDLERS`, and tests covering:

- the match case
- the mismatch case
- an unparseable prediction (should be `malformed`, not `wrong`)
- an unparseable ground truth (should be `malformed` — a corpus bug must be visible, not
  charged to the model)

## Contributing to the synthetic corpus

Only fabricated data. No document, name, identifier or image derived from any real source,
redacted or otherwise. The generator is seeded so the corpus can be rebuilt from the script.

New degradations are welcome if they reproduce something that actually breaks extraction in
the field. Add a tag for it so the effect can be sliced out in reports.

## Style

- `ruff` for lint and import order, 100 columns.
- Comments explain *why*, not *what*. If a line encodes a decision or a trap, say so.
- No new runtime dependencies without a reason in the PR description.
