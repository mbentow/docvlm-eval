#!/usr/bin/env bash
# Local pre-push check: run everything CI runs, before pushing.
#
#   ./release.sh
#
# CI (.github/workflows/ci.yml) is the authority. This exists so the same gates
# run on your machine in one command, including the one that matters most: the
# README quotes specific numbers, and they are re-derived from the committed run
# data rather than trusted. A repository whose headline claim has silently
# drifted from its own evidence is worse than one with no numbers at all.
set -euo pipefail

cd "$(dirname "$0")"

echo "-> tests"
python -m pytest -q

echo "-> lint and format"
ruff check .
ruff format --check .

echo "-> corpora validate"
docvlm-eval validate --corpus corpora/synthetic-forms
docvlm-eval validate --corpus corpora/synthetic-forms-hard

echo "-> end to end on the mock backend (no GPU, no model)"
docvlm-eval run \
  --corpus corpora/synthetic-forms \
  --config configs/mock-baseline.yaml \
  --runs-dir /tmp/runs --cache-dir /tmp/cache \
  --bootstrap 300 --overwrite --fail-under 0.70

echo "-> committed runs still re-render"
docvlm-eval report --run hard-qwen3vl-8b --runs-dir examples/runs \
  --format md --bootstrap 400 > /dev/null
docvlm-eval diff -b hard-qwen3vl-30b -k hard-qwen3vl-8b \
  --runs-dir examples/runs --bootstrap 400 > /dev/null

echo "-> do the README numbers still hold?"
docvlm-eval report --run hard-qwen3vl-8b --runs-dir examples/runs \
  --format json --bootstrap 0 \
  | python -c "import json,sys; m=json.load(sys.stdin); \
      assert round(m['macro_accuracy'],3)==0.781, m['macro_accuracy']; \
      assert round(m['all_fields_correct'],3)==0.583, m['all_fields_correct']; \
      assert m['hallucination_rate']==0.0, m['hallucination_rate']; \
      print('   README ok')"

echo
echo "all gates passed - safe to push"
