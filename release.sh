#!/usr/bin/env bash
# Run every gate CI runs, then (optionally) commit and tag a release.
#
#   ./release.sh --check  # gates only: the pre-push check, touches nothing
#   ./release.sh          # gates, commit, annotated tag
#   ./release.sh --push   # ... and push the branch and the tag
#
# The version is READ FROM pyproject.toml, never typed here. An earlier version
# of this script hardcoded "v0.3.0" in three places, so cutting 0.4.0 meant
# editing the release tool itself — the kind of chore that eventually gets
# skipped, leaving a tag that disagrees with the package it contains.
#
# CI (.github/workflows/ci.yml) is the authority; the gates below mirror it one
# for one. Same reason the README numbers are re-derived instead of asserted
# from memory: a claim in a README is a test that nobody runs.
set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-release}"
case "$MODE" in
  --check|release|--push) ;;
  *) echo "usage: $0 [--check|--push]" >&2; exit 64 ;;
esac

PY="${PY:-.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"
# A venv built on another OS (a Linux .venv synced to a Mac) exists but cannot
# run. Fail here with the fix, not halfway through the gates.
"$PY" -c 'import docvlm_eval' 2>/dev/null || {
  echo "!! $PY cannot import docvlm_eval. Rebuild: uv venv && uv pip install -e '.[dev]'" >&2
  exit 1
}
cli() { "$PY" -m docvlm_eval.cli "$@"; }

VERSION="$("$PY" - <<'EOF'
import pathlib, re
src = pathlib.Path("pyproject.toml").read_text(encoding="utf-8")
print(re.search(r'^version\s*=\s*"([^"]+)"', src, re.M).group(1))
EOF
)"
TAG="v${VERSION}"

# The package version and the tag must agree, or `pip show` and `git describe`
# tell different stories about the same code.
INSTALLED="$("$PY" -c 'import docvlm_eval; print(docvlm_eval.__version__)')"
if [[ "$INSTALLED" != "$VERSION" ]]; then
  echo "!! pyproject says ${VERSION}, docvlm_eval.__version__ says ${INSTALLED}" >&2
  exit 1
fi

if [[ "$MODE" != "--check" ]]; then
  echo "→ releasing ${TAG} (from pyproject.toml)"
  if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
    echo "!! ${TAG} already exists. Bump the version in pyproject.toml first." >&2
    exit 1
  fi
  if ! grep -q "^## ${VERSION} " CHANGELOG.md; then
    echo "!! CHANGELOG.md has no '## ${VERSION}' section." >&2
    exit 1
  fi
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "→ tests, lint, format"
"$PY" -m pytest -q
"$PY" -m ruff check .
"$PY" -m ruff format --check .

echo "→ corpora validate"
cli validate --corpus corpora/synthetic-forms
cli validate --corpus corpora/synthetic-forms-hard

echo "→ end to end on the mock backend (no GPU, no model)"
cli run --corpus corpora/synthetic-forms --config configs/mock-baseline.yaml \
  --runs-dir "$TMP/runs" --cache-dir "$TMP/cache" \
  --bootstrap 300 --overwrite --fail-under 0.70 > /dev/null

echo "→ committed runs still re-render"
cli report --run hard-qwen3vl-8b --runs-dir examples/runs \
  --format md --bootstrap 400 > /dev/null
cli diff -b hard-qwen3vl-30b -k hard-qwen3vl-8b \
  --runs-dir examples/runs --bootstrap 400 > /dev/null

echo "→ do the README numbers still reproduce?"
cli report --run hard-qwen3vl-8b --runs-dir examples/runs \
  --format json --bootstrap 0 \
  | "$PY" -c "import json,sys; m=json.load(sys.stdin); \
      assert round(m['macro_accuracy'],3)==0.781, m['macro_accuracy']; \
      assert round(m['all_fields_correct'],3)==0.583, m['all_fields_correct']; \
      assert m['hallucination_rate']==0.0, m['hallucination_rate']; \
      print('   README ok')"

if [[ "$MODE" == "--check" ]]; then
  echo "✓ all gates passed — safe to push"
  exit 0
fi

# A stale lock from a crashed git is safe to remove; one held by a running git
# is not. Only the first kind gets cleaned.
for lock in .git/index.lock .git/HEAD.lock; do
  [[ -e "$lock" ]] || continue
  if pgrep -x git >/dev/null; then
    echo "!! $lock exists and a git process is running. Wait for it, then retry." >&2
    exit 1
  fi
  echo "   removing stale $lock (no git process alive)"
  rm -f "$lock"
done

echo "→ commit"
git add -A
if [[ -f COMMIT_MESSAGE.txt ]]; then
  git reset -q COMMIT_MESSAGE.txt
  git commit -F COMMIT_MESSAGE.txt && rm -f COMMIT_MESSAGE.txt
elif git diff --cached --quiet; then
  echo "   nothing staged — tagging the current HEAD"
else
  git commit -m "docvlm-eval ${VERSION}"
fi

echo "→ annotated tag ${TAG}, with this release's changelog section as the message"
# Everything from this version's heading to the next one: the tag message and
# the changelog can then never drift apart, because there is only one source.
"$PY" - "$VERSION" > "$TMP/release-msg" <<'EOF'
import pathlib, re, sys
v = sys.argv[1]
text = pathlib.Path("CHANGELOG.md").read_text(encoding="utf-8")
m = re.search(rf"^## {re.escape(v)}[^\n]*\n(.*?)(?=^## |\Z)", text, re.S | re.M)
print(f"v{v}\n\n{m.group(1).strip()}")
EOF
git tag -a "$TAG" -F "$TMP/release-msg"

git --no-pager log --oneline -3
git --no-pager tag -n1 "$TAG"

if [[ "$MODE" == "--push" ]]; then
  echo "→ push"
  git push origin HEAD --follow-tags
fi

echo "✓ ${TAG} ready"
