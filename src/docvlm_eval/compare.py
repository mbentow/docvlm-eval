"""Field-level scoring.

One function decides every number this tool prints, so it is the one place a
silent bug would be fatal. It is also the most heavily tested module.

The outcome taxonomy is the product:

===============  ==============================================================
``correct``      match, including "truth empty and model correctly said nothing"
``missing``      truth has a value, model returned nothing
``hallucinated`` truth is empty, model invented a value
``wrong``        both have values, they disagree
``malformed``    the value does not fit the declared type
``refused``      the backend errored or the model declined
===============  ==============================================================
"""

from __future__ import annotations

from typing import Any

from rapidfuzz.distance import Levenshtein

from docvlm_eval.normalize import (
    apply_chain,
    digits_only,
    is_empty,
    parse_bool,
    parse_date,
    parse_number,
)
from docvlm_eval.schema import Compare, ExtractionSchema, FieldSpec
from docvlm_eval.types import CaseResult, FieldResult, Outcome


def similarity(a: str, b: str) -> float:
    """Normalised Levenshtein similarity in ``[0, 1]``."""
    if not a and not b:
        return 1.0
    return Levenshtein.normalized_similarity(a, b)


def compare_field(
    name: str,
    truth: Any,
    predicted: Any,
    spec: FieldSpec,
    *,
    present_in_output: bool = True,
) -> FieldResult:
    """Score one field.

    ``present_in_output`` distinguishes "the model omitted the key" from "the
    model returned the key with an empty value". Both are ``missing`` when the
    truth has a value — the distinction shows up in ``detail``.
    """
    truth_empty = is_empty(truth)
    pred_empty = is_empty(predicted)

    if truth_empty and pred_empty:
        return FieldResult(name, Outcome.CORRECT, truth, predicted, 1.0, "both empty")
    if truth_empty and not pred_empty:
        return FieldResult(
            name,
            Outcome.HALLUCINATED,
            truth,
            predicted,
            0.0,
            "ground truth is empty; model returned a value",
        )
    if not truth_empty and pred_empty:
        detail = "key absent from output" if not present_in_output else "empty value returned"
        return FieldResult(name, Outcome.MISSING, truth, predicted, 0.0, detail)

    handler = _HANDLERS.get(spec.compare, _cmp_text)
    return handler(name, truth, predicted, spec)


# --------------------------------------------------------------------------- #
# Per-mode handlers
# --------------------------------------------------------------------------- #


def _cmp_exact(name: str, truth: Any, pred: Any, spec: FieldSpec) -> FieldResult:
    t, p = str(truth), str(pred)
    if t == p:
        return FieldResult(name, Outcome.CORRECT, truth, pred, 1.0, "exact")
    return FieldResult(name, Outcome.WRONG, truth, pred, similarity(t, p), "exact mismatch")


def _cmp_text(name: str, truth: Any, pred: Any, spec: FieldSpec) -> FieldResult:
    t = apply_chain(str(truth), spec.normalize)
    p = apply_chain(str(pred), spec.normalize)
    if t == p:
        return FieldResult(name, Outcome.CORRECT, truth, pred, 1.0, "normalized match")
    score = similarity(t, p)
    if spec.fuzzy_threshold < 1.0 and score >= spec.fuzzy_threshold:
        return FieldResult(name, Outcome.CORRECT, truth, pred, score, f"fuzzy match {score:.3f}")
    return FieldResult(name, Outcome.WRONG, truth, pred, score, f"similarity {score:.3f}")


def _cmp_digits(name: str, truth: Any, pred: Any, spec: FieldSpec) -> FieldResult:
    t, p = digits_only(str(truth)), digits_only(str(pred))
    if not t:
        # A corpus bug, not a model failure. Charging it to the model would cap
        # the field's accuracy for reasons no amount of prompting can fix.
        return FieldResult(name, Outcome.MALFORMED, truth, pred, 0.0, "no digits in ground truth")
    if not p:
        return FieldResult(
            name, Outcome.MALFORMED, truth, pred, 0.0, "no digits in predicted value"
        )
    if t == p:
        return FieldResult(name, Outcome.CORRECT, truth, pred, 1.0, "digits match")
    score = similarity(t, p)
    edits = Levenshtein.distance(t, p)
    return FieldResult(name, Outcome.WRONG, truth, pred, score, f"{edits} digit(s) off")


def _cmp_number(name: str, truth: Any, pred: Any, spec: FieldSpec) -> FieldResult:
    t, p = parse_number(truth), parse_number(pred)
    if t is None:
        return FieldResult(name, Outcome.MALFORMED, truth, pred, 0.0, "unparseable ground truth")
    if p is None:
        return FieldResult(name, Outcome.MALFORMED, truth, pred, 0.0, "not a number")
    delta = abs(t - p)
    if delta <= spec.tolerance:
        return FieldResult(name, Outcome.CORRECT, truth, pred, 1.0, f"delta {delta:g}")
    return FieldResult(name, Outcome.WRONG, truth, pred, 0.0, f"delta {delta:g}")


def _cmp_date(name: str, truth: Any, pred: Any, spec: FieldSpec) -> FieldResult:
    t = parse_date(truth, dayfirst=spec.dayfirst)
    p = parse_date(pred, dayfirst=spec.dayfirst)
    if t is None:
        return FieldResult(name, Outcome.MALFORMED, truth, pred, 0.0, "unparseable ground truth")
    if p is None:
        return FieldResult(name, Outcome.MALFORMED, truth, pred, 0.0, "not a date")
    days = abs((p - t).days)
    if days <= spec.tolerance_days:
        return FieldResult(name, Outcome.CORRECT, truth, pred, 1.0, f"{days}d apart")
    return FieldResult(name, Outcome.WRONG, truth, pred, 0.0, f"{days}d apart")


def _cmp_bool(name: str, truth: Any, pred: Any, spec: FieldSpec) -> FieldResult:
    t, p = parse_bool(truth), parse_bool(pred)
    if t is None:
        return FieldResult(name, Outcome.MALFORMED, truth, pred, 0.0, "unparseable ground truth")
    if p is None:
        return FieldResult(name, Outcome.MALFORMED, truth, pred, 0.0, "not a boolean")
    if t == p:
        return FieldResult(name, Outcome.CORRECT, truth, pred, 1.0, "bool match")
    return FieldResult(name, Outcome.WRONG, truth, pred, 0.0, f"expected {t}, got {p}")


def _cmp_set_text(name: str, truth: Any, pred: Any, spec: FieldSpec) -> FieldResult:
    t_items = _as_list(truth)
    p_items = _as_list(pred)
    if p_items is None:
        return FieldResult(name, Outcome.MALFORMED, truth, pred, 0.0, "not a list")

    t_norm = {apply_chain(str(x), spec.normalize) for x in t_items if not is_empty(x)}
    p_norm = {apply_chain(str(x), spec.normalize) for x in p_items if not is_empty(x)}

    hits = len(t_norm & p_norm)
    precision = hits / len(p_norm) if p_norm else 0.0
    recall = hits / len(t_norm) if t_norm else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    detail = f"P {precision:.2f} R {recall:.2f} F1 {f1:.2f} ({hits}/{len(t_norm)})"

    if t_norm == p_norm:
        return FieldResult(name, Outcome.CORRECT, truth, pred, 1.0, "set match · " + detail)
    if not (t_norm & p_norm) and len(p_norm) > len(t_norm):
        # Nothing in common and the model returned more than it should have:
        # this is the "read the whole pre-printed menu" failure, not a near miss.
        return FieldResult(name, Outcome.HALLUCINATED, truth, pred, f1, "over-read · " + detail)
    return FieldResult(name, Outcome.WRONG, truth, pred, f1, detail)


def _as_list(value: Any) -> list[Any] | None:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        # A model asked for a list sometimes returns "a; b; c".
        parts = [p.strip() for p in value.replace("\n", ";").split(";")]
        return [p for p in parts if p]
    return None


_HANDLERS = {
    Compare.EXACT: _cmp_exact,
    Compare.TEXT: _cmp_text,
    Compare.DIGITS: _cmp_digits,
    Compare.NUMBER: _cmp_number,
    Compare.DATE: _cmp_date,
    Compare.BOOL: _cmp_bool,
    Compare.SET_TEXT: _cmp_set_text,
}


# --------------------------------------------------------------------------- #
# Case-level scoring
# --------------------------------------------------------------------------- #


def score_case(
    case_id: str,
    tags: list[str],
    truth: dict[str, Any],
    predicted: dict[str, Any] | None,
    schema: type[ExtractionSchema],
    *,
    error: str = "",
    latency_ms: float = 0.0,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    raw_output: str = "",
    cached: bool = False,
) -> CaseResult:
    """Score every schema field of one document.

    A backend error is not a zero — it is ``refused`` on every field, reported
    separately. Folding refusals into "wrong" makes an unreliable backend look
    like an inaccurate model.
    """
    specs = schema.specs()
    result = CaseResult(
        case_id=case_id,
        tags=list(tags),
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        error=error,
        raw_output=raw_output,
        cached=cached,
    )

    if predicted is None:
        detail = error or "no output"
        result.fields = [
            FieldResult(name, Outcome.REFUSED, truth.get(name), None, 0.0, detail) for name in specs
        ]
        return result

    for name, spec in specs.items():
        result.fields.append(
            compare_field(
                name,
                truth.get(name),
                predicted.get(name),
                spec,
                present_in_output=name in predicted,
            )
        )
    return result
