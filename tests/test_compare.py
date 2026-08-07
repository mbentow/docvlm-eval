"""Scoring tests.

This is the module where a silent bug is most expensive: it would not crash, it
would just change every number the tool reports. The tests below are written as
assertions about *behaviour a user would notice*, not about implementation.
"""

from __future__ import annotations

import pytest

from docvlm_eval import Compare, ExtractionSchema, field
from docvlm_eval.compare import compare_field, score_case
from docvlm_eval.schema import FieldSpec
from docvlm_eval.types import Outcome


class Doc(ExtractionSchema):
    name: str | None = field(None, compare=Compare.TEXT, fuzzy_threshold=0.9)
    crm: str | None = field(None, compare=Compare.DIGITS, critical=True)
    when: str | None = field(None, compare=Compare.DATE)
    urgent: bool | None = field(None, compare=Compare.BOOL)
    exams: list[str] = field(default_factory=list, compare=Compare.SET_TEXT)


TEXT = FieldSpec(compare=Compare.TEXT)
DIGITS = FieldSpec(compare=Compare.DIGITS, normalize=())
DATE = FieldSpec(compare=Compare.DATE, normalize=())
BOOL = FieldSpec(compare=Compare.BOOL, normalize=())
NUM = FieldSpec(compare=Compare.NUMBER, normalize=(), tolerance=0.01)
SET = FieldSpec(compare=Compare.SET_TEXT)


# --------------------------------------------------------------------------- #
# The taxonomy — the reason this tool exists
# --------------------------------------------------------------------------- #


def test_missing_and_hallucinated_are_not_the_same_thing():
    missing = compare_field("f", "Maria", None, TEXT)
    hallucinated = compare_field("f", None, "Maria", TEXT)
    assert missing.outcome is Outcome.MISSING
    assert hallucinated.outcome is Outcome.HALLUCINATED
    assert missing.outcome is not hallucinated.outcome


def test_both_empty_is_correct_not_ignored():
    """Correctly declining to fill a field is a right answer, not a non-answer.

    Scoring it as "no data" would reward a model that fills every field.
    """
    result = compare_field("f", None, None, TEXT)
    assert result.outcome is Outcome.CORRECT


@pytest.mark.parametrize("stand_in", ["null", "N/A", "-", "  ", "não informado"])
def test_model_prose_for_empty_counts_as_empty(stand_in):
    """Models say "N/A" instead of omitting a key. Treating that as a value
    would inflate the hallucination rate with what is really an abstention."""
    assert compare_field("f", None, stand_in, TEXT).outcome is Outcome.CORRECT
    assert compare_field("f", "Maria", stand_in, TEXT).outcome is Outcome.MISSING


def test_missing_key_vs_empty_value_both_missing_but_distinguishable():
    absent = compare_field("f", "x", None, TEXT, present_in_output=False)
    empty = compare_field("f", "x", "", TEXT, present_in_output=True)
    assert absent.outcome is empty.outcome is Outcome.MISSING
    assert "absent" in absent.detail
    assert "absent" not in empty.detail


# --------------------------------------------------------------------------- #
# Normalisation is per field, never global
# --------------------------------------------------------------------------- #


def test_text_ignores_case_accents_and_punctuation():
    assert compare_field("f", "João da Silva", "JOAO DA SILVA.", TEXT).outcome is Outcome.CORRECT


def test_text_fuzzy_only_when_the_field_allows_it():
    strict = FieldSpec(compare=Compare.TEXT)
    lenient = FieldSpec(compare=Compare.TEXT, fuzzy_threshold=0.85)
    assert compare_field("f", "Marina Ferrazzo", "Marina Ferrazo", strict).outcome is Outcome.WRONG
    assert (
        compare_field("f", "Marina Ferrazzo", "Marina Ferrazo", lenient).outcome is Outcome.CORRECT
    )


def test_digits_strips_formatting_but_not_a_wrong_digit():
    """The failure this guards against: a licence number read with one digit
    off points at a different person, and downstream systems match it with full
    confidence. Formatting differences are noise; a digit is not."""
    assert compare_field("crm", "12345-RS", "CRM 12.345 / RS", DIGITS).outcome is Outcome.CORRECT
    off_by_one = compare_field("crm", "12345", "12845", DIGITS)
    assert off_by_one.outcome is Outcome.WRONG
    assert "1 digit" in off_by_one.detail


def test_digits_with_no_digits_is_malformed_not_wrong():
    """The model returned something that is not a number at all. That is a
    schema failure, not a reading error, and it is counted separately."""
    result = compare_field("crm", "12345", "CRM do solicitante", DIGITS)
    assert result.outcome is Outcome.MALFORMED


def test_a_model_declaring_illegibility_is_abstaining_not_malformed():
    """`ilegivel` is the model saying "I could not read it". That is a missing
    field — an operator retypes it — not a broken value."""
    assert compare_field("crm", "12345", "ilegível", DIGITS).outcome is Outcome.MISSING


def test_date_compares_as_date_across_formats():
    assert compare_field("d", "2026-03-14", "14/03/2026", DATE).outcome is Outcome.CORRECT
    assert compare_field("d", "2026-03-14", "14 de março de 2026", DATE).outcome is Outcome.CORRECT


def test_date_dayfirst_is_a_declared_choice():
    """``04/03/2026`` is 4 March in Brazil and 3 April in the US. The field
    declares which, and the ISO ground truth is never reinterpreted."""
    ptbr = FieldSpec(compare=Compare.DATE, normalize=(), dayfirst=True)
    us = FieldSpec(compare=Compare.DATE, normalize=(), dayfirst=False)
    assert compare_field("d", "2026-03-04", "04/03/2026", ptbr).outcome is Outcome.CORRECT
    assert compare_field("d", "2026-03-04", "04/03/2026", us).outcome is Outcome.WRONG
    assert compare_field("d", "2026-03-04", "03/04/2026", us).outcome is Outcome.CORRECT


def test_iso_ground_truth_is_never_reinterpreted_by_dayfirst():
    """Regression guard: with naive parsing, dayfirst=True turned every ISO
    ground-truth date around and shifted the whole corpus."""
    from docvlm_eval.normalize import parse_date

    assert parse_date("2026-03-04", dayfirst=True).isoformat() == "2026-03-04"
    assert parse_date("2026-03-04", dayfirst=False).isoformat() == "2026-03-04"


def test_date_tolerance():
    loose = FieldSpec(compare=Compare.DATE, normalize=(), tolerance_days=2)
    assert compare_field("d", "2026-03-14", "2026-03-15", loose).outcome is Outcome.CORRECT
    assert compare_field("d", "2026-03-14", "2026-03-20", loose).outcome is Outcome.WRONG


def test_date_unparseable_prediction_is_malformed():
    assert compare_field("d", "2026-03-14", "sem data", DATE).outcome is Outcome.MALFORMED


@pytest.mark.parametrize(
    ("truth", "pred"), [(True, "sim"), (True, "URGENTE"), (False, "rotina"), (False, "no")]
)
def test_bool_handles_how_models_actually_answer(truth, pred):
    assert compare_field("u", truth, pred, BOOL).outcome is Outcome.CORRECT


def test_bool_false_is_a_value_not_an_absence():
    """`False` must not be treated as empty — otherwise every correctly-read
    "ROTINA" would be scored as a missing field."""
    assert compare_field("u", False, False, BOOL).outcome is Outcome.CORRECT
    assert compare_field("u", False, True, BOOL).outcome is Outcome.WRONG


def test_number_tolerance():
    assert compare_field("n", 1234.56, "1.234,56", NUM).outcome is Outcome.CORRECT
    assert compare_field("n", 1234.56, "1234.60", NUM).outcome is Outcome.WRONG


# --------------------------------------------------------------------------- #
# Sets — measuring whether the model knows when to stop
# --------------------------------------------------------------------------- #


def test_set_ignores_order():
    assert compare_field("e", ["a", "b"], ["b", "a"], SET).outcome is Outcome.CORRECT


def test_set_partial_match_is_wrong_not_correct():
    """A document where the model found 2 of 3 exams still needs a human."""
    result = compare_field("e", ["a", "b", "c"], ["a", "b"], SET)
    assert result.outcome is Outcome.WRONG
    assert 0 < result.score < 1


def test_set_over_reading_the_whole_menu_is_flagged_as_hallucination():
    """The pre-printed-form failure: the model transcribes every option on the
    page instead of the one that was ticked."""
    result = compare_field("e", ["holter"], ["opt1", "opt2", "opt3", "opt4"], SET)
    assert result.outcome is Outcome.HALLUCINATED


def test_set_accepts_a_delimited_string_from_a_sloppy_backend():
    assert compare_field("e", ["a", "b"], "a; b", SET).outcome is Outcome.CORRECT


def test_set_non_list_is_malformed():
    assert compare_field("e", ["a"], {"exam": "a"}, SET).outcome is Outcome.MALFORMED


# --------------------------------------------------------------------------- #
# Case level
# --------------------------------------------------------------------------- #


def test_backend_error_is_refused_on_every_field_not_wrong():
    """Folding backend failures into "wrong" makes an unreliable server look
    like an inaccurate model."""
    result = score_case("1", [], {"name": "x"}, None, Doc, error="connection reset")
    assert {f.outcome for f in result.fields} == {Outcome.REFUSED}
    assert not result.all_fields_correct


def test_all_fields_correct_needs_every_field():
    truth = {"name": "Ana", "crm": "123", "when": "2026-01-02", "urgent": True, "exams": ["a"]}
    perfect = score_case("1", [], truth, dict(truth), Doc)
    assert perfect.all_fields_correct

    almost = score_case("2", [], truth, {**truth, "crm": "124"}, Doc)
    assert not almost.all_fields_correct


def test_extra_keys_from_the_model_are_ignored():
    truth = {"name": "Ana", "crm": "123", "when": "2026-01-02", "urgent": True, "exams": ["a"]}
    result = score_case("1", [], truth, {**truth, "confidence": 0.9, "notes": "hi"}, Doc)
    assert result.all_fields_correct
    assert len(result.fields) == len(Doc.model_fields)
