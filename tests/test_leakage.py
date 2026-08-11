"""Prompt echo — telling a copied answer from a common one."""

from __future__ import annotations

from docvlm_eval.leakage import Echo, leakage_report, literals, render
from docvlm_eval.types import CaseResult, FieldResult, Outcome


def case(cid: str, predicted: str, truth: str):
    ok = predicted.casefold() == truth.casefold()
    return CaseResult(
        case_id=cid,
        fields=[
            FieldResult(
                "exam",
                Outcome.CORRECT if ok else Outcome.WRONG,
                truth=truth,
                predicted=predicted,
                score=1.0 if ok else 0.0,
            )
        ],
    )


# --------------------------------------------------------------------------- #
# Pulling literals out of a prompt
# --------------------------------------------------------------------------- #


def test_finds_quoted_examples():
    assert "US ABD TOTAL" in literals("do not expand abbreviations (write 'US ABD TOTAL')")


def test_finds_uppercase_domain_codes_without_quotes():
    assert "RX TORAX PA" in literals("Codes look like RX TORAX PA on the form.")


def test_single_uppercase_words_are_instructions_not_answers():
    """JSON/ONLY/NOT are how prompts shout, not things a model would copy as an
    answer. Including them would bury the real signal under boilerplate."""
    found = literals("Reply ONLY with JSON. Do NOT invent fields.")
    assert not any(w in found for w in ("JSON", "ONLY", "NOT"))


# --------------------------------------------------------------------------- #
# The central distinction
# --------------------------------------------------------------------------- #

PROMPT = "For example, write 'ECO ABDOMEN TOTAL' when you see it."


def test_common_answer_is_not_a_leak_even_when_frequent():
    """The example is the most common true answer, and the model says it a lot
    and is right. Raw frequency would scream; this must not."""
    cases = [case(f"t{i}", "ECO ABDOMEN TOTAL", "ECO ABDOMEN TOTAL") for i in range(12)]
    cases += [case(f"o{i}", "RM JOELHO", "RM JOELHO") for i in range(8)]
    rep = leakage_report(cases, PROMPT)
    assert rep.clean, "an example that is simply the common answer must not be flagged"


def test_over_produced_and_wrong_is_a_leak():
    """Same literal, opposite story: the model says it on documents where it is
    not the answer, and gets them wrong."""
    cases = [case(f"bad{i}", "ECO ABDOMEN TOTAL", "RM JOELHO") for i in range(10)]
    cases += [case(f"ok{i}", "MAMOGRAFIA", "MAMOGRAFIA") for i in range(10)]
    rep = leakage_report(cases, PROMPT)
    assert not rep.clean
    top = rep.suspicious[0]
    assert top.literal == "ECO ABDOMEN TOTAL"
    assert top.accuracy_gap < 0
    assert top.lift is None, "truth never contains it → lift undefined, not infinite"


def test_over_produced_but_still_accurate_is_not_flagged():
    """Lift above the threshold, accuracy unharmed. Flagging here is exactly the
    false alarm that would cost someone a working prompt."""
    cases = [case(f"h{i}", "ECO ABDOMEN TOTAL", "ECO ABDOMEN TOTAL") for i in range(9)]
    cases += [case("x", "ECO ABDOMEN TOTAL", "RM JOELHO")]
    cases += [case(f"o{i}", "MAMOGRAFIA", "MAMOGRAFIA") for i in range(10)]
    rep = leakage_report(cases, PROMPT)
    echo = next(e for e in rep.echoes if e.literal == "ECO ABDOMEN TOTAL")
    assert echo.lift is not None and echo.lift > 1.0
    assert not echo.suspicious


def test_a_handful_of_hits_is_noise_not_evidence():
    cases = [case("a", "ECO ABDOMEN TOTAL", "RM JOELHO")]
    cases += [case(f"o{i}", "MAMOGRAFIA", "MAMOGRAFIA") for i in range(10)]
    rep = leakage_report(cases, PROMPT)
    assert rep.clean, "one wrong echo cannot support a claim about the prompt"


def test_literals_the_model_never_said_are_not_reported():
    cases = [case(f"o{i}", "MAMOGRAFIA", "MAMOGRAFIA") for i in range(6)]
    rep = leakage_report(cases, PROMPT)
    assert rep.echoes == []
    assert rep.literals_scanned >= 1
    assert "Nothing the model said came from the prompt" in render(rep)


# --------------------------------------------------------------------------- #
# Arithmetic and edges
# --------------------------------------------------------------------------- #


def test_lift_compares_prediction_rate_to_truth_rate():
    e = Echo("X", n_predicted=6, n_truth=3, n_cases=12, accuracy_echoed=0.5, accuracy_other=0.9)
    assert e.rate_predicted == 0.5
    assert e.rate_truth == 0.25
    assert e.lift == 2.0


def test_matching_ignores_case_and_surrounding_text():
    cases = [case(f"b{i}", "pedido: eco abdomen total (urgente)", "RM JOELHO") for i in range(10)]
    cases += [case(f"o{i}", "MAMOGRAFIA", "MAMOGRAFIA") for i in range(10)]
    assert not leakage_report(cases, PROMPT).clean


def test_empty_input_does_not_explode():
    rep = leakage_report([], PROMPT)
    assert rep.n_cases == 0
    assert "No cases" in render(rep)
    assert leakage_report([case("a", "X", "X")], "").echoes == []
