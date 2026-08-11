"""Is the model reading the document, or repeating the prompt?

Prompts for document extraction are full of literals: format hints, worked
examples, enumerations of valid codes. Anything you write inside the
instructions is a string the model can emit without ever looking at the page —
and it will look like a normal answer, because it *is* a normal answer. It just
did not come from the image.

This shows up when the document is hard. The model is under pressure to fill a
required field, finds nothing legible, and returns the most available token
sequence it has: the example you gave it. Accuracy barely moves, because the
example was chosen to be typical and typical answers are often right. What
moves is *where* the model is right — it stops being right for the right reason.

Why counting is not enough
--------------------------
The obvious check — "how often does the model output my example?" — proves
nothing on its own, and acting on it is how you break a working pipeline.

An example in a prompt is usually chosen *because* it is the common case. So it
should appear often in the output. Measured on a real deployment (200k
extractions of medical imaging orders): the string used as an inline example
appeared in 4.4% of predictions, which looks alarming until you check the
ground truth, where the same procedure is 7.7% of all orders. The model was
saying it *less* often than reality. There was no leak.

So this module never reports a raw count as evidence. Every literal is scored
against two things:

    lift      how much more often the model says it than the truth contains it
    accuracy  whether cases that echo the literal are *less* accurate than the rest

Lift alone still is not enough — a legitimately over-eager model can have lift
above 1 and be right anyway. The deciding evidence is the second one. A literal
is only flagged when the model over-produces it **and** gets those cases wrong
more often than its own baseline. That is the signature of an answer that came
from the instructions rather than the page: available, plausible, unearned.

The same 200k deployment, scored this way: predictions containing the example
were 90.2% correct against 80.6% for everything else. Higher, not lower. The
check clears it, and the pipeline is left alone.

What it cannot tell you
-----------------------
* A leak that happens to be correct is invisible here, and that is deliberate:
  if the answer is right, "for the wrong reason" is a research question, not an
  operational one. Where it matters, add a case where the example is *not* the
  answer and watch it fail.
* Paraphrased leakage — the model emitting a normalised form of your example
  rather than the literal — is not detected. Exact and case-insensitive
  substring only. Fuzzy matching here would trade a precise signal for a noisy
  one, and this module exists to avoid false alarms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .types import CaseResult, Outcome

MIN_HITS = 3
"""Below this many echoing cases, accuracy on the subset is noise. Reported for
transparency, never flagged."""

LIFT_FLAG = 1.5
"""Model says it 50% more often than the truth contains it. Necessary, not
sufficient — see the module docstring."""

_QUOTED = re.compile(r"""["'`“‘«]([^"'`”’»\n]{3,60})["'`”’»]""")
_CAPS = re.compile(r"\b([A-Z][A-Z0-9./-]{1,}(?:\s+[A-Z][A-Z0-9./-]{1,}){1,5})\b")


def literals(prompt: str, *, min_len: int = 4) -> list[str]:
    """Strings a model could copy out of the prompt.

    Two sources, both deliberate:

    * quoted spans — how worked examples are almost always written;
    * runs of two or more ALL-CAPS tokens — how domain codes and abbreviations
      appear ("US ABD TOTAL", "RX TORAX PA"). Single caps words are excluded:
      "JSON", "NOT", "ONLY" are instructions, not answers, and including them
      buries the signal.

    Returned longest-first so that a containing literal is considered before the
    fragment inside it.
    """
    found: set[str] = set()
    for rx in (_QUOTED, _CAPS):
        for m in rx.finditer(prompt or ""):
            s = " ".join(m.group(1).split())
            if len(s) >= min_len:
                found.add(s)
    return sorted(found, key=lambda s: (-len(s), s))


def _contains(value, needle: str) -> bool:
    if value is None:
        return False
    return needle.casefold() in str(value).casefold()


def _case_has(case: CaseResult, needle: str, *, predicted: bool) -> bool:
    return any(_contains(f.predicted if predicted else f.truth, needle) for f in case.fields)


def _accuracy(cases: list[CaseResult]) -> float:
    fields = [f for c in cases for f in c.fields]
    if not fields:
        return 0.0
    return sum(f.outcome is Outcome.CORRECT for f in fields) / len(fields)


@dataclass
class Echo:
    """One prompt literal, measured against the run."""

    literal: str
    n_predicted: int
    n_truth: int
    n_cases: int
    accuracy_echoed: float
    accuracy_other: float

    @property
    def rate_predicted(self) -> float:
        return self.n_predicted / self.n_cases if self.n_cases else 0.0

    @property
    def rate_truth(self) -> float:
        return self.n_truth / self.n_cases if self.n_cases else 0.0

    @property
    def lift(self) -> float | None:
        """Over-production versus reality. ``None`` when the truth never
        contains it — undefined rather than infinite, because a literal absent
        from the ground truth of a small corpus is usually a corpus fact, not a
        model fact."""
        if not self.rate_truth:
            return None
        return self.rate_predicted / self.rate_truth

    @property
    def accuracy_gap(self) -> float:
        """Negative means echoing cases do worse — the direction that matters."""
        return self.accuracy_echoed - self.accuracy_other

    @property
    def suspicious(self) -> bool:
        """Over-produced **and** less accurate. Both, never either.

        The `accuracy_gap < 0` half is what keeps this from firing on a prompt
        example that is simply the common answer — the failure mode described in
        the module docstring, and the one that would cost you a working prompt.
        """
        if self.n_predicted < MIN_HITS:
            return False
        over = self.lift is None or self.lift >= LIFT_FLAG
        return over and self.accuracy_gap < 0


@dataclass
class LeakageReport:
    n_cases: int
    literals_scanned: int
    echoes: list[Echo]

    @property
    def suspicious(self) -> list[Echo]:
        return sorted([e for e in self.echoes if e.suspicious], key=lambda e: e.accuracy_gap)

    @property
    def clean(self) -> bool:
        return not self.suspicious


def leakage_report(cases: list[CaseResult], prompt: str) -> LeakageReport:
    """Score every literal in ``prompt`` against the predictions in ``cases``.

    Only literals the model actually emitted are kept: a prompt has dozens of
    quoted spans and reporting all of them at rate zero would bury the two that
    matter.
    """
    lits = literals(prompt)
    echoes: list[Echo] = []
    for lit in lits:
        hit = [c for c in cases if _case_has(c, lit, predicted=True)]
        if not hit:
            continue
        miss = [c for c in cases if c not in hit]
        echoes.append(
            Echo(
                literal=lit,
                n_predicted=len(hit),
                n_truth=sum(_case_has(c, lit, predicted=False) for c in cases),
                n_cases=len(cases),
                accuracy_echoed=_accuracy(hit),
                accuracy_other=_accuracy(miss),
            )
        )
    echoes.sort(key=lambda e: -e.n_predicted)
    return LeakageReport(n_cases=len(cases), literals_scanned=len(lits), echoes=echoes)


def render(report: LeakageReport) -> str:
    if not report.n_cases:
        return "No cases to check for prompt echo."
    out = [
        f"Prompt echo — {report.literals_scanned} literal(s) in the prompt, "
        f"{len(report.echoes)} of them emitted by the model."
    ]
    if not report.echoes:
        return out[0] + "\n  Nothing the model said came from the prompt text."
    out.append(
        f"  {'literal':<28}{'said':>6}{'true':>6}{'lift':>7}{'acc(echo)':>11}{'acc(rest)':>11}"
    )
    for e in report.echoes[:12]:
        lift = "  n/a" if e.lift is None else f"{e.lift:>5.2f}"
        flag = "  <-- suspect" if e.suspicious else ""
        out.append(
            f"  {e.literal[:27]:<28}{e.n_predicted:>6}{e.n_truth:>6}{lift:>7}"
            f"{e.accuracy_echoed:>11.3f}{e.accuracy_other:>11.3f}{flag}"
        )
    if report.clean:
        out.append(
            "  No literal is both over-produced and less accurate — "
            "frequency alone is not evidence of a leak."
        )
    else:
        out.append(
            f"  {len(report.suspicious)} literal(s) look copied rather than read: "
            "the model says them more than reality contains them AND is "
            "less accurate when it does."
        )
    return "\n".join(out)
