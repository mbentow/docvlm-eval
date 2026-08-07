"""Schema for the public synthetic corpus.

This is an *example* schema over invented documents. The schemas used in
production deployments are not part of this repository.

Read the comparison policies, not just the types — they are where the honesty
of an evaluation lives:

* ``patient_name`` accepts a fuzzy match, because an operator reading the result
  would accept ``MARINA FERRAZZO`` for ``Marina Ferrazzo``;
* ``doctor_crm`` does **not**. One wrong digit points at a different doctor, and
  the system downstream will match it with full confidence and be silently
  wrong. It is also marked ``critical``: a hallucinated licence number is a
  different class of problem from a missing one;
* ``exams`` is a set, scored strictly. Reading three exams on a single-exam
  request is an error — it means the model did not know when to stop.
"""

from docvlm_eval import Compare, ExtractionSchema, field


class MedicalRequest(ExtractionSchema):
    patient_name: str | None = field(
        None,
        description="Full name of the patient, exactly as written",
        compare=Compare.TEXT,
        fuzzy_threshold=0.90,
    )
    doctor_crm: str | None = field(
        None,
        description="Medical licence number of the requesting doctor (digits and state)",
        compare=Compare.DIGITS,
        critical=True,
    )
    insurer: str | None = field(
        None,
        description="Health insurance company; null if the form says none",
        compare=Compare.TEXT,
        fuzzy_threshold=0.88,
    )
    member_id: str | None = field(
        None,
        description="Insurance member/card number; null if absent or unreadable",
        compare=Compare.DIGITS,
        critical=True,
    )
    exams: list[str] = field(
        default_factory=list,
        description="Every exam explicitly requested on the form",
        compare=Compare.SET_TEXT,
        normalize=["collapse_ws", "lower", "strip_accents", "strip_punct", "collapse_ws"],
    )
    request_date: str | None = field(
        None,
        description="Date the request was signed, as written (DD/MM/YYYY)",
        compare=Compare.DATE,
        dayfirst=True,
    )
    urgent: bool | None = field(
        None,
        description="True if the form is marked URGENTE, false if ROTINA",
        compare=Compare.BOOL,
    )


SCHEMA = MedicalRequest
