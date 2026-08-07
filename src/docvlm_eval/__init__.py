"""docvlm-eval — field-level evaluation for document extraction with vision models.

Public API::

    from docvlm_eval import ExtractionSchema, field, Compare

    class Invoice(ExtractionSchema):
        number: str | None = field(None, compare=Compare.DIGITS)
        total: float | None = field(None, compare=Compare.NUMBER, tolerance=0.01)
"""

from docvlm_eval.schema import Compare, ExtractionSchema, FieldSpec, field, spec_of
from docvlm_eval.types import Outcome

__all__ = [
    "Compare",
    "ExtractionSchema",
    "FieldSpec",
    "Outcome",
    "field",
    "spec_of",
]

__version__ = "0.2.0"
