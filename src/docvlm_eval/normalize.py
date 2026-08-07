"""Normalisers.

Normalisation is declared **per field**, never globally. A patient name should
ignore case and accents; a licence number should be reduced to its digits; a
date must be parsed and compared as a date, not as a string. Applying one global
rule to all of them is how an evaluator quietly reports the wrong number.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from datetime import date, datetime

from dateutil import parser as dateparser

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_NON_DIGIT = re.compile(r"\D+")


def strip_accents(text: str) -> str:
    """``"Coração"`` -> ``"Coracao"``."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def collapse_ws(text: str) -> str:
    """Squash runs of whitespace and trim the ends."""
    return _WS.sub(" ", text).strip()


def lower(text: str) -> str:
    return text.lower()


def upper(text: str) -> str:
    return text.upper()


def strip_punct(text: str) -> str:
    """Drop punctuation, keeping word characters and spaces."""
    return _PUNCT.sub(" ", text)


def digits_only(text: str) -> str:
    """``"CRM 12.345/RS"`` -> ``"12345"``."""
    return _NON_DIGIT.sub("", text)


def strip_titles(text: str) -> str:
    """Remove common Brazilian clinical name prefixes (``Dr.``, ``Dra.``, ``Prof.``)."""
    return re.sub(r"^\s*(dr|dra|drs|prof|profa)\.?\s+", "", text, flags=re.IGNORECASE)


#: Named normalisers usable from a schema declaration or a YAML config.
NORMALIZERS: dict[str, Callable[[str], str]] = {
    "lower": lower,
    "upper": upper,
    "strip_accents": strip_accents,
    "collapse_ws": collapse_ws,
    "strip_punct": strip_punct,
    "digits_only": digits_only,
    "strip_titles": strip_titles,
}

#: A sensible default chain for free text.
DEFAULT_TEXT_CHAIN = ("collapse_ws", "lower", "strip_accents", "strip_punct", "collapse_ws")


def apply_chain(value: str, chain: tuple[str, ...] | list[str]) -> str:
    """Run a named normaliser chain in order. Unknown names raise ``KeyError``."""
    out = value
    for name in chain:
        try:
            out = NORMALIZERS[name](out)
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(
                f"unknown normalizer {name!r}; available: {sorted(NORMALIZERS)}"
            ) from exc
    return out


_ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")

#: Month names a pt-BR document (or a model writing about one) actually uses.
#: dateutil only knows English, so without this "14 de março de 2026" is scored
#: as malformed — an evaluator bug that looks like a model failure.
_PT_MONTHS = {
    "janeiro": "January",
    "jan": "January",
    "fevereiro": "February",
    "fev": "February",
    "marco": "March",
    "mar": "March",
    "abril": "April",
    "abr": "April",
    "maio": "May",
    "mai": "May",
    "junho": "June",
    "jun": "June",
    "julho": "July",
    "jul": "July",
    "agosto": "August",
    "ago": "August",
    "setembro": "September",
    "set": "September",
    "outubro": "October",
    "out": "October",
    "novembro": "November",
    "nov": "November",
    "dezembro": "December",
    "dez": "December",
}


def parse_date(value: object, dayfirst: bool = True) -> date | None:
    """Best-effort date parsing. ``dayfirst=True`` matches pt-BR documents.

    ISO ``YYYY-MM-DD`` is always parsed as ISO, whatever ``dayfirst`` says.
    Ground truth is normally written that way, and letting ``dayfirst`` flip it
    silently turns 4 March into 3 April on every date in the corpus.

    Returns ``None`` when the value cannot be read as a date — the caller turns
    that into a ``MALFORMED`` outcome rather than silently comparing strings.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None

    iso = _ISO_DATE.match(text)
    if iso:
        try:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        except ValueError:
            return None

    lowered = strip_accents(text.lower())
    lowered = re.sub(r"\bde\b", " ", lowered)
    for pt, en in _PT_MONTHS.items():
        lowered = re.sub(rf"\b{pt}\b", en, lowered)

    for candidate in (lowered, text):
        try:
            parsed = dateparser.parse(candidate, dayfirst=dayfirst)
        except (ValueError, OverflowError, TypeError):
            continue
        if parsed is not None:
            return parsed.date()
    return None


_TRUE = {"true", "1", "yes", "y", "sim", "s", "verdadeiro", "urgente"}
_FALSE = {"false", "0", "no", "n", "nao", "não", "falso", "rotina"}


def parse_bool(value: object) -> bool | None:
    """Parse the many ways a VLM says yes. ``None`` when it is not a boolean."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = strip_accents(str(value).strip().lower())
    if text in {strip_accents(t) for t in _TRUE}:
        return True
    if text in {strip_accents(t) for t in _FALSE}:
        return False
    return None


def parse_number(value: object) -> float | None:
    """Parse a number written the pt-BR way (``1.234,56``) or the en way."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"[^\d,.\-]", "", text)
    if "," in text and "." in text:
        # Whichever separator comes last is the decimal one.
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


#: Accent-stripped, lowercased strings a model writes instead of omitting a
#: field. Counting these as values would inflate the hallucination rate with
#: what is really an abstention.
_EMPTY_STANDINS = frozenset(
    {
        "",
        "null",
        "none",
        "nil",
        "n/a",
        "na",
        "-",
        "--",
        "nao informado",
        "nao consta",
        "ilegivel",
        "not available",
        "unknown",
    }
)


def is_empty(value: object) -> bool:
    """Empty means: ``None``, empty/whitespace string, empty collection.

    Models also like to write ``"null"``, ``"n/a"`` or ``"-"`` instead of
    omitting a field. Those count as empty; treating them as values would
    inflate the hallucination rate with what is really an abstention.
    """
    if value is None:
        return True
    if isinstance(value, str):
        cleaned = strip_accents(value.strip().lower())
        return cleaned in _EMPTY_STANDINS
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False
