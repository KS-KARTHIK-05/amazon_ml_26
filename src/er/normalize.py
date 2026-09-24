"""Raw-preserving multilingual normalization views.

Every function here derives an additional *view* of the source text; none of
them overwrite or discard the original string. Callers keep `raw_name` /
`raw_address` untouched alongside these derived fields, per
docs/IMPLEMENTATION_SPEC.md's "Data contracts" section.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from unidecode import unidecode

# Legal-form tokens to strip only from the *end* of a legal-reduced view, case-
# and accent-folded. Deliberately conservative: word-boundary matches only, so
# we don't eat a business named e.g. "Coco" by matching "Co".
_LEGAL_FORM_TOKENS = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd",
    "limited", "llc", "llp", "lp", "plc", "pvt", "private", "gmbh", "sa",
    "sarl", "bv", "ag", "pty", "pc", "group", "holdings", "enterprises",
    "enterprise", "industries", "international", "intl",
}

_NUMERIC_RE = re.compile(r"\d+")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)  # words, keeps non-ASCII scripts


def is_missing(text: str | None) -> bool:
    """True for None, empty string, or whitespace-only text."""
    return text is None or text.strip() == ""


def casefold_unicode(text: str) -> str:
    """NFKC-normalize then Unicode casefold (script-aware, not ASCII-only)."""
    if text is None:
        return ""
    return unicodedata.normalize("NFKC", text).casefold()


def fold_latin_accents(text: str) -> str:
    """ASCII transliteration (e.g. 'café' -> 'cafe'). Lossy by design; used as
    an additional alternate view, never as a replacement for the raw text."""
    if text is None:
        return ""
    return unidecode(text)


def tokenize(text: str) -> tuple[str, ...]:
    """Unicode-aware word tokens (letters/digits in any script), casefolded."""
    if text is None:
        return ()
    return tuple(_TOKEN_RE.findall(casefold_unicode(text)))


def token_set(text: str) -> frozenset[str]:
    return frozenset(tokenize(text))


def numeric_tokens(text: str) -> tuple[str, ...]:
    """Digit runs, e.g. house numbers / PIN codes / unit numbers."""
    if text is None:
        return ()
    return tuple(_NUMERIC_RE.findall(text))


def legal_form_reduced(text: str) -> str:
    """Casefolded, accent-folded token sequence with trailing legal-form
    tokens stripped. Repeated so "X Pvt Ltd" -> "x", not just "x pvt"."""
    tokens = list(tokenize(fold_latin_accents(text)) if text else ())
    while tokens and tokens[-1] in _LEGAL_FORM_TOKENS:
        tokens.pop()
    return " ".join(tokens)


def initials(text: str) -> str:
    """Compact initials view, e.g. 'International South Consultants' -> 'isc'."""
    tokens = tokenize(fold_latin_accents(text)) if text else ()
    return "".join(t[0] for t in tokens if t)


@dataclass(frozen=True)
class NormalizedText:
    raw: str
    missing: bool
    casefold: str
    ascii_fold: str
    tokens: tuple[str, ...]
    token_set: frozenset[str] = field(compare=False)
    numeric_tokens: tuple[str, ...]
    legal_reduced: str
    initials: str


def normalize_text(raw: str | None) -> NormalizedText:
    raw = raw if raw is not None else ""
    missing = is_missing(raw)
    toks = tokenize(raw)
    return NormalizedText(
        raw=raw,
        missing=missing,
        casefold=casefold_unicode(raw),
        ascii_fold=fold_latin_accents(raw),
        tokens=toks,
        token_set=frozenset(toks),
        numeric_tokens=numeric_tokens(raw),
        legal_reduced=legal_form_reduced(raw),
        initials=initials(raw),
    )


@dataclass(frozen=True)
class NormalizedAddress:
    raw: str
    missing: bool
    tokens: tuple[str, ...]
    token_set: frozenset[str] = field(compare=False)
    numeric_tokens: tuple[str, ...]


def normalize_address(raw: str | None) -> NormalizedAddress:
    raw = raw if raw is not None else ""
    toks = tokenize(raw)
    return NormalizedAddress(
        raw=raw,
        missing=is_missing(raw),
        tokens=toks,
        token_set=frozenset(toks),
        numeric_tokens=numeric_tokens(raw),
    )


def normalize_country(raw: str | None) -> str:
    """Country stays an open string; only casefold+strip for comparison."""
    if raw is None:
        return ""
    return raw.strip().casefold()
