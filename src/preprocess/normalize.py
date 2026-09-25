"""Text normalization for business name/address matching.

Two views are produced for every record:
- raw: minimal whitespace cleanup only, original scripts/casing preserved.
  Kept because the cross-encoder/LLM stages benefit from original text.
- normalized: lowercased, Latin accents stripped (non-Latin scripts, e.g.
  Devanagari/Bengali/etc., are left completely untouched -- their combining
  vowel signs and viramas are not accent marks and must survive), legal-suffix
  and street abbreviations expanded, punctuation collapsed to spaces, leading
  zeros stripped from purely-numeric tokens.

Indian-script transliteration (Phase 6) is a separate, later step: this module
leaves non-Latin words as-is so the baseline blocker/matcher aren't blocked
on it.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Latin-only accent stripping, via a precomputed str.translate table so it's
# O(1) per character at runtime with no per-call unicodedata lookups. The
# ranges cover Latin-1 Supplement, Latin Extended-A/B, and Latin Extended
# Additional -- i.e. every accented Latin letter and ligature (é, ü, ñ, œ,
# the French ordinal indicator º, ...). Devanagari (U+0900+), Bengali
# (U+0980+), and the other Indian scripts are all far outside these ranges,
# so they pass through completely unchanged.
_ACCENT_RANGES = [(0x00A0, 0x02AF), (0x1E00, 0x1EFF)]


def _build_latin_accent_table() -> dict[int, str]:
    table: dict[int, str] = {}
    for lo, hi in _ACCENT_RANGES:
        for cp in range(lo, hi + 1):
            ch = chr(cp)
            decomposed = unicodedata.normalize("NFKD", ch)
            base = "".join(c for c in decomposed if not unicodedata.combining(c))
            if base and base != ch and all(ord(c) < 0x250 for c in base):
                table[cp] = base
    return table


_LATIN_ACCENT_TABLE = _build_latin_accent_table()


def strip_latin_accents(text: str) -> str:
    return text.translate(_LATIN_ACCENT_TABLE)


# ---------------------------------------------------------------------------
# Abbreviation expansion. Expand to one canonical long form so "Corp" and
# "Corporation" collapse to the same normalized token. Applied with word
# boundaries on whitespace-delimited tokens (case-insensitive, operates after
# lowercasing), stripping a trailing period first so "rd." and "rd" match
# the same rule.

NAME_ABBREVIATIONS = {
    "corp": "corporation",
    "inc": "incorporated",
    "ltd": "limited",
    "pvt": "private",
    "co": "company",
    "llc": "llc",
    "llp": "llp",
    "plc": "plc",
    "assoc": "associates",
    "assn": "association",
    "intl": "international",
    "mfg": "manufacturing",
    "svcs": "services",
    "svc": "service",
    "grp": "group",
    "&": "and",
}

ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "cir": "circle",
    "hwy": "highway",
    "pkwy": "parkway",
    "pl": "place",
    "sq": "square",
    "ter": "terrace",
    "apt": "apartment",
    "bldg": "building",
    "fl": "floor",
    "ste": "suite",
    "no": "number",
    "po": "po",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
}

_TOKEN_RE = re.compile(r"[^\W_]+|\W+", re.UNICODE)


def _expand_tokens(text: str, abbrev: dict[str, str]) -> str:
    out = []
    for tok in _TOKEN_RE.findall(text):
        key = tok.strip(".")
        if key and key in abbrev:
            out.append(abbrev[key])
        else:
            out.append(tok)
    return "".join(out)


def _build_punct_blank_table() -> dict[int, str]:
    """Blank out punctuation/symbol code points (Unicode category P*/S*, plus
    control chars) across the BMP, leaving every letter and mark -- crucially
    including Indian-script combining vowel signs and viramas (category Mn),
    which Python's regex ``\\w`` does NOT treat as word characters and would
    otherwise shred apart -- completely untouched. This is a one-time O(65536)
    build; str.translate() with it is then O(n) at C speed with no per-call
    unicodedata lookups.
    """
    table: dict[int, str] = {}
    for cp in range(0x0000, 0x10000):
        cat = unicodedata.category(chr(cp))[0]
        if cat in ("P", "S", "C"):
            table[cp] = " "
    return table


_PUNCT_BLANK_TABLE = _build_punct_blank_table()
_MULTI_SPACE = re.compile(r"\s+")
_LEADING_ZERO = re.compile(r"\b0+(\d)")


def clean_whitespace(text: str) -> str:
    return _MULTI_SPACE.sub(" ", text).strip()


def normalize_name(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower()
    text = strip_latin_accents(text)
    text = _expand_tokens(text, NAME_ABBREVIATIONS)
    text = text.translate(_PUNCT_BLANK_TABLE)
    text = clean_whitespace(text)
    return text


def normalize_address(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower()
    text = strip_latin_accents(text)
    text = _expand_tokens(text, ADDRESS_ABBREVIATIONS)
    text = text.translate(_PUNCT_BLANK_TABLE)
    text = _LEADING_ZERO.sub(r"\1", text)
    text = clean_whitespace(text)
    return text


def normalize_country(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return clean_whitespace(text).lower()


def clean_raw(text: str) -> str:
    """Minimal cleanup for the raw view: whitespace only, scripts/casing kept."""
    return clean_whitespace(unicodedata.normalize("NFKC", text or ""))


def build_record_text(name: str, address: str, country: str, normalized: bool) -> str:
    if normalized:
        name_t = normalize_name(name)
        addr_t = normalize_address(address)
        country_t = normalize_country(country)
    else:
        name_t = clean_raw(name)
        addr_t = clean_raw(address)
        country_t = clean_raw(country)
    return f"{name_t} | {addr_t} | {country_t}"


if __name__ == "__main__":
    examples = [
        ("Acme Robotics Incorporated", "12 MG Rd., Pune, Maharashtra", "India"),
        ("NK Infrastructure LLP", "NK-Infrastructure L.L.P.", "India"),
        ("Marina Ecole France Sarl", "63 R. de Dieppe, Lille, Hauts-de-France", "France"),
        ("SCI Ptit Àmicale", "18 Rue Jenzay, Dunkerque, Nord", "France"),
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi", "India"),
    ]
    for name, addr, country in examples:
        print("raw: ", build_record_text(name, addr, country, normalized=False))
        print("norm:", build_record_text(name, addr, country, normalized=True))
        print()
