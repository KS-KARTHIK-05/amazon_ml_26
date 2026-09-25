"""Pairwise similarity features for (S1, candidate) pairs.

Cheap, vectorizable-per-row features used by the baseline classifier (Phase 3)
and reused later to blend with/calibrate the cross-encoder (Phase 5), per the
plan. Nothing here needs a GPU or a trained model.
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from src.blocking.key_blocker import address_tokens, distinctive_name_tokens
from src.preprocess.normalize import normalize_address, normalize_country, normalize_name

FEATURE_NAMES = [
    "name_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "addr_ratio",
    "addr_token_sort_ratio",
    "name_jaccard",
    "addr_jaccard",
    "addr_digit_shared",
    "country_match",
    "name_len_diff",
    "addr_len_diff",
    "name_has_text",
    "addr_has_text",
]


@dataclass
class RecordNorm:
    name: str
    address: str
    country: str
    name_tokens: frozenset
    addr_tokens: frozenset

    @classmethod
    def from_raw(cls, name: str, address: str, country: str) -> "RecordNorm":
        n = normalize_name(name)
        a = normalize_address(address)
        c = normalize_country(country)
        return cls(
            name=n,
            address=a,
            country=c,
            name_tokens=frozenset(distinctive_name_tokens(n)),
            addr_tokens=frozenset(address_tokens(a)),
        )


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def pair_features(r1: RecordNorm, r2: RecordNorm) -> list[float]:
    name_digit_1 = {t for t in r1.addr_tokens if t.isdigit()}
    name_digit_2 = {t for t in r2.addr_tokens if t.isdigit()}
    shared_digits = 1.0 if (name_digit_1 & name_digit_2) else 0.0

    return [
        fuzz.ratio(r1.name, r2.name) / 100.0,
        fuzz.token_sort_ratio(r1.name, r2.name) / 100.0,
        fuzz.token_set_ratio(r1.name, r2.name) / 100.0,
        fuzz.ratio(r1.address, r2.address) / 100.0,
        fuzz.token_sort_ratio(r1.address, r2.address) / 100.0,
        _jaccard(r1.name_tokens, r2.name_tokens),
        _jaccard(r1.addr_tokens, r2.addr_tokens),
        shared_digits,
        1.0 if r1.country == r2.country else 0.0,
        abs(len(r1.name) - len(r2.name)) / max(len(r1.name), len(r2.name), 1),
        abs(len(r1.address) - len(r2.address)) / max(len(r1.address), len(r2.address), 1),
        1.0 if r1.name else 0.0,
        1.0 if r1.address else 0.0,
    ]


if __name__ == "__main__":
    r1 = RecordNorm.from_raw("Premio Pharma LLP", "12 MG Road, Pune, Maharashtra", "India")
    r2 = RecordNorm.from_raw("PP", "12 M.G. Rd, Pune", "India")
    r3 = RecordNorm.from_raw("Premio Foods Pvt Ltd", "88 FC Road, Pune", "India")
    for name, feats in [("true match", pair_features(r1, r2)), ("look-alike", pair_features(r1, r3))]:
        print(name, dict(zip(FEATURE_NAMES, feats)))
