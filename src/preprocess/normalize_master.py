"""Country-aware normalization of the master table, vectorized in Polars.

Order matters:
1. strip invisible chars (ZWJ/ZWNJ...), transliterate Indic words with the
   learned table (translit.py) -- must happen before accent folding, since
   NFKD + mark removal would otherwise damage Indic vowel signs
2. NFKC, lowercase, fold Latin accents (NFKD minus U+0300-036F)
3. names: drop "(India)"/"(France)" tags, strip web prefixes/TLDs, undo
   letter-digit swaps inside words (m0tors -> motors), punctuation -> space,
   join runs of single letters (n k -> nk, l l c -> llc), expand
   abbreviations, derive name_core (no legal forms/honorifics/filler)
4. addresses: split on commas into components; per component remove c/o,
   ordinals and leading zeros, punctuation -> space, expand abbreviations with
   the country's map (single-token state codes protected), map whole
   components (US state names -> codes, India state codes -> names); drop
   null markers

Countries are processed per label found in the data (open set): US / India /
France get their own maps, anything else gets the generic map.

Usage: python3 -m src.preprocess.normalize_master train|test
"""

from __future__ import annotations

import re
import sys
import time

import polars as pl

from src.data.load import CACHE_ROOT
from src.preprocess import lexicons as L
from src.preprocess.translit import INDIC_RE, INVISIBLE_RE, TABLE_PATH

_SPECIAL = {
    "ß": "ss", "æ": "ae", "œ": "oe", "ø": "o", "ł": "l", "đ": "d", "þ": "th",
    "ı": "i", "’": "'", "‘": "'", "`": "'", "´": "'", "ʼ": "'", "º": "o", "ª": "a",
}
_LEET = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t"}
_SINGLE_RUN_RE = r"(^| )\p{L}( \p{L})+( |$)"
_UNIT_WORDS = {"apt", "apartment", "ste", "suite", "unit", "fl", "floor", "bldg", "building", "rm", "room"}


def _translit_mapping() -> tuple[list[str], list[str]]:
    table = pl.read_csv(TABLE_PATH, separator="\t")
    table = table.with_columns(pl.col("indic").str.len_chars().alias("n")).sort("n", descending=True)
    return table["indic"].to_list(), [f" {x} " for x in table["latin"].to_list()]


def _base_clean(col: str, indic_pats: list[str], indic_repl: list[str]) -> pl.Expr:
    e = pl.col(col).fill_null("").str.replace_all(INVISIBLE_RE, "")
    e = pl.when(e.str.contains(INDIC_RE)).then(e.str.replace_many(indic_pats, indic_repl, leftmost=True)).otherwise(e)
    e = e.str.normalize("NFKC").str.to_lowercase()
    e = e.str.replace_many(list(_SPECIAL), list(_SPECIAL.values()))
    e = e.str.normalize("NFKD").str.replace_all(r"[̀-ͯ]", "")
    return e


def _join_single_letter_runs(s: str) -> str:
    out, buf = [], []
    for tok in s.split(" "):
        if len(tok) == 1 and tok.isalpha():
            buf.append(tok)
            continue
        out.append("".join(buf)) if len(buf) > 1 else out.extend(buf)
        buf = []
        out.append(tok)
    out.append("".join(buf)) if len(buf) > 1 else out.extend(buf)
    return " ".join(t for t in out if t)


def _apply_single_runs(df: pl.DataFrame, col: str) -> pl.DataFrame:
    mask = df[col].str.contains(_SINGLE_RUN_RE)
    if not mask.any():
        return df
    fixed = df.filter(mask).with_columns(
        pl.col(col).map_elements(_join_single_letter_runs, return_dtype=pl.Utf8)
    )
    return pl.concat([df.filter(~mask), fixed]).sort("_row")


def _squash(e: pl.Expr) -> pl.Expr:
    return e.str.replace_all(r"\s+", " ").str.strip_chars()


def normalize_names(df: pl.DataFrame, country: str, pats, repl) -> pl.DataFrame:
    amp = L.AMPERSAND.get(country, " and ")
    apostrophe = " " if country == "france" else ""
    e = _base_clean("business_name", pats, repl)
    e = e.str.replace_all(r"\((india|france|usa|us|u\.s\.a\.?)\)", " ")
    e = e.str.replace_all(r"https?://|www\.", " ")
    e = e.str.replace_all(r"\.(com|net|org|co\.in|in|co|fr|biz|info|us|io)\b", " ")
    for d, ch in _LEET.items():
        e = e.str.replace_all(rf"(\p{{L}}){d}(\p{{L}})", f"${{1}}{ch}${{2}}")
    e = e.str.replace_all(r"(\p{L}{2})5\b", "${1}s").str.replace_all(r"(\p{L}{2})0\b", "${1}o")  # wing5 -> wings
    e = e.str.replace_all(r"[&+]", amp).str.replace_all("'", apostrophe)
    e = _squash(e.str.replace_all(r"[^\p{L}\p{N}\s]", " "))
    df = df.with_columns(e.alias("name_norm"))
    df = _apply_single_runs(df, "name_norm")

    drop = sorted(L.NAME_CORE_DROP | {country})
    tokens = pl.col("name_norm").str.split(" ").list.eval(
        pl.element().replace(L.NAME_TOKEN_MAP).filter(pl.element() != "")
    )
    df = df.with_columns(tokens.alias("_ntok"))
    core = pl.col("_ntok").list.eval(pl.element().filter(~pl.element().is_in(drop)))
    return df.with_columns(
        pl.col("_ntok").list.join(" ").alias("name_norm"),
        pl.when(core.list.len() > 0).then(core).otherwise(pl.col("_ntok")).list.join(" ").alias("name_core"),
    ).drop("_ntok")


def normalize_addresses(df: pl.DataFrame, country: str, pats, repl) -> pl.DataFrame:
    token_map = L.ADDRESS_TOKEN_MAPS.get(country, L.GENERIC_ADDRESS_TOKEN_MAP)
    comp_map = L.COMPONENT_MAPS.get(country, {})
    codes = sorted(L.STATE_CODES.get(country, set()))
    apostrophe = " " if country == "france" else ""

    e = _base_clean("business_address", pats, repl)
    e = e.str.replace_all(r"\bc\s*/\s*o\b", " ")
    e = e.str.replace_all(r"\bn\s*°", " ")  # French "N°33"
    e = e.str.replace_all(r"\bh\s*[./]?\s*no\b\.?", " house ")
    if country == "us":
        e = e.str.replace_all(r"\bu\s*\.?\s*s\b\.?", "us")
    e = e.str.replace_all("'", apostrophe)

    comps = (
        df.select("_row", e.str.split(",").alias("comp"))
        .explode("comp")
        .with_columns(pl.int_range(pl.len()).over("_row").alias("_pos"))
    )
    c = pl.col("comp")
    c = c.str.replace_all(r"(\d+)(st|nd|rd|th)\b", "${1}")
    c = c.str.replace_all(r"[^\p{L}\p{N}\s]", " ")
    c = c.str.replace_all(r"(\p{L})(\d)", "${1} ${2}").str.replace_all(r"(\d)(\p{L}{2,})", "${1} ${2}")
    c = c.str.replace_all(r"\b0+(\d)", "${1}")
    comps = comps.with_columns(
        _squash(c).str.split(" ").list.eval(
            pl.element().filter((pl.element() != "") & ~pl.element().is_in(sorted(L.ADDRESS_DROP_TOKENS)))
        ).alias("tok")
    )

    if country == "us":  # "st" is saint unless it ends the component or precedes a unit/number
        nxt = pl.element().shift(-1)
        saint = (pl.element() == "st") & nxt.is_not_null() & ~nxt.is_in(sorted(_UNIT_WORDS)) & ~nxt.str.contains(r"^\d")
        comps = comps.with_columns(
            pl.col("tok").list.eval(pl.when(saint).then(pl.lit("saint")).otherwise(pl.element()))
        )

    protected = (pl.col("tok").list.len() == 1) & pl.col("tok").list.first().is_in(codes)
    mapped = pl.col("tok").list.eval(pl.element().replace(token_map)).list.join(" ")
    comps = comps.with_columns(
        pl.when(protected).then(pl.col("tok").list.join(" ")).otherwise(mapped).alias("comp")
    )
    if comp_map:
        comps = comps.with_columns(pl.col("comp").replace(comp_map))
    comps = comps.filter(pl.col("comp") != "")

    grouped = comps.sort(["_row", "_pos"]).group_by("_row", maintain_order=True).agg(
        pl.col("comp").alias("addr_components")
    )
    df = df.join(grouped, on="_row", how="left").with_columns(
        pl.col("addr_components").fill_null(pl.lit([], dtype=pl.List(pl.Utf8)))
    )
    return df.with_columns(
        pl.col("addr_components").list.join(", ").alias("addr_norm"),
        pl.col("addr_components").list.last().fill_null("").alias("addr_last"),
        pl.col("addr_components").list.join(" ").str.extract_all(r"\d+").list.unique(maintain_order=True).alias("addr_numbers"),
    )


def normalize_split(split: str) -> pl.DataFrame:
    master = pl.read_parquet(CACHE_ROOT / split / "master.parquet")
    pats, repl = _translit_mapping()
    parts = []
    for country_label in sorted(master["country"].unique().to_list()):
        t0 = time.time()
        country = country_label.strip().lower()
        df = master.filter(pl.col("country") == country_label).with_row_index("_row")
        df = normalize_names(df, country, pats, repl)
        df = normalize_addresses(df, country, pats, repl)
        parts.append(df.drop("_row"))
        print(f"[{split}/{country_label}] {df.height} rows in {time.time() - t0:.1f}s")
    return pl.concat(parts)


def write_normalized(split: str) -> pl.DataFrame:
    out = normalize_split(split)
    out.write_parquet(CACHE_ROOT / split / "master_norm.parquet")
    out.with_columns(
        pl.col("addr_components").list.join(" | "),
        pl.col("addr_numbers").list.join(" "),
    ).write_csv(CACHE_ROOT / split / f"master_{split}_norm.tsv", separator="\t")
    return out


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    t0 = time.time()
    out = write_normalized(split)
    print(f"total {out.height} rows in {time.time() - t0:.1f}s")
    leftover = out.filter(pl.col("name_norm").str.contains(INDIC_RE) | pl.col("addr_norm").str.contains(INDIC_RE)).height
    print(f"rows still containing Indic script after normalization: {leftover}")
