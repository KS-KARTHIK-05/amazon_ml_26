"""Indic-script -> Latin word table, learned from the training ground truth.

Why data-driven: S1 is 100% Latin, and ~35-40% of India S2/S3 rows carry
Indic script (9 scripts). There are only ~1.5k distinct Indic words across
train+test, and every test word also occurs in train, so the matched S1
records literally tell us how each word is spelled in English
("प्राइवेट" -> "private", "राम" -> "ram", "தமிழ்நாடு" -> "tamil nadu").
AI4Bharat IndicXlit would be the model alternative, but its fairseq
dependency does not build on Python 3.12.

Scoring for an Indic word w and a Latin candidate t (a unigram or bigram of
the matched S1 field -- name for name words, address for address words):

    assoc = (P(t | w) - P(t)) / (1 - P(t))
    score = assoc * (0.3 + 0.7 * sim(romanize(w), t))

- assoc: how much more often t shows up in the matched S1 field when w is
  present than it does at baseline. Raw P(t | w) alone crowns ubiquitous
  tokens ("private" is in ~60% of Indian company names, so it co-occurs with
  "फूड" 62% of the time purely by chance); assoc is ~0 for those and ~1 for
  the true spelling ("food").
- sim: similarity between t and a rule-based romanization of w
  (indic-transliteration, MIT); breaks ties phonetically (e.g. "उत्तर" always
  co-occurs with both "uttar" and "pradesh"). Weighted gently so genuine
  translations with little phonetic overlap ("পশ্চিমবঙ্গ" -> "west bengal")
  still win on assoc.

Words with no training evidence (only seen in unmatched records) fall back to
the cleaned rule-based romanization.

Output: cache/translit_table.tsv (one row per Indic word, human reviewable).
Usage:  python3 -m src.preprocess.translit
"""

from __future__ import annotations

import math
import re
import unicodedata

import polars as pl
from indic_transliteration import sanscript
from rapidfuzz import fuzz, process

from src.data.load import CACHE_ROOT

INDIC_RE = r"[ऀ-෿]"
INVISIBLE_RE = r"[​-‍﻿­]"
WORD_RE = r"[\p{L}\p{M}\p{N}]+"
TABLE_PATH = CACHE_ROOT / "translit_table.tsv"

MIN_P = 0.05  # ignore candidates that co-occur in <5% of a word's pairs
TOP_CANDIDATES = 40
FALLBACK_SNAP_MIN = 90  # snap only on near-identical phonetic skeletons; a wrong snap onto a real word (bakery->"bikaner") risks false matches, a raw romanization is harmless

# Unicode block -> sanscript scheme for the rule-based romanization anchor.
_SCRIPT_BLOCKS = [
    (0x0900, 0x097F, sanscript.DEVANAGARI, "devanagari"),
    (0x0980, 0x09FF, sanscript.BENGALI, "bengali"),
    (0x0A00, 0x0A7F, sanscript.GURMUKHI, "gurmukhi"),
    (0x0A80, 0x0AFF, sanscript.GUJARATI, "gujarati"),
    (0x0B00, 0x0B7F, sanscript.ORIYA, "oriya"),
    (0x0B80, 0x0BFF, sanscript.TAMIL, "tamil"),
    (0x0C00, 0x0C7F, sanscript.TELUGU, "telugu"),
    (0x0C80, 0x0CFF, sanscript.KANNADA, "kannada"),
    (0x0D00, 0x0D7F, sanscript.MALAYALAM, "malayalam"),
]


def detect_script(word: str) -> tuple[str | None, str]:
    for ch in word:
        cp = ord(ch)
        for lo, hi, scheme, name in _SCRIPT_BLOCKS:
            if lo <= cp <= hi:
                return scheme, name
    return None, "unknown"


def romanize(word: str) -> str:
    """Rule-based phonetic romanization, folded to plain lowercase ASCII."""
    scheme, _ = detect_script(word)
    if scheme is None:
        return ""
    iast = sanscript.transliterate(word, scheme, sanscript.IAST)
    folded = unicodedata.normalize("NFKD", iast)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return "".join(c for c in folded.lower() if c.isascii() and c.isalpha())


def _sim(roman: str, cand: str) -> float:
    a = roman
    b = cand.replace(" ", "")
    if not a or not b:
        return 0.0
    # Indic romanization keeps the inherent vowel ("rama", "gujarata"), so
    # also score against the variant with a trailing 'a' dropped.
    s = max(fuzz.ratio(a, b), fuzz.ratio(a.rstrip("a"), b) if len(a) > 3 else 0)
    return s / 100.0


def _tokens(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .str.replace_all(INVISIBLE_RE, "")
        .str.to_lowercase()
        .str.extract_all(WORD_RE)
    )


def _latin_unigrams_and_bigrams(col: str) -> pl.Expr:
    toks = pl.col(col).list.eval(pl.element().filter(~pl.element().str.contains(INDIC_RE)))
    bigrams = toks.list.eval(
        pl.concat_str([pl.element(), pl.element().shift(-1)], separator=" ").drop_nulls()
    )
    return pl.concat_list([toks, bigrams]).list.unique()


def _cooccurrence(pairs: pl.DataFrame, s1_col: str, m_col: str) -> pl.DataFrame:
    df = pairs.select(
        _tokens(m_col).alias("m_tok"),
        _tokens(s1_col).alias("s1_tok"),
    ).with_columns(
        pl.col("m_tok").list.eval(pl.element().filter(pl.element().str.contains(INDIC_RE))).list.unique(),
    )
    df = df.filter(pl.col("m_tok").list.len() > 0).with_columns(
        _latin_unigrams_and_bigrams("s1_tok").alias("cand")
    )
    n_total = df.height
    baseline = (
        df.select(pl.col("cand").explode())
        .drop_nulls()
        .group_by("cand")
        .len()
        .select("cand", (pl.col("len") / n_total).alias("p0"))
    )
    n_per_word = df.select(pl.col("m_tok").explode()).group_by("m_tok").len().rename({"len": "n_pairs"})
    co = (
        df.select("m_tok", "cand")
        .explode("m_tok")
        .explode("cand")
        .drop_nulls("cand")
        .group_by(["m_tok", "cand"])
        .len()
    )
    return (
        co.join(n_per_word, on="m_tok")
        .join(baseline, on="cand")
        .with_columns((pl.col("len") / pl.col("n_pairs")).alias("p"))
        .with_columns(
            ((pl.col("p") - pl.col("p0")) / (1 - pl.col("p0")).clip(lower_bound=1e-9)).alias("assoc")
        )
    )


def build_pairs() -> pl.DataFrame:
    master = pl.read_parquet(CACHE_ROOT / "train" / "master.parquet")
    s1 = master.filter(pl.col("source") == "S1").select(
        pl.col("entity_id").alias("cluster_id"),
        pl.col("business_name").alias("s1_name"),
        pl.col("business_address").alias("s1_addr"),
    )
    members = master.filter(
        (pl.col("source") != "S1")
        & pl.col("cluster_id").is_not_null()
        & (pl.col("business_name") + pl.col("business_address")).str.contains(INDIC_RE)
    ).select(
        "cluster_id",
        pl.col("business_name").alias("m_name"),
        pl.col("business_address").alias("m_addr"),
    )
    return members.join(s1, on="cluster_id")


def build_table() -> pl.DataFrame:
    pairs = build_pairs()
    print(f"training pairs with Indic text: {pairs.height}")

    co = pl.concat(
        [
            _cooccurrence(pairs, "s1_name", "m_name").with_columns(pl.lit("name").alias("field")),
            _cooccurrence(pairs, "s1_addr", "m_addr").with_columns(pl.lit("address").alias("field")),
        ]
    )
    co = (
        co.filter((pl.col("p") >= MIN_P) & (pl.col("assoc") > 0))
        .sort(["m_tok", "field", "assoc"], descending=[False, False, True])
        .group_by(["m_tok", "field"], maintain_order=True)
        .head(TOP_CANDIDATES)
    )

    rows = []
    for (word, field), grp in co.group_by(["m_tok", "field"], maintain_order=True):
        roman = romanize(word)
        _, script = detect_script(word)
        best = None
        for cand, p, assoc, n_pairs in zip(grp["cand"], grp["p"], grp["assoc"], grp["n_pairs"]):
            sim = _sim(roman, cand)
            score = assoc * (0.3 + 0.7 * sim)
            if best is None or score > best[0]:
                best = (score, cand, p, assoc, sim, n_pairs)
        score, cand, p, assoc, sim, n_pairs = best
        rows.append(
            dict(indic=word, field=field, script=script, latin=cand, romanized=roman,
                 method="aligned", score=round(score, 4), p=round(p, 4),
                 assoc=round(assoc, 4), sim=round(sim, 4), n_pairs=n_pairs)
        )

    table = pl.DataFrame(rows)
    # A word can appear in both names and addresses; keep its stronger field.
    table = table.sort("score", descending=True).unique(subset="indic", keep="first")

    # Words never seen in a matched pair (in train they only occur in
    # unmatched distractor records): romanize, then snap to the nearest real
    # word in the S1 name vocabulary ("ilektraniks" -> "electronics").
    vocab = full_vocabulary()
    missing = sorted(vocab - set(table["indic"].to_list()))
    s1_words, s1_freq = s1_name_vocabulary()
    s1_keys = [phonetic_key(w) for w in s1_words]
    fallback_rows = []
    for word in missing:
        roman = romanize(word)
        _, script = detect_script(word)
        cleaned = clean_fallback(roman)
        latin, sim, method = snap_to_vocab(cleaned, s1_words, s1_keys, s1_freq)
        fallback_rows.append(
            dict(indic=word, field="unknown", script=script, latin=latin,
                 romanized=roman, method=method, score=0.0, p=0.0, assoc=0.0, sim=round(sim, 4), n_pairs=0)
        )
    if fallback_rows:
        table = pl.concat([table, pl.DataFrame(fallback_rows, schema=table.schema)])
    return table.sort("n_pairs", descending=True)


def clean_fallback(roman: str) -> str:
    """Drop the inherent final vowel the rule-based scheme keeps ('rama' -> 'ram')."""
    return roman[:-1] if len(roman) > 3 and roman.endswith("a") else roman


_PHONETIC_RULES = [
    (re.compile(r"ph"), "f"),
    (re.compile(r"ck"), "k"),
    (re.compile(r"c(?=[eiy])"), "s"),
    (re.compile(r"c"), "k"),
    (re.compile(r"g(?=[ei])"), "j"),
    (re.compile(r"x"), "ks"),
    (re.compile(r"q"), "k"),
    (re.compile(r"w"), "v"),
    (re.compile(r"z"), "s"),
    (re.compile(r"m(?=[tdkg])"), "n"),  # anusvara before a stop is romanized as 'm'
    (re.compile(r"[aeiouyh]"), ""),
    (re.compile(r"(.)\1+"), r"\1"),
]


def phonetic_key(word: str) -> str:
    """Consonant skeleton that is robust to English-vs-Indic spelling of the
    same loanword: 'jewellers' and 'jvelars' both -> 'jvlrs'."""
    key = word.lower()
    for pattern, repl in _PHONETIC_RULES:
        key = pattern.sub(repl, key)
    return key or word


def snap_to_vocab(
    roman: str, words: list[str], keys: list[str], freq: list[int]
) -> tuple[str, float, str]:
    key = phonetic_key(roman)
    hits = process.extract(key, keys, scorer=fuzz.ratio, limit=15, score_cutoff=FALLBACK_SNAP_MIN)
    best = None
    for _key, key_score, idx in hits:
        combined = 0.7 * key_score + 0.3 * fuzz.ratio(roman, words[idx]) + 2 * math.log10(freq[idx])
        if best is None or combined > best[0]:
            best = (combined, idx, key_score)
    if best is None:
        return roman, 0.0, "fallback_raw"
    _, idx, key_score = best
    return words[idx], key_score / 100.0, "fallback_snapped"


def s1_name_vocabulary(min_count: int = 100) -> tuple[list[str], list[int]]:
    """Frequent Latin words from India S1 business names (train + test).
    India only, because Indic script only ever appears in India records --
    this keeps US surnames and French words out of the snap targets. A
    min_count keeps generic business vocabulary and drops rare look-alikes."""
    counts = []
    for split in ("train", "test"):
        m = pl.read_parquet(CACHE_ROOT / split / "master.parquet", columns=["source", "country", "business_name"])
        counts.append(
            m.filter((pl.col("source") == "S1") & (pl.col("country") == "India"))
            .select(pl.col("business_name").str.to_lowercase().str.extract_all(r"[a-z]{3,}").explode().alias("w"))
            .drop_nulls()
        )
    vc = pl.concat(counts).group_by("w").len().filter(pl.col("len") >= min_count)
    return vc["w"].to_list(), vc["len"].to_list()


def full_vocabulary() -> set[str]:
    """Every distinct Indic word (invisible chars stripped) in train + test."""
    vocab: set[str] = set()
    for split in ("train", "test"):
        m = pl.read_parquet(CACHE_ROOT / split / "master.parquet", columns=["business_name", "business_address"])
        for col in ("business_name", "business_address"):
            words = (
                m.select(
                    pl.col(col)
                    .filter(pl.col(col).str.contains(INDIC_RE))
                    .str.replace_all(INVISIBLE_RE, "")
                    .str.extract_all(WORD_RE)
                    .explode()
                )
                .drop_nulls()
                .unique()
                .filter(pl.col(col).str.contains(INDIC_RE))
            )
            vocab |= set(words[col].to_list())
    return vocab


if __name__ == "__main__":
    import time

    t0 = time.time()
    table = build_table()
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(TABLE_PATH, separator="\t")
    print(f"built {table.height} entries in {time.time() - t0:.1f}s -> {TABLE_PATH}")
    pl.Config.set_tbl_rows(60)
    pl.Config.set_fmt_str_lengths(30)
    print(table.group_by("method").len())
    print(table.head(40))
    print("lowest-scoring aligned entries (review these):")
    print(table.filter(pl.col("method") == "aligned").sort("score").head(30))
    fb = table.filter(pl.col("method").str.starts_with("fallback"))
    print("fallback entries grouped by chosen latin:")
    print(fb.group_by(["latin", "method"]).agg(pl.col("romanized").alias("romanized_variants")).sort("latin"))
