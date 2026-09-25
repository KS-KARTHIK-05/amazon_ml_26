"""Vectorized pair features for (S1, candidate) pairs from the blocker.

- blocking scores: key_cos, name_score, addr_score, conj_score, key_shared
- string similarity via rapidfuzz.process.cpdist (element-wise, multithreaded
  C++) on core name, full normalized name and address
- address number overlap, last-component (state/region) agreement
- per-S1 relative features: how this candidate compares with the other
  candidates of the same S1 (rank, gap to best) -- the strongest precision
  signal, since the matcher must pick the right records among look-alikes

No country one-hot: country is an open set (France only appears in test).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

REC_COLS = ["entity_id", "source", "name_norm", "name_core", "addr_norm", "addr_numbers", "addr_last"]

_STRING_FEATURES = [
    ("nc_ratio", fuzz.ratio, "name_core"),
    ("nc_tset", fuzz.token_set_ratio, "name_core"),
    ("nc_tsort", fuzz.token_sort_ratio, "name_core"),
    ("nc_partial", fuzz.partial_ratio, "name_core"),
    ("nc_jw", JaroWinkler.normalized_similarity, "name_core"),
    ("nn_ratio", fuzz.ratio, "name_norm"),
    ("ad_ratio", fuzz.ratio, "addr_norm"),
    ("ad_tset", fuzz.token_set_ratio, "addr_norm"),
    ("ad_partial", fuzz.partial_ratio, "addr_norm"),
]

FEATURES = [
    "key_cos", "name_score", "addr_score", "conj_score", "key_shared",
    *[f for f, _, _ in _STRING_FEATURES],
    "nc_jacc", "num_inter", "num_jacc", "first_num_eq", "s1_has_num", "c_has_num",
    "last_eq", "c_addr_empty", "s1_addr_empty", "nc_len_s1", "nc_len_c", "nc_len_diff", "c_is_s3",
    "n_cands", "rank_cos", "rank_tset", "gap_cos", "gap_tset", "gap_ad_tset", "rank_combo",
]


def _prefixed(rec: pl.DataFrame, prefix: str, id_name: str) -> pl.DataFrame:
    return rec.select(pl.col("entity_id").alias(id_name), *[pl.col(c).alias(prefix + c) for c in REC_COLS[1:]])


def build_features(pairs: pl.DataFrame, rec: pl.DataFrame, chunk: int = 4_000_000) -> pl.DataFrame:
    """pairs: s1_id, cand_id + blocking scores. rec: normalized records (REC_COLS)."""
    j = (
        pairs.join(_prefixed(rec, "s_", "s1_id"), on="s1_id")
        .join(_prefixed(rec, "c_", "cand_id"), on="cand_id")
    )

    parts = []
    for start in range(0, j.height, chunk):
        b = j.slice(start, chunk)
        cols = {}
        for name, scorer, field in _STRING_FEATURES:
            a, c = b[f"s_{field}"].to_list(), b[f"c_{field}"].to_list()
            cols[name] = cpdist(a, c, scorer=scorer, workers=-1, dtype=np.float32)
        parts.append(b.with_columns(pl.Series(k, v) for k, v in cols.items()))
    j = pl.concat(parts)

    toks = lambda c: pl.col(c).str.split(" ").list.unique()
    inter = pl.col("s_addr_numbers").list.set_intersection("c_addr_numbers").list.len()
    union = pl.col("s_addr_numbers").list.set_union("c_addr_numbers").list.len()
    j = j.with_columns(
        (toks("s_name_core").list.set_intersection(toks("c_name_core")).list.len()
         / toks("s_name_core").list.set_union(toks("c_name_core")).list.len()).cast(pl.Float32).alias("nc_jacc"),
        inter.cast(pl.Float32).alias("num_inter"),
        pl.when(union > 0).then(inter / union).otherwise(0.0).cast(pl.Float32).alias("num_jacc"),
        (pl.col("s_addr_numbers").list.first() == pl.col("c_addr_numbers").list.first()).fill_null(False).cast(pl.Float32).alias("first_num_eq"),
        (pl.col("s_addr_numbers").list.len() > 0).cast(pl.Float32).alias("s1_has_num"),
        (pl.col("c_addr_numbers").list.len() > 0).cast(pl.Float32).alias("c_has_num"),
        (pl.col("s_addr_last") == pl.col("c_addr_last")).cast(pl.Float32).alias("last_eq"),
        (pl.col("c_addr_norm") == "").cast(pl.Float32).alias("c_addr_empty"),
        (pl.col("s_addr_norm") == "").cast(pl.Float32).alias("s1_addr_empty"),
        pl.col("s_name_core").str.len_chars().cast(pl.Float32).alias("nc_len_s1"),
        pl.col("c_name_core").str.len_chars().cast(pl.Float32).alias("nc_len_c"),
        (pl.col("c_source") == "S3").cast(pl.Float32).alias("c_is_s3"),
        pl.col("key_cos").fill_null(0.0),
    ).with_columns((pl.col("nc_len_s1") - pl.col("nc_len_c")).abs().alias("nc_len_diff"))

    g = "s1_id"
    j = j.with_columns(
        pl.len().over(g).cast(pl.Float32).alias("n_cands"),
        pl.col("key_cos").rank("min", descending=True).over(g).cast(pl.Float32).alias("rank_cos"),
        pl.col("nc_tset").rank("min", descending=True).over(g).cast(pl.Float32).alias("rank_tset"),
        (pl.col("key_cos").max().over(g) - pl.col("key_cos")).alias("gap_cos"),
        (pl.col("nc_tset").max().over(g) - pl.col("nc_tset")).alias("gap_tset"),
        (pl.col("ad_tset").max().over(g) - pl.col("ad_tset")).alias("gap_ad_tset"),
        (pl.col("nc_tset") + pl.col("ad_tset")).rank("min", descending=True).over(g).cast(pl.Float32).alias("rank_combo"),
    )
    return j.select("s1_id", "cand_id", *[pl.col(f).cast(pl.Float32) for f in FEATURES])
