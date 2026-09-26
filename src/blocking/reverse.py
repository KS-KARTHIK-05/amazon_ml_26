"""Reverse (record -> S1) blocking, and forward key scores for arbitrary pairs.

Forward blocking asks, for each S1, which S2/S3 records look like it. A record
whose name is common among S2/S3 ("golden solutions", often with an empty
address) is crowded out of every S1's top-k by dozens of look-alikes, even
when only one or two S1s in the country carry that name. Each S2/S3 record
belongs to at most one S1, so the reverse question -- which S1s does this
record look like, with IDF measured over the S1 pool -- ranks exactly those
records well. The union of both directions is the candidate set.

On held-out train entities, top-3 by key cosine + top-2 per key family in the
reverse direction recovers ~32% of the true matches forward blocking misses
(blocking F0.5 ceiling 0.985 -> 0.990 India, 0.988 -> 0.992 US).
"""

from __future__ import annotations

import gc

import numpy as np
import polars as pl

from src.blocking.blocker import FAM, record_keys_chunked
from src.blocking.sparse_blocker import overlap_topk

REV_DEFAULTS = dict(
    rev_top_k=3,          # S1s per record by reverse key cosine
    rev_top_k_family=2,   # ... and per key family (name / address / name x locality)
    rev_max_df=5000,      # keys shared by more S1s than this are dropped
    rev_keys=24,          # rarest keys used per query record
    n_threads=16,
)
REV_COLS = ["r_key_cos", "r_name_score", "r_addr_score", "r_conj_score", "r_rank"]


def reverse_candidates(s1: pl.DataFrame, cand: pl.DataFrame, params: dict | None = None) -> pl.DataFrame:
    """s1 / cand: normalized rows of ONE country. Returns s1_id, cand_id and the
    reverse scores; r_rank is the S1's rank among this record's selections."""
    p = {**REV_DEFAULTS, **(params or {})}
    q = cand.with_row_index("rid")
    idx = s1.with_row_index("rid")
    # same scores as blocker.key_overlap_candidates (verified pair-for-pair), as sparse products
    r = overlap_topk(q, idx, top_k=p["rev_top_k"], top_k_family=p["rev_top_k_family"],
                     max_df=p["rev_max_df"], keys_per_query=p["rev_keys"], n_threads=p["n_threads"])
    r = r.with_columns(pl.col("key_cos").rank("ordinal", descending=True).over("rid").cast(pl.Float32).alias("r_rank"))
    return (
        r.join(q.select("rid", pl.col("entity_id").alias("cand_id")), on="rid")
        .join(idx.select(pl.col("rid").alias("crid"), pl.col("entity_id").alias("s1_id")), on="crid")
        .select("s1_id", "cand_id",
                pl.col("key_cos").alias("r_key_cos"), pl.col("name_score").alias("r_name_score"),
                pl.col("addr_score").alias("r_addr_score"), pl.col("conj_score").alias("r_conj_score"), "r_rank")
    )


def forward_key_scores(pairs: pl.DataFrame, s1: pl.DataFrame, cand: pl.DataFrame, max_df: int,
                       keys_per_s1: int, s1_chunk: int = 100_000) -> pl.DataFrame:
    """Forward blocking scores (key_cos, name/addr/conj score, key_shared) for
    given (s1_id, cand_id) pairs, computed exactly as key_overlap_candidates
    does: IDF over all candidates of the country, each S1's rarest keys,
    candidate norms over all of its vocabulary keys. Pairs sharing no key get 0."""
    s1 = s1.join(pairs.select(pl.col("s1_id").alias("entity_id")).unique(), on="entity_id", how="semi").with_row_index("srid")
    cand = cand.with_row_index("crid")
    ck = record_keys_chunked(cand.rename({"crid": "rid"}))
    vocab = (
        ck.group_by("key").agg(pl.len().alias("df")).filter(pl.col("df") <= max_df)
        .with_columns(np.log((cand.height + 1) / (pl.col("df") + 1)).cast(pl.Float32).alias("idf"))
    )
    need = pairs.select("cand_id").unique().join(cand.select(pl.col("entity_id").alias("cand_id"), "crid"), on="cand_id")
    ck = ck.rename({"rid": "crid"}).join(need.select("crid"), on="crid", how="semi").join(vocab.select("key", "idf"), on="key")
    cnorm = ck.group_by("crid").agg((pl.col("idf") ** 2).sum().sqrt().alias("cnorm"))
    ck = ck.select("crid", "key")
    gc.collect()
    sk = (
        record_keys_chunked(s1.rename({"srid": "rid"})).join(vocab, on="key")
        .sort(["rid", "df"]).group_by("rid", maintain_order=True).head(keys_per_s1)
        .select(pl.col("rid").alias("srid"), "key", "idf", "fam")
    )
    snorm = sk.group_by("srid").agg((pl.col("idf") ** 2).sum().sqrt().alias("snorm"))
    ids = (
        pairs.select("s1_id", "cand_id")
        .join(s1.select(pl.col("entity_id").alias("s1_id"), "srid"), on="s1_id")
        .join(need, on="cand_id")
    )
    out = []
    for lo in range(0, s1.height, s1_chunk):
        b = ids.filter(pl.col("srid").is_between(lo, lo + s1_chunk - 1))
        agg = (
            b.select("srid", "crid").join(sk, on="srid").join(ck, on=["crid", "key"], how="semi")
            .group_by(["srid", "crid"]).agg(
                (pl.col("idf") ** 2).sum().alias("dot"),
                pl.col("idf").filter(pl.col("fam") == FAM["n"]).sum().alias("name_score"),
                pl.col("idf").filter(pl.col("fam").is_in([FAM["a"], FAM["c"], FAM["w"]])).sum().alias("addr_score"),
                pl.col("idf").filter(pl.col("fam") == FAM["x"]).sum().alias("conj_score"),
                pl.len().cast(pl.UInt16).alias("key_shared"),
            )
            .join(snorm, on="srid").join(cnorm, on="crid")
            .with_columns((pl.col("dot") / (pl.col("snorm") * pl.col("cnorm"))).cast(pl.Float32).alias("key_cos"))
        )
        cols = ["key_cos", "name_score", "addr_score", "conj_score", "key_shared"]
        out.append(b.join(agg.select("srid", "crid", *cols), on=["srid", "crid"], how="left")
                   .with_columns(pl.col(cols).fill_null(0)).select("s1_id", "cand_id", *cols))
    return pl.concat(out)
