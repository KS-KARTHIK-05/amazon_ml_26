"""Candidate generation (blocking) over the normalized master table.

Per country (open set), the union of two sources:

A. IDF-weighted key overlap, computed with Polars joins (no per-row Python).
   Keys per record:
     n:<core-name token>   a:<house/unit number>
     c:<whole address component>   w:<address word, 4+ letters>
   Keys shared by more than `max_df` candidates carry no signal and are
   dropped; each S1 record uses at most `keys_per_s1` of its rarest keys.
   A candidate's score is the summed IDF of the keys it shares with the S1
   record; the top `top_k` per S1 are kept.

B. Character 3-gram TF-IDF cosine on the core name, top `top_n` per S1 above
   `min_sim`, via sparse_dot_topn (computes per-row top-n without ever
   materializing the full product -- the naive sparse matmul did, and OOM'd).
   Catches typo'd names that share no exact token.

Output rows: s1_id, cand_id, key_cos, name_score, addr_score, key_shared,
name_sim (null when a source did not propose the pair). Nothing here uses
ground truth.
"""

from __future__ import annotations

import gc
import time

import numpy as np
import polars as pl
from sparse_dot_topn import sp_matmul_topn

from src.blocking.key_blocker import ADDRESS_STOPWORDS
from src.blocking.ngram_tfidf import NgramTfidf

DEFAULTS = dict(
    top_k=30,          # by key cosine
    top_k_family=15,   # by name-key score, and by address-key score
    max_df=5000,
    keys_per_s1=24,
    top_n=20,          # name tf-idf neighbours
    min_sim=0.45,
    tfidf_max_df=0.005,
    use_tfidf=1,
    s1_batch=10_000,   # key-overlap join batch: rows ~ batch x keys x posting length (100k OOM'd at 27GB)
    tfidf_batch=100_000,
    n_threads=16,
)


# --------------------------------------------------------------------------
# A. key overlap

FAM = {"n": 0, "a": 1, "c": 2, "w": 3, "x": 4}


def record_keys(df: pl.DataFrame, conjunctions: bool = True) -> pl.DataFrame:
    """Long table (rid, key u64, fam u8) of distinct blocking keys per record.
    fam: n=name token, a=address number, c=address component, w=address word,
    x=name token x locality word conjunction."""
    k = pl.col("k")
    stop = sorted(ADDRESS_STOPWORDS)
    words = pl.col("addr_norm").str.extract_all(r"[a-z]{4,}").list.eval(
        pl.element().filter(~pl.element().is_in(stop))).list.unique()
    names = pl.col("name_core").str.split(" ").list.eval(
        pl.element().filter(pl.element().str.len_chars() >= 2)).list.unique(maintain_order=True)
    base = df.select("rid", names.alias("nt"), words.alias("wd"), "addr_numbers",
                     pl.col("addr_components").list.eval(pl.element().filter(pl.element().str.len_chars() >= 3)).alias("ac"))

    def fam(frame: pl.DataFrame, col: str, tag: str) -> pl.DataFrame:
        return (
            frame.select("rid", pl.col(col).alias("k")).explode("k").drop_nulls()
            .select("rid", (pl.lit(tag + ":") + k).hash(seed=7).alias("key"), pl.lit(FAM[tag], dtype=pl.UInt8).alias("fam"))
        )

    parts = [fam(base, "nt", "n"), fam(base, "addr_numbers", "a"), fam(base, "ac", "c"), fam(base, "wd", "w")]
    if conjunctions:
        # name token x locality word: rare even when both halves are common
        # ("sri balaji traders" in a busy building). Bounded to the first 3
        # name tokens x 4 words of the last 3 address components (city /
        # district / state area) -- the full cross product OOM'd on 6M rows.
        loc = df.select(
            "rid",
            pl.col("name_core").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 3))
            .list.unique(maintain_order=True).list.head(3).alias("nt"),
            pl.col("addr_components").list.tail(3).list.join(" ").str.extract_all(r"[a-z]{4,}")
            .list.eval(pl.element().filter(~pl.element().is_in(stop))).list.unique(maintain_order=True).list.head(4).alias("lw"),
        )
        x = (
            loc.explode("nt").drop_nulls("nt").explode("lw").drop_nulls("lw")
            .select("rid", pl.struct("nt", "lw").hash(seed=11).alias("key"), pl.lit(FAM["x"], dtype=pl.UInt8).alias("fam"))
        )
        parts.append(x)
    return pl.concat(parts).unique(["rid", "key"])


def record_keys_chunked(df: pl.DataFrame, chunk: int = 1_000_000) -> pl.DataFrame:
    """record_keys in row chunks: the exploded string intermediates only ever
    exist for one chunk; what is kept is the compact (u32, u64, u8) table.
    Keys are per record, so chunking by row is exact."""
    parts = []
    for start in range(0, df.height, chunk):
        parts.append(record_keys(df.slice(start, chunk)))
        gc.collect()
    return pl.concat(parts, rechunk=True)


def _top_per_rid(df: pl.DataFrame, col: str, k: int) -> pl.DataFrame:
    return (
        df.filter(pl.col(col) > 0)
        .sort(["rid", col], descending=[False, True])
        .group_by("rid", maintain_order=True).head(k)
        .select("rid", "crid")
    )


def key_overlap_candidates(s1: pl.DataFrame, cand: pl.DataFrame, top_k: int, top_k_family: int,
                           max_df: int, keys_per_s1: int, s1_batch: int, **_) -> pl.DataFrame:
    """Selection is the union of three rankings, so one key family can't crowd
    out the other: raw IDF sums favour records with long addresses sharing
    many generic street/city words (other businesses at the same building),
    which pushed true matches sharing the name + part of the address below
    the cutoff.
      - key_cos: cosine over all shared keys (length-normalized)  -> top_k
      - name_score: IDF sum of shared name keys                   -> top_k_family
      - addr_score: IDF sum of shared address keys                -> top_k_family
      - conj_score: IDF sum of shared name x address-word keys     -> top_k_family
    """
    s1_keys = record_keys_chunked(s1)
    cand_keys = record_keys_chunked(cand)

    n_cand = cand.height
    vocab = (
        cand_keys.group_by("key").agg(pl.len().alias("df"), pl.col("fam").first())
        .filter(pl.col("df") <= max_df)
        .with_row_index("kid")
        .with_columns(np.log((n_cand + 1) / (pl.col("df") + 1)).cast(pl.Float32).alias("idf"))
    )
    cand_k = cand_keys.join(vocab.select("key", "kid", "idf"), on="key").select("kid", pl.col("rid").alias("crid"), "idf")
    cand_norm = cand_k.group_by("crid").agg((pl.col("idf") ** 2).sum().sqrt().alias("cnorm"))
    cand_k = cand_k.select("kid", "crid")
    s1_k = (
        s1_keys.join(vocab.select("key", "kid", "df", "idf", pl.col("fam").alias("vfam")), on="key")
        .sort(["rid", "df"])
        .group_by("rid", maintain_order=True).head(keys_per_s1)
        .select("rid", "kid", "idf", pl.col("vfam").alias("fam"))
    )
    s1_norm = s1_k.group_by("rid").agg((pl.col("idf") ** 2).sum().sqrt().alias("snorm"))

    del s1_keys, cand_keys
    gc.collect()
    out = []
    rids = s1["rid"]
    n_batches = (s1.height + s1_batch - 1) // s1_batch
    t_start = time.time()
    for bi, start in enumerate(range(0, s1.height, s1_batch)):
        if n_batches > 5 and (bi % max(n_batches // 20, 1) == 0 or bi == n_batches - 1):
            el = time.time() - t_start
            eta = el / max(bi, 1) * (n_batches - bi)
            print(f"      key-overlap batch {bi + 1}/{n_batches}  elapsed {el:.0f}s  eta {eta:.0f}s", flush=True)
        lo, hi = rids[start], rids[min(start + s1_batch, s1.height) - 1]
        batch = s1_k.filter(pl.col("rid").is_between(lo, hi))
        agg = (
            batch.join(cand_k, on="kid")
            .group_by(["rid", "crid"])
            .agg(
                (pl.col("idf") ** 2).sum().alias("dot"),
                pl.col("idf").filter(pl.col("fam") == FAM["n"]).sum().alias("name_score"),
                pl.col("idf").filter(pl.col("fam").is_in([FAM["a"], FAM["c"], FAM["w"]])).sum().alias("addr_score"),
                pl.col("idf").filter(pl.col("fam") == FAM["x"]).sum().alias("conj_score"),
                pl.len().cast(pl.UInt16).alias("key_shared"),
            )
            .join(s1_norm, on="rid")
            .join(cand_norm, on="crid")
            .with_columns((pl.col("dot") / (pl.col("snorm") * pl.col("cnorm"))).cast(pl.Float32).alias("key_cos"))
            .drop("dot", "snorm", "cnorm")
        )
        keep = pl.concat([
            _top_per_rid(agg, "key_cos", top_k),
            _top_per_rid(agg, "name_score", top_k_family),
            _top_per_rid(agg, "addr_score", top_k_family),
            _top_per_rid(agg, "conj_score", top_k_family),
        ]).unique()
        out.append(agg.join(keep, on=["rid", "crid"], how="semi"))
    return pl.concat(out)


# --------------------------------------------------------------------------
# B. char n-gram TF-IDF on the core name

def name_tfidf_candidates(s1: pl.DataFrame, cand: pl.DataFrame, top_n: int, min_sim: float,
                          tfidf_max_df: float, n_threads: int, tfidf_batch: int, **_) -> pl.DataFrame:
    s1_batch = tfidf_batch
    vec = NgramTfidf(n=3, min_df=2, max_df=tfidf_max_df).fit(cand["name_core"])
    b_t = vec.transform(cand["name_core"]).T.tocsr()

    out = []
    s1_rids = s1["rid"].to_numpy()
    for start in range(0, s1.height, s1_batch):
        a = vec.transform(s1["name_core"].slice(start, s1_batch))
        c = sp_matmul_topn(a, b_t, top_n=top_n, threshold=min_sim, n_threads=n_threads).tocoo()
        out.append(pl.DataFrame({
            "rid": s1_rids[start + c.row].astype(np.uint32),
            "crid": c.col.astype(np.uint32),
            "name_sim": c.data.astype(np.float32),
        }))
    return pl.concat(out)


# --------------------------------------------------------------------------

def block_partition(s1: pl.DataFrame, cand: pl.DataFrame, params: dict | None = None,
                    verbose: bool = True) -> pl.DataFrame:
    """s1 / cand: normalized rows of ONE country. Returns candidate pairs by entity id."""
    p = {**DEFAULTS, **(params or {})}
    s1 = s1.with_row_index("rid")
    cand = cand.with_row_index("rid")

    t0 = time.time()
    a = key_overlap_candidates(s1, cand, **p)
    t1 = time.time()
    if p["use_tfidf"]:
        b = name_tfidf_candidates(s1, cand, **p)
        pairs = a.join(b, on=["rid", "crid"], how="full", coalesce=True)
    else:
        b = pl.DataFrame(schema={"rid": pl.UInt32, "crid": pl.UInt32, "name_sim": pl.Float32})
        pairs = a.with_columns(pl.lit(None, dtype=pl.Float32).alias("name_sim"))
    t2 = time.time()
    pairs = (
        pairs.join(s1.select("rid", pl.col("entity_id").alias("s1_id")), on="rid")
        .join(cand.select(pl.col("rid").alias("crid"), pl.col("entity_id").alias("cand_id")), on="crid")
        .select("s1_id", "cand_id", "key_cos", "name_score", "addr_score", "conj_score", "key_shared", "name_sim")
    )
    if verbose:
        print(f"    keys: {a.height} pairs in {t1 - t0:.1f}s | name tf-idf: {b.height} pairs in {t2 - t1:.1f}s "
              f"| union: {pairs.height} ({pairs.height / max(s1.height, 1):.1f}/S1)")
    return pairs
