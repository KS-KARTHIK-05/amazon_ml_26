"""Key-overlap blocking as sparse matrix products (same result as
blocker.key_overlap_candidates, ~5-10x faster, less memory).

key_overlap_candidates joins every query key against every record sharing it
and sums per (query, record) in Polars -- hundreds of millions of rows per
10k-query batch. The same scores are products of sparse matrices over the key
vocabulary, and sparse_dot_topn computes each row's top-n of such a product
in multi-threaded C++ without materializing it:

    key_cos    = (Q_idf / |Q|) . (R_idf / |R|)^T      top_k
    name_score = Q_idf[name keys] . R_bin[name keys]^T  top_k_family
    addr_score = Q_idf[addr keys] . R_bin[addr keys]^T  top_k_family
    conj_score = Q_idf[conj keys] . R_bin[conj keys]^T  top_k_family

Q = queries (each keeps its `keys_per_query` rarest keys), R = the indexed
records; IDF and max_df are measured over R. Works in both directions:
forward (S1 -> S2/S3 records) and reverse (records -> S1).
"""

from __future__ import annotations

import gc

import numpy as np
import polars as pl
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

from src.blocking.blocker import FAM, record_keys_chunked

FAMILIES = {"name_score": [FAM["n"]], "addr_score": [FAM["a"], FAM["c"], FAM["w"]], "conj_score": [FAM["x"]]}
SCORE_COLS = ["key_cos", "name_score", "addr_score", "conj_score", "key_shared"]


def _csr(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, shape: tuple[int, int]) -> sp.csr_matrix:
    return sp.csr_matrix((vals.astype(np.float32), (rows, cols)), shape=shape)


def key_matrices(q: pl.DataFrame, idx: pl.DataFrame, max_df: int, keys_per_query: int):
    """Long key tables -> (vocab, query keys, index keys) with integer key ids."""
    ik = record_keys_chunked(idx)                               # rid, key, fam
    vocab = (
        ik.group_by("key").agg(pl.len().alias("df"), pl.col("fam").first())
        .filter(pl.col("df") <= max_df).with_row_index("kid")
        .with_columns(np.log((idx.height + 1) / (pl.col("df") + 1)).cast(pl.Float32).alias("idf"))
    )
    ik = ik.join(vocab.select("key", "kid", "idf", pl.col("fam").alias("vfam")), on="key").select(
        "rid", "kid", "idf", pl.col("vfam").alias("fam"))
    qk = (
        record_keys_chunked(q).join(vocab.select("key", "kid", "df", "idf", pl.col("fam").alias("vfam")), on="key")
        .sort(["rid", "df"]).group_by("rid", maintain_order=True).head(keys_per_query)
        .select("rid", "kid", "idf", pl.col("vfam").alias("fam"))
    )
    return vocab.select("kid", "fam"), qk, ik


def overlap_topk(q: pl.DataFrame, idx: pl.DataFrame, top_k: int, top_k_family: int, max_df: int,
                 keys_per_query: int, n_threads: int = 16, q_chunk: int = 200_000) -> pl.DataFrame:
    """q / idx must carry a dense `rid` (0..n-1). Returns rid (query), crid (index record)
    and the five scores for the union of the four top-k rankings."""
    vocab, qk, ik = key_matrices(q, idx, max_df, keys_per_query)
    n_q, n_i, n_k = q.height, idx.height, vocab.height
    fam_of_key = vocab.sort("kid")["fam"].to_numpy()

    def cols(df, c):
        return df[c].to_numpy()

    # index side: cosine-normalized idf, and binary
    i_rid, i_kid, i_idf = cols(ik, "rid"), cols(ik, "kid"), cols(ik, "idf")
    inorm = np.sqrt(np.bincount(i_rid, weights=i_idf.astype(np.float64) ** 2, minlength=n_i)).astype(np.float32)
    R_cos = _csr(i_rid, i_kid, i_idf / np.maximum(inorm[i_rid], 1e-12), (n_i, n_k)).T.tocsr()
    R_bin = {name: _csr(i_rid, i_kid, np.isin(fam_of_key[i_kid], f).astype(np.float32), (n_i, n_k)).T.tocsr()
             for name, f in FAMILIES.items()}
    for m in R_bin.values():
        m.eliminate_zeros()
    R_idf = _csr(i_rid, i_kid, i_idf, (n_i, n_k))          # for scoring selected pairs
    del ik, i_rid, i_kid, i_idf
    gc.collect()

    q_rid, q_kid, q_idf = cols(qk, "rid"), cols(qk, "kid"), cols(qk, "idf")
    qnorm = np.sqrt(np.bincount(q_rid, weights=q_idf.astype(np.float64) ** 2, minlength=n_q)).astype(np.float32)
    Q_idf = _csr(q_rid, q_kid, q_idf, (n_q, n_k))
    del qk
    gc.collect()

    fam_cols = {name: np.isin(fam_of_key, f) for name, f in FAMILIES.items()}
    out = []
    for lo in range(0, n_q, q_chunk):
        hi = min(lo + q_chunk, n_q)
        Qc = Q_idf[lo:hi]
        Qcos = sp.diags(1.0 / np.maximum(qnorm[lo:hi], 1e-12)).astype(np.float32) @ Qc
        sel = [sp_matmul_topn(Qcos.tocsr(), R_cos, top_n=top_k, threshold=1e-9, n_threads=n_threads).tocoo()]
        for name in FAMILIES:
            Qf = Qc @ sp.diags(fam_cols[name].astype(np.float32))
            Qf = Qf.tocsr()
            Qf.eliminate_zeros()
            sel.append(sp_matmul_topn(Qf, R_bin[name], top_n=top_k_family, threshold=1e-9, n_threads=n_threads).tocoo())
        r = np.concatenate([m.row for m in sel]).astype(np.int64) + lo
        c = np.concatenate([m.col for m in sel]).astype(np.int64)
        pairs = np.unique(r * n_i + c)
        out.append(_score_pairs(pairs // n_i, pairs % n_i, Q_idf, R_idf, qnorm, inorm, fam_cols))
        del sel, Qc, Qcos
        gc.collect()
    return pl.concat(out)


def _score_pairs(r: np.ndarray, c: np.ndarray, Q_idf, R_idf, qnorm, inorm, fam_cols, chunk: int = 2_000_000) -> pl.DataFrame:
    """Exact scores for given (query row, index row) pairs via row gathers."""
    res = []
    fam_mat = sp.csr_matrix(np.stack([fam_cols[n] for n in FAMILIES], axis=1).astype(np.float32))
    for lo in range(0, len(r), chunk):
        rr, cc = r[lo:lo + chunk], c[lo:lo + chunk]
        A = Q_idf[rr]
        B = R_idf[cc]
        B.data[:] = 1.0
        shared = A.multiply(B).tocsr()                      # idf of shared keys
        dot = np.asarray(shared.multiply(shared).sum(axis=1)).ravel()
        fam_sum = np.asarray((shared @ fam_mat).todense())
        res.append(pl.DataFrame({
            "rid": rr.astype(np.uint32), "crid": cc.astype(np.uint32),
            "key_cos": (dot / np.maximum(qnorm[rr] * inorm[cc], 1e-12)).astype(np.float32),
            "name_score": fam_sum[:, 0].astype(np.float32),
            "addr_score": fam_sum[:, 1].astype(np.float32),
            "conj_score": fam_sum[:, 2].astype(np.float32),
            "key_shared": np.diff(shared.indptr).astype(np.uint16),
        }))
    return pl.concat(res)
