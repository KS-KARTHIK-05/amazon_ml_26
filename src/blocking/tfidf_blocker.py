"""Baseline blocker: country-partitioned character n-gram TF-IDF + chunked
sparse cosine top-K.

This is deliberately unsupervised (no ground truth used in fitting) so the
exact same code fits a fresh vectorizer per split: train (to measure recall
against ground truth) and test (to produce the submitted candidate_pairs.tsv)
never share a vocabulary, which matters because France only exists in test.

Country grouping is dynamic (whatever distinct strings appear in that split),
never hardcoded to {US, India}, per the problem statement's requirement.

This also doubles as the main track's TF-IDF safety net (Phase 4): its output
candidate_pairs is unioned with the fine-tuned-embedding FAISS candidates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_NGRAM_RANGE = (3, 5)
DEFAULT_MAX_DF = 0.05
DEFAULT_MIN_DF = 2
DEFAULT_MAX_FEATURES = 300_000
DEFAULT_TOP_K = 30
DEFAULT_S1_CHUNK = 4_000


def fit_vectorizer(
    texts: list[str],
    ngram_range=DEFAULT_NGRAM_RANGE,
    max_df=DEFAULT_MAX_DF,
    min_df=DEFAULT_MIN_DF,
    max_features=DEFAULT_MAX_FEATURES,
) -> TfidfVectorizer:
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
        dtype=np.float32,
        lowercase=False,  # text is already normalized/lowercased upstream
    )
    vec.fit(texts)
    return vec


@dataclass
class TopKResult:
    # row i of `s1_ids` -> candidate_ids[indptr[i]:indptr[i+1]], scores likewise
    s1_ids: list[str]
    candidate_ids: np.ndarray  # object array of matched id strings, ragged via indptr
    scores: np.ndarray
    indptr: np.ndarray


def chunked_topk(
    s1_matrix: sp.csr_matrix,
    cand_matrix: sp.csr_matrix,
    cand_ids: np.ndarray,
    k: int = DEFAULT_TOP_K,
    chunk_size: int = DEFAULT_S1_CHUNK,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return (per-row candidate-id arrays, per-row score arrays), top-k by cosine.

    Both matrices must already be L2-normalized (TfidfVectorizer does this by
    default), so cosine similarity is just the dot product.
    """
    cand_matrix_t = cand_matrix.T.tocsr()
    n = s1_matrix.shape[0]
    out_ids: list[np.ndarray] = []
    out_scores: list[np.ndarray] = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sims = s1_matrix[start:end] @ cand_matrix_t  # sparse (chunk x n_cand)
        sims = sims.tocsr()
        for r in range(sims.shape[0]):
            row = sims.getrow(r)
            if row.nnz == 0:
                out_ids.append(np.empty(0, dtype=object))
                out_scores.append(np.empty(0, dtype=np.float32))
                continue
            data = row.data
            idx = row.indices
            if row.nnz > k:
                top = np.argpartition(-data, k - 1)[:k]
            else:
                top = np.arange(row.nnz)
            order = top[np.argsort(-data[top])]
            out_ids.append(cand_ids[idx[order]])
            out_scores.append(data[order].astype(np.float32))
    return out_ids, out_scores


def block_country_partition(
    s1_ids: np.ndarray,
    s1_texts: list[str],
    cand_ids: np.ndarray,
    cand_texts: list[str],
    k: int = DEFAULT_TOP_K,
    chunk_size: int = DEFAULT_S1_CHUNK,
    vectorizer: TfidfVectorizer | None = None,
) -> dict[str, list[str]]:
    """Fit (or reuse) a vectorizer on s1_texts+cand_texts and return top-k candidates per S1 id."""
    if len(s1_ids) == 0 or len(cand_ids) == 0:
        return {s1_id: [] for s1_id in s1_ids}

    if vectorizer is None:
        vectorizer = fit_vectorizer(s1_texts + cand_texts)

    s1_matrix = vectorizer.transform(s1_texts).tocsr()
    cand_matrix = vectorizer.transform(cand_texts).tocsr()

    ids_per_row, _scores_per_row = chunked_topk(s1_matrix, cand_matrix, cand_ids, k=k, chunk_size=chunk_size)
    return {s1_id: list(ids) for s1_id, ids in zip(s1_ids, ids_per_row)}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "..")
    from src.data.load import load_split
    from src.preprocess.normalize import build_record_text

    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
    country_filter = sys.argv[2] if len(sys.argv) > 2 else "US"

    print(f"Loading train split, sampling {n_sample} S1 rows from country={country_filter} ...")
    data = load_split("train")
    s1 = data["source1"].filter(pl.col("country") == country_filter)
    s2 = data["source2"].filter(pl.col("country") == country_filter)
    s3 = data["source3"].filter(pl.col("country") == country_filter)
    print(f"partition sizes: S1={s1.height} S2={s2.height} S3={s3.height}")

    s1 = s1.sample(n=min(n_sample, s1.height), seed=42, shuffle=True)

    t0 = time.time()
    s1_texts = [
        build_record_text(n, a, c, normalized=True)
        for n, a, c in zip(s1["business_name"], s1["business_address"], s1["country"])
    ]
    s2_texts = [
        build_record_text(n, a, c, normalized=True)
        for n, a, c in zip(s2["business_name"], s2["business_address"], s2["country"])
    ]
    s3_texts = [
        build_record_text(n, a, c, normalized=True)
        for n, a, c in zip(s3["business_name"], s3["business_address"], s3["country"])
    ]
    print(f"normalize {s1.height + s2.height + s3.height} texts: {time.time() - t0:.1f}s")

    cand_ids = np.concatenate([s2["entity_id"].to_numpy(), s3["entity_id"].to_numpy()])
    cand_texts = s2_texts + s3_texts

    t0 = time.time()
    result = block_country_partition(
        s1["entity_id"].to_numpy(), s1_texts, cand_ids, cand_texts, k=DEFAULT_TOP_K
    )
    dt = time.time() - t0
    n_cand_pool = len(cand_ids)
    print(
        f"blocked {len(result)} S1 rows against {n_cand_pool} candidates in {dt:.1f}s "
        f"({dt / len(result) * 1000:.2f} ms/row); "
        f"projected full-US-train ({s1.height and 1323633}): "
        f"{dt / len(result) * 1323633 / 60:.1f} min (fit cost not scaled)"
    )
    example_id = s1["entity_id"][0]
    print(f"example candidates for {example_id}: {result[example_id][:5]}")
