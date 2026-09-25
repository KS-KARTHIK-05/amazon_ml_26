"""Baseline blocker: inverted-index blocking on cheap keys from name and
address, country-partitioned.

A flat character n-gram TF-IDF cosine over the full candidate pool is
infeasible at this scale: even with aggressive max_df filtering, common
n-grams ("ltd", "inc", street-name fragments) blow the sparse matmul up to
billions of nonzero entries for a few thousand query rows (measured; see
tfidf_blocker.py's benchmark failure). Real record-linkage systems handle
this with blocking keys instead of a full similarity matrix: group records by
a cheap key so only records sharing a key are ever compared, exactly what the
challenge's own video describes ("records group through similar name or
shared address"). No dense/sparse matrix ever spans the full candidate pool.

Two key families, unioned per query:
- address digit-tokens: any digit run in the address (house/building number,
  unit number, PIN/ZIP). Two records sharing a distinctive digit run very
  likely share a street address.
- distinctive name tokens: normalized-name tokens that aren't common legal /
  filler words. Two records sharing a distinctive word very likely name the
  same business.

A key whose posting list is larger than `max_posting` is dropped entirely
(the key blocking analogue of TF-IDF's max_df) since an overly common key
carries no discriminative signal and would blow up candidate pool size.
Within the union of surviving postings, a cheap token-Jaccard score picks the
final top-K -- computed only over the bounded per-query candidate pool, never
the full corpus.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict

import numpy as np
import polars as pl
from tqdm import tqdm

DEFAULT_MAX_POSTING = 6000
DEFAULT_TOP_K = 30
DEFAULT_MIN_TOKEN_LEN = 3

# Common business-name filler words: not useful as a blocking key on their own
# (huge posting lists) but still contribute to the Jaccard re-ranking score.
NAME_STOPWORDS = {
    "the", "and", "of", "for", "inc", "incorporated", "llc", "llp", "ltd",
    "limited", "private", "pvt", "company", "co", "corp", "corporation",
    "group", "services", "service", "international", "associates", "assoc",
    "enterprises", "enterprise", "solutions", "holdings", "plc", "trust",
    "partners", "partnership",
}

# Generic street-type / directional / unit words expanded by normalize_address's
# ADDRESS_ABBREVIATIONS -- ultra common, no discriminative value as a key, but
# still contribute to the Jaccard re-ranking score.
ADDRESS_STOPWORDS = {
    "road", "street", "avenue", "boulevard", "drive", "lane", "court",
    "circle", "highway", "parkway", "place", "square", "terrace",
    "apartment", "building", "floor", "suite", "number", "north", "south",
    "east", "west", "po", "unit", "near", "opp", "behind", "sector", "block",
}

_DIGIT_RE = re.compile(r"\d+")
_ALPHA_TOKEN_RE = re.compile(r"[a-z]+")


def name_tokens(norm_name: str) -> list[str]:
    return [t for t in norm_name.split(" ") if t]


def distinctive_name_tokens(norm_name: str, min_len: int = DEFAULT_MIN_TOKEN_LEN) -> list[str]:
    return [t for t in name_tokens(norm_name) if len(t) >= min_len and t not in NAME_STOPWORDS]


def address_digit_tokens(norm_address: str) -> list[str]:
    return [d for d in _DIGIT_RE.findall(norm_address) if len(d) >= 2]


def address_alpha_tokens(norm_address: str, min_len: int = 4) -> list[str]:
    """Locality/street-name words from the address -- city, state, street name,
    landmark -- excluding generic street-type/directional filler words. This is
    what lets two records match when they share a street or city name but no
    house number survived (missing/omitted, per the EDA's address-missing
    category), or vice versa lets digit tokens carry a match when the text
    otherwise diverges."""
    return [
        t
        for t in _ALPHA_TOKEN_RE.findall(norm_address)
        if len(t) >= min_len and t not in ADDRESS_STOPWORDS
    ]


def address_tokens(norm_address: str) -> list[str]:
    """Digit runs and distinctive words combined -- one key family for the address."""
    return address_digit_tokens(norm_address) + address_alpha_tokens(norm_address)


def build_inverted_index(
    keys_per_row: list[list[str]], max_posting: int = DEFAULT_MAX_POSTING
) -> dict[str, np.ndarray]:
    postings: dict[str, list[int]] = defaultdict(list)
    for row_idx, keys in enumerate(keys_per_row):
        for k in set(keys):  # dedupe within a row so a repeated token isn't double-listed
            postings[k].append(row_idx)
    return {
        k: np.asarray(v, dtype=np.int32)
        for k, v in postings.items()
        if len(v) <= max_posting
    }


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def block_one(
    query_tokens: set[str],
    query_addr_tokens: set[str],
    name_index: dict[str, np.ndarray],
    addr_index: dict[str, np.ndarray],
    cand_tokens: list[set[str]],
    cand_addr_tokens: list[set[str]],
    k: int,
) -> list[int]:
    pool: set[int] = set()
    for t in query_tokens:
        arr = name_index.get(t)
        if arr is not None:
            pool.update(arr.tolist())
    for d in query_addr_tokens:
        arr = addr_index.get(d)
        if arr is not None:
            pool.update(arr.tolist())

    if not pool:
        return []
    if len(pool) <= k:
        return list(pool)

    scored = []
    for idx in pool:
        score = _jaccard(query_tokens, cand_tokens[idx]) + _jaccard(query_addr_tokens, cand_addr_tokens[idx])
        scored.append((score, idx))
    scored.sort(key=lambda x: -x[0])
    return [idx for _score, idx in scored[:k]]


_WORKER_STATE: dict = {}


def _worker_block_chunk(rows: list[tuple[int, str, str]]) -> list[tuple[int, list[int]]]:
    ws = _WORKER_STATE
    out = []
    for row_pos, name, addr in rows:
        q_tokens = set(distinctive_name_tokens(name))
        q_addr = set(address_tokens(addr))
        idxs = block_one(
            q_tokens, q_addr, ws["name_index"], ws["addr_index"], ws["cand_name_tokens"], ws["cand_addr_tokens"], ws["k"]
        )
        out.append((row_pos, idxs))
    return out


def block_country_partition(
    s1_ids: np.ndarray,
    s1_names_norm: list[str],
    s1_addrs_norm: list[str],
    cand_ids: np.ndarray,
    cand_names_norm: list[str],
    cand_addrs_norm: list[str],
    k: int = DEFAULT_TOP_K,
    max_posting: int = DEFAULT_MAX_POSTING,
    n_jobs: int = 1,
) -> dict[str, list[str]]:
    if len(s1_ids) == 0 or len(cand_ids) == 0:
        return {s1_id: [] for s1_id in s1_ids}

    t0 = time.time()
    cand_name_tokens = [set(distinctive_name_tokens(n)) for n in cand_names_norm]
    cand_addr_tokens = [set(address_tokens(a)) for a in cand_addrs_norm]

    name_index = build_inverted_index(
        [list(s) for s in cand_name_tokens], max_posting=max_posting
    )
    addr_index = build_inverted_index(
        [list(s) for s in cand_addr_tokens], max_posting=max_posting
    )
    t_index = time.time() - t0

    t0 = time.time()
    n = len(s1_ids)
    progress = tqdm(total=n, desc="blocking", unit="row", smoothing=0.1, mininterval=2.0)
    if n_jobs <= 1:
        out_idxs: list[list[int]] = []
        for name, addr in zip(s1_names_norm, s1_addrs_norm):
            q_tokens = set(distinctive_name_tokens(name))
            q_addr = set(address_tokens(addr))
            out_idxs.append(block_one(q_tokens, q_addr, name_index, addr_index, cand_name_tokens, cand_addr_tokens, k))
            progress.update(1)
    else:
        # fork-based multiprocessing: set globals BEFORE creating the Pool so
        # forked workers inherit the (large, read-only) index/candidate-token
        # lists via copy-on-write, with no per-task or per-worker pickling.
        global _WORKER_STATE
        _WORKER_STATE = dict(
            name_index=name_index,
            addr_index=addr_index,
            cand_name_tokens=cand_name_tokens,
            cand_addr_tokens=cand_addr_tokens,
            k=k,
        )
        import multiprocessing as mp

        rows = list(enumerate(zip(s1_names_norm, s1_addrs_norm)))
        rows = [(pos, name, addr) for pos, (name, addr) in rows]
        # Small chunks so the progress bar updates every ~15-20s instead of
        # every few minutes; IPC overhead per chunk is negligible next to the
        # per-row work at this size.
        chunk_size = max(100, min(2000, n // (n_jobs * 50)))
        chunks = [rows[i : i + chunk_size] for i in range(0, n, chunk_size)]

        out_idxs = [None] * n
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_jobs) as pool:
            for chunk_result in pool.imap_unordered(_worker_block_chunk, chunks):
                for pos, idxs in chunk_result:
                    out_idxs[pos] = idxs
                progress.update(len(chunk_result))
        _WORKER_STATE = {}
    progress.close()
    t_query = time.time() - t0

    out = {s1_id: [str(cand_ids[i]) for i in idxs] for s1_id, idxs in zip(s1_ids, out_idxs)}
    print(f"  [index build: {t_index:.1f}s, query loop: {t_query:.1f}s for {n} rows "
          f"({t_query / n * 1000:.2f} ms/row, n_jobs={n_jobs})]")
    return out


if __name__ == "__main__":
    import sys

    from src.data.load import load_split
    from src.preprocess.normalize import normalize_address, normalize_name

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
    s1_names = [normalize_name(n) for n in s1["business_name"]]
    s1_addrs = [normalize_address(a) for a in s1["business_address"]]
    s2_names = [normalize_name(n) for n in s2["business_name"]]
    s2_addrs = [normalize_address(a) for a in s2["business_address"]]
    s3_names = [normalize_name(n) for n in s3["business_name"]]
    s3_addrs = [normalize_address(a) for a in s3["business_address"]]
    print(f"normalize {s1.height + s2.height + s3.height} texts: {time.time() - t0:.1f}s")

    cand_ids = np.concatenate([s2["entity_id"].to_numpy(), s3["entity_id"].to_numpy()])
    cand_names = s2_names + s3_names
    cand_addrs = s2_addrs + s3_addrs

    t0 = time.time()
    result = block_country_partition(
        s1["entity_id"].to_numpy(), s1_names, s1_addrs, cand_ids, cand_names, cand_addrs, k=DEFAULT_TOP_K
    )
    dt = time.time() - t0
    n_cand_pool = len(cand_ids)
    n_empty = sum(1 for v in result.values() if not v)
    avg_cands = np.mean([len(v) for v in result.values()])
    print(
        f"blocked {len(result)} S1 rows against {n_cand_pool} candidates in {dt:.1f}s "
        f"({dt / len(result) * 1000:.2f} ms/row); empty={n_empty} ({n_empty / len(result):.1%}); "
        f"avg candidates/row={avg_cands:.1f}"
    )
    full_size = {"US": 1_323_633, "India": 883_188}.get(country_filter, s1.height)
    print(f"projected full {country_filter}-train S1 ({full_size}): {dt / len(result) * full_size / 60:.1f} min")
    example_id = s1["entity_id"][0]
    print(f"example candidates for {example_id}: {result[example_id][:5]}")
