"""Evaluate blocking on the held-out fold of train.

Queries = held-out S1 records; candidate pool = ALL S2/S3 records of the same
country (matched and unmatched), exactly as at test time. Reports, per
country and overall:
- pair_recall: share of true (S1, S2/S3) pairs present in the candidates
- full_recall: share of matched S1 entities with ALL true matches present
- cands_per_s1: mean candidate list length
- f05_ceiling: macro F0.5 a perfect matcher would reach on these candidates
  (singletons included, scoring 1.0) -- the cap blocking puts on the score
- same numbers for source A (key overlap) and B (name tf-idf) alone

Usage: python3 -m src.eval.blocking_recall [--sample N] [--param k=v ...]
"""

from __future__ import annotations

import argparse
import time

import polars as pl

from src.blocking.blocker import DEFAULTS, block_partition
from src.data.load import CACHE_ROOT

COLS = ["entity_id", "source", "country", "cluster_id", "fold", "name_core", "addr_norm",
        "addr_components", "addr_numbers"]


def _metrics(cands: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.DataFrame) -> dict:
    """cands: s1_id, cand_id. truth: s1_id, cand_id. s1_ids: all queried S1 ids."""
    hit = truth.join(cands.select("s1_id", "cand_id").unique(), on=["s1_id", "cand_id"], how="semi")
    per = (
        s1_ids.join(truth.group_by("s1_id").len().rename({"len": "n_true"}), on="s1_id", how="left")
        .join(hit.group_by("s1_id").len().rename({"len": "n_hit"}), on="s1_id", how="left")
        .fill_null(0)
        .with_columns(
            # perfect matcher predicts exactly the true matches that were blocked
            pl.when(pl.col("n_true") == 0).then(1.0)
            .when(pl.col("n_hit") == 0).then(0.0)
            .otherwise(
                (1.25 * (pl.col("n_hit") / pl.col("n_true")))
                / (0.25 + pl.col("n_hit") / pl.col("n_true"))
            ).alias("f05")
        )
    )
    matched = per.filter(pl.col("n_true") > 0)
    return {
        "pair_recall": round(hit.height / max(truth.height, 1), 4),
        "full_recall": round(float((matched["n_hit"] == matched["n_true"]).mean()), 4),
        "cands_per_s1": round(cands.height / max(s1_ids.height, 1), 1),
        "f05_ceiling": round(float(per["f05"].mean()), 4),
    }


def evaluate(sample: int | None, params: dict) -> pl.DataFrame:
    m = pl.read_parquet(CACHE_ROOT / "train" / "master_norm.parquet", columns=COLS)
    rows = []
    for country in sorted(m["country"].unique().to_list()):
        mc = m.filter(pl.col("country") == country)
        s1 = mc.filter((pl.col("source") == "S1") & (pl.col("fold") == "held_out"))
        if sample:
            s1 = s1.sample(min(sample, s1.height), seed=0)
        cand = mc.filter(pl.col("source") != "S1")
        truth = cand.filter(pl.col("cluster_id").is_in(s1["entity_id"].implode())).select(
            pl.col("cluster_id").alias("s1_id"), pl.col("entity_id").alias("cand_id")
        )
        print(f"[{country}] {s1.height} held-out S1 vs {cand.height} candidates, {truth.height} true pairs")
        t0 = time.time()
        pairs = block_partition(s1, cand, params)
        dt = time.time() - t0
        ids = s1.select(pl.col("entity_id").alias("s1_id"))
        for name, sub in [
            ("union", pairs),
            ("A_keys", pairs.filter(pl.col("key_cos").is_not_null())),
            ("B_name_tfidf", pairs.filter(pl.col("name_sim").is_not_null())),
        ]:
            rows.append({"country": country, "source": name, **_metrics(sub, truth, ids),
                         "sec": round(dt, 1) if name == "union" else None})
    out = pl.DataFrame(rows)
    pl.Config.set_tbl_rows(30)
    print(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--param", action="append", default=[], help="override, e.g. top_k=60")
    args = ap.parse_args()
    params = {}
    for kv in args.param:
        k, v = kv.split("=")
        params[k] = type(DEFAULTS[k])(float(v)) if isinstance(DEFAULTS[k], int) else type(DEFAULTS[k])(v)
    evaluate(args.sample, params)
