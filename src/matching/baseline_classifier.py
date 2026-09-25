"""Baseline matching classifier: LightGBM over the pairwise features in
features.py, trained on baseline-blocker candidate pairs labeled from ground
truth.

Pipeline:
1. Explode candidates_baseline.parquet (source1_entity_id -> candidate_ids)
   into one row per (s1_id, candidate_id) pair, restricted to a fold (fit /
   held_out) via split.parquet.
2. Label a pair 1 if candidate_id is in that S1's ground-truth matched_ids.
3. Look up raw name/address/country for both sides and compute pair_features.
4. Train LightGBM on the FIT fold (optionally S1-subsampled for speed),
   evaluate on the FULL held-out fold: pick the probability threshold that
   maximizes held-out macro F0.5 (via eval.f05_score), after one-owner-per-
   candidate assignment (a candidate that clears threshold for multiple S1
   rows goes to the highest-scoring one only).
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl

from src.data.load import CACHE_ROOT, load_split
from src.data.split import load_or_make_split
from src.matching.features import FEATURE_NAMES, RecordNorm, pair_features

MODEL_DIR = CACHE_ROOT.parent / "models"


def _entity_lookup(*frames: pl.DataFrame) -> dict[str, tuple[str, str, str]]:
    """entity_id -> (business_name, business_address, country), across frames."""
    lookup: dict[str, tuple[str, str, str]] = {}
    for df in frames:
        for eid, name, addr, country in zip(
            df["entity_id"], df["business_name"], df["business_address"], df["country"]
        ):
            lookup[eid] = (name, addr, country)
    return lookup


def _explode_candidates(candidates: pl.DataFrame, s1_ids_allowed: set[str] | None) -> pl.DataFrame:
    df = candidates
    if s1_ids_allowed is not None:
        df = df.filter(pl.col("source1_entity_id").is_in(list(s1_ids_allowed)))
    df = df.filter(pl.col("candidate_ids").list.len() > 0)
    return df.explode("candidate_ids").rename({"candidate_ids": "candidate_entity_id"})


def build_pair_table(
    pairs: pl.DataFrame,
    entity_lookup: dict[str, tuple[str, str, str]],
    gt_lookup: dict[str, set[str]] | None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Returns (X features, y labels or None, s1_ids, cand_ids) for the given pairs."""
    n = pairs.height
    X = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.empty(n, dtype=np.int8) if gt_lookup is not None else None
    s1_ids = pairs["source1_entity_id"].to_list()
    cand_ids = pairs["candidate_entity_id"].to_list()

    # cache RecordNorm per entity id to avoid recomputing for repeated ids
    cache: dict[str, RecordNorm] = {}

    def get_norm(eid: str) -> RecordNorm:
        rn = cache.get(eid)
        if rn is None:
            name, addr, country = entity_lookup[eid]
            rn = RecordNorm.from_raw(name, addr, country)
            cache[eid] = rn
        return rn

    for i, (s1_id, cand_id) in enumerate(zip(s1_ids, cand_ids)):
        r1 = get_norm(s1_id)
        r2 = get_norm(cand_id)
        X[i] = pair_features(r1, r2)
        if y is not None:
            y[i] = 1 if cand_id in gt_lookup.get(s1_id, ()) else 0

    return X, y, s1_ids, cand_ids


def load_train_candidate_pairs(fold: str, s1_sample: int | None, seed: int = 42) -> pl.DataFrame:
    candidates = pl.read_parquet(CACHE_ROOT / "train" / "candidates_baseline.parquet")
    split = load_or_make_split()
    fold_ids = set(split.filter(pl.col("fold") == fold)["source1_entity_id"].to_list())

    if s1_sample is not None and len(fold_ids) > s1_sample:
        rng = np.random.default_rng(seed)
        fold_ids = set(rng.choice(sorted(fold_ids), size=s1_sample, replace=False).tolist())

    return _explode_candidates(candidates, fold_ids)


def train_and_evaluate(fit_sample_s1: int = 200_000, seed: int = 42) -> dict:
    data = load_split("train")
    entity_lookup = _entity_lookup(data["source1"], data["source2"], data["source3"])
    gt_lookup = {
        row["source1_entity_id"]: set(row["matched_ids"])
        for row in data["ground_truth"].iter_rows(named=True)
    }

    print("Building FIT pair table...")
    t0 = time.time()
    fit_pairs = load_train_candidate_pairs("fit", s1_sample=fit_sample_s1, seed=seed)
    X_fit, y_fit, _, _ = build_pair_table(fit_pairs, entity_lookup, gt_lookup)
    print(f"  {X_fit.shape[0]} pairs, {y_fit.mean():.3f} positive rate, {time.time() - t0:.1f}s")

    print("Building HELD_OUT pair table...")
    t0 = time.time()
    held_pairs = load_train_candidate_pairs("held_out", s1_sample=None)
    X_held, y_held, s1_held, cand_held = build_pair_table(held_pairs, entity_lookup, gt_lookup)
    print(f"  {X_held.shape[0]} pairs, {y_held.mean():.3f} positive rate, {time.time() - t0:.1f}s")

    print("Training LightGBM...")
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        objective="binary",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_fit, y_fit, feature_name=FEATURE_NAMES)

    proba_held = clf.predict_proba(X_held)[:, 1]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_DIR / "baseline_classifier.joblib")

    return {
        "clf": clf,
        "held_pairs": held_pairs,
        "s1_held": s1_held,
        "cand_held": cand_held,
        "proba_held": proba_held,
        "y_held": y_held,
    }


if __name__ == "__main__":
    import sys

    fit_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    result = train_and_evaluate(fit_sample_s1=fit_sample)
    print("importances:", dict(zip(FEATURE_NAMES, result["clf"].feature_importances_.tolist())))
    print("held-out pair AUC-ish sanity: mean proba pos vs neg:",
          result["proba_held"][result["y_held"] == 1].mean(),
          result["proba_held"][result["y_held"] == 0].mean())
