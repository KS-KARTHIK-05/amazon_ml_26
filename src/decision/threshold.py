"""Threshold tuning + one-owner-per-candidate assignment.

Given per-pair match probabilities, pick the probability threshold that
maximizes macro F0.5 on a labeled set (held-out), after resolving the
constraint that a single S2/S3 record can only be given to one S1 entity
(confirmed true in the ground truth: no S2/S3 id is linked to more than one
S1). Ties keep the highest-probability assignment.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.eval.f05_score import macro_f05


def assign_one_owner(
    s1_ids: list[str], cand_ids: list[str], proba: np.ndarray, threshold: float
) -> dict[str, list[str]]:
    """Among pairs with proba >= threshold, give each candidate to its highest-proba S1."""
    best_for_cand: dict[str, tuple[float, str]] = {}
    for s1_id, cand_id, p in zip(s1_ids, cand_ids, proba):
        if p < threshold:
            continue
        cur = best_for_cand.get(cand_id)
        if cur is None or p > cur[0]:
            best_for_cand[cand_id] = (p, s1_id)

    predictions: dict[str, list[str]] = defaultdict(list)
    for cand_id, (_p, s1_id) in best_for_cand.items():
        predictions[s1_id].append(cand_id)
    return dict(predictions)


def tune_threshold(
    s1_ids: list[str],
    cand_ids: list[str],
    proba: np.ndarray,
    truth: dict[str, set[str]],
    thresholds: np.ndarray | None = None,
) -> dict:
    if thresholds is None:
        thresholds = np.concatenate([np.linspace(0.05, 0.5, 10), np.linspace(0.5, 0.98, 25)])

    best = {"threshold": None, "macro_f05": -1.0, "metrics": None}
    for t in thresholds:
        preds = assign_one_owner(s1_ids, cand_ids, proba, t)
        metrics = macro_f05(preds, truth)
        if metrics["macro_f05"] > best["macro_f05"]:
            best = {"threshold": float(t), "macro_f05": metrics["macro_f05"], "metrics": metrics}
    return best


if __name__ == "__main__":
    # tiny synthetic sanity check
    s1_ids = ["S1-1", "S1-1", "S1-2", "S1-3"]
    cand_ids = ["S2-1", "S2-2", "S2-1", "S3-1"]  # S2-1 contested between S1-1 and S1-2
    proba = np.array([0.9, 0.2, 0.95, 0.6])
    truth = {"S1-1": {"S2-1"}, "S1-2": set(), "S1-3": {"S3-1"}}

    preds = assign_one_owner(s1_ids, cand_ids, proba, threshold=0.5)
    print("predictions:", preds)
    # S2-1 is contested (0.9 for S1-1 vs 0.95 for S1-2); one-owner keeps the
    # higher-probability side (S1-2) and drops it from S1-1, even though S1-1
    # is the true match here -- exactly the failure mode threshold tuning and
    # a well-calibrated classifier need to make rare.
    assert preds.get("S1-2") == ["S2-1"]
    assert "S2-1" not in preds.get("S1-1", [])

    best = tune_threshold(s1_ids, cand_ids, proba, truth)
    print("best:", best)
