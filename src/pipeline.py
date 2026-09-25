"""Orchestration CLI chaining blocking -> features -> classifier -> decision -> output.

Subcommands:
    train-baseline    Train the baseline LightGBM classifier on train candidates,
                      tune the decision threshold on held-out, save both.
    predict-baseline  Score the test split with the saved baseline classifier +
                      threshold, write output/matching_results.tsv and
                      output/candidate_pairs.tsv, run the real validator.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from src.data.load import CACHE_ROOT, load_split
from src.data.split import load_or_make_split
from src.decision.threshold import assign_one_owner, tune_threshold
from src.decision.writer import run_validator, write_candidate_pairs, write_matching_results
from src.matching.baseline_classifier import MODEL_DIR, _entity_lookup, build_pair_table, train_and_evaluate
from src.matching.features import FEATURE_NAMES

STUDENT_RESOURCE_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = STUDENT_RESOURCE_ROOT / "output"
STATE_PATH = MODEL_DIR / "baseline_state.json"


def cmd_train_baseline(args):
    result = train_and_evaluate(fit_sample_s1=args.fit_sample)
    print("Tuning threshold on held-out...")
    truth = {
        row["source1_entity_id"]: set(row["matched_ids"])
        for row in load_split("train")["ground_truth"].iter_rows(named=True)
    }
    # score against the FULL held-out fold (including S1 rows with zero candidates)
    split = load_or_make_split()
    held_ids = set(split.filter(pl.col("fold") == "held_out")["source1_entity_id"].to_list())
    held_truth = {sid: truth[sid] for sid in held_ids}

    best = tune_threshold(result["s1_held"], result["cand_held"], result["proba_held"], held_truth)
    print("Best threshold on held-out:", best)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps({"threshold": best["threshold"], "metrics": best["metrics"]}, indent=2))
    print(f"Saved threshold state to {STATE_PATH}")


def cmd_predict_baseline(args):
    state = json.loads(STATE_PATH.read_text())
    threshold = state["threshold"]
    print(f"Using threshold={threshold} (held-out macro F0.5={state['metrics']['macro_f05']:.4f})")

    clf = joblib.load(MODEL_DIR / "baseline_classifier.joblib")

    data = load_split("test")
    s1_all = data["source1"]
    required_s1_ids = s1_all["entity_id"].to_list()
    entity_lookup = _entity_lookup(data["source1"], data["source2"], data["source3"])

    candidates = pl.read_parquet(CACHE_ROOT / "test" / "candidates_baseline.parquet")
    pairs = candidates.filter(pl.col("candidate_ids").list.len() > 0).explode("candidate_ids").rename(
        {"candidate_ids": "candidate_entity_id"}
    )
    print(f"Scoring {pairs.height} candidate pairs...")

    t0 = time.time()
    X, _y, s1_ids, cand_ids = build_pair_table(pairs, entity_lookup, gt_lookup=None)
    print(f"  features built in {time.time() - t0:.1f}s")

    proba = clf.predict_proba(X)[:, 1]
    predictions = assign_one_owner(s1_ids, cand_ids, proba, threshold)

    candidate_lists = {
        row["source1_entity_id"]: row["candidate_ids"] for row in candidates.iter_rows(named=True)
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_matching_results(OUTPUT_DIR / "matching_results.tsv", predictions, required_s1_ids)
    write_candidate_pairs(OUTPUT_DIR / "candidate_pairs.tsv", candidate_lists, required_s1_ids)

    n_matched = sum(1 for v in predictions.values() if v)
    print(f"Wrote predictions: {n_matched} S1 entities with >=1 match out of {len(required_s1_ids)}")

    passed, output = run_validator(
        OUTPUT_DIR / "matching_results.tsv", OUTPUT_DIR / "candidate_pairs.tsv"
    )
    print(output)
    print("VALIDATION PASSED" if passed else "VALIDATION FAILED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train-baseline")
    p_train.add_argument("--fit-sample", type=int, default=200_000)
    p_train.set_defaults(func=cmd_train_baseline)

    p_pred = sub.add_parser("predict-baseline")
    p_pred.set_defaults(func=cmd_predict_baseline)

    args = parser.parse_args()
    args.func(args)
