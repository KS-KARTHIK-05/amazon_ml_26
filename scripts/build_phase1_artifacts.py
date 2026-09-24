"""Phase 1 CLI: ingest sources, parse ground truth, build families and splits.

Run from the repository root:

    .venv/bin/python scripts/build_phase1_artifacts.py --split train

Writes:
  artifacts/normalized/{train,test}_source{1,2,3}.parquet
  artifacts/splits/train_families.json  (family_id, country, split membership)
  reports/phase1_build_report.json      (measured counts, timings, integrity)
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from er.io import ingest_source, iter_ground_truth_chunks, iter_source_chunks  # noqa: E402
from er.splits import SplitConfig, assign_splits, save_split_manifest, split_counts  # noqa: E402
from er.truth import (  # noqa: E402
    TruthParseResult,
    build_positive_families,
    multi_owner_targets,
    parse_ground_truth_chunk,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def peak_rss_gib() -> float:
    # ru_maxrss is KiB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = {"split": args.split, "seed": args.seed, "stages": {}}
    dataset_dir = os.path.join(ROOT, "student_resource", "dataset", args.split)
    out_dir = os.path.join(ROOT, "artifacts", "normalized")
    os.makedirs(out_dir, exist_ok=True)

    t_start = time.time()

    # --- Ingest S1/S2/S3 ---
    for source_num, source_label in [(1, "S1"), (2, "S2"), (3, "S3")]:
        t0 = time.time()
        in_path = os.path.join(dataset_dir, f"{args.split}_source{source_num}.tsv")
        out_path = os.path.join(out_dir, f"{args.split}_source{source_num}.parquet")
        stats = ingest_source(in_path, source_label, out_path)
        report["stages"][f"ingest_{source_label}"] = {
            "row_count": stats.row_count,
            "seconds": round(time.time() - t0, 2),
            "out_path": os.path.relpath(out_path, ROOT),
        }
        print(f"ingested {source_label}: {stats.row_count} rows in "
              f"{report['stages'][f'ingest_{source_label}']['seconds']}s")

    # --- Country lookup for S1 (for stratified splits) ---
    t0 = time.time()
    s1_country: dict[str, str] = {}
    s1_path = os.path.join(dataset_dir, f"{args.split}_source1.tsv")
    for chunk in iter_source_chunks(s1_path):
        for eid, country in zip(chunk["entity_id"], chunk["country"]):
            s1_country[eid] = country
    report["stages"]["s1_country_index"] = {
        "count": len(s1_country), "seconds": round(time.time() - t0, 2),
    }
    print(f"S1 country index: {len(s1_country)} entities in "
          f"{report['stages']['s1_country_index']['seconds']}s")

    if args.split == "train":
        # --- Ground truth ---
        t0 = time.time()
        gt_path = os.path.join(dataset_dir, "train_ground_truth.tsv")
        result = TruthParseResult()
        for chunk in iter_ground_truth_chunks(gt_path):
            rows = list(zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]))
            parse_ground_truth_chunk(rows, result)
        n_edges = len(result.edges)
        n_errors = len(result.errors)
        report["stages"]["parse_ground_truth"] = {
            "s1_count": len(result.s1_truth),
            "edge_count": n_edges,
            "error_count": n_errors,
            "seconds": round(time.time() - t0, 2),
        }
        print(f"ground truth: {len(result.s1_truth)} S1 rows, {n_edges} edges, "
              f"{n_errors} errors in {report['stages']['parse_ground_truth']['seconds']}s")
        if n_errors:
            examples = result.errors[:10]
            report["stages"]["parse_ground_truth"]["error_examples"] = examples

        singleton_count = sum(1 for t in result.s1_truth.values() if not t)
        report["stages"]["parse_ground_truth"]["singleton_count"] = singleton_count
        print(f"singletons (no true match): {singleton_count}")

        # --- Multi-owner targets ---
        owners = multi_owner_targets(result)
        report["stages"]["multi_owner_targets"] = {
            "count": len(owners),
            "examples": dict(list(owners.items())[:5]),
        }
        print(f"multi-owner targets (shared across >1 S1): {len(owners)}")

        # --- Positive family components ---
        t0 = time.time()
        families = build_positive_families(result)
        n_families = len(set(families.values()))
        report["stages"]["build_families"] = {
            "s1_count": len(families),
            "family_count": n_families,
            "seconds": round(time.time() - t0, 2),
        }
        print(f"families: {n_families} components over {len(families)} S1 entities "
              f"in {report['stages']['build_families']['seconds']}s")

        # --- Splits ---
        t0 = time.time()
        family_country: dict[str, str] = {}
        for s1_id, family_id in families.items():
            family_country.setdefault(family_id, s1_country.get(s1_id, ""))
        config = SplitConfig(seed=args.seed)
        assignment = assign_splits(family_country, config)
        counts = split_counts(assignment)
        split_out_path = os.path.join(ROOT, "artifacts", "splits", "train_families.json")
        os.makedirs(os.path.dirname(split_out_path), exist_ok=True)
        save_split_manifest(assignment, config, split_out_path)
        report["stages"]["assign_splits"] = {
            "family_count": len(assignment),
            "counts": counts,
            "seconds": round(time.time() - t0, 2),
            "out_path": os.path.relpath(split_out_path, ROOT),
        }
        print(f"splits: {counts} in {report['stages']['assign_splits']['seconds']}s")

        # Persist s1 -> family_id mapping for downstream phases.
        s1_family_path = os.path.join(ROOT, "artifacts", "splits", "s1_family_map.json")
        with open(s1_family_path, "w", encoding="utf-8") as f:
            json.dump(families, f)
        # Persist full s1 truth (as sorted lists) for downstream metric computation.
        s1_truth_path = os.path.join(ROOT, "artifacts", "splits", "s1_truth.json")
        with open(s1_truth_path, "w", encoding="utf-8") as f:
            json.dump({k: sorted(v) for k, v in result.s1_truth.items()}, f)

    report["total_seconds"] = round(time.time() - t_start, 2)
    report["peak_rss_gib"] = round(peak_rss_gib(), 3)
    print(f"TOTAL: {report['total_seconds']}s, peak RSS {report['peak_rss_gib']} GiB")

    report_path = os.path.join(ROOT, "reports", f"phase1_build_report_{args.split}.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"report written to {os.path.relpath(report_path, ROOT)}")


if __name__ == "__main__":
    main()
