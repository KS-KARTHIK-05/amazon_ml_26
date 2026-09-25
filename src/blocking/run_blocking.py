"""Full-scale baseline blocking driver: run key_blocker over every country in
a split, checkpointing per-country results to Parquet so a crash/OOM never
loses more than one country's worth of work.

Usage:
    python3 -m src.blocking.run_blocking train --n-jobs 4
    python3 -m src.blocking.run_blocking test --n-jobs 4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from src.blocking.key_blocker import DEFAULT_MAX_POSTING, DEFAULT_TOP_K, block_country_partition
from src.data.load import CACHE_ROOT, load_split
from src.preprocess.normalize import normalize_address, normalize_name


def block_split(
    split: str,
    k: int = DEFAULT_TOP_K,
    max_posting: int = DEFAULT_MAX_POSTING,
    n_jobs: int = 4,
    force: bool = False,
) -> pl.DataFrame:
    out_dir = CACHE_ROOT / split / "blocking"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = CACHE_ROOT / split / "candidates_baseline.parquet"

    data = load_split(split)
    s1_all, s2_all, s3_all = data["source1"], data["source2"], data["source3"]
    countries = sorted(set(s1_all["country"].unique().to_list()))
    print(f"[{split}] countries found: {countries}")

    per_country_frames = []
    for country in countries:
        country_path = out_dir / f"{country}.parquet"
        if country_path.exists() and not force:
            print(f"[{split}/{country}] cached, skipping")
            per_country_frames.append(pl.read_parquet(country_path))
            continue

        t_start = time.time()
        s1 = s1_all.filter(pl.col("country") == country)
        s2 = s2_all.filter(pl.col("country") == country)
        s3 = s3_all.filter(pl.col("country") == country)
        print(f"[{split}/{country}] S1={s1.height} S2={s2.height} S3={s3.height}")

        s1_names = [normalize_name(x) for x in s1["business_name"]]
        s1_addrs = [normalize_address(x) for x in s1["business_address"]]
        cand_ids = pl.concat([s2["entity_id"], s3["entity_id"]]).to_numpy()
        cand_names = [normalize_name(x) for x in s2["business_name"]] + [
            normalize_name(x) for x in s3["business_name"]
        ]
        cand_addrs = [normalize_address(x) for x in s2["business_address"]] + [
            normalize_address(x) for x in s3["business_address"]
        ]

        result = block_country_partition(
            s1["entity_id"].to_numpy(),
            s1_names,
            s1_addrs,
            cand_ids,
            cand_names,
            cand_addrs,
            k=k,
            max_posting=max_posting,
            n_jobs=n_jobs,
        )

        frame = pl.DataFrame(
            {
                "source1_entity_id": list(result.keys()),
                "candidate_ids": list(result.values()),
            }
        )
        frame.write_parquet(country_path)
        per_country_frames.append(frame)
        print(f"[{split}/{country}] done in {time.time() - t_start:.1f}s, wrote {country_path}")

    combined = pl.concat(per_country_frames)
    combined.write_parquet(final_path)
    print(f"[{split}] combined {combined.height} rows -> {final_path}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["train", "test"])
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-posting", type=int, default=DEFAULT_MAX_POSTING)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    block_split(args.split, k=args.k, max_posting=args.max_posting, n_jobs=args.n_jobs, force=args.force)
