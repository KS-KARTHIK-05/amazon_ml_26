"""Build one master table per split integrating S1, S2 and S3.

Columns: entity_id, source (S1/S2/S3), business_name, business_address,
country, and for train also:
- cluster_id: the S1 entity this record belongs to (S1 rows -> themselves,
  matched S2/S3 rows -> their ground-truth S1, unmatched S2/S3 -> null)
- fold: fit / held_out, inherited from the cluster's S1 (null if unmatched)

Written as Parquet (fast, used by the pipeline) and TSV (for inspection).

Usage: python3 -m src.data.master train|test
"""

from __future__ import annotations

import sys

import polars as pl

from src.data.load import CACHE_ROOT, load_split
from src.data.split import load_or_make_split


def build_master(split: str) -> pl.DataFrame:
    data = load_split(split)
    frames = [
        data[src].drop("row_id").with_columns(pl.lit(tag).alias("source"))
        for src, tag in (("source1", "S1"), ("source2", "S2"), ("source3", "S3"))
    ]
    master = pl.concat(frames).select(
        ["entity_id", "source", "business_name", "business_address", "country"]
    )

    if split == "train":
        gt = data["ground_truth"]
        member_of = (
            gt.select(["source1_entity_id", "matched_ids"])
            .explode("matched_ids")
            .drop_nulls("matched_ids")
            .rename({"matched_ids": "entity_id", "source1_entity_id": "cluster_id"})
        )
        dup = member_of.group_by("entity_id").len().filter(pl.col("len") > 1).height
        if dup:
            raise ValueError(f"{dup} S2/S3 ids are linked to more than one S1 in ground truth")

        master = master.join(member_of, on="entity_id", how="left").with_columns(
            pl.when(pl.col("source") == "S1")
            .then(pl.col("entity_id"))
            .otherwise(pl.col("cluster_id"))
            .alias("cluster_id")
        )
        folds = load_or_make_split().select(
            pl.col("source1_entity_id").alias("cluster_id"), "fold"
        )
        master = master.join(folds, on="cluster_id", how="left")

    return master


def write_master(split: str) -> pl.DataFrame:
    master = build_master(split)
    out_dir = CACHE_ROOT / split
    master.write_parquet(out_dir / "master.parquet")
    master.write_csv(out_dir / f"master_{split}.tsv", separator="\t")
    return master


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    m = write_master(split)
    print(m.shape)
    print(m.group_by(["source", "country"]).len().sort(["source", "country"]))
    if split == "train":
        print(
            m.group_by("source")
            .agg(
                pl.col("cluster_id").is_not_null().sum().alias("in_a_cluster"),
                pl.col("cluster_id").is_null().sum().alias("unmatched"),
            )
            .sort("source")
        )
        print(m.group_by("fold").len().sort("fold"))
    print(m.sample(5, seed=0))
