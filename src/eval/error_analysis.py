"""Where does held-out macro F0.5 go? Decompose 1 - F0.5 per S1 entity.

Uses the cached train-sample candidates/features and the saved matcher.
Each held-out S1 entity's loss (1 - its F0.5) is attributed to one bucket:
- singleton_fp      : true singleton, we predicted something
- missed_all_blocked: matched entity, nothing predicted, true match was in candidates
- missed_all_unblk  : matched entity, nothing predicted, no true match in candidates
- has_fp            : matched entity, at least one false positive predicted
- partial_fn_block  : only FNs, some because the true match was never a candidate
- partial_fn_model  : only FNs, all missing ones were candidates the model rejected

Usage: python3 -m src.eval.error_analysis [threshold]
"""

from __future__ import annotations

import json
import sys

import lightgbm as lgb
import polars as pl

from src.data.load import CACHE_ROOT
from src.matching.pair_features import FEATURES

ROOT = CACHE_ROOT.parent
WORK = CACHE_ROOT / "run"


def held_out_scored() -> tuple[pl.DataFrame, pl.DataFrame]:
    ids = pl.read_parquet(WORK / "train_sample.parquet").filter(pl.col("fold") == "held_out")
    feats = pl.concat([pl.read_parquet(p) for p in sorted((WORK / "train").glob("feats_*.parquet"))])
    feats = feats.join(ids.select(pl.col("entity_id").alias("s1_id"), "country"), on="s1_id")
    booster = lgb.Booster(model_file=str(ROOT / "models" / "lgb_matcher.txt"))
    feats = feats.with_columns(pl.Series("p", booster.predict(feats.select(FEATURES).to_numpy())))
    return feats, ids


def decompose(feats: pl.DataFrame, ids: pl.DataFrame, t: float) -> pl.DataFrame:
    master = pl.scan_parquet(CACHE_ROOT / "train" / "master_norm.parquet")
    n_true = (
        master.filter((pl.col("source") != "S1") & pl.col("cluster_id").is_not_null())
        .group_by("cluster_id").len().rename({"cluster_id": "s1_id", "len": "n_true"}).collect()
    )
    pred = (
        feats.filter(pl.col("p") >= t).sort("p", descending=True).unique("cand_id", keep="first")
        .group_by("s1_id").agg(pl.len().alias("n_pred"), pl.col("label").sum().alias("n_tp"))
    )
    blocked = feats.group_by("s1_id").agg(pl.col("label").sum().alias("n_blocked_true"))
    per = (
        ids.select(pl.col("entity_id").alias("s1_id"), "country")
        .join(n_true, on="s1_id", how="left").join(pred, on="s1_id", how="left")
        .join(blocked, on="s1_id", how="left").fill_null(0)
        .with_columns(
            (pl.col("n_tp") / pl.col("n_pred")).fill_nan(0).alias("P"),
            (pl.col("n_tp") / pl.col("n_true")).fill_nan(0).alias("R"),
        )
        .with_columns(
            pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
            .when(pl.col("n_tp") == 0).then(0.0)
            .otherwise(1.25 * pl.col("P") * pl.col("R") / (0.25 * pl.col("P") + pl.col("R"))).alias("f05")
        )
        .with_columns(
            pl.when(pl.col("n_true") == 0).then(pl.lit("singleton_fp" ))
            .when((pl.col("n_pred") == 0) & (pl.col("n_blocked_true") > 0)).then(pl.lit("missed_all_blocked"))
            .when(pl.col("n_pred") == 0).then(pl.lit("missed_all_unblk"))
            .when(pl.col("n_pred") > pl.col("n_tp")).then(pl.lit("has_fp"))
            .when(pl.col("n_blocked_true") < pl.col("n_true")).then(pl.lit("partial_fn_block"))
            .otherwise(pl.lit("partial_fn_model")).alias("bucket")
        )
    )
    return per


if __name__ == "__main__":
    t = float(sys.argv[1]) if len(sys.argv) > 1 else json.loads((ROOT / "models" / "matcher_state.json").read_text())["threshold"]
    feats, ids = held_out_scored()
    per = decompose(feats, ids, t)
    n = per.height
    print(f"threshold {t} | held-out S1 {n} | macro F0.5 {per['f05'].mean():.4f}")
    lost = (
        per.with_columns((1 - pl.col("f05")).alias("loss"))
        .filter(pl.col("loss") > 0)
        .group_by("bucket").agg(pl.len().alias("entities"), (pl.col("loss").sum() / n).alias("f05_lost"))
        .sort("f05_lost", descending=True)
    )
    pl.Config.set_tbl_rows(20)
    print(lost)
    print("by country:", per.group_by("country").agg(pl.col("f05").mean().round(4), pl.len()).sort("country").to_dicts())
    per.write_parquet(WORK / "held_out_per_entity.parquet")
    feats.select("s1_id", "cand_id", "label", "p").write_parquet(WORK / "held_out_scored.parquet")
