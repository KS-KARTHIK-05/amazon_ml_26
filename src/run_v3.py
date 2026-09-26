"""v3 = v2 + reverse (record -> S1) blocking.

Forward candidates are reused from the v2 run (cache/run_v2/*/cands_*.parquet);
this adds the reverse pairs, gives the new ones the same forward key scores
the blocker would have computed, and attaches reverse-side evidence to every
pair. Features and the two-stage matcher then run unchanged on cache/run_v3.

Usage:
    python3 -u -m src.run_v3 rev_train merge_train rev_test merge_test
    python3 -u -m src.run_submission feat_train feat_test
    python3 -u -m src.two_stage stage1 score1 context stage2 predict
"""

from __future__ import annotations

import gc
import os
import shutil
import sys
import time

import polars as pl

from src.blocking.blocker import DEFAULTS
from src.blocking.reverse import REV_COLS, forward_key_scores, reverse_candidates
from src.data.load import CACHE_ROOT
from src.run_submission import BLOCK_COLS, WORK, _countries, _log, _train_sample_ids

WORK_V2 = CACHE_ROOT / "run_v2"
# reverse width: v3 = 3 / 2; v4 = 10 / 5 (held-out ceiling +0.0013 for +13 pairs/S1)
REV_PARAMS: dict = {"rev_top_k": int(os.environ.get("ER_REV_K", "3")),
                    "rev_top_k_family": int(os.environ.get("ER_REV_KF", "2"))}
# Only records whose best forward key cosine (over every S1 list they appear
# in) is below this are queried in reverse: on held-out, those ~30% of records
# hold 99-100% of the true matches reverse blocking recovers, so the reverse
# pass costs ~2h instead of ~7h. A record that already strongly matches some
# S1 is well placed; the weakly placed ones are the crowded-out ones.
REV_QUERY_MAX_COS = 0.5


def _strongly_placed(fwd: pl.DataFrame) -> pl.DataFrame:
    """cand_id of records the forward pass placed strongly (best key_cos >= cutoff)."""
    return (fwd.group_by("cand_id").agg(pl.col("key_cos").max().alias("best"))
            .filter(pl.col("best") >= REV_QUERY_MAX_COS).select("cand_id"))


def _load(split: str, country: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    mc = (
        pl.scan_parquet(CACHE_ROOT / split / "master_norm.parquet")
        .filter(pl.col("country") == country).select(BLOCK_COLS).collect()
    )
    s1 = mc.filter(pl.col("source") == "S1")
    if split == "train":
        s1 = s1.join(_train_sample_ids().select("entity_id"), on="entity_id", how="semi")
    return s1, mc.filter(pl.col("source") != "S1")


def stage_rev(split: str) -> None:
    for country in _countries(split):
        out = WORK / split / f"rev_{country}.parquet"
        if out.exists():
            _log(f"rev {split}/{country}: cached")
            continue
        t0 = time.time()
        s1, cand = _load(split, country)
        strong = _strongly_placed(pl.read_parquet(WORK_V2 / split / f"cands_{country}.parquet", columns=["cand_id", "key_cos"]))
        n_all = cand.height
        cand = cand.join(strong.rename({"cand_id": "entity_id"}), on="entity_id", how="anti")
        _log(f"rev {split}/{country}: {cand.height} weakly placed of {n_all} records vs {s1.height} S1")
        r = reverse_candidates(s1, cand, REV_PARAMS)
        out.parent.mkdir(parents=True, exist_ok=True)
        r.write_parquet(out)
        _log(f"rev {split}/{country}: {r.height} pairs ({r.height / max(s1.height, 1):.1f}/S1) in {time.time() - t0:.0f}s")
        del s1, cand, r
        gc.collect()


def stage_merge(split: str) -> None:
    for country in _countries(split):
        out = WORK / split / f"cands_{country}.parquet"
        if out.exists():
            _log(f"merge {split}/{country}: cached")
            continue
        t0 = time.time()
        # phases kept apart so the US train merge (77M forward pairs) stays well under RAM:
        # 1) new pairs and placement from the forward pair ids only
        fwd_path = WORK_V2 / split / f"cands_{country}.parquet"
        rev = pl.read_parquet(WORK / split / f"rev_{country}.parquet")
        ids = pl.read_parquet(fwd_path, columns=["s1_id", "cand_id", "key_cos"])
        new = rev.select("s1_id", "cand_id").join(ids.select("s1_id", "cand_id"), on=["s1_id", "cand_id"], how="anti")
        strong = _strongly_placed(ids)
        del ids
        gc.collect()
        # 2) forward key scores for the new pairs
        s1, cand = _load(split, country)
        scored = forward_key_scores(new, s1, cand, DEFAULTS["max_df"], DEFAULTS["keys_per_s1"])
        del s1, cand
        gc.collect()
        # 3) combine
        fwd = pl.read_parquet(fwd_path)
        scored = scored.with_columns(pl.lit(None, dtype=pl.Float32).alias("name_sim"))
        pairs = (
            pl.concat([fwd.with_columns(pl.lit(1.0, dtype=pl.Float32).alias("fwd_hit")),
                       scored.select(fwd.columns).with_columns(pl.lit(0.0, dtype=pl.Float32).alias("fwd_hit"))],
                      how="vertical_relaxed")
            .join(rev, on=["s1_id", "cand_id"], how="left")
            .with_columns(
                pl.col("r_rank").is_not_null().cast(pl.Float32).alias("rev_hit"),
                pl.col([c for c in REV_COLS if c != "r_rank"]).fill_null(0.0).cast(pl.Float32),
                pl.col("r_rank").fill_null(99.0).cast(pl.Float32),
            )
            .join(strong.with_columns(pl.lit(0.0, dtype=pl.Float32).alias("rev_queried")), on="cand_id", how="left")
            .with_columns(pl.col("rev_queried").fill_null(1.0))
        )
        pairs.write_parquet(out)
        n_s1 = pairs["s1_id"].n_unique()
        _log(f"merge {split}/{country}: forward {fwd.height} + new {new.height} = {pairs.height} "
             f"({fwd.height / n_s1:.1f} -> {pairs.height / n_s1:.1f}/S1) in {time.time() - t0:.0f}s")
        del fwd, rev, new, scored, pairs
        gc.collect()


def _init() -> None:
    """v3 reuses v2's entity sample / fold assignment exactly."""
    WORK.mkdir(parents=True, exist_ok=True)
    if not (WORK / "train_sample.parquet").exists():
        shutil.copy(WORK_V2 / "train_sample.parquet", WORK / "train_sample.parquet")


STAGES = {"rev_train": lambda: stage_rev("train"), "merge_train": lambda: stage_merge("train"),
          "rev_test": lambda: stage_rev("test"), "merge_test": lambda: stage_merge("test")}

if __name__ == "__main__":
    _init()
    for name in sys.argv[1:]:
        t0 = time.time()
        _log(f"=== {name} ===")
        STAGES[name]()
        _log(f"=== {name} done in {time.time() - t0:.0f}s ===")
