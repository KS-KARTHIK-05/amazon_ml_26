"""Cross-encoder scores for the stage-2 feature `ce` (GPU).

Scores only uncertain-band pairs (stage-1 p1 in BAND) of the entities stage 2
actually uses: its training/validation S1s, the held-out S1s, and every test
S1. The cross-encoder itself was trained on band pairs of OTHER fit-fold S1s,
so its scores on these entities are honest (never seen in its training).

Usage (after two_stage score1; model dir from the pilot):
    ER_CE_MODEL=<dir> python3 -u -m src.ce_stage
then two_stage stage2 / predict with ER_CE=1.
"""

from __future__ import annotations

import os
import time

import polars as pl

from src.data.load import CACHE_ROOT
from src.matching.cross_encoder import BAND, TEXT_COLS, attach_text, record_text, score, text_format
from src.run_submission import WORK, _log
from src.two_stage import S2_TRAIN_S1, SEED, _fold_map, _parts

CE_MODEL = os.environ.get("ER_CE_MODEL", "")
# pairs scored: stage-1 p1 above this (v2 CE covers the "sure" pairs too); default = the v1 band
CE_MIN_P1 = float(os.environ.get("ER_CE_MIN_P1", "-1"))


def _stage2_entities() -> pl.DataFrame:
    """Exactly the train entities two_stage.stage2 trains/validates/evaluates on."""
    folds = _fold_map()
    fit = folds.filter(pl.col("fold") == "fit").sample(fraction=1.0, seed=SEED + 7)
    return pl.concat([fit.head(S2_TRAIN_S1 + 40_000).select("s1_id"),
                      folds.filter(pl.col("fold") == "held_out").select("s1_id")])


def run() -> None:
    assert CE_MODEL, "set ER_CE_MODEL to the trained cross-encoder directory"
    lo, hi = BAND if CE_MIN_P1 < 0 else (CE_MIN_P1, 1.01)
    fmt = text_format(CE_MODEL)
    for split in ("train", "test"):
        out_dir = WORK / split / os.environ.get("ER_CE_DIR", "ce")
        out_dir.mkdir(parents=True, exist_ok=True)
        keep = _stage2_entities() if split == "train" else None
        text = record_text(pl.read_parquet(CACHE_ROOT / split / "master_norm.parquet",
                                           columns=["entity_id", *TEXT_COLS[fmt]]), fmt)
        for part in _parts(split, "p1"):
            out = out_dir / part.name
            if out.exists():
                continue
            t0 = time.time()
            b = pl.scan_parquet(part).filter((pl.col("p1") > lo) & (pl.col("p1") < hi)).collect()
            if keep is not None:
                b = b.join(keep, on="s1_id", how="semi")
            b = attach_text(b.select("s1_id", "cand_id"), text)
            ce = score(CE_MODEL, b) if b.height else []
            b.select("s1_id", "cand_id").with_columns(pl.Series("ce", ce, dtype=pl.Float32)).write_parquet(out)
            _log(f"ce {split}/{part.name}: {b.height} band pairs in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run()
