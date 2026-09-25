"""Group split of train S1 entities into fit/held-out, stratified on singleton status.

Every S1 entity and all of its ground-truth matches must stay together on one
side of the split (trivially true here since ground truth is already one row
per S1 entity), and the singleton rate must be preserved in both halves so the
held-out F0.5 is not biased by an unrepresentative match-rate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from .load import CACHE_ROOT, load_split

SEED = 42
HELD_OUT_FRACTION = 0.10


def make_split(seed: int = SEED, held_out_fraction: float = HELD_OUT_FRACTION) -> pl.DataFrame:
    """Return a DataFrame with columns source1_entity_id, n_matches, is_singleton, fold."""
    gt = load_split("train")["ground_truth"]
    df = gt.select(["source1_entity_id", "n_matches"]).with_columns(
        (pl.col("n_matches") == 0).alias("is_singleton")
    )

    rng = np.random.default_rng(seed)
    fold = np.full(df.height, "fit", dtype=object)

    is_singleton = df["is_singleton"].to_numpy()
    for stratum in (True, False):
        idx = np.where(is_singleton == stratum)[0]
        rng.shuffle(idx)
        n_held = int(round(len(idx) * held_out_fraction))
        fold[idx[:n_held]] = "held_out"

    df = df.with_columns(pl.Series("fold", fold))
    return df


def save_split(cache_path: Path | None = None, **kwargs) -> pl.DataFrame:
    cache_path = cache_path or (CACHE_ROOT / "train" / "split.parquet")
    df = make_split(**kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache_path)
    return df


def load_or_make_split(cache_path: Path | None = None, force: bool = False, **kwargs) -> pl.DataFrame:
    cache_path = cache_path or (CACHE_ROOT / "train" / "split.parquet")
    if cache_path.exists() and not force:
        return pl.read_parquet(cache_path)
    return save_split(cache_path=cache_path, **kwargs)


if __name__ == "__main__":
    df = save_split(force=True) if False else load_or_make_split(force=True)
    print(df.group_by(["fold", "is_singleton"]).agg(pl.len()).sort(["fold", "is_singleton"]))
    total = df.height
    held = df.filter(pl.col("fold") == "held_out").height
    print(f"held_out: {held} / {total} = {held / total:.4f}")
    print(
        "singleton rate overall:",
        df["is_singleton"].mean(),
        "fit:",
        df.filter(pl.col("fold") == "fit")["is_singleton"].mean(),
        "held_out:",
        df.filter(pl.col("fold") == "held_out")["is_singleton"].mean(),
    )
