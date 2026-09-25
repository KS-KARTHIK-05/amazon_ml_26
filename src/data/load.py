"""Load raw challenge TSVs into cached Parquet with a stable row_id per source.

The raw files use tab as the delimiter but still follow CSV quoting rules
(RFC4180 double-quote escaping shows up in a handful of rows whose address
text itself contains a literal `"`), so we parse with quoting enabled rather
than disabling it. Empty address fields come back as null; we keep them as
empty strings so downstream code has one representation for "no address".
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

SOURCE_SCHEMA = {
    "entity_id": pl.Utf8,
    "business_name": pl.Utf8,
    "business_address": pl.Utf8,
    "country": pl.Utf8,
}

GROUND_TRUTH_SCHEMA = {
    "source1_entity_id": pl.Utf8,
    "matched_entity_ids": pl.Utf8,
}


def _read_tsv(path: Path, schema: dict) -> pl.DataFrame:
    return pl.read_csv(
        path,
        separator="\t",
        quote_char='"',
        schema=schema,
        empty_string_is_null=False,
    )


def load_source(path: Path) -> pl.DataFrame:
    """Load one *_source{1,2,3}.tsv file.

    Returns columns: row_id (u32, 0-based position), entity_id, business_name,
    business_address (empty string when missing), country.
    """
    df = _read_tsv(path, SOURCE_SCHEMA)
    df = df.with_columns(
        [
            pl.col("business_name").fill_null(""),
            pl.col("business_address").fill_null(""),
            pl.col("country").fill_null(""),
        ]
    )
    df = df.with_row_index("row_id")
    df = df.with_columns(pl.col("row_id").cast(pl.UInt32))
    return df.select(["row_id", "entity_id", "business_name", "business_address", "country"])


def load_ground_truth(path: Path) -> pl.DataFrame:
    """Load train_ground_truth.tsv.

    Returns columns: source1_entity_id, matched_ids (List[Utf8], empty list
    for singletons), n_matches (u32).
    """
    df = _read_tsv(path, GROUND_TRUTH_SCHEMA)
    df = df.with_columns(pl.col("matched_entity_ids").fill_null(""))
    df = df.with_columns(
        pl.when(pl.col("matched_entity_ids") == "")
        .then(pl.lit([], dtype=pl.List(pl.Utf8)))
        .otherwise(pl.col("matched_entity_ids").str.split(","))
        .alias("matched_ids")
    )
    df = df.with_columns(pl.col("matched_ids").list.len().cast(pl.UInt32).alias("n_matches"))
    return df.select(["source1_entity_id", "matched_ids", "n_matches"])


def cache_parquet(df: pl.DataFrame, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache_path)


def load_or_cache_source(raw_path: Path, cache_path: Path, force: bool = False) -> pl.DataFrame:
    if cache_path.exists() and not force:
        return pl.read_parquet(cache_path)
    df = load_source(raw_path)
    cache_parquet(df, cache_path)
    return df


def load_or_cache_ground_truth(raw_path: Path, cache_path: Path, force: bool = False) -> pl.DataFrame:
    if cache_path.exists() and not force:
        return pl.read_parquet(cache_path)
    df = load_ground_truth(raw_path)
    cache_parquet(df, cache_path)
    return df


DATASET_ROOT = Path(__file__).resolve().parents[4] / "dataset"
CACHE_ROOT = Path(__file__).resolve().parents[4] / "cache"


def load_split(split: str, force: bool = False) -> dict[str, pl.DataFrame]:
    """Load source1/2/3 (and ground_truth for train) for a split ('train'/'test')."""
    raw_dir = DATASET_ROOT / split
    cache_dir = CACHE_ROOT / split
    out = {}
    for src in ("source1", "source2", "source3"):
        raw_path = raw_dir / f"{split}_{src}.tsv"
        cache_path = cache_dir / f"{src}.parquet"
        out[src] = load_or_cache_source(raw_path, cache_path, force=force)
    if split == "train":
        raw_path = raw_dir / "train_ground_truth.tsv"
        cache_path = cache_dir / "ground_truth.parquet"
        out["ground_truth"] = load_or_cache_ground_truth(raw_path, cache_path, force=force)
    return out


if __name__ == "__main__":
    import sys

    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    dfs = load_split(split, force=True)
    for name, df in dfs.items():
        print(name, df.shape)
        print(df.head(3))
