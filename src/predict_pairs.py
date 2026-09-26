"""Predict supplied pairs using an existing normalized master and LightGBM model.

No blocking selection, normalization, or training is run. Blocking *features*
are recomputed with the original country-wide candidate vocabulary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from src.blocking.blocker import DEFAULTS, FAM, record_keys_chunked
from src.eval.f05_score import macro_f05
from src.matching.pair_features import BASE_COLS, FEATURES, add_s1_name_stats, build_features


def read_table(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path, separator="," if path.suffix.lower() == ".csv" else "\t",
                       infer_schema=False)


def read_pairs(path: Path) -> pl.DataFrame:
    df = read_table(path)
    if {"s1_id", "cand_id"} <= set(df.columns):
        df = df.select("s1_id", "cand_id")
    elif {"source1_entity_id", "candidate_entity_ids"} <= set(df.columns):
        df = df.select(pl.col("source1_entity_id").alias("s1_id"),
                       pl.col("candidate_entity_ids").fill_null("").str.split(",").alias("cand_id")).explode("cand_id")
    else:
        raise ValueError("Expected s1_id,cand_id or source1_entity_id,candidate_entity_ids columns")
    df = df.with_columns(pl.all().cast(pl.String).str.strip_chars())
    if df.filter(pl.col("s1_id").is_null() | (pl.col("s1_id") == "")).height:
        raise ValueError("Candidate file contains an empty S1 ID")
    return df.filter(pl.col("cand_id").is_not_null() & (pl.col("cand_id") != "")).unique(maintain_order=True)


def key_context(records: pl.DataFrame, query_ids: pl.Series, max_df: int, keys_per_s1: int):
    """Same IDF, key selection and norms as key_overlap_candidates."""
    s1 = records.filter((pl.col("source") == "S1") & pl.col("entity_id").is_in(query_ids.implode()))
    cand = records.filter(pl.col("source").is_in(["S2", "S3"]))
    ck = record_keys_chunked(cand.with_row_index("rid")).join(
        cand.with_row_index("rid").select("rid", pl.col("entity_id").alias("cand_id")), on="rid").drop("rid")
    vocab = ck.group_by("key").agg(pl.len().alias("df")).filter(pl.col("df") <= max_df).with_columns(
        np.log((cand.height + 1) / (pl.col("df") + 1)).cast(pl.Float32).alias("idf"))
    ck = ck.join(vocab, on="key")
    cn = ck.group_by("cand_id").agg((pl.col("idf") ** 2).sum().sqrt().alias("cnorm"))
    sk = record_keys_chunked(s1.with_row_index("rid")).join(
        s1.with_row_index("rid").select("rid", pl.col("entity_id").alias("s1_id")), on="rid")
    sk = sk.join(vocab, on="key").sort(["rid", "df"]).group_by("rid", maintain_order=True).head(keys_per_s1)
    sn = sk.group_by("s1_id").agg((pl.col("idf") ** 2).sum().sqrt().alias("snorm"))
    return sk.select("s1_id", "key", "idf", "fam"), ck.select("cand_id", "key"), sn, cn


CONTEXT_PARTS = ("sk", "ck", "sn", "cn")


def cached_key_context(records: pl.DataFrame, country: str, master: Path, query_ids: pl.Series,
                       max_df: int, keys_per_s1: int, refresh: bool = False):
    """key_context over all S1s of a country, stored beside the master so later
    runs on any pair file of the same split skip the key building. Rebuilt when
    the master is newer than the cache or with --refresh-cache."""
    folder = master.parent / "pair_context" / f"{country}_df{max_df}_k{keys_per_s1}"
    paths = {n: folder / f"{n}.parquet" for n in CONTEXT_PARTS}
    fresh = all(p.exists() for p in paths.values()) and \
        min(p.stat().st_mtime for p in paths.values()) > master.stat().st_mtime
    if refresh or not fresh:
        print(f"{country}: building key statistics cache -> {folder}", flush=True)
        all_s1 = records.filter(pl.col("source") == "S1")["entity_id"]
        folder.mkdir(parents=True, exist_ok=True)
        for n, part in zip(CONTEXT_PARTS, key_context(records, all_s1, max_df, keys_per_s1)):
            part.write_parquet(paths[n])
    else:
        print(f"{country}: loading cached key statistics from {folder}", flush=True)
    wanted = query_ids.implode()
    sk, ck, sn, cn = (pl.read_parquet(paths[n]) for n in CONTEXT_PARTS)
    return sk.filter(pl.col("s1_id").is_in(wanted)), ck, sn.filter(pl.col("s1_id").is_in(wanted)), cn


def score_keys(pairs: pl.DataFrame, context) -> pl.DataFrame:
    sk, ck, sn, cn = context
    shared = pairs.join(sk, on="s1_id").join(ck, on=["cand_id", "key"], how="semi")
    scores = shared.group_by("s1_id", "cand_id").agg(
        (pl.col("idf") ** 2).sum().alias("dot"),
        pl.col("idf").filter(pl.col("fam") == FAM["n"]).sum().alias("name_score"),
        pl.col("idf").filter(pl.col("fam").is_in([FAM["a"], FAM["c"], FAM["w"]])).sum().alias("addr_score"),
        pl.col("idf").filter(pl.col("fam") == FAM["x"]).sum().alias("conj_score"),
        pl.len().alias("key_shared"),
    ).join(sn, on="s1_id").join(cn, on="cand_id").with_columns(
        (pl.col("dot") / (pl.col("snorm") * pl.col("cnorm"))).cast(pl.Float32).alias("key_cos"))
    cols = ["key_cos", "name_score", "addr_score", "conj_score", "key_shared"]
    return pairs.join(scores.select("s1_id", "cand_id", *cols), on=["s1_id", "cand_id"], how="left").with_columns(
        pl.col(cols).fill_null(0))


def run(args) -> None:
    threshold = args.threshold
    if threshold is None:
        state = args.state or args.model.with_name("matcher_state.json")
        threshold = float(json.loads(state.read_text())["threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    if args.batch_s1 < 1:
        raise ValueError("--batch-s1 must be positive")
    model = lgb.Booster(model_file=str(args.model))
    # predict with the model's own feature list: v1 lgb_matcher.txt uses a
    # subset of today's FEATURES, v2 stage1 models use all of them
    model_features = model.feature_name()
    unknown = [f for f in model_features if f not in FEATURES]
    if unknown:
        raise ValueError(f"Model needs features this checkout cannot build: {unknown[:5]} "
                         "(stage2.txt needs the two_stage context pipeline, not this script)")
    cols = list(dict.fromkeys(BASE_COLS + ["country", "addr_components"]))
    master = pl.read_parquet(args.master, columns=cols)
    if master["entity_id"].n_unique() != master.height:
        raise ValueError("Master contains duplicate entity IDs")
    pairs = read_pairs(args.pairs)
    queries = master.filter(pl.col("source") == "S1").select(pl.col("entity_id").alias("s1_id"), "country")
    candidates = master.filter(pl.col("source").is_in(["S2", "S3"])).select(pl.col("entity_id").alias("cand_id"), pl.col("country").alias("cand_country"))
    checked = pairs.join(queries, on="s1_id", how="left").join(candidates, on="cand_id", how="left")
    invalid = checked.filter(pl.col("country").is_null() | pl.col("cand_country").is_null() | (pl.col("country") != pl.col("cand_country")))
    if invalid.height:
        raise ValueError(f"{invalid.height} pairs have unknown IDs, wrong sources, or different countries: {invalid.head(5).to_dicts()}")
    scored = []
    for (country,), group in checked.group_by("country"):
        records = add_s1_name_stats(master.filter(pl.col("country") == country))
        ids = group["s1_id"].unique()
        print(f"{country}: {len(ids)} S1 records", flush=True)
        context = cached_key_context(records, country, args.master, ids, args.max_df, args.keys_per_s1,
                                     args.refresh_cache)
        for start in range(0, len(ids), args.batch_s1):
            batch = group.filter(pl.col("s1_id").is_in(ids.slice(start, args.batch_s1).implode())).select("s1_id", "cand_id")
            # name tf-idf similarity only exists inside blocking; left null (has_name_sim = 0)
            batch = score_keys(batch, context).with_columns(pl.lit(None, dtype=pl.Float32).alias("name_sim"))
            features = build_features(batch, records)
            scored.append(features.select("s1_id", "cand_id").with_columns(
                pl.Series("p", model.predict(features.select(model_features).to_numpy()))))
    scores = pl.concat(scored) if scored else pl.DataFrame(schema={"s1_id": pl.String, "cand_id": pl.String, "p": pl.Float64})
    accepted = scores.filter(pl.col("p") >= threshold).sort(["p", "s1_id", "cand_id"], descending=[True, False, False]).unique("cand_id", keep="first")
    scores = scores.join(accepted.select("s1_id", "cand_id").with_columns(pl.lit(True).alias("accepted")),
                         on=["s1_id", "cand_id"], how="left").with_columns(pl.col("accepted").fill_null(False))
    args.output.mkdir(parents=True, exist_ok=True)
    scores.write_parquet(args.output / "pair_scores.parquet")
    matches = accepted.group_by("s1_id").agg("cand_id")
    predictions = dict(zip(matches["s1_id"], matches["cand_id"].to_list()))
    with (args.output / "matching_results.tsv").open("w", encoding="utf-8", newline="") as out:
        out.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in queries["s1_id"]:
            out.write(f"{s1_id}\t{','.join(predictions.get(s1_id, []))}\n")
    print(f"Scored {scores.height} pairs; accepted {accepted.height}; threshold={threshold}")
    if args.truth:
        truth_df = read_table(args.truth)
        truth = {}
        for row in truth_df.select("source1_entity_id", "matched_entity_ids").iter_rows(named=True):
            key = row["source1_entity_id"]
            if key in truth:
                raise ValueError(f"Duplicate truth S1 ID: {key}")
            truth[key] = [x.strip() for x in (row["matched_entity_ids"] or "").split(",") if x.strip()]
        if not truth or set(truth) - set(queries["s1_id"]):
            raise ValueError("Truth must be nonempty and its S1 IDs must exist in the master")
        metrics = macro_f05(predictions, truth)
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--master", type=Path, required=True, help="Existing full master_norm.parquet for this split")
    parser.add_argument("--model", type=Path, required=True, help="Existing lgb_matcher.txt")
    parser.add_argument("--state", type=Path, help="Default: matcher_state.json beside model")
    parser.add_argument("--threshold", type=float, help="Override saved threshold")
    parser.add_argument("--truth", type=Path, help="Optional labeled TSV/CSV: source1_entity_id,matched_entity_ids")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-s1", type=int, default=1000)
    parser.add_argument("--max-df", type=int, default=DEFAULTS["max_df"])
    parser.add_argument("--keys-per-s1", type=int, default=DEFAULTS["keys_per_s1"])
    parser.add_argument("--refresh-cache", action="store_true",
                        help="Rebuild the cached key statistics beside the master")
    run(parser.parse_args())