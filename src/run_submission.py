"""End-to-end: blocking -> pair features -> LightGBM (GPU) -> F0.5 threshold
-> output/matching_results.tsv + output/candidate_pairs.tsv -> validator.

Prerequisites (already cached by earlier steps):
    python3 -m src.data.master train|test
    python3 -m src.preprocess.translit
    python3 -m src.preprocess.normalize_master train|test

Usage:
    python3 -u -m src.run_submission all          # every stage
    python3 -u -m src.run_submission block_train  # or one stage at a time:
        block_train feat_train train block_test feat_test predict
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from src.blocking.blocker import block_partition
from src.data.load import CACHE_ROOT
from src.decision.writer import run_validator, write_candidate_pairs, write_matching_results
from src.matching.pair_features import FEATURES, REC_COLS, build_features

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "output"
MODEL_DIR = ROOT / "models"
WORK = CACHE_ROOT / "run"
BLOCK_PARAMS = {"use_tfidf": 0}
FIT_S1_PER_COUNTRY = 150_000
HELD_S1_PER_COUNTRY = 50_000
SEED = 42
FEATURE_S1_CHUNK = 150_000

BLOCK_COLS = ["entity_id", "source", "country", "name_core", "addr_norm", "addr_components", "addr_numbers"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _countries(split: str) -> list[str]:
    return sorted(pl.scan_parquet(CACHE_ROOT / split / "master_norm.parquet").select("country").unique().collect()["country"].to_list())


def _train_sample_ids() -> pl.DataFrame:
    path = WORK / "train_sample.parquet"
    if path.exists():
        return pl.read_parquet(path)
    s1 = (
        pl.scan_parquet(CACHE_ROOT / "train" / "master_norm.parquet")
        .filter(pl.col("source") == "S1").select("entity_id", "country", "fold").collect()
    )
    parts = []
    for (country, fold), g in s1.group_by(["country", "fold"]):
        n = FIT_S1_PER_COUNTRY if fold == "fit" else HELD_S1_PER_COUNTRY
        parts.append(g.sample(min(n, g.height), seed=SEED))
    out = pl.concat(parts)
    WORK.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path)
    return out


# --------------------------------------------------------------------------

def stage_block(split: str) -> None:
    only = _train_sample_ids()["entity_id"] if split == "train" else None
    for country in _countries(split):
        out = WORK / split / f"cands_{country}.parquet"
        if out.exists():
            _log(f"block {split}/{country}: cached")
            continue
        t0 = time.time()
        mc = (
            pl.scan_parquet(CACHE_ROOT / split / "master_norm.parquet")
            .filter(pl.col("country") == country).select(BLOCK_COLS).collect()
        )
        s1 = mc.filter(pl.col("source") == "S1")
        if only is not None:
            s1 = s1.filter(pl.col("entity_id").is_in(only.implode()))
        cand = mc.filter(pl.col("source") != "S1")
        del mc
        _log(f"block {split}/{country}: {s1.height} S1 vs {cand.height} candidates")
        pairs = block_partition(s1, cand, BLOCK_PARAMS).drop("name_sim")
        out.parent.mkdir(parents=True, exist_ok=True)
        pairs.write_parquet(out)
        _log(f"block {split}/{country}: {pairs.height} pairs ({pairs.height / max(s1.height, 1):.1f}/S1) in {time.time() - t0:.0f}s")
        del s1, cand, pairs
        gc.collect()


def stage_features(split: str) -> None:
    for country in _countries(split):
        out = WORK / split / f"feats_{country}.parquet"
        if out.exists():
            _log(f"features {split}/{country}: cached")
            continue
        t0 = time.time()
        cols = REC_COLS + (["cluster_id"] if split == "train" else [])
        rec = (
            pl.scan_parquet(CACHE_ROOT / split / "master_norm.parquet")
            .filter(pl.col("country") == country).select(cols).collect()
        )
        pairs = pl.read_parquet(WORK / split / f"cands_{country}.parquet")
        # chunk by S1 entity (all of an S1's candidates stay in one chunk, so the
        # per-S1 relative features are exact) to bound the text join's memory
        s1_ids = pairs.select("s1_id").unique(maintain_order=True)["s1_id"]
        chunks = []
        for start in range(0, len(s1_ids), FEATURE_S1_CHUNK):
            sub = pairs.filter(pl.col("s1_id").is_in(s1_ids.slice(start, FEATURE_S1_CHUNK).implode()))
            chunks.append(build_features(sub, rec))
            gc.collect()
        feats = pl.concat(chunks)
        del chunks
        if split == "train":
            feats = feats.join(
                rec.select(pl.col("entity_id").alias("cand_id"), pl.col("cluster_id").alias("_cl")), on="cand_id", how="left"
            ).with_columns((pl.col("_cl") == pl.col("s1_id")).fill_null(False).cast(pl.Int8).alias("label")).drop("_cl")
        feats.write_parquet(out)
        _log(f"features {split}/{country}: {feats.height} rows in {time.time() - t0:.0f}s")
        del rec, pairs, feats
        gc.collect()


# --------------------------------------------------------------------------

def macro_f05_at(scored: pl.DataFrame, truth_n: pl.DataFrame, t: float) -> dict:
    """Vectorized macro F0.5 with one-owner assignment.
    scored: s1_id, cand_id, p, label.  truth_n: s1_id, n_true (ALL evaluated S1)."""
    pred = (
        scored.filter(pl.col("p") >= t)
        .sort("p", descending=True)
        .unique("cand_id", keep="first")          # each S2/S3 record goes to one S1 only
        .group_by("s1_id").agg(pl.len().alias("n_pred"), pl.col("label").sum().alias("n_tp"))
    )
    per = truth_n.join(pred, on="s1_id", how="left").fill_null(0).with_columns(
        (pl.col("n_tp") / pl.col("n_pred")).fill_nan(0).alias("P"),
        (pl.col("n_tp") / pl.col("n_true")).fill_nan(0).alias("R"),
    ).with_columns(
        pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
        .when(pl.col("n_tp") == 0).then(0.0)
        .otherwise(1.25 * pl.col("P") * pl.col("R") / (0.25 * pl.col("P") + pl.col("R")))
        .alias("f05")
    )
    single = per.filter(pl.col("n_true") == 0)
    return {
        "t": t,
        "macro_f05": float(per["f05"].mean()),
        "precision": float(pred["n_tp"].sum() / max(pred["n_pred"].sum(), 1)),
        "recall": float(pred["n_tp"].sum() / max(per["n_true"].sum(), 1)),
        "singleton_acc": float(single["f05"].mean()) if single.height else None,
    }


def stage_train() -> None:
    feats = pl.concat([pl.read_parquet(WORK / "train" / f"feats_{c}.parquet") for c in _countries("train")])
    ids = _train_sample_ids()
    feats = feats.join(ids.select(pl.col("entity_id").alias("s1_id"), "fold"), on="s1_id")
    fit, held = feats.filter(pl.col("fold") == "fit"), feats.filter(pl.col("fold") == "held_out")

    # early-stopping slice: 10% of fit S1 entities (grouped, so no leakage)
    es_ids = fit.select("s1_id").unique().sample(fraction=0.1, seed=SEED)
    es = fit.join(es_ids, on="s1_id", how="semi")
    tr = fit.join(es_ids, on="s1_id", how="anti")
    _log(f"train pairs={tr.height} (pos {tr['label'].mean():.3f}) | es={es.height} | held-out pairs={held.height}")

    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  max_bin=63, device_type="gpu", verbose=-1, seed=SEED)
    dtr = lgb.Dataset(tr.select(FEATURES).to_numpy(), tr["label"].to_numpy(), feature_name=FEATURES, free_raw_data=True)
    des = lgb.Dataset(es.select(FEATURES).to_numpy(), es["label"].to_numpy(), reference=dtr)
    t0 = time.time()
    try:
        booster = lgb.train(params, dtr, 2000, valid_sets=[des],
                            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    except lgb.basic.LightGBMError as e:
        _log(f"GPU training failed ({e}); falling back to CPU")
        params["device_type"] = "cpu"
        booster = lgb.train(params, dtr, 2000, valid_sets=[des],
                            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    _log(f"trained {booster.best_iteration} rounds on {params['device_type']} in {time.time() - t0:.0f}s")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL_DIR / "lgb_matcher.txt"), num_iteration=booster.best_iteration)

    # threshold on held-out S1 entities -- ALL of them, including singletons and
    # entities whose candidate list came back empty
    held_scored = held.select("s1_id", "cand_id", "label").with_columns(
        pl.Series("p", booster.predict(held.select(FEATURES).to_numpy(), num_iteration=booster.best_iteration))
    )
    master = pl.scan_parquet(CACHE_ROOT / "train" / "master_norm.parquet")
    held_ids = ids.filter(pl.col("fold") == "held_out").select(pl.col("entity_id").alias("s1_id"))
    n_true = (
        master.filter((pl.col("source") != "S1") & pl.col("cluster_id").is_not_null())
        .group_by("cluster_id").len().rename({"cluster_id": "s1_id", "len": "n_true"}).collect()
    )
    truth_n = held_ids.join(n_true, on="s1_id", how="left").fill_null(0)

    grid = np.round(np.arange(0.30, 0.96, 0.02), 2)
    results = [macro_f05_at(held_scored, truth_n, float(t)) for t in grid]
    best = max(results, key=lambda r: r["macro_f05"])
    for r in results:
        if abs(r["t"] - best["t"]) < 0.07:
            _log(f"  t={r['t']:.2f}  F0.5={r['macro_f05']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}  singleton_acc={r['singleton_acc']:.4f}")
    _log(f"BEST held-out macro F0.5 = {best['macro_f05']:.4f} at threshold {best['t']:.2f}")

    imp = sorted(zip(FEATURES, booster.feature_importance("gain")), key=lambda x: -x[1])[:12]
    _log("top features by gain: " + ", ".join(f"{f}" for f, _ in imp))
    (MODEL_DIR / "matcher_state.json").write_text(json.dumps({"threshold": best["t"], "held_out": best,
                                                              "best_iteration": booster.best_iteration,
                                                              "device": params["device_type"]}, indent=2))


def stage_predict() -> None:
    state = json.loads((MODEL_DIR / "matcher_state.json").read_text())
    t = state["threshold"]
    booster = lgb.Booster(model_file=str(MODEL_DIR / "lgb_matcher.txt"))

    scored_parts, cand_parts = [], []
    for country in _countries("test"):
        f = pl.read_parquet(WORK / "test" / f"feats_{country}.parquet")
        p = booster.predict(f.select(FEATURES).to_numpy())
        scored_parts.append(f.select("s1_id", "cand_id").with_columns(pl.Series("p", p)))
        cand_parts.append(pl.read_parquet(WORK / "test" / f"cands_{country}.parquet", columns=["s1_id", "cand_id"]))
        del f
        gc.collect()
    scored = pl.concat(scored_parts)
    cands = pl.concat(cand_parts)

    matches = (
        scored.filter(pl.col("p") >= t).sort("p", descending=True).unique("cand_id", keep="first")
        .group_by("s1_id").agg(pl.col("cand_id").sort_by("p", descending=True))
    )
    cand_lists = cands.group_by("s1_id").agg("cand_id")

    required = (
        pl.scan_parquet(CACHE_ROOT / "test" / "master.parquet")
        .filter(pl.col("source") == "S1").select("entity_id").collect()["entity_id"].to_list()
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_matching_results(OUT_DIR / "matching_results.tsv", dict(zip(matches["s1_id"], matches["cand_id"])), required)
    write_candidate_pairs(OUT_DIR / "candidate_pairs.tsv", dict(zip(cand_lists["s1_id"], cand_lists["cand_id"])), required)

    n_nonempty = matches.height
    _log(f"test: {len(required)} S1 | {n_nonempty} with >=1 match ({n_nonempty / len(required):.1%}) | "
         f"{matches['cand_id'].list.len().sum()} matched ids | {cands.height} candidate pairs | threshold {t}")
    passed, report = run_validator(OUT_DIR / "matching_results.tsv", OUT_DIR / "candidate_pairs.tsv")
    print(report)
    _log("VALIDATION PASSED" if passed else "VALIDATION FAILED")


STAGES = {
    "block_train": lambda: stage_block("train"),
    "feat_train": lambda: stage_features("train"),
    "train": stage_train,
    "block_test": lambda: stage_block("test"),
    "feat_test": lambda: stage_features("test"),
    "predict": stage_predict,
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or ["all"]
    order = list(STAGES) if wanted == ["all"] else wanted
    for name in order:
        t0 = time.time()
        _log(f"=== {name} ===")
        STAGES[name]()
        _log(f"=== {name} done in {time.time() - t0:.0f}s ===")
