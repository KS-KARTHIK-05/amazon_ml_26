"""v2 matcher: stage-1 pairwise LightGBM (out-of-fold) + stage-2 entity-context
LightGBM, threshold tuned for macro F0.5, then test prediction.

Why two stages: stage 1 scores each (S1, record) pair in isolation, so it
can't see that a *different* S1 claims the record more strongly (the
"Caveran vs Kaveran at the same building" false merges), or that a record
with no address has the same name as this S1's confident matches (the
"Saint Cáthedral, empty address" misses). Stage 2 adds exactly that context,
computed from stage-1 probabilities.

Stage 1 is trained twice, once per half of the fit-fold S1 entities; each
half's pairs are scored by the model that never saw them (honest OOF inputs
for stage 2). Held-out and test pairs get the average of both models.

Usage (after run_submission block_* and feat_* stages):
    python3 -u -m src.two_stage stage1 score1 context stage2 predict
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from src.data.load import CACHE_ROOT
from src.decision.writer import run_validator, write_candidate_pairs, write_matching_results
from src.matching.pair_features import FEATURES
from src.run_submission import MODEL_DIR, OUT_DIR, WORK, _countries, _log, _train_sample_ids, macro_f05_at

SEED = 42
N_FOLDS = int(os.environ.get("ER_FOLDS", "2"))   # stage-1 out-of-fold models
S1_PER_MODEL = 350_000 if N_FOLDS == 2 else 420_000   # training entities per stage-1 model (RAM-bound)
S2_TRAIN_S1 = 350_000          # stage-2 training entities (fit fold); ~24M pairs keeps RAM ~14GB
LGB_BASE = dict(objective="binary", learning_rate=0.1, num_leaves=255, min_data_in_leaf=500,
                feature_fraction=0.85, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
                max_bin=63, device_type="cpu", verbose=-1, seed=SEED)

CONTEXT = [
    "p1", "p1_max_s1", "p1_2nd_s1", "p1_gap_top", "p1_rank_s1", "n_hi_s1", "n_mid_s1", "p1_sum_s1",
    "p_other", "p1_margin", "n_claims", "is_top_owner",
    "sim_top1_name", "sim_top1_addr", "sim_top2_name", "sim_top2_addr",
]
# ER_CE=1: cross-encoder score on uncertain-band pairs (src.ce_stage) as a stage-2 feature
USE_CE = os.environ.get("ER_CE", "0") == "1"
S2_FEATURES = FEATURES + CONTEXT + (["ce", "has_ce"] if USE_CE else [])


def _read_s2(part: Path, cols: list[str]) -> pl.DataFrame:
    base = [c for c in cols if c not in ("ce", "has_ce")]
    f = pl.read_parquet(part, columns=base)
    if USE_CE:
        ce_part = part.parent.parent / os.environ.get("ER_CE_DIR", "ce") / part.name
        ce = pl.read_parquet(ce_part) if ce_part.exists() else pl.DataFrame(
            schema={"s1_id": pl.String, "cand_id": pl.String, "ce": pl.Float32})
        f = f.join(ce, on=["s1_id", "cand_id"], how="left").with_columns(
            pl.col("ce").is_not_null().cast(pl.Float32).alias("has_ce"), pl.col("ce").fill_null(-1.0))
    return f.select(cols)


def _parts(split: str, sub: str) -> list[Path]:
    return sorted((WORK / split / sub).glob("*.parquet"))


def _kfold(expr: pl.Expr) -> pl.Expr:
    return (expr.hash(seed=SEED) % N_FOLDS).cast(pl.Int8)


def _fold_map() -> pl.DataFrame:
    """fold: fit / held_out split; k: stage-1 out-of-fold group of fit entities."""
    ids = _train_sample_ids()
    return ids.select(pl.col("entity_id").alias("s1_id"), "fold").with_columns(_kfold(pl.col("s1_id")).alias("k"))


def _model_path(k: int) -> Path:
    # never the v3 "stage1_half" names: their half convention is the reverse of fold k here
    return MODEL_DIR / f"stage1_k{N_FOLDS}_{k}.txt"


def _train(params: dict, X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray, names: list[str],
           rounds: int = 3000) -> lgb.Booster:
    dtr = lgb.Dataset(X, y, feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(Xv, yv, reference=dtr)
    cb = [lgb.early_stopping(60, verbose=False), lgb.log_evaluation(250)]
    try:
        b = lgb.train(params, dtr, rounds, valid_sets=[dva], callbacks=cb)
        _log(f"trained on {params['device_type']}")
        return b
    except lgb.basic.LightGBMError as e:
        _log(f"{params['device_type']} training failed ({e}); CPU fallback")
        return lgb.train({**params, "device_type": "cpu"}, dtr, rounds, valid_sets=[dva], callbacks=cb)


# --------------------------------------------------------------------------
# stage 1

def stage1() -> None:
    folds = _fold_map()
    fit = folds.filter(pl.col("fold") == "fit")
    for h in range(N_FOLDS):
        path = _model_path(h)
        if path.exists():
            _log(f"stage1 model {h}: cached")
            continue
        # model h is the out-of-fold scorer for fold h (for 2 folds this is v3's
        # "train on one half, score the other" with the halves' names swapped)
        ids = fit.filter(pl.col("k") != h).sample(fraction=1.0, seed=SEED + h)
        tr_ids = ids.head(S1_PER_MODEL).select("s1_id")
        va_ids = ids.slice(S1_PER_MODEL, 40_000).select("s1_id")
        tr, va = [], []
        for part in _parts("train", "feats"):
            f = pl.read_parquet(part, columns=["s1_id", *FEATURES, "label"])
            tr.append(f.join(tr_ids, on="s1_id", how="semi"))
            va.append(f.join(va_ids, on="s1_id", how="semi"))
        tr, va = pl.concat(tr), pl.concat(va)
        _log(f"stage1 model {h}/{N_FOLDS}: train {tr.height} pairs (pos {tr['label'].mean():.3f}), valid {va.height}")
        t0 = time.time()
        b = _train(LGB_BASE, tr.select(FEATURES).to_numpy(), tr["label"].to_numpy(),
                   va.select(FEATURES).to_numpy(), va["label"].to_numpy(), FEATURES)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        b.save_model(str(path), num_iteration=b.best_iteration)
        _log(f"stage1 model {h}/{N_FOLDS}: {b.best_iteration} rounds in {time.time() - t0:.0f}s")
        del tr, va
        gc.collect()


def score1() -> None:
    """p1 for every pair: fit pairs from the model that never saw their fold, others averaged."""
    m = [lgb.Booster(model_file=str(_model_path(h))) for h in range(N_FOLDS)]
    folds = _fold_map()
    for split in ("train", "test"):
        out_dir = WORK / split / "p1"
        out_dir.mkdir(parents=True, exist_ok=True)
        for part in _parts(split, "feats"):
            out = out_dir / part.name
            if out.exists():
                continue
            f = pl.read_parquet(part, columns=["s1_id", "cand_id", *FEATURES])
            X = f.select(FEATURES).to_numpy()
            if split == "train":
                h = f.select("s1_id").join(folds, on="s1_id", how="left")
                is_fit = (h["fold"] == "fit").to_numpy()
                k = h["k"].fill_null(-1).to_numpy()
                p = np.zeros(len(X))
                # fit pairs: only the model that never saw their fold
                for j in range(N_FOLDS):
                    sel = is_fit & (k == j)
                    if sel.any():
                        p[sel] = m[j].predict(X[sel])
                rest = ~is_fit
                if rest.any():
                    p[rest] = np.mean([mj.predict(X[rest]) for mj in m], axis=0)
            else:
                p = np.mean([mj.predict(X) for mj in m], axis=0)
            f.select("s1_id", "cand_id").with_columns(pl.Series("p1", p.astype(np.float32))).write_parquet(out)
            _log(f"score1 {split}/{part.name}: {f.height} pairs")
            del f, X
            gc.collect()


# --------------------------------------------------------------------------
# stage-2 context

def _cand_competition(split: str, country: str) -> pl.DataFrame:
    """Per candidate record: best / second-best stage-1 score over ALL S1s."""
    p = pl.concat([pl.read_parquet(x) for x in _parts(split, "p1") if x.name.startswith(country + "_")])
    top = p.sort("p1", descending=True).group_by("cand_id", maintain_order=True).agg(
        pl.col("p1").first().alias("c_top1"),
        pl.col("s1_id").first().alias("c_top1_s1"),
        pl.col("p1").slice(1, 1).first().fill_null(0.0).alias("c_top2"),
        (pl.col("p1") > 0.5).sum().cast(pl.Float32).alias("n_claims"),
    )
    return top


def context() -> None:
    for split in ("train", "test"):
        out_dir = WORK / split / "s2"
        out_dir.mkdir(parents=True, exist_ok=True)
        for country in _countries(split):
            if (out_dir / f"{country}.done").exists():
                _log(f"context {split}/{country}: cached")
                continue
            t0 = time.time()
            comp = _cand_competition(split, country)
            rec = (
                pl.scan_parquet(CACHE_ROOT / split / "master_norm.parquet")
                .filter(pl.col("country") == country).select("entity_id", "name_core", "addr_norm").collect()
            )
            for part in [x for x in _parts(split, "feats") if x.name.startswith(country + "_")]:
                f = pl.read_parquet(part).join(pl.read_parquet(WORK / split / "p1" / part.name), on=["s1_id", "cand_id"])
                f = _add_context(f, comp, rec)
                f.write_parquet(out_dir / part.name)
                del f
                gc.collect()
            (out_dir / f"{country}.done").touch()
            _log(f"context {split}/{country} in {time.time() - t0:.0f}s")
            del comp, rec
            gc.collect()


def _add_context(f: pl.DataFrame, comp: pl.DataFrame, rec: pl.DataFrame) -> pl.DataFrame:
    g = "s1_id"
    f = f.join(comp, on="cand_id", how="left").with_columns(
        pl.when(pl.col("c_top1_s1") == pl.col("s1_id")).then(pl.col("c_top2")).otherwise(pl.col("c_top1")).alias("p_other"),
        (pl.col("c_top1_s1") == pl.col("s1_id")).cast(pl.Float32).alias("is_top_owner"),
    ).with_columns(
        (pl.col("p1") - pl.col("p_other")).alias("p1_margin"),
        pl.col("p1").max().over(g).alias("p1_max_s1"),
        pl.col("p1").sort(descending=True).slice(1, 1).first().over(g).fill_null(0.0).alias("p1_2nd_s1"),
        pl.col("p1").rank("min", descending=True).over(g).cast(pl.Float32).alias("p1_rank_s1"),
        (pl.col("p1") > 0.9).sum().over(g).cast(pl.Float32).alias("n_hi_s1"),
        ((pl.col("p1") > 0.3) & (pl.col("p1") <= 0.9)).sum().over(g).cast(pl.Float32).alias("n_mid_s1"),
        pl.col("p1").sum().over(g).alias("p1_sum_s1"),
    ).with_columns((pl.col("p1_max_s1") - pl.col("p1")).alias("p1_gap_top"))

    # similarity of each candidate to its S1's two most confident candidates
    # (excluding itself): does it look like the variants we already trust?
    ranked = f.select("s1_id", "cand_id", "p1").sort(["s1_id", "p1"], descending=[False, True])
    tops = ranked.group_by("s1_id", maintain_order=True).agg(pl.col("cand_id").head(3).alias("tops"))
    f = f.join(tops, on="s1_id", how="left").with_columns(
        pl.col("tops").list.set_difference(pl.concat_list(pl.col("cand_id"))).list.head(2).alias("tops")
    ).with_columns(
        pl.col("tops").list.get(0, null_on_oob=True).alias("t1"),
        pl.col("tops").list.get(1, null_on_oob=True).alias("t2"),
    ).drop("tops")
    names = rec.select(pl.col("entity_id").alias("k"), "name_core", "addr_norm")
    f = (
        f.join(names.rename({"k": "cand_id", "name_core": "_cn", "addr_norm": "_ca"}), on="cand_id", how="left")
        .join(names.rename({"k": "t1", "name_core": "_t1n", "addr_norm": "_t1a"}), on="t1", how="left")
        .join(names.rename({"k": "t2", "name_core": "_t2n", "addr_norm": "_t2a"}), on="t2", how="left")
    )
    sims = {}
    for out, a, b in [("sim_top1_name", "_cn", "_t1n"), ("sim_top1_addr", "_ca", "_t1a"),
                      ("sim_top2_name", "_cn", "_t2n"), ("sim_top2_addr", "_ca", "_t2a")]:
        va, vb = f[a].fill_null("").to_list(), f[b].fill_null("").to_list()
        v = cpdist(va, vb, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        sims[out] = np.where(f[b].is_null().to_numpy(), -1.0, v).astype(np.float32)
    f = f.with_columns(pl.Series(k, v) for k, v in sims.items())
    return f.drop("_cn", "_ca", "_t1n", "_t1a", "_t2n", "_t2a", "t1", "t2", "c_top1", "c_top1_s1", "c_top2").with_columns(
        [pl.col(c).cast(pl.Float32) for c in CONTEXT]
    )


# --------------------------------------------------------------------------
# stage 2

def _load_s2(split: str, ids: pl.DataFrame | None, cols: list[str]) -> pl.DataFrame:
    out = []
    for part in _parts(split, "s2"):
        f = _read_s2(part, cols)
        out.append(f if ids is None else f.join(ids, on="s1_id", how="semi"))
    return pl.concat(out)


def _truth_n(ids: pl.DataFrame) -> pl.DataFrame:
    n_true = (
        pl.scan_parquet(CACHE_ROOT / "train" / "master_norm.parquet")
        .filter((pl.col("source") != "S1") & pl.col("cluster_id").is_not_null())
        .group_by("cluster_id").len().rename({"cluster_id": "s1_id", "len": "n_true"}).collect()
    )
    return ids.join(n_true, on="s1_id", how="left").fill_null(0)


def _sweep(scored: pl.DataFrame, truth: pl.DataFrame, col: str) -> dict:
    s = scored.rename({col: "p"})
    res = [macro_f05_at(s, truth, float(t)) for t in np.round(np.arange(0.30, 0.97, 0.02), 2)]
    return max(res, key=lambda r: r["macro_f05"])


def stage2() -> None:
    folds = _fold_map()
    fit = folds.filter(pl.col("fold") == "fit").sample(fraction=1.0, seed=SEED + 7)
    tr_ids, va_ids = fit.head(S2_TRAIN_S1).select("s1_id"), fit.slice(S2_TRAIN_S1, 40_000).select("s1_id")
    held_ids = folds.filter(pl.col("fold") == "held_out").select("s1_id")
    cols = ["s1_id", "cand_id", *S2_FEATURES, "label"]
    tr, va = _load_s2("train", tr_ids, cols), _load_s2("train", va_ids, cols)
    _log(f"stage2: train {tr.height} pairs, valid {va.height}")
    t0 = time.time()
    b = _train(LGB_BASE, tr.select(S2_FEATURES).to_numpy(), tr["label"].to_numpy(),
               va.select(S2_FEATURES).to_numpy(), va["label"].to_numpy(), S2_FEATURES)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    b.save_model(str(MODEL_DIR / "stage2.txt"), num_iteration=b.best_iteration)
    _log(f"stage2: {b.best_iteration} rounds in {time.time() - t0:.0f}s")
    del tr, va
    gc.collect()

    held = _load_s2("train", held_ids, cols)
    held = held.with_columns(pl.Series("p2", b.predict(held.select(S2_FEATURES).to_numpy()).astype(np.float32)))
    truth = _truth_n(held_ids)
    base = _sweep(held.select("s1_id", "cand_id", "label", "p1"), truth, "p1")
    best = _sweep(held.select("s1_id", "cand_id", "label", "p2"), truth, "p2")
    _log(f"HELD-OUT (all {held_ids.height} held-out S1)  stage1 only: F0.5={base['macro_f05']:.4f} @ {base['t']}  "
         f"P={base['precision']:.4f} R={base['recall']:.4f}")
    _log(f"HELD-OUT  stage2: F0.5={best['macro_f05']:.4f} @ {best['t']}  P={best['precision']:.4f} "
         f"R={best['recall']:.4f} singleton_acc={best['singleton_acc']:.4f}")
    imp = sorted(zip(S2_FEATURES, b.feature_importance("gain")), key=lambda x: -x[1])[:12]
    _log("stage2 top features: " + ", ".join(f for f, _ in imp))
    (MODEL_DIR / "v2_state.json").write_text(json.dumps(
        {"threshold": best["t"], "held_out_stage2": best, "held_out_stage1": base,
         "stage2_rounds": b.best_iteration}, indent=2))


def predict() -> None:
    state = json.loads((MODEL_DIR / "v2_state.json").read_text())
    t = state["threshold"]
    b = lgb.Booster(model_file=str(MODEL_DIR / "stage2.txt"))
    scored = []
    for part in _parts("test", "s2"):
        f = _read_s2(part, ["s1_id", "cand_id", *S2_FEATURES])
        scored.append(f.select("s1_id", "cand_id").with_columns(
            pl.Series("p", b.predict(f.select(S2_FEATURES).to_numpy()).astype(np.float32))))
        del f
        gc.collect()
    scored = pl.concat(scored)
    matches = (
        scored.filter(pl.col("p") >= t).sort("p", descending=True).unique("cand_id", keep="first")
        .group_by("s1_id").agg(pl.col("cand_id").sort_by("p", descending=True))
    )
    cands = pl.concat([pl.read_parquet(WORK / "test" / f"cands_{c}.parquet", columns=["s1_id", "cand_id"])
                       for c in _countries("test")]).group_by("s1_id").agg("cand_id")
    required = (
        pl.scan_parquet(CACHE_ROOT / "test" / "master.parquet").filter(pl.col("source") == "S1")
        .select("entity_id").collect()["entity_id"].to_list()
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_matching_results(OUT_DIR / "matching_results.tsv", dict(zip(matches["s1_id"], matches["cand_id"])), required)
    write_candidate_pairs(OUT_DIR / "candidate_pairs.tsv", dict(zip(cands["s1_id"], cands["cand_id"])), required)
    _log(f"test: {len(required)} S1 | {matches.height} with >=1 match ({matches.height / len(required):.1%}) | "
         f"{matches['cand_id'].list.len().sum()} matched ids | threshold {t}")
    passed, report = run_validator(OUT_DIR / "matching_results.tsv", OUT_DIR / "candidate_pairs.tsv")
    print(report)
    _log("VALIDATION PASSED" if passed else "VALIDATION FAILED")


STAGES = {"stage1": stage1, "score1": score1, "context": context, "stage2": stage2, "predict": predict}

if __name__ == "__main__":
    for name in sys.argv[1:]:
        t0 = time.time()
        _log(f"=== {name} ===")
        STAGES[name]()
        _log(f"=== {name} done in {time.time() - t0:.0f}s ===")
