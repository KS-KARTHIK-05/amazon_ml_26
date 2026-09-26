"""Per-entity decision that maximizes expected F0.5, instead of a global
probability threshold.

For one S1 with candidate probabilities p_1 >= p_2 >= ... (after one-owner
resolution), predicting the top-k gives, in plug-in expectation,

    E[F0.5 | k] ~= 1.25 * sum_{i<=k} p_i / (0.25 * T + k),   k >= 1
    E[F0.5 | 0]  = P(no true match) ~= prod_i (1 - p_i) * (1 - miss)

where T = sum_i p_i + extra is the expected number of true matches
(`extra` accounts for true matches blocking never retrieved). We pick the k
with the highest value. Because F0.5 is per entity, this naturally adds a
0.6-probability third record when the entity already has confident matches,
and refuses a lone 0.6 candidate for an entity that looks like a singleton.

`power` optionally sharpens/flattens the probabilities (calibration knob).
"""

from __future__ import annotations

import polars as pl


def owner_resolve(scored: pl.DataFrame) -> pl.DataFrame:
    """Each S2/S3 record keeps only its highest-probability S1."""
    return scored.sort("p", descending=True).unique("cand_id", keep="first")


def expected_f_select(scored: pl.DataFrame, extra: float = 0.0, power: float = 1.0,
                      min_p: float = 0.05, singleton_bias: float = 1.0) -> pl.DataFrame:
    """scored: s1_id, cand_id, p (already owner-resolved). Returns selected rows."""
    s = scored.with_columns((pl.col("p") ** power).alias("q")).filter(pl.col("q") >= min_p)
    s = s.sort(["s1_id", "q"], descending=[False, True]).with_columns(
        pl.col("q").cum_sum().over("s1_id").alias("tp_k"),
        pl.int_range(1, pl.len() + 1).over("s1_id").alias("k"),
        (pl.col("q").sum().over("s1_id") + extra).alias("T"),
        ((1 - pl.col("q")).log().sum().over("s1_id").exp() * singleton_bias).alias("f_empty"),
    ).with_columns((1.25 * pl.col("tp_k") / (0.25 * pl.col("T") + pl.col("k"))).alias("ef"))
    best = s.group_by("s1_id").agg(
        pl.col("ef").max().alias("best_ef"),
        pl.col("k").get(pl.col("ef").arg_max()).alias("best_k"),
        pl.col("f_empty").first(),
    ).filter(pl.col("best_ef") > pl.col("f_empty"))
    return s.join(best.select("s1_id", "best_k"), on="s1_id").filter(pl.col("k") <= pl.col("best_k")).select(
        "s1_id", "cand_id", "p"
    )
