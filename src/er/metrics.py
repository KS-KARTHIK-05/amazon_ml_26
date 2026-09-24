"""Exact macro F0.5 scoring and retrieval/oracle metrics.

The challenge metric (student_resource/README.md, "Evaluation Criteria"):
per-S1 F_beta with beta=0.5, macro-averaged over all S1 queries including
singletons. Implements agent.md invariant 10 exactly, including its edge
cases: an empty true set scores 1 only for an empty prediction, else 0.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

BETA_SQUARED = 0.5**2  # 0.25


def f_beta_score(true_set: frozenset | set, pred_set: frozenset | set, beta: float = 0.5) -> float:
    """Per-query F_beta. true_set/pred_set are sets of target entity ids."""
    if not true_set:
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        return 0.0
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    beta_sq = beta * beta
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp
    if denominator == 0:
        return 0.0
    return numerator / denominator


@dataclass(frozen=True)
class QueryScore:
    query_id: str
    true_set: frozenset
    pred_set: frozenset
    score: float


def macro_f05(
    truth: Mapping[str, Iterable[str]],
    predictions: Mapping[str, Iterable[str]],
    query_ids: Iterable[str] | None = None,
) -> tuple[float, list[QueryScore]]:
    """Macro-average F0.5 over ``query_ids`` (default: every key in ``truth``).

    Raises on missing or duplicate query coverage rather than silently
    dropping rows, per agent.md invariant 10 ("reject duplicate/missing query
    rows rather than hiding them").
    """
    if query_ids is None:
        query_ids = list(truth.keys())
    else:
        query_ids = list(query_ids)

    seen = set()
    for qid in query_ids:
        if qid in seen:
            raise ValueError(f"duplicate query id in scoring population: {qid}")
        seen.add(qid)
        if qid not in truth:
            raise ValueError(f"query id missing from truth: {qid}")

    missing_predictions = seen - set(predictions.keys())
    if missing_predictions:
        example = sorted(missing_predictions)[:5]
        raise ValueError(
            f"{len(missing_predictions)} query id(s) missing from predictions, "
            f"e.g. {example}"
        )

    per_query: list[QueryScore] = []
    for qid in query_ids:
        true_set = frozenset(truth[qid])
        pred_set = frozenset(predictions[qid])
        score = f_beta_score(true_set, pred_set, beta=0.5)
        per_query.append(QueryScore(qid, true_set, pred_set, score))

    macro = sum(qs.score for qs in per_query) / len(per_query) if per_query else 0.0
    return macro, per_query


def candidate_recall(
    truth: Mapping[str, Iterable[str]],
    candidates: Mapping[str, Iterable[str]],
) -> dict[str, float]:
    """Micro and macro recall of candidates against full truth, over
    non-singleton queries (queries with at least one true target)."""
    micro_tp = 0
    micro_fn = 0
    macro_scores = []
    for qid, true_ids in truth.items():
        true_set = set(true_ids)
        if not true_set:
            continue
        cand_set = set(candidates.get(qid, ()))
        hit = len(true_set & cand_set)
        micro_tp += hit
        micro_fn += len(true_set) - hit
        macro_scores.append(hit / len(true_set))
    micro = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else float("nan")
    macro = sum(macro_scores) / len(macro_scores) if macro_scores else float("nan")
    return {
        "micro_recall": micro,
        "macro_recall": macro,
        "non_singleton_queries": len(macro_scores),
    }


def complete_family_coverage(
    truth: Mapping[str, Iterable[str]],
    candidates: Mapping[str, Iterable[str]],
) -> float:
    """Fraction of non-singleton queries whose *entire* true set is covered
    by candidates (a stricter, per-query pass/fail complement to recall)."""
    total = 0
    complete = 0
    for qid, true_ids in truth.items():
        true_set = set(true_ids)
        if not true_set:
            continue
        total += 1
        cand_set = set(candidates.get(qid, ()))
        if true_set <= cand_set:
            complete += 1
    return complete / total if total else float("nan")


def oracle_macro_f05(
    truth: Mapping[str, Iterable[str]],
    candidates: Mapping[str, Iterable[str]],
) -> tuple[float, list[QueryScore]]:
    """Best achievable macro F0.5 if the matcher predicted exactly
    ``truth ∩ candidates`` for every query (and nothing for singletons that
    have no candidates to accept, since accepting a false positive there only
    hurts). Uses complete truth; never inject truth into ``candidates``."""
    oracle_predictions = {
        qid: set(true_ids) & set(candidates.get(qid, ()))
        for qid, true_ids in truth.items()
    }
    return macro_f05(truth, oracle_predictions, query_ids=truth.keys())
