"""Macro F0.5 scorer, matching the challenge PDF's definition exactly.

Per S1 entity: F0.5 = 1.25 * P * R / (0.25 * P + R), with a true singleton
(no ground-truth matches) scoring 1.0 for an empty prediction and 0.0 for any
non-empty prediction. The final score is the unweighted mean over all S1
entities (macro-average), singletons included.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def f05_for_entity(predicted: Iterable[str], truth: Iterable[str]) -> float:
    predicted_set = set(predicted)
    truth_set = set(truth)

    if not truth_set:
        return 1.0 if not predicted_set else 0.0

    if not predicted_set:
        return 0.0

    tp = len(predicted_set & truth_set)
    if tp == 0:
        return 0.0

    precision = tp / len(predicted_set)
    recall = tp / len(truth_set)
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom


def macro_f05(
    predictions: Mapping[str, Iterable[str]],
    truth: Mapping[str, Iterable[str]],
) -> dict:
    """Score predictions against truth for every S1 id present in `truth`.

    `predictions` may omit ids (treated as empty prediction) but every id in
    `truth` is scored. Returns a dict with the macro score plus micro
    precision/recall/singleton-accuracy diagnostics.
    """
    scores = []
    tp_total = fp_total = fn_total = 0
    singleton_correct = singleton_total = 0
    matched_correct = matched_total = 0

    for s1_id, true_ids in truth.items():
        true_set = set(true_ids)
        pred_set = set(predictions.get(s1_id, ()))
        score = f05_for_entity(pred_set, true_set)
        scores.append(score)

        tp = len(pred_set & true_set)
        tp_total += tp
        fp_total += len(pred_set - true_set)
        fn_total += len(true_set - pred_set)

        if not true_set:
            singleton_total += 1
            singleton_correct += int(not pred_set)
        else:
            matched_total += 1
            matched_correct += int(score > 0)

    n = len(scores)
    macro = sum(scores) / n if n else 0.0
    micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0

    return {
        "macro_f05": macro,
        "n_entities": n,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "singleton_accuracy": singleton_correct / singleton_total if singleton_total else None,
        "n_singletons": singleton_total,
        "matched_hit_rate": matched_correct / matched_total if matched_total else None,
        "n_matched": matched_total,
    }


def _test_pdf_example():
    score = f05_for_entity(
        predicted=["S2-00047", "S2-00193", "S3-00812"],
        truth=["S2-00047", "S3-00812"],
    )
    assert abs(score - 0.714) < 0.001, f"expected ~0.714, got {score}"

    assert f05_for_entity([], []) == 1.0
    assert f05_for_entity(["S2-1"], []) == 0.0
    assert f05_for_entity([], ["S2-1"]) == 0.0
    assert f05_for_entity(["S2-1"], ["S2-1"]) == 1.0

    result = macro_f05(
        predictions={
            "S1-00001": ["S2-00047", "S2-00193", "S3-00812"],
            "S1-00002": ["S3-00004"],
            "S1-00003": [],
        },
        truth={
            "S1-00001": ["S2-00047", "S3-00812"],
            "S1-00002": ["S3-00004"],
            "S1-00003": [],
        },
    )
    expected_macro = (0.714 + 1.0 + 1.0) / 3
    assert abs(result["macro_f05"] - expected_macro) < 0.001, result
    print("all f05_score tests passed:", result)


if __name__ == "__main__":
    _test_pdf_example()
