import pytest

from er.metrics import f_beta_score, macro_f05, candidate_recall, oracle_macro_f05


# Exact score table from docs/IMPLEMENTATION_SPEC.md Phase 1.
@pytest.mark.parametrize(
    "true_set,pred_set,expected",
    [
        (set(), set(), 1.0),
        (set(), {"S2-1"}, 0.0),
        ({"S2-1", "S2-2"}, set(), 0.0),
        ({"S2-1", "S2-2"}, {"S2-1", "S2-2"}, 1.0),
        ({"S2-1", "S2-2"}, {"S2-1"}, 5 / 6),
        ({"S2-1", "S2-2"}, {"S2-1", "S2-2", "S3-9"}, 5 / 7),
    ],
)
def test_exact_score_table(true_set, pred_set, expected):
    assert f_beta_score(true_set, pred_set) == pytest.approx(expected)


def test_readme_worked_example():
    # student_resource/README.md: pred [S2-47,S2-193,S3-812], truth [S2-47,S3-812]
    true_set = {"S2-47", "S3-812"}
    pred_set = {"S2-47", "S2-193", "S3-812"}
    assert f_beta_score(true_set, pred_set) == pytest.approx(0.714, abs=1e-3)


def test_macro_averaging_across_queries():
    truth = {"S1-1": {"S2-1"}, "S1-2": set(), "S1-3": {"S2-2", "S2-3"}}
    predictions = {"S1-1": {"S2-1"}, "S1-2": set(), "S1-3": {"S2-2"}}
    macro, per_query = macro_f05(truth, predictions)
    expected = (1.0 + 1.0 + 5 / 6) / 3
    assert macro == pytest.approx(expected)
    assert len(per_query) == 3


def test_determinism():
    truth = {"S1-1": {"S2-1", "S2-2"}}
    predictions = {"S1-1": {"S2-1"}}
    m1, _ = macro_f05(truth, predictions)
    m2, _ = macro_f05(truth, predictions)
    assert m1 == m2


def test_rejects_duplicate_query_rows():
    truth = {"S1-1": {"S2-1"}}
    predictions = {"S1-1": {"S2-1"}}
    with pytest.raises(ValueError, match="duplicate"):
        macro_f05(truth, predictions, query_ids=["S1-1", "S1-1"])


def test_rejects_missing_query_from_truth():
    truth = {"S1-1": {"S2-1"}}
    predictions = {"S1-1": {"S2-1"}}
    with pytest.raises(ValueError, match="missing from truth"):
        macro_f05(truth, predictions, query_ids=["S1-1", "S1-2"])


def test_rejects_missing_query_from_predictions():
    truth = {"S1-1": {"S2-1"}, "S1-2": set()}
    predictions = {"S1-1": {"S2-1"}}
    with pytest.raises(ValueError, match="missing from predictions"):
        macro_f05(truth, predictions)


def test_singletons_included_and_scored():
    truth = {"S1-1": set()}
    predictions = {"S1-1": set()}
    macro, per_query = macro_f05(truth, predictions)
    assert macro == 1.0
    assert per_query[0].score == 1.0


def test_full_truth_retained_after_blocking_miss():
    # Candidate generation missed one of two true targets: score reflects the
    # miss, truth is not silently reduced to what was retrieved.
    truth = {"S1-1": {"S2-1", "S2-2"}}
    predictions = {"S1-1": {"S2-1"}}  # S2-2 was never retrieved
    macro, per_query = macro_f05(truth, predictions)
    assert macro == pytest.approx(5 / 6)
    assert per_query[0].true_set == {"S2-1", "S2-2"}


def test_candidate_recall_micro_macro():
    truth = {"S1-1": {"S2-1", "S2-2"}, "S1-2": {"S3-1"}, "S1-3": set()}
    candidates = {"S1-1": {"S2-1"}, "S1-2": {"S3-1", "S3-2"}, "S1-3": {"S2-9"}}
    stats = candidate_recall(truth, candidates)
    assert stats["micro_recall"] == pytest.approx(2 / 3)
    assert stats["macro_recall"] == pytest.approx((0.5 + 1.0) / 2)
    assert stats["non_singleton_queries"] == 2  # S1-3 is a singleton, excluded


def test_oracle_macro_f05_uses_truth_intersect_candidates():
    truth = {"S1-1": {"S2-1", "S2-2"}}
    candidates = {"S1-1": {"S2-1", "S3-9"}}  # S2-2 missed, S3-9 is a distractor
    macro, per_query = oracle_macro_f05(truth, candidates)
    # oracle predicts exactly truth ∩ candidates = {S2-1}
    assert per_query[0].pred_set == {"S2-1"}
    assert macro == pytest.approx(5 / 6)
