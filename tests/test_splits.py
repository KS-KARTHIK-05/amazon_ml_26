from er.splits import (
    HOLDOUT,
    TRAIN,
    TUNING,
    SplitConfig,
    assign_splits,
    split_counts,
    split_fingerprint,
)


def _synthetic_families(n_us=800, n_india=200):
    family_country = {}
    for i in range(n_us):
        family_country[f"S1-us-{i}"] = "US"
    for i in range(n_india):
        family_country[f"S1-in-{i}"] = "India"
    return family_country


def test_determinism_same_seed_same_assignment():
    family_country = _synthetic_families()
    a1 = assign_splits(family_country, SplitConfig(seed=42))
    a2 = assign_splits(family_country, SplitConfig(seed=42))
    assert a1 == a2
    assert split_fingerprint(a1) == split_fingerprint(a2)


def test_different_seed_gives_different_assignment():
    family_country = _synthetic_families()
    a1 = assign_splits(family_country, SplitConfig(seed=42))
    a2 = assign_splits(family_country, SplitConfig(seed=7))
    assert a1 != a2


def test_every_family_assigned_exactly_one_split():
    family_country = _synthetic_families()
    assignment = assign_splits(family_country)
    assert set(assignment.keys()) == set(family_country.keys())
    assert set(assignment.values()) <= {TRAIN, TUNING, HOLDOUT}


def test_stratified_by_country_within_tolerance():
    family_country = _synthetic_families(n_us=800, n_india=200)
    assignment = assign_splits(family_country, SplitConfig(seed=42))
    us_split = [assignment[f] for f in family_country if family_country[f] == "US"]
    india_split = [assignment[f] for f in family_country if family_country[f] == "India"]
    us_train_frac = us_split.count(TRAIN) / len(us_split)
    india_train_frac = india_split.count(TRAIN) / len(india_split)
    assert abs(us_train_frac - 0.8) < 0.02
    assert abs(india_train_frac - 0.8) < 0.02


def test_fingerprint_changes_with_assignment():
    family_country = _synthetic_families()
    a1 = assign_splits(family_country, SplitConfig(seed=42))
    a2 = dict(a1)
    some_key = next(iter(a2))
    a2[some_key] = HOLDOUT if a2[some_key] != HOLDOUT else TRAIN
    assert split_fingerprint(a1) != split_fingerprint(a2)


def test_counts_sum_to_total_families():
    family_country = _synthetic_families()
    assignment = assign_splits(family_country)
    counts = split_counts(assignment)
    assert sum(counts.values()) == len(family_country)
