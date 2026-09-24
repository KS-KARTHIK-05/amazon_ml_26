from er.truth import (
    TruthParseResult,
    build_positive_families,
    multi_owner_targets,
    parse_ground_truth_chunk,
)


def test_parse_basic_and_empty_matches():
    result = TruthParseResult()
    rows = [
        ("S1-1", "S2-1,S3-1"),
        ("S1-2", ""),  # singleton, no matches
        ("S1-3", "S2-2"),
    ]
    parse_ground_truth_chunk(rows, result)
    assert result.s1_truth["S1-1"] == {"S2-1", "S3-1"}
    assert result.s1_truth["S1-2"] == set()
    assert result.s1_truth["S1-3"] == {"S2-2"}
    assert result.errors == []


def test_self_target_is_flagged_as_error():
    result = TruthParseResult()
    parse_ground_truth_chunk([("S1-1", "S1-2")], result)
    assert any("S1 id as a target" in e for e in result.errors)
    # The bad target is dropped, not silently kept as a valid match.
    assert result.s1_truth["S1-1"] == set()


def test_duplicate_s1_row_is_flagged():
    result = TruthParseResult()
    parse_ground_truth_chunk([("S1-1", "S2-1"), ("S1-1", "S2-2")], result)
    assert any("duplicate source1_entity_id" in e for e in result.errors)


def test_duplicate_target_within_row_is_flagged():
    result = TruthParseResult()
    parse_ground_truth_chunk([("S1-1", "S2-1,S2-1")], result)
    assert any("duplicate target id" in e for e in result.errors)


def test_wrong_prefix_target_is_flagged():
    result = TruthParseResult()
    parse_ground_truth_chunk([("S1-1", "X-1")], result)
    assert any("no S2-/S3- prefix" in e for e in result.errors)


def test_positive_family_components_merge_multi_owner_targets():
    result = TruthParseResult()
    rows = [
        ("S1-1", "S2-1"),
        ("S1-2", "S2-1"),  # shares target S2-1 with S1-1 -> same family
        ("S1-3", "S3-9"),  # unrelated singleton match
        ("S1-4", ""),  # true singleton, no matches
    ]
    parse_ground_truth_chunk(rows, result)
    families = build_positive_families(result)
    assert families["S1-1"] == families["S1-2"]
    assert families["S1-3"] != families["S1-1"]
    assert families["S1-4"] != families["S1-1"]
    # Every S1 gets a family, including true singletons.
    assert set(families.keys()) == {"S1-1", "S1-2", "S1-3", "S1-4"}


def test_family_ids_are_deterministic_min_member():
    result = TruthParseResult()
    parse_ground_truth_chunk([("S1-9", "S2-1"), ("S1-2", "S2-1")], result)
    families = build_positive_families(result)
    assert families["S1-9"] == families["S1-2"] == "S1-2"


def test_multi_owner_targets_detected():
    result = TruthParseResult()
    parse_ground_truth_chunk(
        [("S1-1", "S2-1"), ("S1-2", "S2-1"), ("S1-3", "S2-2")], result
    )
    owners = multi_owner_targets(result)
    assert owners == {"S2-1": 2}
