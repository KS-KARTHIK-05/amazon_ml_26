from er.validate import validate_outputs

REQUIRED = {"S1-1", "S1-2", "S1-3"}
VALID_TARGETS = {"S2-1", "S2-2", "S3-1"}


def _write(path, header, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for s1, ids in rows:
            f.write(f"{s1}\t{ids}\n")


def test_valid_output_passes(tmp_path):
    matching = tmp_path / "matching_results.tsv"
    candidate = tmp_path / "candidate_pairs.tsv"
    _write(
        matching, "source1_entity_id\tmatched_entity_ids",
        [("S1-1", "S2-1"), ("S1-2", ""), ("S1-3", "S3-1")],
    )
    _write(
        candidate, "source1_entity_id\tcandidate_entity_ids",
        [("S1-1", "S2-1,S2-2"), ("S1-2", ""), ("S1-3", "S3-1")],
    )
    result = validate_outputs(str(matching), str(candidate), REQUIRED, VALID_TARGETS)
    assert result.ok, result.errors


def test_missing_required_s1_fails(tmp_path):
    matching = tmp_path / "matching_results.tsv"
    candidate = tmp_path / "candidate_pairs.tsv"
    _write(matching, "source1_entity_id\tmatched_entity_ids", [("S1-1", "S2-1")])
    _write(candidate, "source1_entity_id\tcandidate_entity_ids", [("S1-1", "S2-1")])
    result = validate_outputs(str(matching), str(candidate), REQUIRED, VALID_TARGETS)
    assert not result.ok
    assert any("missing a row" in e for e in result.errors)


def test_duplicate_s1_row_fails(tmp_path):
    matching = tmp_path / "matching_results.tsv"
    candidate = tmp_path / "candidate_pairs.tsv"
    _write(
        matching, "source1_entity_id\tmatched_entity_ids",
        [("S1-1", "S2-1"), ("S1-1", "S2-2"), ("S1-2", ""), ("S1-3", "")],
    )
    _write(
        candidate, "source1_entity_id\tcandidate_entity_ids",
        [("S1-1", "S2-1,S2-2"), ("S1-2", ""), ("S1-3", "")],
    )
    result = validate_outputs(str(matching), str(candidate), REQUIRED, VALID_TARGETS)
    assert not result.ok
    assert any("duplicate source1_entity_id" in e for e in result.errors)


def test_self_match_fails(tmp_path):
    matching = tmp_path / "matching_results.tsv"
    candidate = tmp_path / "candidate_pairs.tsv"
    _write(
        matching, "source1_entity_id\tmatched_entity_ids",
        [("S1-1", "S1-2"), ("S1-2", ""), ("S1-3", "")],
    )
    _write(
        candidate, "source1_entity_id\tcandidate_entity_ids",
        [("S1-1", "S1-2"), ("S1-2", ""), ("S1-3", "")],
    )
    result = validate_outputs(str(matching), str(candidate), REQUIRED, VALID_TARGETS)
    assert not result.ok
    assert any("self-matches" in e for e in result.errors)


def test_unknown_target_id_fails_hard_unlike_official_validator(tmp_path):
    matching = tmp_path / "matching_results.tsv"
    candidate = tmp_path / "candidate_pairs.tsv"
    _write(
        matching, "source1_entity_id\tmatched_entity_ids",
        [("S1-1", "S2-999"), ("S1-2", ""), ("S1-3", "")],
    )
    _write(
        candidate, "source1_entity_id\tcandidate_entity_ids",
        [("S1-1", "S2-999"), ("S1-2", ""), ("S1-3", "")],
    )
    result = validate_outputs(str(matching), str(candidate), REQUIRED, VALID_TARGETS)
    assert not result.ok
    assert any("not present in the target pool" in e for e in result.errors)


def test_matched_outside_candidate_set_fails_hard(tmp_path):
    matching = tmp_path / "matching_results.tsv"
    candidate = tmp_path / "candidate_pairs.tsv"
    _write(
        matching, "source1_entity_id\tmatched_entity_ids",
        [("S1-1", "S2-1"), ("S1-2", ""), ("S1-3", "")],
    )
    _write(
        candidate, "source1_entity_id\tcandidate_entity_ids",
        [("S1-1", "S2-2"), ("S1-2", ""), ("S1-3", "")],  # S2-1 never a candidate
    )
    result = validate_outputs(str(matching), str(candidate), REQUIRED, VALID_TARGETS)
    assert not result.ok
    assert any("retrieval-lineage violation" in e for e in result.errors)


def test_intra_list_duplicate_fails(tmp_path):
    matching = tmp_path / "matching_results.tsv"
    candidate = tmp_path / "candidate_pairs.tsv"
    _write(
        matching, "source1_entity_id\tmatched_entity_ids",
        [("S1-1", "S2-1,S2-1"), ("S1-2", ""), ("S1-3", "")],
    )
    _write(
        candidate, "source1_entity_id\tcandidate_entity_ids",
        [("S1-1", "S2-1"), ("S1-2", ""), ("S1-3", "")],
    )
    result = validate_outputs(str(matching), str(candidate), REQUIRED, VALID_TARGETS)
    assert not result.ok
    assert any("duplicate id within" in e for e in result.errors)
