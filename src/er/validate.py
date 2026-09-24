"""Strict output validation — stricter than the supplied official validator.

`student_resource/utils/validate_submission.py` skips ID existence by default
and only *warns* on missing candidate-subset coverage. Per
docs/IMPLEMENTATION_SPEC.md Phase 6, our own release checks must enforce ID
existence, candidate coverage, and subset membership as hard failures. This
module is meant to run in addition to the official validator, not instead of
it, and is designed to work from streamed rows / precomputed ID sets rather
than assuming a small file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DELIM = "\t"
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    row_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def read_ids(path: str) -> set[str]:
    with open(path, encoding="utf-8") as f:
        next(f, None)
        return {line.split(DELIM, 1)[0].strip() for line in f if line.strip()}


def _parse_id_list_file(
    path: str,
    expected_header: list[str],
    col_label: str,
    required_ids: set[str],
    valid_target_ids: set[str] | None,
    result: ValidationResult,
) -> dict[str, set[str]] | None:
    if not os.path.isfile(path):
        result.errors.append(f"File not found: {path}")
        return None
    name = os.path.basename(path)
    mapping: dict[str, set[str]] = {}
    seen: set[str] = set()
    dup_rows: set[str] = set()
    intra_dupes: set[str] = set()
    self_matches: set[str] = set()
    wrong_prefix: set[str] = set()
    unknown: set[str] = set()

    with open(path, encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            result.errors.append(f"{name} is empty.")
            return None
        cols = [c.strip().lower() for c in header_line.rstrip("\n").split(DELIM)]
        if cols != expected_header:
            result.errors.append(
                f"{name}: unexpected header {cols}; expected {expected_header}."
            )
            return None

        for line_num, line in enumerate(f, start=2):
            s1, tab, rest = line.partition(DELIM)
            if not tab:
                if s1.strip():
                    result.errors.append(
                        f"{name}: malformed row (no tab) at line {line_num}: {line.rstrip()!r}"
                    )
                continue
            result.row_count += 1
            if s1 in seen:
                dup_rows.add(s1)
            seen.add(s1)

            rest = rest.rstrip("\n")
            ids = rest.split(",") if rest.strip() else []
            id_set = set(ids)
            if len(ids) != len(id_set):
                intra_dupes.add(s1)
            mapping[s1] = id_set
            for target_id in id_set:
                if target_id.startswith("S1-"):
                    self_matches.add(target_id)
                elif not target_id.startswith(("S2-", "S3-")):
                    wrong_prefix.add(target_id)
                elif valid_target_ids is not None and target_id not in valid_target_ids:
                    unknown.add(target_id)

    findings = [
        (dup_rows, f"{name}: duplicate source1_entity_id rows: {sorted(dup_rows)[:5]}"),
        (intra_dupes, f"{name}: duplicate id within a {col_label} list for: {sorted(intra_dupes)[:5]}"),
        (self_matches, f"{name}: {col_label} contains S1 self-matches: {sorted(self_matches)[:5]}"),
        (wrong_prefix, f"{name}: {col_label} contains ids without S2-/S3- prefix: {sorted(wrong_prefix)[:5]}"),
        (unknown, f"{name}: {col_label} references ids not present in the target pool: {sorted(unknown)[:5]}"),
        (required_ids - seen, f"{name}: required S1 entities missing a row: {sorted(required_ids - seen)[:5]}"),
        (seen - required_ids, f"{name}: rows use an S1 id outside the required set: {sorted(seen - required_ids)[:5]}"),
    ]
    for offenders, message in findings:
        if offenders:
            result.errors.append(message)
    return mapping


def validate_outputs(
    matching_path: str,
    candidate_path: str,
    required_s1_ids: set[str],
    valid_target_ids: set[str] | None,
) -> ValidationResult:
    """Hard-fail validation of matching_results.tsv and candidate_pairs.tsv.

    Unlike the official validator, both ID existence (if ``valid_target_ids``
    is given) and matched-subset-of-candidate coverage are hard failures here.
    """
    result = ValidationResult()
    matched = _parse_id_list_file(
        matching_path, MATCHING_HEADER, "matched_entity_ids",
        required_s1_ids, valid_target_ids, result,
    )
    candidate_result = ValidationResult()
    candidates = _parse_id_list_file(
        candidate_path, CANDIDATE_HEADER, "candidate_entity_ids",
        required_s1_ids, valid_target_ids, candidate_result,
    )
    result.errors.extend(candidate_result.errors)
    result.row_count += candidate_result.row_count

    if matched is not None and candidates is not None:
        offenders = {
            s1: mids - candidates.get(s1, set())
            for s1, mids in matched.items()
            if mids - candidates.get(s1, set())
        }
        if offenders:
            example = sorted(offenders.items())[:5]
            result.errors.append(
                f"{len(offenders)} S1 entity(ies) have matched ids outside their "
                f"exported candidate set (retrieval-lineage violation): {example}"
            )
    return result
