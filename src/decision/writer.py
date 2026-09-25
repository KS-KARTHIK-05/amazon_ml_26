"""Write matching_results.tsv / candidate_pairs.tsv and run the real validator.

Both files share the same shape: one row per required S1 id, a header, and a
comma-joined (no quoting) list of S2/S3 ids in the second column, empty for
no matches. Every S1 id in `required_s1_ids` must appear exactly once, even
if `id_lists` has no entry for it.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

STUDENT_RESOURCE_ROOT = Path(__file__).resolve().parents[4]


def write_id_list_tsv(
    path: Path,
    id_lists: Mapping[str, Iterable[str]],
    required_s1_ids: Iterable[str],
    id_col_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{id_col_name}\n")
        for s1_id in required_s1_ids:
            ids = id_lists.get(s1_id, ())
            # de-dup while preserving order, guard against accidental self-matches
            seen = set()
            clean = []
            for i in ids:
                if i in seen or i == s1_id:
                    continue
                seen.add(i)
                clean.append(i)
            f.write(f"{s1_id}\t{','.join(clean)}\n")


def write_matching_results(
    path: Path, matches: Mapping[str, Iterable[str]], required_s1_ids: Iterable[str]
) -> None:
    write_id_list_tsv(path, matches, required_s1_ids, "matched_entity_ids")


def write_candidate_pairs(
    path: Path, candidates: Mapping[str, Iterable[str]], required_s1_ids: Iterable[str]
) -> None:
    write_id_list_tsv(path, candidates, required_s1_ids, "candidate_entity_ids")


def run_validator(
    matching_path: Path,
    candidate_path: Path | None = None,
    test_dir: Path | None = None,
    check_ids: bool = False,
) -> tuple[bool, str]:
    """Shell out to the real utils/validate_submission.py. Returns (passed, output)."""
    test_dir = (test_dir or (STUDENT_RESOURCE_ROOT / "dataset" / "test")).resolve()
    cmd = [
        sys.executable,
        str(STUDENT_RESOURCE_ROOT / "utils" / "validate_submission.py"),
        "--matching",
        str(Path(matching_path).resolve()),
        "--test-dir",
        str(test_dir),
    ]
    if candidate_path is not None:
        cmd += ["--candidate", str(Path(candidate_path).resolve())]
    if check_ids:
        cmd.append("--check-ids")

    proc = subprocess.run(cmd, cwd=STUDENT_RESOURCE_ROOT, capture_output=True, text=True)
    passed = proc.returncode == 0
    return passed, proc.stdout + proc.stderr
