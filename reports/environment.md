# Environment report

Date: 2026-09-25. Generated during Phase 0 preflight.

## Compute

- Python: 3.11.16 (`.venv/bin/python`), managed via `.venv` + `requirements.txt`.
- CPU: 20 logical threads (`nproc`).
- RAM: ~31 GiB total, ~18 GiB available at preflight time (`free -h`).
- Disk: 357 GiB filesystem, 67 GiB free at `/` (`df -h .`).
- GPU: NVIDIA GeForce RTX 3060, 12288 MiB VRAM, driver 555.58.02, CUDA 12.5.
  Verified with an actual tensor op, not just package presence:
  `torch.cuda.is_available()` → `True`; `(torch.randn(1000,1000,device='cuda') @ ...).sum()`
  executed successfully and returned a finite value. `torch==2.6.0+cu124` installed.

## Dependencies

`requirements.txt` (pre-existing) pinned `torch`/`numpy`/`pandas`/`pyarrow`/`boto3`
etc. but no modeling/text libraries. Installed into `.venv` for this pipeline:
`lightgbm`, `scikit-learn`, `rapidfuzz`, `unidecode`, `pytest`. Not yet re-pinned
into a frozen requirements file — do that once the Phase 3 baseline is stable so
we pin only what's actually used.

## Dataset facts (confirmed by direct inspection, not assumed)

Folder: `student_resource/dataset/{train,test}/` (singular "dataset", matches
`PROJECT_STATUS.md`).

| File | Rows (incl. header) | Size |
| --- | ---: | ---: |
| train_source1.tsv | 2,206,822 | 201 MiB |
| train_source2.tsv | 5,034,617 | 467 MiB |
| train_source3.tsv | 5,285,604 | 481 MiB |
| train_ground_truth.tsv | 2,206,822 | 122 MiB |
| test_source1.tsv | 1,732,545 | 167 MiB |
| test_source2.tsv | 4,887,274 | 486 MiB |
| test_source3.tsv | 5,082,317 | 483 MiB |

Row counts equal `PROJECT_STATUS.md`'s record counts minus one header row each.
`train_ground_truth.tsv` has exactly one row per `train_source1.tsv` entity
(2,206,821 data rows each) — consistent with "every S1 gets a ground-truth row,
possibly empty."

Schema confirmed by direct header read: source files are
`entity_id, business_name, business_address, country`; ground truth is
`source1_entity_id, matched_entity_ids` (comma-joined S2-/S3- IDs, empty when none).
Sample rows show Devanagari-script business names in `train_source2.tsv`
(India), confirming multilingual content requiring Unicode-preserving handling,
not ASCII-only normalization.

SHA-256 fingerprints of the seven source files are in `reports/dataset_sha256.txt`
(computed once here in ~1.3s; do not re-hash on every command — reuse this file
and re-verify only if a file's size/mtime looks inconsistent).

## Resource budget for this session's pilots

Following `agent.md`'s defaults, not yet overridden:

- One memory-heavy job at a time.
- ≤30 minutes wall time per pilot job.
- ≤16 GiB RSS, and ≤75% of the ~18 GiB currently available (~13.5 GiB) — use the
  tighter of the two.
- ≥15 GiB free disk reserved at all times (67 GiB currently free, comfortable).
- No paid cloud spending; GPU use is local-only (RTX 3060, no cloud billing).

## What exists vs. what doesn't (avoid confusing analysis tools with a pipeline)

- `sample_dataset.py`, `data_var.py`: prior analysis/sampling scripts. Their
  `data_var.py` variation-category labels are heuristic, not verified ground
  truth, and must never be used as supervised matching labels.
- `train.py`: currently only a CUDA availability check — not a matching pipeline.
- No `src/er/` modules existed before this session. Phase 1 implementation begins
  now: `io.py`, `normalize.py`, `truth.py`, `splits.py`, `metrics.py`, `validate.py`
  plus tests, per `docs/IMPLEMENTATION_SPEC.md`.

## Next command

Implement and test Phase 1 modules (`src/er/{io,normalize,truth,splits,metrics,validate}.py`
and `tests/`), then run:

```bash
.venv/bin/python -m pytest tests/ -q
```
