# Verified project status

Updated: 2026-09-25 (Phase 1 implementation session). This file records
evidence, not aspirational progress.

## Completed

- Read the challenge statement and dataset schemas.
- Created and ran `sample_dataset.py` for a random 50-S1 linked sample.
- Created and ran `data_var.py`: audited 50,000 S1 entities and 172,910 labelled
  pairs; exported a diverse 50-S1 sample with 83 S2 and 91 S3 records.
- Wrote `data_variations_report.md` and `modeling_strategy.md`.
- Reviewed the video's visual slides; audio was not transcribed.
- Inspected a balanced 270-record test sample and six-query, 60-candidate
  demonstration. All demonstration candidates remain UNLABELLED.
- Prepared Claude instructions and implementation specifications. No model
  training was performed as part of preparing these instructions.
- **Phase 0 (environment/preflight):** confirmed RTX 3060 12 GiB, driver
  555.58.02, CUDA 12.5, actual `torch.cuda` tensor op executed successfully
  (not just package presence). 20 CPU threads, ~31 GiB RAM, 65+ GiB free disk.
  Installed `lightgbm`, `scikit-learn`, `rapidfuzz`, `unidecode`, `pytest` into
  `.venv`. SHA-256 fingerprints of all 7 raw dataset files computed once
  (`reports/dataset_sha256.txt`). Full report: `reports/environment.md`.
- **Phase 1 (correctness foundation) implemented and tested:**
  `src/er/{io,normalize,truth,splits,metrics,validate}.py`, 53 tests in
  `tests/` (`.venv/bin/python -m pytest tests/ -q` → 53 passed), including the
  exact score table (empty/empty=1, empty/wrong=0, 5/6, 5/7 cases), macro
  averaging, duplicate/missing query rejection, Unicode round-trip
  (Devanagari), self-target/duplicate-row/wrong-prefix truth-parsing errors,
  positive-family union-find over shared targets, deterministic
  country-stratified splits with fingerprinting, atomic Parquet ingestion with
  row order/missingness preservation, and strict validator hard-fails
  (ID-existence, candidate-subset lineage) that the official validator only
  warns on by default.
- **Ran the real Phase 1 pipeline on the full train and test splits** via
  `scripts/build_phase1_artifacts.py` (not a fixture/sample):
  - Ingested all 6 source files (train S1/S2/S3, test S1/S2/S3) to Parquet
    under `artifacts/normalized/` (gitignored — regenerable, ~1.9 GiB).
    Train: 2,206,821 / 5,034,616 / 5,285,603 rows. Test: 1,732,544 /
    4,887,273 / 5,082,316 rows. Total ingest time ~38s combined, peak RSS
    5.04 GiB (train run) / 1.15 GiB (test run) — well inside the 16 GiB budget.
  - Parsed `train_ground_truth.tsv` in full: 2,206,821 S1 rows, 7,638,365
    match edges, **0 parse errors**.
  - **123,247 S1 entities (5.6%) are true singletons** (no ground-truth match).
  - **Verified all 7,638,365 target references exist in the ingested S2/S3
    tables — 0 unknown target IDs** (explicit integrity check, not assumed).
  - **Measured finding: 0 multi-owner targets.** No S2/S3 record is claimed by
    more than one S1 in the training ground truth, so every positive-family
    union-find component is trivially just its own S1 (2,206,821 families =
    2,206,821 S1 entities). The union-find code still handles the general
    multi-owner case correctly (tested in `tests/test_truth.py`); this is
    just what the actual data does. Re-verify this assumption if it matters
    for a future design choice — it was measured once, not guaranteed stable
    across a resupplied dataset.
  - Built a deterministic, country-stratified 80/10/10 family split (seed 42):
    train 1,765,456 / tuning 220,682 / holdout 220,683 families. Fingerprint
    and full membership persisted to `artifacts/splits/train_families.json`
    (gitignored; regenerate with the exact command below — it's a pure
    function of the seed and the ground-truth file, so it reproduces
    byte-for-byte).
  - Confirmed test set country distribution directly: India 809,986 / US
    663,106 / **France 259,452** (matches the prior sample-based estimate in
    this file's "Inputs and scale" table, now confirmed from the full file
    rather than a sample).
  - Full machine-readable reports: `reports/phase1_build_report_{train,test}.json`.

## Not yet implemented or measured

- Phase 2 retrieval/blocking (lexical name+address indexes, candidate
  generation, recall measurement) — not started.
- Phase 3 LightGBM baseline training and evaluation — not started.
- Trained pair classifier, embeddings, neural reranker, or submission predictions.
- Natural full-pool candidate recall, local macro F0.5, or our leaderboard score.
- Best model/run: NONE. Experiment ledger (`experiments/ledger.jsonl`): not
  created yet — no model run exists to log.
- `train.py` is still only a CUDA availability check; the real pipeline lives
  in `src/er/` + `scripts/build_phase1_artifacts.py` now.

## Inputs and scale

| Split | S1 records | S2 records | S3 records |
| --- | ---: | ---: | ---: |
| Train | 2,206,821 | 5,034,616 | 5,285,603 |
| Test | 1,732,544 | 4,887,273 | 5,082,316 |

Train countries: US, India. Test adds France (259,452 S1 records). Full country
counts are in `student_resource/dataset/variance_sampled/audit_summary.json`.

Hardware confirmed this session (`reports/environment.md`): RTX 3060 12 GiB
(driver 555.58.02, CUDA 12.5, verified with an actual executed tensor op),
20 CPU threads, ~31 GiB RAM, 65+ GiB free disk. `.venv` extended with
`lightgbm`, `scikit-learn`, `rapidfuzz`, `unidecode`, `pytest`.

## Leaderboard context

User-reported reference: 27 teams; leading public score 0.964733. Not independently
verified and not comparable directly with a local validation score. The provided
README describes public/private portions of the same test set; additional hidden
input datasets have not been confirmed.

## Next action

Start Phase 2 (`docs/IMPLEMENTATION_SPEC.md`): country/source-partitioned
lexical name+address retrieval over S2/S3, character 3-5 gram + token routes,
union with source quotas, measured against the full training target pool.
Use the `train` split membership already persisted in
`artifacts/splits/train_families.json` — do not touch `holdout`-labelled
families for anything except a final, one-time evaluation.

## Handoff fields to update during implementation

- Current phase and remaining acceptance criteria: Phase 1 complete and
  tested; Phase 2 (candidate generation / blocking) not started.
- Last passing tests and exact commands:
  `.venv/bin/python -m pytest tests/ -q` → 53 passed (repo root, needs
  `conftest.py`'s `src/` path insert, already present).
- Latest run ID, config, split fingerprint, and target-index scope: no named
  run yet (Phase 1 artifact build isn't a model run). Split config: seed=42,
  80/10/10 country-stratified by family; fingerprint recorded inside
  `artifacts/splits/train_families.json` (regenerate via
  `.venv/bin/python scripts/build_phase1_artifacts.py --split train`, then
  `--split test`). Both are gitignored/regenerable, not committed.
- Best development run and measured metrics: none (no matcher trained yet).
- Last failed hypothesis and evidence: none yet — no experiment has run.
- Resource budget and active job/checkpoint: none active. Defaults from
  `reports/environment.md` apply (≤16 GiB RSS, ≤30 min/pilot job, one heavy
  job at a time, ≥15 GiB disk reserve).
- Next concrete action and exact resume command: implement
  `src/er/indexes/` + `src/er/retrieve.py` (Phase 2), then run a smoke
  (1,000-2,000 families) and pilot (50,000-100,000 families) candidate
  generation against the full S2/S3 train target pool and report recall.
- Final holdout status: NOT OPENED. 220,683 families are assigned `holdout`
  in `artifacts/splits/train_families.json`; no code has read their metrics.
