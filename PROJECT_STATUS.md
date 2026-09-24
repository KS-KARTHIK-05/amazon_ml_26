# Verified project status

Updated: 2026-09-25. This file records evidence, not aspirational progress.

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
- Found Claude Code 2.1.282 installed; no Claude coding/model session was launched.

## Not yet implemented or measured

- Production preprocessing, full retrieval indexes, grouped model splits.
- Exact pipeline evaluator and strict scalable output validation.
- Trained pair classifier, embeddings, neural reranker, or submission predictions.
- Natural full-pool candidate recall, local macro F0.5, or our leaderboard score.
- Best model/run: NONE. Experiment ledger: not created yet.
- `train.py` is only a CUDA availability check.

## Inputs and scale

| Split | S1 records | S2 records | S3 records |
| --- | ---: | ---: | ---: |
| Train | 2,206,821 | 5,034,616 | 5,285,603 |
| Test | 1,732,544 | 4,887,273 | 5,082,316 |

Train countries: US, India. Test adds France (259,452 S1 records). Full country
counts are in `student_resource/dataset/variance_sampled/audit_summary.json`.

Hardware observed previously: approximately 31 GiB RAM; GPU driver query failed;
about 68 GiB free disk at that time. Recheck during preflight. A CUDA package list
does not prove a usable GPU. Existing `.venv` and requirements need inspection.

## Leaderboard context

User-reported reference: 27 teams; leading public score 0.964733. Not independently
verified and not comparable directly with a local validation score. The provided
README describes public/private portions of the same test set; additional hidden
input datasets have not been confirmed.

## Next action

Read `CLAUDE_START_PROMPT.md`; implement Phase 0 and then the Phase 1 correctness
foundation from `docs/IMPLEMENTATION_SPEC.md`. Continue through the bounded pilot
as requested. Do not start with a large neural training run or a guessed submission.

## Handoff fields to update during implementation

- Current phase and remaining acceptance criteria:
- Last passing tests and exact commands:
- Latest run ID, config, split fingerprint, and target-index scope:
- Best development run and measured metrics:
- Last failed hypothesis and evidence:
- Resource budget and active job/checkpoint:
- Next concrete action and exact resume command:
- Final holdout status: NOT CREATED / NOT OPENED / FINAL EVALUATED (record which):
