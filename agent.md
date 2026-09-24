# Shared implementation rules

These rules guide implementation work requested by the user. They do not turn a
documentation-only request into authorization to train models or submit files.

## Objective and evidence

- Optimize reproducible end-to-end macro F0.5 and generalization, with a competitive
  leaderboard result as the goal. First place cannot be promised.
- The user reported 27 teams and a leading score of 0.964733 on 2026-09-25. Treat
  this as an unverified, moving public reference, not a local acceptance threshold.
- The supplied README describes public/private scoring on different portions of
  the provided test set. A separate unseen dataset is not confirmed. Nevertheless,
  inference must work on new input directories, source sizes, IDs, and countries.
- Distinguish actual measurements, heuristic observations, proposals, and unknowns.
  Test candidates and `data_var.py` variation tags are not verified matching labels.
- Use supplied data for entity resolution. No external business lookup, geocoding,
  identity databases, or external-data augmentation. Software documentation and
  permitted pretrained model downloads are different from looking up businesses.
- Verify and record checkpoint license, revision, and parameter count. Respect the
  statement's MIT/Apache-2.0 and at-most-8B-parameter model constraint.

## Non-negotiable modeling invariants

1. One S1 may match zero, one, or many S2/S3 records, including multiple per source.
   No forced top-1 result, one-to-one assignment, or automatic bucket merging.
2. Return every S1, including France and singletons. Countries are open strings.
3. Preserve source data and original IDs/text. Normalize into additional views;
   never overwrite inputs or collapse distinct target IDs just because text agrees.
4. Keep Unicode marks meaningful to Indian scripts; keep alternate Latin accent
   folding. Numeric tokens, units, `bis`, and address suffixes require care.
5. Use overlapping name/address/alias/multilingual retrieval. Do not require exact
   city, state, house number, or name agreement in every route.
6. Fit learned transformations and models on the training fold only. Split by
   positive entity family before training or hard-negative mining. Group shared
   positive components; exclude held-out family records from supervised negatives.
7. No ground-truth injection into validation/test candidates. Injected training
   positives must be flagged and excluded from natural retrieval-recall reports.
8. Evaluate against a realistic target pool, including distractors. A tiny closed
   sample result must not be labelled full-pool recall or competition accuracy.
9. Use the complete true match set for scoring. Missing candidates remain false
   negatives. Never compute end-to-end scores only on successfully retrieved pairs.
10. Exact metric: for nonempty truth, `1.25*TP/(1.25*TP + 0.25*FN + FP)`;
    empty truth scores 1 only for an empty prediction, otherwise 0. Macro-average
    over all S1 queries. Reject duplicate/missing query rows rather than hiding them.
11. Keep a tuning split and an untouched final holdout. Never relabel the final
    holdout as another tuning set after seeing its results. Preserve a failed test.
12. Keep a CPU lexical + pair-classifier baseline. Add complexity only with a
    measured ablation, reproducible benefit, and realistic runtime/memory cost.
13. Preserve retrieval lineage. Final matches must be valid S2/S3 IDs in the exact
    candidate set supplied to the matching stage for that S1.

## Engineering and experiment discipline

- Prefer modular Python with typed interfaces, explicit configurations, meaningful
  tests, pinned project dependencies, and CLI commands. Do not rewrite unrelated work.
- Record dataset/split fingerprints, code revision or code hash, seeds, environment,
  configuration, model revision, commands, scope, timings, and peak memory per run.
- Store raw records once. Use integer row keys, sharded indexes, batch feature
  generation, and incremental output. No all-pairs join, dense global similarity
  matrix, or giant Python object graph of the full candidate dataset.
- Cache only with keys that include input, preprocessing, split, model, and relevant
  configuration identities. Reject stale cache mismatches rather than silently reuse.
- Write outputs atomically; checkpoint long stages. Resuming must not duplicate or
  omit query IDs. Do not overwrite the best model/output with an unverified run.
- Measure before expanding. Initial defaults: one heavy job at a time, up to
  30 minutes per pilot job, up to 16 GiB RSS and less when available RAM is lower,
  leave at least 15 GiB free disk, no paid cloud spending. Recheck and configure
  these budgets during preflight; do not silently shrink evaluation scope to fit.
- Routine local implementation, tests, and bounded experiments may proceed under
  the user's implementation request. Do not ask again at every phase. External
  uploads, paid resources, destructive changes, or larger budgets need authorization.
- No mandatory multi-agent setup. One implementation owner controls shared files.
  If the user requests delegation, assign bounded work with disjoint ownership;
  independent metric/leakage review is more useful than duplicating expensive runs.
- Use normal tool permissions. Do not suggest disabling all permission checks as
  a way to make progress.
- Inspect the official validator's actual behavior. Its defaults skip ID existence
  and only warn on match/candidate inconsistency. Our release checks must enforce
  ID existence, candidate coverage, and subset membership as hard failures.
- Make predictions from input content, not memorized IDs or sample-specific rules.
  Test altered IDs, row order, unseen country values, and new data paths.
- Update `PROJECT_STATUS.md` at meaningful checkpoints and before context handoff.
  Include exact reproduction commands, current best run, failed hypotheses, and
  next unfinished step. Existing promises are not evidence of completed work.

## Evidence required at a checkpoint

Report tests and commands actually executed; training/evaluation population and
target-index scope; candidate micro/macro recall and full-family coverage; oracle
and actual macro F0.5; singleton false-positive rate; worst slices; runtime/memory;
and whether the run supersedes the incumbent. Use `not measured` for unavailable
metrics. No model is promoted on public leaderboard movement alone.
