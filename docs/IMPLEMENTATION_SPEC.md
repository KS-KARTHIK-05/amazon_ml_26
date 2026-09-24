# Implementation contract and experiment plan

## Purpose and authority

Implement the architecture in `modeling_strategy.md` under the durable rules in
`agent.md`. This file supplies phase deliverables, interfaces, and acceptance
criteria. The user's current request determines how many phases to execute.
Commands and modules described below are target interfaces, not existing code.
Do not report them as runnable until implemented and actually exercised.

The benchmark to beat is the user's reported public score of 0.964733. Local
scores are not on the same population. Optimize evidence of generalization;
never manufacture a claim that local F0.5 above that number means rank 1.

## Proposed code and artifact structure

```text
src/er/
  cli.py                 # one documented CLI, explicit input/config/output paths
  io.py                  # streaming TSV ingestion, schemas, atomic outputs
  normalize.py           # raw-preserving multilingual text views
  truth.py               # positive edges, family ownership, integrity checks
  splits.py              # fixed family/component splits and fingerprints
  indexes/               # lexical and optional dense retrieval backends
  retrieve.py            # route union, source quotas, cap, candidate lineage
  features.py            # batched pair evidence without ID leakage
  train.py               # hard negatives, classifier fitting, model serialization
  calibrate.py           # development-only calibration and threshold selection
  metrics.py             # exact macro F0.5 and retrieval/oracle metrics
  evaluate.py            # complete query population, slice reports and comparisons
  predict.py             # resumable batched zero/one/many output
  validate.py            # strict scalable ID and subset validation
tests/                   # correctness, leakage, normalization, integration
configs/                 # smoke, pilot, baseline, controlled challengers
artifacts/<run_id>/       # immutable configuration, provenance, models and reports
experiments/ledger.jsonl  # append-only run summaries, including failures
experiments/incumbent.json
reports/                 # environment, model cards, evaluation and error evidence
output/<run_id>/          # candidate_pairs.tsv and matching_results.tsv
```

Use a minimal project environment or carefully extend `.venv`; do not blindly
replace the existing requirements or remove unrelated scripts. Pin dependencies
needed to reproduce the implemented pipeline. Keep final packaging requirements
separate from the incidental existing environment.

## Data contracts

**Record table:** numeric `row_id`, original `entity_id`, `source`, `country`, raw
name/address, versioned normalized views, script/missingness/quality fields.
`row_id` is an internal key; neither it nor entity ID digits are predictive features.

**Truth edges:** `(s1_row_id, target_row_id, target_source, positive_family_id)`.
Keep a separate S1 truth table including empty lists. Check whether target IDs
have multiple owners. Group connected positive components before any split.

**Candidate table:** `(s1_row_id, target_row_id, target_source, retrieval_routes,
route_ranks, route_scores, final_candidate_rank, run_id)`; unique ID pairs.
Maintain a query manifest so zero-candidate S1 entities cannot disappear.

**Prediction table:** candidate ID pair, model score, calibration version,
threshold policy, accepted flag. Candidate scores must have provenance; a fuzzy
similarity or cosine is not automatically a probability.

All caches need fingerprints of relevant data, preprocessing, split, index,
model, and configuration. Atomically publish completed artifacts and keep a
manifest of finished batches. An interrupted run must be safely resumable.

## Phase 0 — Inspect and establish the execution budget

1. Read the statement, current files, status, and dataset headers. Report which
   modules actually exist. Do not confuse analysis scripts with a trained pipeline.
2. Check Python/dependencies, RAM/disk, available CPU threads, and GPU. For CUDA,
   execute a tiny tensor operation if available; do not infer it from installed
   CUDA wheels or a claimed GPU model. Start with the CPU path if unavailable.
3. Write `reports/environment.md` with date and observations. Verify the existing
   country/source counts; compute and persist dataset fingerprints once, avoiding
   repeated full hashing on every command.
4. Create explicit resource settings: initially <=30 minutes per pilot job,
   <=16 GiB RSS and at most 75% of currently available RAM, >=15 GiB disk reserve,
   and one memory-heavy job at a time. Benchmarks may propose larger budgets;
   do not launch paid resources or assume unlimited GPU access.
5. Define configurations and exact run commands. Use 1,000–2,000 families for a
   smoke execution, 50,000–100,000 for a pilot. Distinguish fixture target pools,
   reduced target pools, and full-target evaluation in every report.

**Acceptance:** reproducible environment/config report, confirmed inputs,
resource estimates, and a concrete next command. No model-performance claim.

## Phase 1 — Build correctness and split integrity before optimization

Implement ingestion, raw-preserving text views, truth parsing, split creation,
exact scoring, strict output validation, and their tests. Use dynamic country
values. Use separate missingness features; avoid converting empty strings into
literal model tokens such as `nan` unintentionally.

Default split experiment: deterministic family-level 80% train / 10% tuning /
10% final holdout, stratified by country and approximate match multiplicity where
practical. The final holdout is sealed for selection purposes: tuning commands
must not calculate or display its metrics. Persist split membership and a hash.
Record a final-evaluation event after model selection is frozen. Do not repeatedly
reopen it, regenerate a luckier split, or silently tune on its failures.

Learn normalization dictionaries, supervised encoders, and hard negatives only
from training families. Held-out-family records cannot appear in supervised
training even as negatives. A label-free evaluation index can include them and
must include their targets. Document what label-free corpus statistics are fitted
on which data. No test-derived pseudo-labels in the baseline.

**Exact score tests:**

| True set | Predicted set | Expected score |
| --- | --- | ---: |
| empty | empty | 1 |
| empty | one wrong target | 0 |
| two true targets | empty | 0 |
| two true targets | both targets | 1 |
| two true targets | one correct target | 5/6 |
| two true targets | both plus one false target | 5/7 |

Also test macro averaging across queries, full truth retained after blocking,
determinism, duplicate/missing query rejection, unknown target IDs, S1 self-targets,
cross-split ownership, new country strings, Unicode round trips, missing fields,
and changed input row order. Remap all entity ID digits and confirm decisions
remain equivalent after mapping back.

Normalize into multiple views: case-folded Unicode, Latin-accent-folded, legal-form
reduced, compact initials/domain/DBA views, token sequence/count/set, address
tokens, and conservative numeric variants. Keep source text intact. Never turn
the heuristic `data_var.py` labels into ground-truth matching supervision.

**Acceptance:** meaningful tests pass; splits and fingerprints exist; standalone
evaluation covers all queries; strict validation fails on intentionally invalid
outputs. A dummy model may exercise plumbing but must be labelled a dummy.

## Phase 2 — Build and measure natural candidate generation

Build country/source-partitioned target indexes over S2 and S3. Start with lexical
name and address retrieval, plus selective exact/alias routes. Character 3–5 grams
should improve typo tolerance; token routes should handle word reordering. Pick
and benchmark a concrete compact index implementation. Do not implement the full
system as a dense similarity matrix or a nested Python scan per query.

Independently retrieve by name and by address. A candidate with zero name overlap
may still enter through address evidence. Missing-address queries need a name
fallback. Initials alone or common city/legal tokens must not expand into huge
unranked cross joins. Avoid discarding distinctive rare words just to prune vocab.

Union routes, deduplicate target IDs, preserve separate source budgets, record
route lineage, and compare final budgets of 25/50/100 candidates per source.
Rank-fusion and protected route quotas are starting hypotheses. Measure recall
before and after every cap; do not silently favor name-only candidates.

Evaluate pilot queries against the full training target pool when resources allow.
If full-pool indexing exceeds the agreed budget, keep working on a checkpointable
index or a scoped engineering test and state the limitation explicitly. Do not
publish its reduced-pool recall as full-pool performance.

**Required reports:** micro and macro candidate recall over non-singletons;
complete-family coverage; zero-hit rate; S2/S3 recall; mean/p95/p99/max candidates;
singleton candidate traffic; route-only recovered positives; oracle macro F0.5;
latency, RSS, disk and index-build time; scope and denominator of every statistic.
Slice by country, script change, missing address, initialism/alias, numeric change,
common names, and positive multiplicity. Some labels are heuristic slices.

The oracle predicts exactly `truth ∩ candidates` and no negatives for each query.
It uses complete truth and includes singletons. Never inject truth into these
candidate sets. Track unretrieved positives as an explicit failure artifact.

Early retrieval aspirations are ~98–99% recall. For a competitive final system,
investigate whether >=99.5% micro recall and >=0.99 oracle macro F0.5 are attainable
at acceptable cost. These are working aspirations, not measured scores, guarantees,
or reasons to conceal weak slices. Judge progress by end-to-end recoverable loss.

**Acceptance:** runnable candidate pipeline and honest full-scope report; a
sampled missed-positive review; documented cap sensitivity. If recall is weak,
establish the baseline and prioritize the missing route instead of increasing
classifier size to compensate for absent candidates.

## Phase 3 — Train and evaluate a reproducible CPU baseline

Train a shared LightGBM binary matcher with source as a feature. Begin with
normalized name/address character and token comparisons, rare-token agreement,
containment, length ratios, numeric agreement/conflict, legal-form/initials/alias
views, missingness, scripts, parse confidence, and retrieval route scores/ranks.
Do not use source IDs as features. Keep features usable for unseen countries.

Use all training positives and roughly 20–50 mined negatives per query for the
pilot, adjusted to the budget. Mine from actual shortlists: same-name/different-
address cases, same-address/different-name cases, confusing lexical neighbors,
and singleton shortlists. A small random-negative component is optional. Use
full truth to avoid marking another true variant negative. Evaluate optional
query weights; do not let huge buckets silently dominate.

Missed positives may be separately added for matcher training, with an explicit
injected flag. They may not inflate natural retrieval metrics or appear by fiat
in evaluation candidates. Calibration and threshold choice use the unaltered
tuning candidate distribution, including all queries and retrieval misses.

Tune the acceptance policy against actual end-to-end macro F0.5. Start with a
global threshold. Source/quality-specific calibration is an ablation, not a
default collection of many overfit thresholds. No special France threshold
without labels. Multiple accepted targets per source and zero matches are valid.

**Acceptance:** a genuine completed training run; exact reproduction command;
saved model and feature schema; tuning per-query scores/predictions; macro F0.5,
precision/recall, singleton false-positive rate and slice reports; resource costs;
and the gap between candidate oracle and actual score. No placeholder metrics.
This measured baseline is the first milestone, even if it is far below 0.964733.

## Phase 4 — Improve the measured loss, one hypothesis at a time

Maintain an error budget with query counts and macro-score loss contributions:

| Dominant failure | Next experiment |
| --- | --- |
| True target not retrieved | Add/fix name, address, alias, transliteration or multilingual route; change cap only with recall/cost evidence |
| True target present but rejected | Review quality features, hard-positive coverage, calibration and thresholds |
| Wrong target accepted | Add hard negatives and conflict features; inspect common-name and singleton errors |
| Missing address or cross-script names dominate | Improve name-only/multilingual retrieval and corresponding pair evidence |
| Strong development score, weak country transfer | Reduce country memorization; inspect multilingual fine-tuning and normalization assumptions |

Recommended ablations, ordered by measured need:

1. Country-aware abbreviation/Unicode/alias views with a normalization regression suite.
2. Additional retrieval routes and source/route-aware budgets, including a compact
   multilingual embedding baseline such as multilingual-e5-small if verified eligible.
3. Train-fold-only multi-positive contrastive fine-tuning with family-aware negatives.
   Validate held-out-country transfer and refresh versioned embeddings/indexes.
4. Targeted hard-negative refresh and classifier/feature ablations.
5. Optional small multilingual cross-encoder on the actual difficult candidate
   distribution, with bounded pair counts and recorded incremental compute cost.
6. Calibration or a small complementary ensemble based on out-of-fold predictions.
   Do not union accepted matches indiscriminately: precision and singletons matter.

Use one primary change per experiment. For ensembling, keep calibration/stacking
folds separated from the predictions used to fit the combiner. Save paired
per-query scores for the incumbent and challenger on the same tuning queries.
Use a family-level paired bootstrap for uncertainty where feasible; report the
mean delta, interval, and slice deltas. Repeated tuning still creates selection
bias, so the untouched final holdout remains necessary.

An improvement is not established by a sixth-decimal score change on one noisy
split. Confirm promising changes on another grouped development split or seed,
and inspect regressions before promotion. Preserve the incumbent and log failed
and inconclusive experiments. Do not change the metric or denominator to win.

## Phase 5 — Generalization and final evaluation

- Run US→India and India→US held-out-country stress tests. They detect dependence
  on familiar language/country patterns; they do not estimate France performance.
- Evaluate corruption/missing-field slices from genuine held-out examples. Any
  synthetic stress cases must use train-derived transformations, remain clearly
  separated, and not replace real held-out scores.
- Confirm that unlabelled French records and generic unseen-country values flow
  through preprocessing, routing, retrieval, scoring, and output without dropping.
- Freeze model, preprocessing, retrieval budgets, features, and thresholds before
  one named final holdout evaluation. Record that the holdout is now opened.
- If the final test disappoints, report it. Do not tune on it and continue calling
  it untouched. Further development requires an explicit new validation protocol
  and acknowledgment that the original holdout has been consumed.
- A full-label refit for deployment must be a separate artifact. Preserve the
  last unbiased evaluation and do not assign that score to new weights as if the
  refit itself had been evaluated on unseen labels. Choose calibration/refit
  policy through development cross-validation before final release.

Do not reverse-engineer hidden/public membership or optimize per-record decisions
from repeated public scores. Use a small number of submissions for preselected,
locally justified changes. Record reported public/private scores separately from
local metrics; a moving leaderboard is not the experiment's source of truth.

## Phase 6 — Full inference and release integrity

Implement checkpointed batched inference over a supplied `--test-dir`; do not
hard-code row counts or IDs. One hundred total candidates per query already
means 173,254,400 pairs for the current test set. Fifty float32 features for all
those pairs occupy about 32.27 GiB before other data. Batch and release memory.

Record the candidate set at the actual input to the matching stage, after all
blocker filters/caps. If the matching model is a cascade that scores a wider set
before refining a subset, document that stage boundary and export its real input;
do not describe an earlier rejected retrieval pool as the model's input.

Release checks must hard-fail on missing/extra/duplicate S1 rows, wrong schema,
duplicate target IDs in a list, S1 targets, unknown S2/S3 IDs, or a predicted match
outside its exported candidate set. Use a scalable join/sort/partition strategy
for full checks; do not rely on fitting hundreds of millions of Python sets in RAM.

The supplied validator defaults to skipping target-ID existence; its candidate
absence/subset checks are warnings. Run it as an additional compatibility check,
not as the only proof of correctness. Do not modify the supplied validator to
hide warnings. Example command from the repository root, after outputs exist:

```bash
python3 student_resource/utils/validate_submission.py \
  --matching output/<run_id>/matching_results.tsv \
  --candidate output/<run_id>/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test \
  --check-ids
```

Replace `<run_id>` with a real run name. This check may be memory-heavy; if needed,
run the official check on representative shards with corresponding source files
and document its scope, while our strict scalable validator checks the full output.
Never call a shard-only check a full official-validator pass.

Final package follows the supplied template: `output/` with both TSVs,
`code/business_entity_resolution/src/` with reproducible code, README and pinned
dependencies, and the filled `Documentation_template.md`. Include required model
assets or deterministic allowed download instructions with exact revisions.
Test reproduction from a clean environment and new data path. Prepare the package;
portal submission is a separate user action unless explicitly authorized.

## Required run manifest and promotion record

Every run, including failures, should record:

```json
{
  "run_id": "unique-name",
  "status": "planned|running|completed|failed|budget_stopped",
  "hypothesis": "one measurable change",
  "parent_run_id": null,
  "code_fingerprint": "actual hash or revision",
  "config_path": "actual path",
  "dataset_fingerprint": "actual fingerprint",
  "split_fingerprint": "actual fingerprint",
  "evaluation_scope": "smoke|reduced_pool|full_pool_tuning|final_holdout",
  "query_count": null,
  "target_pool_count": null,
  "seed": 42,
  "model_revision_and_license": null,
  "commands": [],
  "candidate_recall_micro": null,
  "candidate_recall_macro": null,
  "complete_family_coverage": null,
  "oracle_macro_f05": null,
  "macro_f05": null,
  "singleton_false_positive_rate": null,
  "candidate_count_mean_p95_p99": null,
  "slice_report_path": null,
  "paired_comparison_path": null,
  "runtime_seconds": null,
  "peak_rss_gib": null,
  "disk_gib": null,
  "promoted": false,
  "reason": "not evaluated yet"
}
```

Null means not measured, never zero. A completed engineering smoke run is not a
model benchmark. Promotion requires reproducible comparison on the same scope,
no unresolved integrity/leakage defect, explained slice regressions, and acceptable
cost. When progress stalls, use the measured error budget to select the next
experiment; do not repeatedly launch similar models without a falsifiable hypothesis.
