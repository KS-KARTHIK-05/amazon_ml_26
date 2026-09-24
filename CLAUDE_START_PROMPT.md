# First implementation prompt

You are implementing our Amazon ML Challenge 2026 business entity resolution
system in this repository. Build a competitive, reproducible pipeline for the
official macro F0.5 objective. The team reports a public leading score of 0.964733
across 27 teams. Treat it as motivation and a moving external reference, not as
a comparable local score or a guarantee of first place.

Read `CLAUDE.md` (including its `agent.md` import), `PROJECT_STATUS.md`,
`docs/IMPLEMENTATION_SPEC.md`, and the relevant parts of `modeling_strategy.md`
and `data_variations_report.md`. Inspect the actual code and supplied README.
The existing sampling scripts are analysis tools; there is no trained baseline.

Your first milestone is a real, measured lexical-retrieval + LightGBM baseline,
with a correct metric and leak-free validation. Execute Phases 0–3 of the spec
within the configured pilot budgets. Do not stop at a plan, scaffold, notebook
outline, dummy model, or unexecuted command list. Give a short implementation
sequence and then implement and run it. Continue normal local work without
asking for confirmation at every step. Respect tool permissions and actual
resource limits; do not launch paid compute or submit externally.

Start by confirming the schema, environment, usable compute, and existing files.
Record resource limits and use a CPU path if CUDA is unavailable. Preserve the
source data and unrelated work. Establish a minimal reproducible dependency set.

Build and test these foundations before optimizing:

1. Streaming TSV ingestion and raw-preserving multilingual normalization, with
   explicit missingness and open-set country handling, including France.
2. Ground-truth edge parsing and deterministic positive-family/component splits.
   Keep tuning and final holdout separate. Exclude held-out family records from
   supervised training, including mined negatives. Do not open final metrics now.
3. Exact macro F0.5 over every S1, including singletons and blocking misses. Write
   the specified hand-calculated tests and leakage/integrity regression tests.
4. Country/source-partitioned name and address retrieval with overlapping routes,
   alias/initial views, source quotas, deduplication by target ID, and versioned
   caches. Avoid all-pairs joins and dense global similarity matrices.
5. Natural pre/post-cap candidate metrics against a realistically sized target
   pool. Never inject truth into evaluation candidates. Clearly label reduced-pool
   engineering runs and preserve every missed positive in an error artifact.
6. Batched comparison features and a genuine LightGBM classifier trained on true
   positives plus hard retrieved negatives, including singleton negatives.
7. Threshold selection on tuning queries using complete end-to-end macro F0.5;
   zero, one, and many matches per S1 must all be possible.
8. Reproducible artifacts, an append-only experiment ledger, and strict output
   integrity checks. A default PASS from the supplied validator is insufficient
   because it skips ID existence and only warns on candidate-subset violations.

Use a small smoke execution first, then a 50,000–100,000-family pilot if it fits.
Evaluate pilot retrieval against the full training S2/S3 target pool where feasible.
If the resource budget prevents that, finish the executable stages, checkpoint,
report the exact scope and bottleneck, and provide the next concrete resume
command. Do not silently substitute easy targets or fabricate benchmark results.

For the first milestone, prioritize correctness, candidate coverage, and measured
end-to-end behavior. Do not start with a large generative model, expensive neural
reranking of hundreds of millions of pairs, test pseudo-labels, or guesses about
hidden labels. Dense retrieval and more complex models come after the baseline's
error analysis identifies a benefit worth testing.

At the checkpoint, deliver:

- Implemented modules, configurations, and exact commands that actually ran.
- Test results and proof that metrics include all queries, full truth, and singletons.
- Dataset/split fingerprints, query count, target-index size, and evaluation scope.
- Candidate micro/macro recall, complete-family coverage, pre/post-cap comparison,
  candidate counts, and oracle macro F0.5.
- Actual tuning macro F0.5, singleton false-positive rate, country/source/hard-case
  slices, plus runtime, peak memory, and disk use. Say `not measured` where needed.
- Saved baseline model/config, or an explicit incomplete status and resumable state.
- The largest error categories and the next three experiments ordered by likely
  recoverable macro-score loss per unit of compute.
- Updated `PROJECT_STATUS.md` and experiment records. Do not claim a leaderboard
  rank, hidden-test score, or successful full pipeline without measured evidence.

If this milestone completes within the session, summarize its evidence and the
highest-value next experiment. Do not quietly turn this bounded first milestone
into an unlimited optimization run. Subsequent work will continue from the saved
state and the experiment prompts in `docs/CLAUDE_WORKFLOW.md`.
