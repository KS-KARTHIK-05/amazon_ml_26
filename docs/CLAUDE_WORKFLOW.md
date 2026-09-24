# Working effectively with Claude Code

## Start correctly

The installed CLI was checked as version 2.1.282 on 2026-09-25. Recheck if using a
different machine. Launch from this repository so relative data paths and project
instructions resolve correctly:

```bash
cd /home/srmist/Desktop/Amazon_ML
claude --effort high
```

Then paste:

```text
Read CLAUDE_START_PROMPT.md and execute its first implementation milestone.
```

`CLAUDE.md` is the uppercase project entry point and imports `agent.md`. The
lowercase `claude.md` is only a pointer, not a second set of instructions.
`AGENTS.md` points other coding tools at the same shared rules. An ordinary
`agent.md` is not a custom Claude subagent definition.

Run `/context` to check the loaded memory files if instructions seem missing.
The explicit import avoids depending on version-specific coexistence rules for
CLAUDE.md and AGENTS.md. Do not run in modes that disable project instructions.
The official documentation describes project files, imports, and their limits:
[Claude Code memory](https://code.claude.com/docs/en/memory).

The command above uses the configured model; it does not change your model or
account. Use your strongest available reasoning model for metric/split design,
architecture, and difficult reviews. Routine module work can use another model
if the same tests and acceptance criteria remain in force. More reasoning is
not a substitute for executing experiments and checking their evidence.

## Use concrete milestones

The first session should build and measure the baseline. Later sessions should
name the next phase or a bounded experiment. Avoid repeatedly asking only to
"make it top 1"; it supplies no useful way to identify the next error or verify
a claimed improvement.

If you want a design-only review before implementation, use Plan Mode explicitly.
For the supplied first milestone, the specification is already written: a short
local plan followed by implementation is appropriate. Do not leave Claude in
Plan Mode and expect it to edit and run code.

Require tests and real run artifacts as completion evidence. This follows the
official guidance to give Claude verifiable outcomes and keep investigations
scoped. [Claude Code best practices](https://code.claude.com/docs/en/best-practices).

## Prompt: resume after interruption or a new session

```text
Read CLAUDE.md, PROJECT_STATUS.md, and the latest completed and failed run
manifests. Inspect actual files before trusting the status notes. Resume the next
unfinished acceptance criterion from docs/IMPLEMENTATION_SPEC.md. Reuse valid
checkpoints and caches; do not restart completed scans or experiments without a
reason. Keep the final holdout unopened. Execute the next bounded step and update
the status with exact commands, measured results, and remaining work.
```

Save status and run IDs before clearing context. Use a new session or `/clear`
when changing to a genuinely different task or when repeated corrections have
made the conversation confusing. Do not rely on a long chat as the only record
of split membership, failed experiments, or the current best model.

## Prompt: independent correctness and leakage review

Use a fresh session after the baseline exists. Keep one writer for the implementation.

```text
Read the repository instructions and independently audit the implemented pipeline.
Do not edit production code or launch heavy training. Inspect the actual metric,
split ownership, supervised negative mining, cache keys, query coverage, and
candidate export boundary. Run bounded correctness checks where useful.

Try to falsify the reported result: does evaluation retain blocking misses and
singletons, does training touch held-out families, is full-pool scope real, are
test candidates ever used as labels, and does the threshold see the final holdout?
Check Unicode, unseen-country handling, resume duplication, unknown target IDs,
and matches outside exported candidates. Trace the reported score to its actual
artifacts. Return actionable defects with file/line references and a minimal
reproduction. Distinguish inspected facts from hypotheses. Do not announce that
the pipeline is correct merely because a supplied validator returned PASS.
```

Fix demonstrated defects before more model search. An independent review is
valuable because the same session that wrote the pipeline may repeat its own
assumptions. No parallel-agent configuration is required. If delegating later,
keep work bounded, ownership disjoint, and heavy jobs serialized.

## Prompt: improve retrieval

```text
Use the current incumbent and the fixed tuning queries. Inspect missed-positive
and cap-loss artifacts. Propose at most three falsifiable retrieval experiments
targeting the largest macro-score loss. Keep target-index scope comparable and
the final holdout closed. Execute the highest-value experiment within the current
budget. Compare natural pre/post-cap micro/macro recall, complete-family coverage,
oracle macro F0.5, hard-case slices, candidates/query, runtime and memory. Do not
inject ground truth into candidates or quietly reduce the target pool. Promote
only with recorded evidence; otherwise retain the incumbent and log the failure.
```

## Prompt: improve the matcher

```text
Freeze the incumbent retrieval configuration and tuning population. Inspect
present-but-rejected positives, accepted wrong matches, and singleton errors.
Choose one feature, hard-negative, model, or calibration change with a measurable
hypothesis. Train only on allowed families; preserve the untouched holdout. Compare
paired per-query macro F0.5 and error slices on the same candidate distribution,
including blocking misses. Report uncertainty, runtime, and regression examples.
Do not promote on pair accuracy, a balanced-sample score, or a tiny unexplained
aggregate change. Save all run artifacts and the decision.
```

## Prompt: investigate generalization

```text
Audit dependence on familiar names, countries, address formats and ID patterns.
Run bounded US-to-India and India-to-US development stress tests and genuine
held-out missing-field/script-change slices. Check that French and generic new
country labels are processed without dropping queries. Do not claim these tests
measure France accuracy, and do not create test pseudo-labels. Keep final holdout
metrics closed. Recommend changes only where the evidence identifies a failure.
```

## Prompt: run a bounded optimization session

```text
Continue from the current incumbent. Run at most three sequential, hypothesis-led
development experiments within the configured resource budget. Choose them using
the error budget rather than model size or novelty. Change one primary factor at
a time, retain reproducible paired comparisons and failed runs, and keep the final
holdout closed. End with the best supported configuration, the exact improvement
and uncertainty, remaining loss, and the next most valuable experiment. Do not
launch paid resources, increase budgets, or submit externally.
```

## Prompt: finalize a selected model

Use this only after the model/configuration selection is frozen.

```text
Freeze and fingerprint the selected pipeline and threshold policy. Follow Phase 5
for one named final-holdout evaluation; record that it is now consumed and report
its result even if disappointing. Do not retune on it. Apply the preselected refit
policy, if any, as a separate artifact with accurate evaluation provenance.

Run checkpointed inference on the supplied test directory, preserving every S1
including France and empty matches. Export the actual matcher-input candidate
lists and final predictions. Execute strict full-output ID/subset/coverage checks
and document the scope of official-validator checks. Package the reproducible
code, pinned dependencies/model revisions, both TSVs, and methodology. Verify a
fresh-path reproduction smoke run. Prepare the package without uploading it.
```

## How to interpret progress toward the leaderboard goal

Maintain three distinct numbers: development score, one-time final local holdout
score, and reported public leaderboard score. The same numeric value does not
mean the same generalization. A local 0.970 does not establish that it beats a
public 0.964733 from another population.

Ask for evidence beyond an aggregate score:

| Question | Evidence |
| --- | --- |
| Can the matcher possibly reach the goal? | Candidate oracle F0.5 and missed-family analysis |
| Is the improvement real? | Paired per-query comparison, uncertainty, another development split/seed |
| Does it harm difficult cases? | Script-change, missing-address, common-name, country/source, singleton slices |
| Will it finish on full test? | Measured index/inference throughput, peak RSS, disk projection, resumable batches |
| Can we reproduce it later? | Immutable run config, data/split/code fingerprints, model revisions and exact commands |
| Is it overfitting feedback? | Frozen validation protocol and sparse, hypothesis-driven public submissions |

The rules supplied locally describe the private leaderboard as the remaining
portion of the same test file. Do not invent another hidden dataset or tailor the
pipeline to presumed membership. A data-path-driven, open-country pipeline is
still important for final reproduction and any later data the organizers supply.

## Avoid expensive detours

- Do not train a larger model to fix positives that never entered the shortlist.
- Do not start a full 173-million-pair neural run before a throughput benchmark.
- Do not mistake a varied 50-entity sample for an unbiased validation benchmark.
- Do not make a bucket imply a match, or force one match for a singleton.
- Do not change three subsystems at once and then attribute the gain to one.
- Do not keep expanding CLAUDE.md with experiment history; use the ledger/status.
- Do not repeatedly expose the final holdout or use public scores to infer labels.
- Do not retain only the newest model. Keep the best evidenced incumbent and its
  working predictions until a challenger has earned promotion.

Before increasing compute, ask Claude for the measured bottleneck and a specific
resource proposal. Local job time/memory settings and Claude usage limits are
different budgets; a shell timeout does not cap account spending. The defaults
are deliberately for pilots, not a promise that the whole competition run fits
in thirty minutes.
