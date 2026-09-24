# Amazon ML Challenge 2026 — Claude Code entry point

@agent.md

## Start of a session

1. Read `PROJECT_STATUS.md` and inspect the actual code before proposing work.
2. For implementation, read the relevant phase in `docs/IMPLEMENTATION_SPEC.md`.
3. Use `CLAUDE_START_PROMPT.md` for the first implementation session; subsequent
   sessions should continue from recorded artifacts and the next unfinished phase.
4. Read `modeling_strategy.md` for the architecture and `data_variations_report.md`
   for evidence. Neither document reports a trained-model benchmark.
5. Keep this entry point short. Put experiment history in run artifacts and
   `PROJECT_STATUS.md`, not in continuously expanding instructions.

## Current repository facts

- Data: `student_resource/dataset/{train,test}/`; folder name is singular.
- Existing analysis scripts: `sample_dataset.py`, `data_var.py`.
- Diverse training examples: `student_resource/dataset/variance_sampled/`.
- Unlabelled test examples: `analysis/test_review/`.
- `train.py` currently only checks CUDA; it is not a matching pipeline.
- `.venv` and `requirements.txt` exist. Inspect them; do not assume the ML
  pipeline's required dependencies are present or that CUDA works.
- Official statement: `student_resource/README.md`, `ML_Problem.pdf`.
- Official output checker: `student_resource/utils/validate_submission.py`.
- Methodology template: `student_resource/Documentation_template.md`.

## Delivery behavior

For an implementation request, build and execute the requested phase rather than
returning only a plan. Continue through runnable checks within the current scope
and resource budget. At each checkpoint report real measurements, artifact paths,
failures, and the next concrete action. Never invent a score or claim a leaderboard
position from a local validation result.
