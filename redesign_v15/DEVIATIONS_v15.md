# Deviations from the frozen analysis plan v15

Each deviation is recorded here with its date and its reason, before the results it affects exist. The frozen plan (`redesign_v15/ANALYSIS_PLAN_v15.md`, tag `v15-frozen`, commit 530d3f7) is not edited.

## D1 (2026-11-09): execution on a local computer instead of GitHub-hosted runners

**When.** Recorded before any Stage X or Stage C computation. No v15 result existed at the time.

**Plan text affected.** Section 15 (execution) and Section 6.1, on how v15b is recorded:
- The plan specifies GitHub-hosted CPU runners.
- It also specifies that v15b is recorded by the `x_aggregate` job in the `X-analysis` artifact.

**What happened.** The workflow `v15` was pushed together with the tag `v15-frozen`. Several manual "Run workflow" attempts on GitHub produced no workflow run (the API reported 0 runs), and the cause could not be identified in time.

**What is done instead.** The same chain runs on the first author's laptop through `v15/run_local.py`:
- **Hardware:** AMD Ryzen 7 7435HS (8 cores, 16 threads), 16 GB RAM, Windows 11.
- **Commands:** `run_local.py` executes the same commands as the workflow with `stage = all`, in the same order and with the same decision logic. The steps are G0, then Stage X on 13 generators, then the aggregate with gates G1/G2 and M by the rule of Section 7. If both gates pass, it continues with screening (M + 4 spares), S2, Stage C (R3 for the first 20 generators) and the C aggregate.
- **Stops:** the chain stops if any Stage X generator fails, and Stage C requires S2, exactly as in the workflow.
- **Software:** the plan's pinned versions are installed locally: Python 3.13.0, numpy 2.2.1, scipy 1.15.0, scikit-learn 1.6.0, pandas 2.2.3 and torch 2.11.0. The local torch is the CUDA build (2.11.0+cu128), and it runs on the CPU only (`CUDA_VISIBLE_DEVICES=-1`), as on the runners.
- **Threads:** each parallel job is limited to (number of logical CPUs / number of workers) threads.
- **Recording v15b:** v15b (gates and M) is recorded in `results_v15/analysis_X.json` and in the run log `results_v15/local_run/run_local.log`, instead of the `X-analysis` artifact.

**Additional check.** Gate G0 also runs check G0.6 against the stored v2 in-band corpus. That corpus is available only locally (`rigorous_rerun/outputs_v2/cache/s7_near_near.npz`), so on GitHub runners this check would have been skipped. Running it is strictly more validation.

**Not changed:**
- the hypotheses, estimands, tests, α levels and decision rules;
- the generators, seed ranges, cohorts, training seeds, grids and features;
- the analysis code and the rule for M.

All v15 results come from this one local environment.

**Consequence for the paper.** The paper states that the pre-registered chain was executed locally, after the public tag `v15-frozen`, and cites this deviation. The public timestamp of the tag (pushed on 2026-11-08) remains the evidence that the plan and code came before the results. The local run log, the models, the predictions and the labels are archived with a DOI.
