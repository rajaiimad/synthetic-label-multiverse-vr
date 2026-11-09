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

## D2 (2026-11-10): added secondary outcome S5, the tuning budget

**When.** Approved by the authors on 2026-11-10 and specified, implemented and committed the same night:
- before any Stage C computation;
- before any Stage X detector result was inspected. Stage X restarted at 02:02 with `--stop-after X`, and only its progress log had been read.

**Why.** An independent pre-run assessment (GO/NO-GO gate, recommendation REVISE, item R2) found one confounder that the plan does not address. The primary grids (16 configurations per detector) are equal in size but not in coverage. A reviewer can therefore ask whether the "winner" between MPNN and Raw-HGB reflects the grid rather than the label choices. A check added after the results would be post hoc, so it is added now.

**What is added** (secondary outcome; no hypothesis test; nothing in Sections 2–9 of the plan changes):
- **Generators.** The first 10 Stage C generators in draw order, both detectors and both targets (U and A), with the same 4 training seeds, splits, features and test corpora as the primary run. This is a subset of the R3 generators, chosen before any outcome exists.
- **Extra configurations.** For each detector, 16 extra configurations drawn once by random search over wider ranges, with seed 15005 (`v15/make_s5_grids.py`). They are frozen in `v15/s5_grids.json`:
  - Raw-HGB: learning rate log-uniform on [0.02, 0.3]; leaves log-uniform on [8, 128]; minimum leaf size log-uniform on [5, 400]; class weight {none, balanced}; boosting iterations {300, 600}, with the number of rounds chosen on validation as in the primary run.
  - MPNN: width {32, 64, 128}; learning rate log-uniform on [3·10⁻⁴, 10⁻²]; positive weighting {none, square root, full}; weight decay log-uniform on [10⁻⁵, 10⁻²]; the extra edge feature {no, yes}.
- **Selection.** Over the union of the primary 16 and the extra 16 (32 per detector), by validation PR-AUC on the training target. An extra configuration must be strictly better, so ties keep the primary choice, as the primary selection keeps the first best configuration in grid order. The primary predictions and selections are reused and never modified.
- **Estimands** (frame level, per target; `analyse_v15.s5_summary`):
  - the number and share of the 10 generators whose winner (the sign of λ) is unchanged;
  - the mean change in λ, with a 95% t-interval;
  - each detector's mean log gain in PR-AUC, with a 95% t-interval;
  - as context, the number of generators whose primary λ lies within 1.96 within-generator SDs of zero.
- **Pre-specified reading.** If the winner changes on 3 or more of the 10 generators for either target, the paper states that the primary conclusions describe winners under the fixed 16-configuration budget and are sensitive to that budget. Otherwise it states that the winners were stable in this check.

**Code.**
- `train_v15.py`: new function `train_s5` and flag `--s5`, run after the primary models of a generator.
- `analyse_v15.py`: S5 conditions in `summarise`, and `s5_summary` in `aggregate`.
- `run_local.py` and the workflow pass `--s5` to the first 10 generators.
- The diff only adds code, apart from three lines that extend the R3 command, the list of secondary conditions, and the step name. The primary training, selection and analysis paths are unchanged.
- Stage X jobs that start after this change load the new files. Their code paths are identical, because S5 runs only with `--s5` in Stage C.

**Tests, before commit.**
- On the separate smoke corpus (seeds ≥ 9,000,000; 2 configurations), both branches of the selection rule occurred. When an extra configuration won, its saved model reproduced the predictions to within 10⁻⁶. When the primary configuration was kept, the predictions were identical to the primary ones.
- Unit checks of `s5_summary`: counting, the threshold of 3 or more changes, the gains and the missing-data handling.
- The smoke outputs were deleted after the test.

**Cost.** About 10–20% more Stage C compute.
