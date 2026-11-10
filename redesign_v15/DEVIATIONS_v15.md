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

## D3 (2026-11-10): code completed to match the plan text (implementation gaps found by a pre-result audit)

**When.** On the night of D2, before any Stage C computation and before any Stage X detector result was inspected.

**What the audit found.** A line-by-line comparison of `analyse_v15.py` and `train_v15.py` with the frozen plan found four estimands that the plan names but the code did not compute. The plan text is not changed; the code now does what the plan says.

| Plan text | Before | Now |
|---|---|---|
| 8.2 and P8: H2 robustness "raw-difference scale" | not computed | `H2.raw_scale`: the mean, SD, t-test p and 95% t-interval of δ on the raw PR-AUC difference scale |
| 8.4 S3: "H2 and ψ_T are recomputed with these λ" | mean shift with a bootstrap CI, and the naive sign share only | adds the H2 t-test (t, p, t-interval) and the model-based ψ_T for (λ_U, λ_A0), with its bootstrap CI. The within-generator covariance `cov_UA0_two` is now stored per generator |
| P7: "the full-knowledge oracle is reported only as a loose upper bound" | not computed, and the benign-plan score was not saved | `meta.ceiling_A_full`: the PR-AUC of P(Y_A = 1 \| s, s0). The benign-plan score `score_b0` is saved in `labels_test.npz` |
| 8.3: secondary estimates with 95% CIs | S4 share without an interval | S4 share with a 95% bootstrap interval over generators |

**Robustness fix.** The U–A covariance of the within-generator bootstrap now uses only the replicates in which both values are finite (`fcov`), as `np.nanvar` already did for the variances. It is identical to `np.cov` when every replicate is finite; this was tested. Non-finite replicates occur only in tiny corpora: in the smoke test (4 test cohorts), a replicate without attack-caused positives made the original covariance NaN.

**Unchanged:** the primary estimands, the tests, the decision rules, training and selection.

**Stage X.** The three Stage X generators whose corpora were built before this change (cutoff, d2, fovfree) lack `score_b0` and `ceiling_A_full`. Stage X is exploratory, and these quantities can be recomputed exactly by regenerating the deterministic corpora.

**Tests.**
- On 6 smoke generators (separate seeds, deleted afterwards), the complete chain ran: summarise, then aggregate with S3, S4, S5 and H2 on the raw scale.
- On two 20-cohort corpora with K = 32 (separate seeds 9,200,000+), the full-knowledge A ceiling was at least the attainable ceiling: 0.2967 vs 0.2958, and 0.0325 vs 0.0313. On the 4-cohort smoke corpus the order was reversed (0.354 vs 0.355), a small-sample artefact.

## D4 (2026-11-10): screening cap raised so that screening continues until M + 4 draws qualify, as the plan says

**When.** 2026-11-10, about 7 minutes into Stage C screening. At that point:
- 23 draws (k = 100–122) had been screened, and 9 qualified;
- no Stage C corpus for training had been generated and no detector had been trained;
- no Stage C output exists other than these oracle screening values.

**Plan text.** Section 6.2: "Draws continue in order until M generators qualify, plus 4 spare qualifying draws."

**Code before.** `screen_v15.py` stopped at k = 399 (`--kmax`, default 399) and exited with an error if fewer than M + 4 = 104 draws had qualified by then. That would have stopped the chain before Stage C.
- The cap was an implementation bound, not a plan quantity. The plan estimated "about 120 draws" from Stage X, where 6 of 8 prior draws qualified.
- In the first 23 Stage C draws the qualification rate was 9/23 (39%). Draws with log-sum-exp aggregation, which did not occur among the 8 prior draws of Stage X, failed in every case so far.
- At about 40%, roughly 260 draws are needed. The cap of 300 draws therefore had a real chance of being reached.

**Change.**
- `--kmax` now defaults to 998. This is the largest k for which `gen_v15.stage_c_ranges` keeps all Stage C seed ranges disjoint: dev < 2,000,000 ≤ test < 3,000,000 ≤ screening < 4,000,000 = the S2 range. `stage_c_ranges` already asserts k < 999.
- Gate check G0.9 now verifies the disjointness of the seed ranges for k = 100–998 instead of 100–399. G0 is re-run before screening resumes.

**Unchanged:**
- the draw order;
- the qualification rule (both ceilings ≥ 0.15 on the 20-cohort screening corpus);
- M, the 4 spares and the selection of the first M completed draws;
- every analysis.

No draw is skipped or reordered, so the qualifying draws are exactly those the plan's rule defines. The only effect of the change is that the screen does not stop early.

**Run record.**
- The chain was paused by the assistant (Ctrl-C at 11:46:49). The run log shows this pause as "PAUSED by the user".
- It was restarted from the commit that contains this deviation.
- Screening restarts from k = 100. It is deterministic, and nothing from the paused screen was kept, because `screen_v15.py` writes its output only at the end.
- The number of draws screened, the number excluded and the reasons are reported in the paper (Table S2), as the plan requires (Section 4.4).

**D4 outcome (added after screening; no detector result exists yet).**
- Screening needed 239 draws (k = 100–338) to obtain 104 qualifying generators: 104 of 239 qualified (44%).
- So the old cap (k ≤ 399) would not have been reached in the event. D4 changed no draw and no selection.
- By aggregation type, the draws that qualified were:
  - sum: 73 of 118;
  - max: 25 of 51;
  - log-sum-exp: 6 of 70.
