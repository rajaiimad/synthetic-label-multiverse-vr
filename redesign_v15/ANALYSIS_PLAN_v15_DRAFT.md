# Analysis plan v15 (DRAFT for author approval)

## Label choices and detector rankings for a policy-compliant avatar attack in multi-user VR

**Status.**
- Draft, 2026-11-08. Not committed.
- No experiment of this plan has been run.
- The manuscript, all results, scripts and earlier plans are unchanged.
- After approval, this file is committed to the release repository before any Stage X or Stage C computation. Any later change is a recorded deviation.

**Supporting files in `redesign_v15/`:**
- `proxy_variance_components.py` and `proxy_components.json`: planning inputs read from existing r6 outputs.
- `power_sim_v15.py`, `power_sim_v15_design.py`, `power_sim_v15_alpha.py`, `power_sim_v15_proxyrandom.py` and their JSON outputs: simulation of the design and of this exact analysis.

---

## 0. What has been seen before this plan (provenance)

**Seen:**
- All r1–r6 results: above-band attack; 13 + 4 fixed generators; cohorts 60000–60059 (dev) and 61000–61059 (test); training seeds {0, 1, 2, 42, 2024}.
- The audit of review 3 (`audit_r6_review3/`): attacker-caused scoring of existing predictions, track-level metrics, transfer recomputation.
- The stage-9 analysis of the base generator v2. A within-band attacker (close, in view, one channel at the top of the benign band) caused 815 attacker-induced unsafe frames, against 687 for the standard above-band attack. Excess-based detectors were near chance on it (rule 0.009, Excess-HGB 0.011 PR-AUC on the attacker target).
- Proxy variance components and the simulations in this folder.

**Not seen:**
- Any in-band result on the family generators.
- Any result on a newly drawn generator.
- Any detector trained on attack-caused labels.

The confirmatory stage uses only new generators, new cohorts and new training seeds (Section 6).

---

## 1. Why a redesign, and what this plan changes at the root

| Root cause in r6 | Consequence | What v15 does |
|---|---|---|
| The research question is trivial for most detector pairs (information access fixes the coarse order) and underpowered for the informative pair | Headline from pairs never in doubt; top pair "cannot tell" | Focus on the comparable-information pair (MPNN, Raw-HGB). Ask how often label choices change its winner, with a design powered for that |
| Claims about 13 fixed generators; tests depend on a 12 × 12 covariance estimated from 60 shared cohorts | Existence claims only; fragile calibration (review 3, 3.2) | The **generator is the unit of inference**: M new generators drawn from the pre-existing prior (plan v7), each with its own independent cohorts and seeds, so ordinary across-generator inference is valid |
| The hypothesis was generated and tested on the same test cohorts | Not confirmatory | Exploratory stage on old data; confirmatory stage on new generators, cohorts and seeds only |
| The label "unsafe" is not the security target; the attack is removable by clipping | Weak security relevance (review 3, 3.5) | A **policy-compliant (in-band) attacker** that clipping cannot stop, and an exact **attack-caused** target alongside the all-unsafe target |
| Transfer headline produced by an untrained rule and a one-sided noise reference | Artefact (review 3, 3.3) | Transfer becomes one secondary outcome among trained detectors only. Noise is estimated by independent replication (split-half) |
| Analysis sprawl (H1, D1–D7, E2–E5, three families) | Unfocused paper | Two primary hypotheses, a small set of pre-declared secondary estimands, and nothing else |

---

## 2. Research question, unit of inference, estimands

**RQ.** When detectors of a policy-compliant avatar attack in multi-user VR can only be evaluated on synthetic labels, how often do the benchmark designer's two label choices change which of two comparable-information detectors performs better, and how does that compare with how often evaluation noise alone changes it? The two choices are:
- the harm generator, drawn from a plausible prior;
- the positive class: all unsafe frames (U) or attack-caused unsafe frames (A).

**Population.**
- Generators drawn from the random-generator prior of plan v7 (`stage7.sample_prior`), unchanged since 2026-10. It was not tuned for this study.
- Restricted to generators that are detectable for both targets (Section 6.2).
- In-band attack, scene and cohort model as in the benchmark.

**Detectors compared.** MPNN (learned message passing) and Raw-HGB (engineered per-peer products with gradient-boosted trees). Both see each peer's full state; they are the only detectors that can see an in-band attack (Section 5).

**Per-generator quantities** (g = generator, t ∈ {U, A}):
- PR^t_{d,g}: frame-level PR-AUC of detector d (tie-aware), mean over training seeds, on the generator's test cohorts. Expectation over cohorts and seeds defines the true value.
- **λ^t_g = log PR^t_{MPNN,g} − log PR^t_{Raw-HGB,g}**: log ratio. Its sign gives the winner. It is invariant to a common rescaling of both detectors' PR-AUC, so prevalence and difficulty differences between U and A do not create artificial shifts.
- **δ_g = λ^A_g − λ^U_g**: how the positive-class choice changes the comparison on generator g.

**Estimands:**

| Symbol | Definition | Answers |
|---|---|---|
| π_U | P_g(λ^U_g > 0), the share of generators on which the MPNN wins (target U) | Does the generator decide the winner? |
| ψ_G | 2 π_U (1 − π_U), the probability that two independent generators have different winners | How often a different generator changes the winner |
| μ_δ | E_g[δ_g] (exp(μ_δ) = multiplicative change in the MPNN/Raw-HGB PR-AUC ratio) | Does the target choice systematically favour one detector? |
| ψ_T | P_g(sign λ^U_g ≠ sign λ^A_g) | How often the target choice changes the winner |
| ψ_N | Probability that two independent replications at the design size (fresh cohorts and seeds, same generator and target) give different winners | Noise reference |
| π_A, ψ_G^A | As π_U and ψ_G under target A | Generator effect under the security target |

The paper's claims are exactly these estimates with their intervals. It makes no claim about human cybersickness, about generators outside the prior, or about detectors other than the two compared.

---

## 3. Threat model (strengthened)

**Attacker.**
- An admitted user who controls only their own avatar.
- Stays within the platform's public benign band on every rendered channel (policy-compliant), so an output policy that clips peer channels to the band has no effect.
- Chooses position and stimulus: moves close to and into the view of co-present users (`_near_pose`) and holds one channel at the top of the band (`near: True`, `near_clamp: True`; the "within-band attacker" of plan v7).
- In exploratory v2 data this attacker caused more attack-induced unsafe frames than the above-band attack (815 vs 687).

**Defender.** A detector on the victim's client or the session server, with the same inputs as before:
- each peer's rendered channels and three geometric features;
- the victim's on-device severity output;
- no labels, noise or generator coefficients.

**Why detection matters here.** Output policies of the kind proposed for AR output safety (Lebeck et al., IEEE S&P 2017; to be verified and cited) can block out-of-band content, but not an adversary who stays in band. Detection enables a response (hiding, muting or distancing the peer) and attribution.

**Out of scope, stated in the paper:**
- above-band attacks (preventable by clipping; summarised from the earlier study);
- colluding or adaptive attackers;
- other scenes, cohort sizes and attack policies;
- any human validity.

**Conceptual weaknesses that remain** are listed in Section 12.

---

## 4. Targets: definitions, validity and selection bias

### 4.1 Structural model

For victim i at frame t of a track, with generator G (per-peer contribution c_G ≥ 0, aggregator A_G ∈ {sum, max, LSE_τ}, gain g_G, noise SD σ_G):

  S_it = o_it + g_G · A_G( { c_G(x_jt, d_ijt, v_ijt) : j ≠ i } ) + ε_it,   ε_it ~ N(0, σ_G²) i.i.d., exogenous

  Y_it = 1[S_it > θ],   θ = 0.30.

On an attacked track with attacker a:
- The attacker's **benign plan** b⁰_a is an exogenous latent sampled before the attack policy acts: its originally sampled pose, plus a benign draw for the attack channel from the uniform benign distribution, using a dedicated random stream `[seed0 + cohort, track, 9]` (as in plan v9).
- The attack policy replaces b⁰_a by b^att_a: the near pose and the channel at the top of the band. Other peers, the victim's own term o and the noise ε are unchanged.

### 4.2 The two targets

- **Target U (all unsafe):** Y^U_it = Y_it(b^att), the observed label.
- **Target A (attack-caused):** for victims i ≠ a on attacked tracks, Y^A_it = Y_it(b^att) · (1 − Y_it(b⁰)). Otherwise Y^A_it = 0.
  - Y_it(b⁰) recomputes S_it with the attacker's contribution under its benign plan, using the **same** ε_it and the same other peers.
  - This is computed exactly inside corpus generation for every aggregator (sum, max, LSE): the aggregator is recomputed with one argument replaced.

### 4.3 Properties

- **P1 (exact identification).** Every term of Y^A is a deterministic function of exogenous variables and the intervention, all known in simulation. No approximation is involved for any aggregator. The existing "attacker-absent" label (r6/Stage-4: S − c_att ≤ θ) is exact only for sum aggregation and answers a different question (Section 4.5).
- **P2 (subset).** Y^A ≤ Y^U for every frame.
- **P3 (meaning).** Y^A = 1 exactly when the attack was necessary for the frame to be unsafe under the realised noise: the unit-level but-for condition (counterfactual necessity in the sense of Halpern and Pearl). Frames unsafe regardless of the attack are negatives under A.
- **P4 (prevented frames).** Because the attack moves the attacker, its contribution to a particular victim can be smaller than under the benign plan (A_G and c_G are monotone in intensity, but geometry differs). Frames with Y(b^att) = 0 and Y(b⁰) = 1 are possible. They are negatives under both targets and are counted and reported. This does not bias Y^A; it is part of the necessity definition.
- **P5 (no sample selection).**
  - The evaluation population is identical for U and A: every victim-frame of the test cohorts.
  - Positives and negatives are fixed by a function of exogenous variables, never by a detector output.
  - The evaluation set is never restricted by Y(b⁰). Excluding unsafe-regardless frames would condition on a function of ε, which changes the joint distribution of features and noise (selection on a collider of the noise and the peer stimuli). That variant appears only as a descriptive three-class table.
- **P6 (no privileged information).**
  - b⁰ is independent of everything a detector observes: the original pose is an independent draw; the near pose is chosen from the other users' poses; the benign channel draw is independent.
  - The uncertainty b⁰ adds to Y^A is therefore irreducible label noise that is the same for every detector. No detector observes b⁰ or ε.
- **P7 (ceilings).**
  - Given the deterministic scores s and s⁰ (attack and benign plan), P(Y^A = 1 | s, s⁰) = Φ((θ − s⁰)/σ) − Φ((θ − s)/σ) if s > s⁰, else 0.
  - The **observable ceiling** integrates this over K = 32 Monte Carlo draws of b⁰ given the observed state. It is used for the detectability filter and reported as the attainable PR-AUC.
  - The full-knowledge oracle is reported only as a loose upper bound.
- **P8 (scale).** Prevalence and attainable PR-AUC are lower under A than under U. All comparisons across targets therefore use the log ratio λ (P8 is why λ, not the raw difference, is primary). The raw difference is reported as a sensitivity.

### 4.4 Remaining sources of bias and how each is handled

| Possible bias | Mechanism | Handling |
|---|---|---|
| Objective mismatch | Detectors trained on U are scored on A | Primary analysis uses **matched training** (train and select on t, score on t). Mismatch is secondary outcome S4 |
| Training sample size | Fewer positives under A, so detectors with different data efficiency may differ | Robustness R3: re-train U models with positives subsampled to the A count (first 20 confirmatory generators) |
| Scale | Lower prevalence shrinks PR-AUC under A | Log ratio λ (P8) |
| Counterfactual choice | "Benign plan" is itself a design choice | S3: the attacker-absent definition as a sensitivity target |
| Population definition | Detectability filter | Pre-specified (Section 6.2); the number of draws screened out is reported |

### 4.5 Sensitivity target A0 (attacker absent)

Y^{A0}_it = Y_it(b^att) · 1[S_it − g_G · (contribution of a) ≤ θ] for sum aggregation, and the exact recomputation with the attacker removed for max and LSE. It answers "would the frame be safe without this peer at all". It over-counts attack-caused frames relative to A whenever a benign peer at the same place would also have caused harm.

### 4.6 Gate G0: code validation before any training

The new generation module (`gen_v15.py`, a new file; `stage3.py` is not modified) must pass:
1. With the in-band option off, it reproduces existing corpora bit-exactly for every existing field (d2, rand00 and rand04 dev corpora; seeds 60000–60004).
2. With the option on, it reproduces the existing v2 near corpus `s7_near_near.npz` (cohorts 70000–70004).
3. For sum aggregation, Y^{A0} equals the r6/Stage-4 attacker-induced label exactly.
4. Y^A ≤ Y^U on every frame; prevented frames counted.
5. The recomputed aggregator equals the stored peer term when no argument is replaced (max and LSE).
6. Seed ranges are asserted disjoint from every range in Section 6.3.

---

## 5. Detectors and training

**Compared:** MPNN and Raw-HGB, with features exactly as in r6:
- MPNN: graph inputs; optional log-angular-velocity feature in its grid (correction C2).
- Raw-HGB: raw and excess channel products with v_ij/(d_ij + 0.1) and 1/(d_ij + 0.1), summed and maximised over peers.

**Not compared, and why:**
- **Rule and Excess-HGB.** They see only the excess over the band, which is zero for an in-band attacker. Stage X documents this (the rule needs no training).
- **GAT, GAT-sum, prototype detectors and E2/E3 generators.** Not needed for the research question; reported in the earlier study.

**Equal tuning:**
- The same 16-configuration grids as r6.
- For each seed: a 45/15 cohort split of the 60-cohort development corpus; configuration selected by validation PR-AUC **on the same target as training**; the test corpus scored once.
- **Trained models are saved**, which r6 did not do.

**Seeds:**
- Stage X: {0, 1, 2, 42, 2024}.
- Stage C: four new seeds {101, 211, 307, 401}.

---

## 6. Exploratory and confirmatory data

### 6.1 Stage X (exploratory): existing generators, cohorts and seeds

- **Generators.** The 13 E1 generators (5 structural, 8 drawn from the prior), with the in-band attack added. Cohorts are the existing dev and test cohorts: identical inputs, only the attacker's behaviour changes. Noise draws are verified identical.
- **Training.** MPNN and Raw-HGB on both targets, 5 seeds, full tuning. The rule is scored on both targets (to document in-band blindness).
- **Outputs (exploratory, labelled as such):**
  - prevalences and ceilings for U, A and A0;
  - prevented-frame counts;
  - λ^U, λ^A, δ per generator;
  - variance components on the log scale;
  - feasibility gates.
- **Gates.** If any of G1–G2 fails, stop and report to the authors before Stage C:
  - **G1 (detectability):** at least 6 of the 8 prior-drawn generators have observable ceilings ≥ 0.15 for both U and A.
  - **G2 (learnability):** on the median generator, both detectors reach PR-AUC ≥ 3 × prevalence under A.
- **What Stage X fixes before Stage C (committed as plan v15b, a pure numeric addendum):**
  - M, by the sample-size rule in Section 7, using only the 8 prior-drawn generators.
  - Nothing else: hypotheses, tests, estimands and decision rules are fixed **now**.

### 6.2 Stage C (confirmatory): new generators, new cohorts, new seeds

**Generator draws.**
- `sample_prior(k)` for k = 100, 101, … (prior random streams 7100+, disjoint from the existing 7000–7011).
- Gains are calibrated label-free exactly as before.

**Detectability screening:**
- Each draw is screened on its own 20-cohort screening corpus.
- It enters the study if the observable ceilings for U and A are both ≥ 0.15.
- Draws continue in order until M generators qualify. Draw order and screening results are logged.
- Screening uses only oracle quantities, never a detector.

**Independence by construction:**

| Component | Stage X | Stage C |
|---|---|---|
| Generators | the 13 existing | new draws k ≥ 100 |
| Dev cohorts | 60000–60059 (shared) | 1,000,000 + 1000k + [0, 60), separate per generator |
| Test cohorts | 61000–61059 (shared) | 2,000,000 + 1000k + [0, 240), separate per generator |
| Screening cohorts | — | 3,000,000 + 1000k + [0, 20) |
| Base generator for S2 | — | dev 4,000,000 + [0, 60) |
| Training seeds | {0, 1, 2, 42, 2024} | {101, 211, 307, 401} |
| Attack policy, grids, features, selection rule | as Stage C | identical |

- Every Stage C generator has its own cohorts, so per-generator estimates are statistically independent. The generator is the unit of analysis, and ordinary across-generator inference (t-test, bootstrap over generators) is valid without estimating any cross-generator covariance.
- Stage C outputs are not inspected until all M generators are complete. No interim analysis.
- The confirmatory analysis script (`analyse_v15.py`) is committed with this plan, before Stage C generation. It runs once.

### 6.3 Seed ranges already in use (asserted disjoint in code)

- 42–101 (main);
- 10,000–10,059 (calibration);
- 20,000–20,119 (lock-box 1);
- 30,000–33,059;
- 40,000–45,039;
- 50,000–54,039;
- 60,000–60,059 and 61,000–61,059 (E1 dev and test);
- 70,000–70,059 (near corpus);
- 80,000–80,119 (lock-box 2).

All Stage C ranges are ≥ 1,000,000.

---

## 7. Sample size and power (independently simulated)

**Simulation model** (`power_sim_v15*.py`). For each simulated study, M generators:
- **Generator level:** true λ^U_g ~ N(μ_U, τ_U²) and δ_g with mean μ_δ, SD τ_δ, correlation ρ_G with λ^U.
- **Cohort level:** variance sdc² · 60/n.
- **Seed level:** variance sds²/S.
- (sdc, sds) pairs are resampled from the 13 empirical per-generator proxies.
- Target A noise is inflated by f_A; errors for U and A correlate with ρ_E = 0.3 (shared test cohorts).

The **exact pre-specified analysis** is applied:
- DerSimonian–Laird fit;
- π = Φ(μ/τ);
- percentile bootstrap over generators;
- intersection-union decision;
- t-test on δ.

**Proxy inputs** (`proxy_components.json`; above-band attack; A is a mismatch proxy). Over the 8 prior-drawn generators, MPNN − Raw-HGB has mean +0.029 and SD 0.077, so τ_U ≈ 0.065 after removing noise. On the log scale, δ has mean +0.038 and SD 0.115 (prior-drawn) or +0.143 and SD 0.225 (all 11). corr(λ^U, δ) = −0.82.

**Findings:**
1. **The number of generators M drives power. Seeds and cohorts matter little beyond moderate values.**
   - At M = 40 with 240 cohorts, H1/H2 power is 0.95/0.88 with 4 seeds, 0.96/0.91 with 6 and 0.94/0.91 with 10 (base scenario; in this comparison H1 used a 90% interval and H2 α = 0.025).
   - Cost rises from 36 to 53 to 88 CPU-h.
   - The earlier proposal of 240 cohorts and 10 seeds was designed for fixed generators and is not appropriate here.
2. **Design trade-off.** Under the prior-only proxy for the target shift (μ_δ 0.028, τ_δ 0.07, ρ_G −0.76), H2 power is 0.50 at M = 40 (S 6), 0.70 at M = 60 (S 6), 0.79 at M = 80 (S 4) and 0.89 at M = 100 (S 4). Spending compute on generators rather than seeds is better.
3. **Calibration (simulated size under the null boundaries):**
   - H1 at 0.025 (95% interval inside (0.10, 0.90)): size 0.013–0.019 at π = 0.10 and 0.90; interval coverage 0.91–0.95.
   - H2 at 0.025: size about 0.03 (Monte Carlo SE 0.006).
4. **Precision at the default design.** ψ_G 95% CI half-width about 0.07; μ_δ half-width about 0.017 (raw scale). The noise flip probability at the design size is ψ_N ≈ 0.15 (raw-scale proxy).

**Default design: M = 80, n = 240 test cohorts, S = 4 seeds.** About 72 CPU-h for the primary training.

| Scenario | H1 power (α 0.025) | H2 power (α 0.025) |
|---|---|---|
| Base (μ_δ 0.04) | 0.996 | 0.99 |
| Prior-only proxy | ≈ 1.00 | 0.79 |
| Weak (τ_U and μ_δ halved) | 0.49 | 0.60 |
| π_U = 0.75 | 0.94 | — |

**Sample-size re-estimation (the only quantity fixed after Stage X):**
1. Re-run `power_sim_v15.py` on the log scale with Stage X estimates from the 8 prior-drawn generators.
2. Shrink μ̂_δ by 25% toward 0 (winner's-curse guard).
3. Choose the smallest M ∈ {60, 70, 80, 90, 100} with H2 power ≥ 0.80. If none reaches it, M = 100 (about 90 CPU-h), and the paper states that H2 is powered at the reported level.
4. M never decreases below 60.

**If effects are weak**, the study is still informative: the estimates of ψ_G, ψ_T and μ_δ have the precision above, and a well-estimated small effect is a result (Section 9).

---

## 8. Confirmatory analysis (fixed now)

### 8.1 Per generator

- Per-seed PR-AUC (tie-aware, as r6) for each detector and target.
- PR^t_{d,g} = mean over the 4 seeds.
- λ^t_g, δ_g as in Section 2.
- Within-generator sampling variance v^t_g of λ^t_g and covariance with the other target: **two-level bootstrap** (B = 2000; resample the 4 seeds with replacement, then the 240 test cohorts; the same draws for both detectors and both targets).

### 8.2 Primary hypotheses

Family-wise α = 0.05, Bonferroni-split: each primary test at 0.025.

**H1: the generator decides the winner (target U).**
- H0: π_U ≤ 0.10 or π_U ≥ 0.90. H1: 0.10 < π_U < 0.90.
- Estimation:
  - DerSimonian–Laird random-effects fit of λ^U_g with known v^U_g gives μ̂, τ̂ and π̂_U = Φ(μ̂/τ̂).
  - REML as a sensitivity.
- Test: percentile bootstrap over generators (B = 4000; resample generators and refit). Reject H0 if the 95% interval of π_U lies inside (0.10, 0.90). This is an intersection-union test of two one-sided tests at 0.025; simulated size ≤ 0.019.
- Effect size: π̂_U and ψ̂_G = 2π̂(1 − π̂), with 95% bootstrap CIs.
- Why 0.10: below it a designer would see the same winner on at least 90% of plausible generators, which practically means "the generator does not decide".

**H2: the positive-class choice shifts the comparison.**
- H0: μ_δ = 0. H1: μ_δ ≠ 0.
- Test: two-sided one-sample t-test of δ_g across the M generators (df M − 1), α = 0.025. Valid because the generators are independent.
- Effect size: μ̂_δ with 97.5% and 95% t-intervals; exp(μ̂_δ) as the multiplicative change in the MPNN/Raw-HGB PR-AUC ratio; standardised mean μ̂_δ / SD(δ).
- Robustness (reported, not a decision): Wilcoxon signed-rank test; raw-difference scale.

### 8.3 Key secondary estimands (estimates and 95% CIs; no confirmatory claims)

- **ψ_T.**
  - Bivariate random-effects model (method of moments) for (λ^U_g, λ^A_g) with the within-generator covariance from 8.1; ψ_T is the bivariate-normal probability of opposite signs.
  - Bootstrap over generators.
  - The naive proportion of generators with opposite observed signs is also reported.
- **ψ_N (noise reference), two ways:**
  - Model-based at the design size: mean over generators of 2Φ(|λ̃_g|/σ_rep)(1 − Φ(|λ̃_g|/σ_rep)), using empirical-Bayes-shrunk λ̃_g and the replication SD.
  - Model-free split-half: seeds {101, 211} with test cohorts [0, 120) vs seeds {307, 401} with cohorts [120, 240). This gives two fully independent half-size replications per generator; report the observed share of generators with different winners (half size).
- **π_A and ψ_G^A:** as H1, under target A.
- Comparisons **ψ_G vs ψ_N** and **ψ_T vs ψ_N**: the paired bootstrap difference with a 95% CI. Reported as estimates.

### 8.4 Secondary outcomes (pre-specified; reported in the paper's robustness section or supplement)

- **S1 (metric).** Repeat 8.2–8.3 with episode-level PR-AUC (episode = cohort, track, victim; score = maximum over frames; label = any positive frame).
- **S2 (generator shift).**
  - MPNN and Raw-HGB trained (per target, same grids and seeds) on the base generator v2 with the in-band attack (dev cohorts 4,000,000+) and scored on each Stage C test corpus.
  - Estimands: median retention ratio PR(shift)/PR(within) per detector and target, with bootstrap CI; and P(the within-generator winner equals the shift winner).
- **S3 (counterfactual definition).** H2 and ψ_T with A0 in place of A.
- **S4 (objective mismatch).** U-trained models scored on A. Estimand: P(winner same as matched A).
- **R3 (positive count).**
  - For the first 20 Stage C generators in draw order, U-models re-trained with unsafe positives randomly subsampled (stream 13) to the A-positive count; rows removed, not relabelled.
  - Estimand: mean change in λ^U with 95% CI.

### 8.5 Model checks (reported)

- Q-Q plot and Shapiro–Wilk of the λ^U_g (descriptive).
- Leave-one-generator-out influence on π̂_U and μ̂_δ.
- Non-parametric bounds that need no normality: k₊/M and k₋/M, where k₊ and k₋ count generators on which the 95% within-generator interval of λ^U lies above or below zero (Holm within direction). These are conservative lower bounds for π_U and 1 − π_U.

### 8.6 Failures and missing data

- A training run that fails is repeated once with the next configuration in grid order.
- A generator that still cannot be trained is replaced by the next qualifying draw; all such events are reported.
- No outcome-dependent exclusion.

### 8.7 Not analysed

Interaction or crossover tests over fixed generators, reversal tests, Friedman/Nemenyi, other detectors, E2/E3 generators, and GAT variants.

---

## 9. Interpretation rules (fixed now)

| H1 | H2 | ψ_T vs ψ_N | Paper's conclusion |
|---|---|---|---|
| Reject | Reject | ψ_T CI above ψ_N | Both label choices change the winner beyond noise. The positive class also favours one detector systematically (direction from μ̂_δ) |
| Reject | Not reject | any | The generator decides the winner; no detectable systematic target shift (report the CI of μ_δ as the bound on it) |
| Not reject, π̂ near 0 or 1 | any | any | Within this prior the generator rarely changes the winner of the top pair. Report ψ_G with its upper confidence bound |
| Not reject, wide CI | any | any | Inconclusive for the generator; report the achieved precision (planned power applies) |

In every case the result is quantified with confidence intervals over a population of generators, not as an existence claim.

---

## 10. Compute and timeline (estimates from r6 logs)

- **Basis.** MPNN tuning 863 s and Raw-HGB 1,095 s per generator for 16 configurations × 5 seeds; corpus generation about 0.35 s per cohort; 3 workers fit in RAM.

| Step | CPU-h | Wall time (3 workers) |
|---|---|---|
| G0 code and checks | 1 | 1 h |
| Stage X: 13 generators × 2 targets × 5 seeds | 14 | 5 h |
| v15b addendum (M), commit | — | 0.5 day of analysis |
| Stage C screening and generation (about 120 draws) | 6 | 2 h |
| Stage C training, M = 80, S = 4, 2 targets | 72 | 24 h |
| S2 base-generator training | 1 | 0.5 h |
| R3 (20 generators) | 9 | 3 h |
| Scoring, bootstraps, analysis | 6 | 3 h |
| **Total** | **≈ 110** | **≈ 2 days machine time** |

**Writing:** about 1 week.

---

## 11. The focused paper this produces

**Working title.** *Who Wins Depends on Who Writes the Labels: Generator and Target Choices in a Synthetic Benchmark of Policy-Compliant Avatar Attacks in Multi-User VR* (to be refined).

**Structure:**
1. Problem.
2. Related work (benchmark construction, simulation studies and multiverse analysis, XR attacks and output safety).
3. Benchmark, in-band threat model and the two targets.
4. Design (prior, independence, power).
5. Results (H1, H2, ψ comparisons, then S1–S4 and R3 briefly).
6. Discussion.
7. Limitations.

**From r6:** one paragraph of exploratory evidence (the earlier fixed-generator study and the audit) explains where the hypotheses came from. Everything else from r6 is not carried over.

---

## 12. Strict self-assessment: remaining weaknesses

1. **ψ_G is relative to the prior.** A broader prior would change it. Mitigation: the prior is the unchanged plan-v7 prior, fixed before this question existed, so it was not chosen to produce a result. The paper frames ψ_G as the effect of a designer's freedom within a plausible prior (multiverse framing). Prior components (aggregation type, distance kernel) are reported as exploratory subgroups. **This is the deepest limitation. It is inherent to any synthetic-label study and is stated, not hidden.**
2. **Two detectors.** The claim is about the winner between two strong, comparable-information detectors. Justification: they are the only ones that can see an in-band attack, and they represent the two dominant paradigms. It is not a claim about all detectors.
3. **No human ground truth.** The study is about benchmark construction, not about VR sickness. This is unchanged and unavoidable.
4. **The counterfactual definition is a design choice.** S3 shows how much it matters.
5. **Weak-effect risk.** If the true effects are about half the proxies, H1/H2 power is about 0.5–0.6. The estimates remain informative (Section 9), and Stage X re-estimates M before Stage C.
6. **Proxies come from the above-band attack.** Stage X is required precisely to replace them, and gates G1–G2 stop the study if the in-band design is infeasible.
7. **One scene, N = 4, one attack policy, static poses, fixed threshold and gain calibration.** Stated as scope.

**None of these is fatal.** The design answers a quantitative, population-level question with fresh data and calibrated tests. The remaining limits concern generalisation beyond the authored prior, and they are inherent to the field's practice that the paper examines.

---

## 13. Why this can produce a stronger, publishable contribution

1. **A real answer instead of "cannot tell".** Estimates with intervals of how often each label choice changes the winner. The power is shown in advance, and every outcome is interpretable (Section 9).
2. **Valid, simple inference.** Independent generators remove the shared-cohort dependence and the covariance-estimation problem that reviews 1–3 attacked.
3. **Genuinely confirmatory.** New generators, cohorts and seeds; the plan and analysis code are committed first; no reuse of the data that generated the hypotheses.
4. **Security relevance.** The attack cannot be stopped by output clipping, and the security target (attack-caused harm) is defined exactly and compared with the safety target.
5. **Focus.** Two primary hypotheses, one coherent message about benchmark construction, and a short list of pre-declared robustness checks.

**Venues:** Computers & Security first (security evaluation methodology); PeerJ Computer Science as the lower-risk alternative.

---

## 14. Decisions needed from the authors before commitment

1. Approve the in-band attacker as the primary threat (Section 3).
2. Approve the two-detector focus (Section 5).
3. Approve the default design (M = 80, n = 240, S = 4) and the re-estimation rule (Section 7).
4. Approve the primary scale (log PR-AUC ratio) and the α split (0.025 each).
5. Approve dropping r6 components not listed in Section 11.
6. After approval: I write `gen_v15.py`, `analyse_v15.py` and the Stage X runner, run G0, commit the plan with the code, and only then run Stage X.
