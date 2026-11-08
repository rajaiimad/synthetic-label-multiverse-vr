"""
Plan v15: simulation of the confirmatory design and of its pre-specified analysis (power, type-I error, CI coverage).

Data-generating model (one simulated study):
  for each of M generators drawn from the prior (independent cohorts and seeds per generator):
    true differences (MPNN - Raw-HGB):  Delta_U ~ N(mu_U, tau_U^2);  delta = Delta_A - Delta_U ~ N(mu_d, tau_d^2),
      with corr(Delta_U, delta) = rho_G
    noise:  se_t^2(n, S) = sdc_t^2 * 60 / n + sds_t^2 / S, (sdc, sds) drawn from the empirical per-generator proxy
      pairs (proxy_components.json); target A noise inflated by f_A (fewer positives)
    observed:  DU_hat = Delta_U + e_U;  DA_hat = Delta_A + e_A;  corr(e_U, e_A) = rho_e (shared test cohorts)
Analysis (exactly as specified in ANALYSIS_PLAN_v15.md):
  H1: random-effects (DerSimonian-Laird) fit of DU_hat with known se; pi = Phi(mu / tau); 90% percentile bootstrap
      CI over generators (B resamples); reject (IUT) if the CI lies inside (0.10, 0.90).
  H2: one-sample t-test of delta_hat = DA_hat - DU_hat across generators, two-sided alpha = 0.025 (Holm over H1, H2
      at the first step; conservative).
  psi_G = 2 pi (1 - pi): CI half-width reported.
  psi_N (noise flip probability at the design size), for reference.
Outputs redesign_v15/power_sim_v15.json and a printed table.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, t as tdist

HERE = Path(__file__).resolve().parent
PX = json.loads((HERE / "proxy_components.json").read_text())
PAIRS = np.array([[v["sd_cohort"], v["sd_seed"]] for v in PX["U"].values()])     # empirical noise pairs (13)
COST_PER_GEN_TARGET_SEED_H = (863 + 1095) / 5 / 3600                               # MPNN + Raw-HGB, 16 configurations, one seed
GEN_S_PER_COHORT = 0.35


def cost(M, n, S):
    return M * 2 * S * COST_PER_GEN_TARGET_SEED_H + M * (60 + n) * GEN_S_PER_COHORT / 3600


def dl(y, v):
    """DerSimonian-Laird on the last axis. y, v: (..., M). Returns mu_RE, tau2."""
    w = 1.0 / v
    sw = w.sum(-1)
    mu_fe = (w * y).sum(-1) / sw
    Q = (w * (y - mu_fe[..., None]) ** 2).sum(-1)
    M = y.shape[-1]
    c = sw - (w ** 2).sum(-1) / sw
    tau2 = np.maximum(0.0, (Q - (M - 1)) / c)
    ws = 1.0 / (v + tau2[..., None])
    mu = (ws * y).sum(-1) / ws.sum(-1)
    return mu, tau2


def pi_of(mu, tau2):
    tau = np.sqrt(tau2)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tau > 0, norm.cdf(mu / np.where(tau > 0, tau, 1.0)), (mu > 0).astype(float))
    return p


def simulate(sc, M, n, S, R=400, B=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(PAIRS), (R, M))
    sdc, sds = PAIRS[idx, 0], PAIRS[idx, 1]
    seU2 = sdc ** 2 * 60 / n + sds ** 2 / S
    seA2 = seU2 * sc["fA"] ** 2
    # true generator effects
    z1, z2 = rng.standard_normal((2, R, M))
    DU = sc["muU"] + sc["tauU"] * z1
    dl_true = sc["mud"] + sc["taud"] * (sc["rhoG"] * z1 + np.sqrt(1 - sc["rhoG"] ** 2) * z2)
    DA = DU + dl_true
    # observation noise
    e1, e2 = rng.standard_normal((2, R, M))
    eU = np.sqrt(seU2) * e1
    eA = np.sqrt(seA2) * (sc["rhoE"] * e1 + np.sqrt(1 - sc["rhoE"] ** 2) * e2)
    DUh, DAh = DU + eU, DA + eA
    # ---- H1: DL + bootstrap over generators
    mu, tau2 = dl(DUh, seU2)
    pih = pi_of(mu, tau2)
    bi = rng.integers(0, M, (R, B, M))
    yb = np.take_along_axis(np.repeat(DUh[:, None, :], B, 1), bi, 2)
    vb = np.take_along_axis(np.repeat(seU2[:, None, :], B, 1), bi, 2)
    mub, t2b = dl(yb, vb)
    pib = pi_of(mub, t2b)
    lo, hi = np.percentile(pib, 5, axis=1), np.percentile(pib, 95, axis=1)
    rejH1 = (lo > 0.10) & (hi < 0.90)
    true_pi = norm.cdf(sc["muU"] / sc["tauU"]) if sc["tauU"] > 0 else float(sc["muU"] > 0)
    cover = np.mean((lo <= true_pi) & (true_pi <= hi))
    psib = 2 * pib * (1 - pib)
    psi_hw = np.mean((np.percentile(psib, 97.5, axis=1) - np.percentile(psib, 2.5, axis=1)) / 2)
    # ---- H2: t-test on delta_hat across generators (alpha 0.025 two-sided)
    d = DAh - DUh
    tt = d.mean(1) / (d.std(1, ddof=1) / np.sqrt(M))
    p2 = 2 * tdist.sf(np.abs(tt), M - 1)
    rejH2 = p2 < 0.025
    d_hw = np.mean(tdist.ppf(0.975, M - 1) * d.std(1, ddof=1) / np.sqrt(M))
    # ---- noise flip probability at the design size: two independent replications of one generator
    sigma_rep = np.sqrt(seU2)
    pf = 2 * norm.cdf(np.abs(DU) / sigma_rep) * norm.sf(np.abs(DU) / sigma_rep)
    return {"powerH1": float(rejH1.mean()), "powerH2": float(rejH2.mean()), "coverage_pi90": float(cover),
            "pi_true": float(true_pi), "pi_hat_mean": float(pih.mean()), "psiG_true": float(2 * true_pi * (1 - true_pi)),
            "psiG_ci95_halfwidth": float(psi_hw), "mu_delta_ci95_halfwidth": float(d_hw), "psiN_design": float(pf.mean()),
            "naive_sign_share_mean": float((DUh > 0).mean()), "cost_cpu_h": cost(M, n, S)}


# proxy-based scenarios (see ANALYSIS_PLAN_v15.md, section 7)
rnd = PX["random"]
dU = np.array([PX["U"][g]["d"] for g in rnd])
base = dict(muU=0.03, tauU=0.065, mud=0.04, taud=0.06, rhoG=-0.5, rhoE=0.3, fA=1.3)
SC = {
    "base": base,
    "weak": {**base, "tauU": 0.035, "mud": 0.02},
    "noisyA": {**base, "fA": 1.8},
    "H1_boundary_pi0.90": {**base, "muU": norm.ppf(0.90) * 0.065},
    "H1_null_pi0.97": {**base, "muU": norm.ppf(0.97) * 0.065},
    "H2_null": {**base, "mud": 0.0},
}
out = {"proxy_dU_random": dU.tolist(), "proxy_dU_mean": float(dU.mean()), "proxy_dU_sd": float(dU.std(ddof=1)),
       "scenarios": SC, "results": []}
designs = [(M, n, S) for M in (20, 30, 40, 50, 60) for n in (60, 120, 240, 480) for S in (4, 6, 10)]
for name in ("base", "weak"):
    for (M, n, S) in designs:
        r = simulate(SC[name], M, n, S, seed=hash((name, M, n, S)) % 2 ** 31)
        out["results"].append({"scenario": name, "M": M, "n": n, "S": S, **r})
for name in ("noisyA", "H1_boundary_pi0.90", "H1_null_pi0.97", "H2_null"):
    for (M, n, S) in [(30, 240, 6), (40, 240, 6), (60, 240, 6), (40, 120, 6), (40, 240, 4)]:
        r = simulate(SC[name], M, n, S, R=600, seed=hash((name, M, n, S)) % 2 ** 31)
        out["results"].append({"scenario": name, "M": M, "n": n, "S": S, **r})
(HERE / "power_sim_v15.json").write_text(json.dumps(out, indent=1))

print(f"proxy random-generator dU mean {dU.mean():+.3f} sd {dU.std(ddof=1):.3f}")
hdr = "scenario              M    n   S  powH1 powH2 cov90  psiG±  muD±  psiN  cpu_h"
print(hdr)
for r in out["results"]:
    if r["scenario"] in ("base", "weak") and not (r["S"] == 6 or (r["M"] == 40 and r["n"] == 240)):
        continue
    print(f"{r['scenario']:20s} {r['M']:3d} {r['n']:4d} {r['S']:3d}  {r['powerH1']:.2f}  {r['powerH2']:.2f}  {r['coverage_pi90']:.2f}"
          f"  {r['psiG_ci95_halfwidth']:.2f}  {r['mu_delta_ci95_halfwidth']:.3f}  {r['psiN_design']:.2f}  {r['cost_cpu_h']:5.1f}")
