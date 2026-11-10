"""
Deviation D5: secondary outcome S6, the meaningful gap.
  Specified and approved by the authors on 2026-11-10, while Stage C was training and before any Stage C detector
  result was inspected. Frame level only (the primary metric). Estimates with 95% intervals; no hypothesis test.

Question: are the winner flips between MPNN and Raw-HGB flips between nearly tied detectors, or between detectors
that differ by a practically meaningful margin?

Margin: c = log(1.10), i.e. one detector's PR-AUC is at least 10% higher than the other's (primary).
Sensitivity: c = log(1.05) and log(1.25). They are reported next to the primary and never substituted for it.

For each c:
  Observed (the analysed generators):
    under U and under A, the number of generators with lambda >= c (MPNN better by the margin), lambda <= -c
    (Raw-HGB better by the margin) and |lambda| < c (near tie); the same with the additional condition that the
    within-generator 95% interval excludes 0 (|lambda| >= 1.96 sqrt(v_two));
    meaningful target flips: sign(lambda^U) != sign(lambda^A) with |lambda^U| >= c and |lambda^A| >= c.
  Model-based (the random-effects models of plan 8.2-8.3; percentile bootstrap over generators with the same
  B = 4000 resamples as analyse_v15.aggregate, generator seed 15):
    pi_plus(c)  = P(lambda^U >= c)  = Phi((mu_U - c) / tau_U)
    pi_minus(c) = P(lambda^U <= -c) = Phi((-c - mu_U) / tau_U)
    psi_G(c) = 2 pi_plus(c) pi_minus(c): two independently drawn generators give opposite winners, each by >= c
    psi_T(c) = P(lambda^U >= c, lambda^A <= -c) + P(lambda^U <= -c, lambda^A >= c), under the bivariate model used
               for psi_T (method of moments with the within-generator covariance)
    psi_N(c) = mean over generators of 2 Phi((l_g - c)/s_g) Phi((-c - l_g)/s_g): two independent replications of
               the same generator give opposite winners, each by >= c (l_g: empirical-Bayes-shrunk lambda^U,
               s_g = sqrt(v_two), as psi_N of plan 8.3)
    psi_G(c) - psi_N(c) and psi_T(c) - psi_N(c), with paired bootstrap intervals.
  At c = 0 these reduce to psi_G, psi_T and psi_N of the plan (checked in the self-test below).

Pre-specified reading (primary c = log 1.10):
  - If the 95% interval of psi_G(c) - psi_N(c) lies above 0, the paper states that the generator changes the winner by
    a meaningful margin more often than noise does. Otherwise it states that, at this margin, generator-driven flips
    cannot be distinguished from noise.
  - The same reading applies to psi_T(c).

  python v15/s6_meaningful_gap.py --stage C        -> results_v15/s6_meaningful_gap_C.json
  python v15/s6_meaningful_gap.py --selftest
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyse_v15 import B_ACROSS, OUT, ci, dl, pi_of, psi_T  # noqa: E402

MARGINS = {"1.05": np.log(1.05), "1.10": np.log(1.10), "1.25": np.log(1.25)}
PRIMARY = "1.10"


def bivar(X, Wm):
    """Mean and between-generator covariance of (lambda^U, lambda^A); identical steps to analyse_v15.psi_T."""
    mu = X.mean(0)
    Sg = np.cov(X, rowvar=False) - Wm
    ev, evec = np.linalg.eigh(Sg)
    return mu, evec @ np.diag(np.maximum(ev, 1e-8)) @ evec.T


def psi_T_c(X, Wm, c):
    mu, Sg = bivar(X, Wm)
    s = np.sqrt(np.diag(Sg))
    F = stats.multivariate_normal(mean=mu, cov=Sg, allow_singular=True).cdf
    a = stats.norm.cdf((-c - mu[1]) / s[1]) - F(np.array([c, -c]))          # lambda^U >= c and lambda^A <= -c
    b = stats.norm.cdf((-c - mu[0]) / s[0]) - F(np.array([-c, c]))          # lambda^U <= -c and lambda^A >= c
    return float(max(a, 0.0) + max(b, 0.0))


def pis(y, v, c):
    mu, t2 = dl(y, v)
    if t2 <= 0:
        return float(mu >= c), float(mu <= -c)
    t = np.sqrt(t2)
    return float(stats.norm.cdf((mu - c) / t)), float(stats.norm.cdf((-c - mu) / t))


def psi_N_c(y, v, c):
    mu, t2 = dl(y, v)
    lt = mu + t2 / (t2 + v) * (y - mu)
    s = np.sqrt(v)
    return float(np.mean(2 * stats.norm.cdf((lt - c) / s) * stats.norm.cdf((-c - lt) / s)))


def observed(lam, v, c):
    clear = np.abs(lam) >= 1.959963984540054 * np.sqrt(v)
    return {"n_mpnn": int((lam >= c).sum()), "n_rawhgb": int((lam <= -c).sum()), "n_near_tie": int((np.abs(lam) < c).sum()),
            "n_mpnn_clear": int(((lam >= c) & clear).sum()), "n_rawhgb_clear": int(((lam <= -c) & clear).sum())}


def s6(stage):
    an = json.loads((OUT / f"analysis_{stage}.json").read_text())
    gens = an["generators"]
    S = {g: json.loads((OUT / stage / g / "summary_v15.json").read_text())["frame"] for g in gens}
    lu = np.array([S[g]["U"]["lam"] for g in gens]); vu = np.array([S[g]["U"]["v_two"] for g in gens])
    la = np.array([S[g]["A"]["lam"] for g in gens]); va = np.array([S[g]["A"]["v_two"] for g in gens])
    cua = np.array([S[g].get("cov_UA_two", 0.0) for g in gens])
    M = len(gens)
    BI = np.random.default_rng(15).integers(0, M, (B_ACROSS, M))             # the resamples of analyse_v15.aggregate
    X = np.c_[lu, la]

    def W(i):
        return np.array([[vu[i].mean(), cua[i].mean()], [cua[i].mean(), va[i].mean()]])

    res = {"stage": stage, "M": M, "deviation": "D5", "margins": {k: float(v) for k, v in MARGINS.items()},
           "primary": PRIMARY}
    for k, c in MARGINS.items():
        pp, pm = pis(lu, vu, c)
        pG, pT, pN = 2 * pp * pm, psi_T_c(X, W(slice(None)), c), psi_N_c(lu, vu, c)
        b = np.array([[*pis(lu[i], vu[i], c), psi_T_c(X[i], W(i), c), psi_N_c(lu[i], vu[i], c)] for i in BI])
        bG = 2 * b[:, 0] * b[:, 1]
        flips = (np.sign(lu) != np.sign(la))
        r = {"observed": {"U": observed(lu, vu, c), "A": observed(la, va, c),
                          "n_target_flips": int(flips.sum()),
                          "n_meaningful_target_flips": int((flips & (np.abs(lu) >= c) & (np.abs(la) >= c)).sum())},
             "model": {"pi_plus": pp, "pi_plus_ci95": ci(b[:, 0]), "pi_minus": pm, "pi_minus_ci95": ci(b[:, 1]),
                       "psi_G": pG, "psi_G_ci95": ci(bG), "psi_T": pT, "psi_T_ci95": ci(b[:, 2]),
                       "psi_N": pN, "psi_N_ci95": ci(b[:, 3]),
                       "psi_G_minus_psi_N": {"est": pG - pN, "ci95": ci(bG - b[:, 3])},
                       "psi_T_minus_psi_N": {"est": pT - pN, "ci95": ci(b[:, 2] - b[:, 3])}}}
        if k == PRIMARY:
            r["reading"] = {"generator_flips_meaningful_beyond_noise": bool(r["model"]["psi_G_minus_psi_N"]["ci95"][0] > 0),
                            "target_flips_meaningful_beyond_noise": bool(r["model"]["psi_T_minus_psi_N"]["ci95"][0] > 0)}
        res[k] = r
    f = OUT / f"s6_meaningful_gap_{stage}.json"
    f.write_text(json.dumps(res, indent=1))
    print(f, json.dumps({k: res[k]["model"]["psi_G_minus_psi_N"] for k in MARGINS}))
    return res


def selftest():
    """At c = 0 the S6 quantities equal psi_G, psi_T and psi_N of analyse_v15; synthetic data, no study output."""
    rng = np.random.default_rng(0)
    M = 40
    lu = rng.normal(0.03, 0.15, M); la = lu + rng.normal(0.02, 0.2, M)
    vu = rng.uniform(0.002, 0.01, M); va = vu * 1.5; cu = 0.3 * np.sqrt(vu * va)
    Wm = np.array([[vu.mean(), cu.mean()], [cu.mean(), va.mean()]])
    X = np.c_[lu, la]
    pp, pm = pis(lu, vu, 0.0)
    pi = pi_of(*dl(lu, vu))
    assert abs(2 * pp * pm - 2 * pi * (1 - pi)) < 1e-12
    assert abs(psi_T_c(X, Wm, 0.0) - psi_T(X, Wm)) < 1e-6
    mu, t2 = dl(lu, vu)
    z = np.abs(mu + t2 / (t2 + vu) * (lu - mu)) / np.sqrt(vu)
    assert abs(psi_N_c(lu, vu, 0.0) - float(np.mean(2 * stats.norm.cdf(z) * stats.norm.sf(z)))) < 1e-12
    for c in (0.05, 0.1, 0.3):                                                  # larger margin, fewer meaningful flips
        assert psi_T_c(X, Wm, c) <= psi_T(X, Wm) + 1e-9 and psi_N_c(lu, vu, c) <= psi_N_c(lu, vu, 0.0) + 1e-12
    print("S6 self-test passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["X", "C"])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.stage:
        s6(a.stage)


if __name__ == "__main__":
    main()
