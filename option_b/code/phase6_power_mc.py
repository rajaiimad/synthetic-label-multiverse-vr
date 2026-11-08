"""
Monte Carlo precision of the R6-B power estimates (plan v14): repeat the simulation of phase6_r6.power with 20 different
seeds (100,000 draws each) and report the mean and SD of the power at the observed MPNN - Raw-HGB pattern and of the
minimum detectable symmetric crossover. A precision check of R6-B, not a new inference.
Run with XRP_GEN=v2 from option_b/code. Writes option_b/results/phase6/power_mc.json.
"""
import json

import numpy as np

import phase4 as P
from phase5_review import load_perseed, two_level

OD = P.OB / "results" / "phase6"


def main():
    f1, r1 = load_perseed("retuned")
    d = np.array([f1[f"{g}|mpnn__MPNN"].mean() - f1[f"{g}|edgefeatraw__HGB"].mean() for g in P.GENS])
    Ds = {"cohort": np.stack([r1[f"{g}|mpnn__MPNN"].mean(0) - r1[f"{g}|edgefeatraw__HGB"].mean(0) for g in P.GENS], 1),
          "two_level": np.stack([two_level(r1[f"{g}|mpnn__MPNN"]) - two_level(r1[f"{g}|edgefeatraw__HGB"]) for g in P.GENS], 1)}
    out = {"n_seeds": 20, "draws": 100_000}
    for lvl, D in Ds.items():
        s = D.std(0, ddof=1)
        L = np.linalg.cholesky(np.corrcoef(D, rowvar=False) + 1e-10 * np.eye(len(d)))
        i, j = np.argsort(np.abs(s - np.median(s)))[:2]          # same placement rule as phase6_r6.power
        res = {}
        for a in (0.05 / 21, 0.05):
            pw, mde = [], []
            for seed in range(20):
                Z0 = np.random.default_rng(100 + seed).standard_normal((100_000, len(d))) @ L.T
                c = np.quantile(Z0.max(1), 1 - a)
                Z = Z0 + d / s
                pw.append(float(np.mean((Z.max(1) > c) & ((-Z).max(1) > c))))
                m = None
                for dl in np.round(np.arange(0.01, 0.41, 0.01), 2):
                    mu = np.zeros(len(d)); mu[i] = dl / s[i]; mu[j] = -dl / s[j]
                    Zm = Z0 + mu
                    if np.mean((Zm.max(1) > c) & ((-Zm).max(1) > c)) >= 0.8:
                        m = float(dl)
                        break
                mde.append(m)
            res[f"alpha_{a:.5f}"] = {"power_observed_mean": float(np.mean(pw)), "power_observed_sd": float(np.std(pw, ddof=1)),
                                     "power_observed_min": min(pw), "power_observed_max": max(pw),
                                     "mde80_values": sorted(set(mde))}
        out[lvl] = res
    (OD / "power_mc.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
