"""Exploratory: H1 and H2 power at M = 60..100 from Stage X estimates (8 prior-drawn generators).
Same model as redesign_v15/power_sim_v15.py, but H1 uses the plan's 95% interval (8.2) and Stage X noise pairs
(n_base 60, S_base 5), as analyse_v15.simulate_power. Read-only on the repo; writes nothing there."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import norm, t as tdist
R0 = Path(sys.argv[1])
ax = json.loads((R0 / "results_v15/analysis_X.json").read_text())["stage_x"]["reestimation"]
prior = [g for g in json.loads((R0 / "results_v15/analysis_X.json").read_text())["generators"] if g.startswith("rand")]
pairs = []
for g in prior:
    s = json.loads((R0 / f"results_v15/X/{g}/summary_v15.json").read_text())["frame"]["U"]
    pairs.append([s["sd_cohort"], s["sd_seed"] or 0.0])
pairs = np.array(pairs)

def dl(y, v):
    w = 1 / v; sw = w.sum(-1); mfe = (w * y).sum(-1) / sw
    Q = (w * (y - mfe[..., None]) ** 2).sum(-1); M = y.shape[-1]
    c = sw - (w ** 2).sum(-1) / sw; t2 = np.maximum(0, (Q - (M - 1)) / c)
    ws = 1 / (v + t2[..., None]); return (ws * y).sum(-1) / ws.sum(-1), t2

def pi_of(mu, t2):
    t = np.sqrt(t2)
    return np.where(t > 0, norm.cdf(mu / np.where(t > 0, t, 1)), (mu > 0).astype(float))

def sim(M, mud, R=1000, B=500, n=240, S=4, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pairs), (R, M)); sdc, sds = pairs[idx, 0], pairs[idx, 1]
    se2 = sdc ** 2 * 60 / n + sds ** 2 / 5 * 5 / S
    z1, z2, e1, e2 = rng.standard_normal((4, R, M))
    DU = ax["muU"] + ax["tauU"] * z1
    dt = mud + ax["taud"] * (ax["rhoG"] * z1 + np.sqrt(1 - ax["rhoG"] ** 2) * z2)
    eU = np.sqrt(se2) * e1
    eA = np.sqrt(se2) * ax["fA"] * (ax["rhoE"] * e1 + np.sqrt(1 - ax["rhoE"] ** 2) * e2)
    DUh = DU + eU; d = dt + eA - eU
    mub, t2b = [], []
    for r in range(R):
        bi = rng.integers(0, M, (B, M)); m, t2 = dl(DUh[r][bi], se2[r][bi]); mub.append(pi_of(m, t2))
    pib = np.array(mub); lo, hi = np.percentile(pib, 2.5, 1), np.percentile(pib, 97.5, 1)
    h1 = ((lo > .10) & (hi < .90)).mean()
    tt = d.mean(1) / (d.std(1, ddof=1) / np.sqrt(M)); h2 = (2 * tdist.sf(abs(tt), M - 1) < .025).mean()
    hw = np.mean(tdist.ppf(.9875, M - 1) * d.std(1, ddof=1) / np.sqrt(M))
    return float(h1), float(h2), float(hw)

print("pi_U point (Stage X) =", round(float(norm.cdf(ax["muU"] / ax["tauU"])), 3))
print("pairs", pairs.round(3).tolist())
for M in (60, 80, 100):
    h1, h2, hw = sim(M, ax["mud_plan_shrunk"])
    print(f"M={M}: H1 power {h1:.2f}  H2 power {h2:.2f}  mean 97.5% half-width of mu_delta {hw:.3f} (log scale)")
