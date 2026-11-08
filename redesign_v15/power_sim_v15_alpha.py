"""Plan v15: H1 at the final level (95% bootstrap interval inside (0.10, 0.90), i.e. each one-sided test at 0.025),
for the default design (M 80, n 240, S 4) and M 60/100. Same model as power_sim_v15.py. Writes power_sim_v15_alpha.json."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import norm
HERE = Path(__file__).resolve().parent
src = (HERE / "power_sim_v15.py").read_text()
ns = {"__file__": str(HERE / "power_sim_v15.py")}
exec(compile(src.split("# proxy-based scenarios")[0], "power_sim_v15", "exec"), ns)
PAIRS, dl, pi_of = ns["PAIRS"], ns["dl"], ns["pi_of"]


def h1(sc, M, n, S, R=800, B=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(PAIRS), (R, M))
    se2 = PAIRS[idx, 0] ** 2 * 60 / n + PAIRS[idx, 1] ** 2 / S
    D = sc["muU"] + sc["tauU"] * rng.standard_normal((R, M)) + np.sqrt(se2) * rng.standard_normal((R, M))
    bi = rng.integers(0, M, (R, B, M))
    mub, t2b = dl(np.take_along_axis(np.repeat(D[:, None], B, 1), bi, 2), np.take_along_axis(np.repeat(se2[:, None], B, 1), bi, 2))
    pib = pi_of(mub, t2b)
    lo, hi = np.percentile(pib, 2.5, 1), np.percentile(pib, 97.5, 1)
    tp = norm.cdf(sc["muU"] / sc["tauU"])
    return {"power95": float(((lo > 0.1) & (hi < 0.9)).mean()), "coverage95": float(((lo <= tp) & (tp <= hi)).mean()), "pi_true": float(tp)}


SC = {"base": dict(muU=0.03, tauU=0.065), "weak": dict(muU=0.03, tauU=0.035),
      "boundary_pi0.90": dict(muU=norm.ppf(0.90) * 0.065, tauU=0.065), "boundary_pi0.10": dict(muU=-norm.ppf(0.90) * 0.065, tauU=0.065),
      "mid_pi0.75": dict(muU=norm.ppf(0.75) * 0.065, tauU=0.065)}
res = []
for name, sc in SC.items():
    for M in (60, 80, 100):
        r = h1(sc, M, 240, 4, seed=M + len(name))
        res.append({"scenario": name, "M": M, "n": 240, "S": 4, **r})
        print(f"{name:16s} M {M:3d}  pi {r['pi_true']:.2f}  H1 power/size {r['power95']:.3f}  coverage95 {r['coverage95']:.2f}", flush=True)
(HERE / "power_sim_v15_alpha.json").write_text(json.dumps(res, indent=1))
