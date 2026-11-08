"""
Plan v15, planning input only: variance components of the MPNN - Raw-HGB PR-AUC difference from EXISTING (r6) data,
used as proxies in the power simulation. Read-only on all existing files; writes redesign_v15/proxy_components.json.

Proxies (above-band attack; the confirmatory study uses the in-band attack, so these are re-estimated in Stage X):
  target U (all unsafe frames): E1 re-tuned predictions, 13 generators (8 drawn from the prior).
  target A (attacker-caused, 'attacker absent' counterfactual, sum-aggregation generators only): the same predictions
    scored against the attacker-caused label (models trained on U, so this is a mismatch proxy).
Per generator: point difference (mean over 5 seeds), seed SD of per-seed differences, cohort-bootstrap SD of the
5-seed-mean difference (B = 1000, shared cohort weights).
Run with XRP_GEN=v2.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

J7 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(J7 / "option_b" / "code"))
os.environ.setdefault("XRP_GEN", "v2")
import phase4 as P  # noqa: E402
from ob_stats import Sorted, cohort_ids  # noqa: E402

RANDOM = ["rand00", "rand02", "rand03", "rand04", "rand05", "rand06", "rand07", "rand11"]
SUMAGG = set(json.loads((P.OUT / "stage7" / "attacker_target.json").read_text()))


def components(y, Pm, Pr):
    g = cohort_ids(len(y))
    per = []
    reps = []
    for a, b in zip(Pm, Pr):
        sa, sb = Sorted(a, y, g), Sorted(b, y, g)
        per.append(sa.ap() - sb.ap())
        reps.append(np.array([sa.ap(w) - sb.ap(w) for w in P.W]))
    per, reps = np.array(per), np.array(reps)            # (5,), (5, B)
    return {"d": float(per.mean()), "sd_seed": float(per.std(ddof=1)), "sd_cohort": float(reps.mean(0).std(ddof=1)),
            "per_seed": per.tolist()}


out = {"U": {}, "A": {}, "random": RANDOM, "note": "proxy only; above-band attack; A is a train-U/score-A mismatch proxy"}
for g in P.GENS:
    t = np.load(P.CACHE / f"s7_{g}_test.npz")
    y = t["y"]
    Pm, Pr = np.load(P.retuned_path(g, "mpnn__MPNN")), np.load(P.retuned_path(g, "edgefeatraw__HGB"))
    out["U"][g] = components(y, Pm, Pr)
    if g in SUMAGG:
        s, a = t["s_noisy"].astype(float), t["atk_term"].astype(float)
        ya = ((s > 0.30) & (s - a <= 0.30)).astype(int)
        out["A"][g] = components(ya, Pm, Pr)
        out["A"][g]["prev"] = float(ya.mean())
    out["U"][g]["prev"] = float(y.mean())
    print(g, {k: round(v["d"], 3) for k, v in ((t_, out[t_][g]) for t_ in ("U", "A") if g in out[t_])},
          "sdU c/s %.3f/%.3f" % (out["U"][g]["sd_cohort"], out["U"][g]["sd_seed"]), flush=True)
(Path(__file__).with_name("proxy_components.json")).write_text(json.dumps(out, indent=1))
