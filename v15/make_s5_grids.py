"""
Deviation D2 (secondary outcome S5, tuning budget): draws the 16 additional configurations per detector once, by random
search over wider ranges than the primary grids, with a fixed seed. Writes v15/s5_grids.json, which is committed and
frozen before any Stage C computation. The primary 16 configurations are unchanged; S5 selects over the union (32).

  python v15/make_s5_grids.py
"""
import json
from pathlib import Path

import numpy as np

SEED = 15_005          # plan v15, outcome S5
N_NEW = 16


def loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def draw(seed=SEED, n=N_NEW):
    rng = np.random.default_rng(seed)
    hgb = [dict(lr=round(loguniform(rng, 0.02, 0.3), 4),
                leaves=int(round(loguniform(rng, 8, 128))),
                msl=int(round(loguniform(rng, 5, 400))),
                cw=[None, "balanced"][int(rng.integers(0, 2))],
                it=[300, 600][int(rng.integers(0, 2))]) for _ in range(n)]
    mpnn = [dict(h=[32, 64, 128][int(rng.integers(0, 3))],
                 lr=float(f"{loguniform(rng, 3e-4, 1e-2):.3g}"),
                 pw=["none", "sqrt", "full"][int(rng.integers(0, 3))],
                 wd=float(f"{loguniform(rng, 1e-5, 1e-2):.3g}"),
                 logd=bool(rng.integers(0, 2))) for _ in range(n)]
    return {"seed": seed, "HGB": hgb, "MPNN": mpnn,
            "ranges": {"HGB": "lr log-U[0.02, 0.3]; leaves round(log-U[8, 128]); min leaf round(log-U[5, 400]); "
                              "class weight {none, balanced}; iterations {300, 600} (rounds chosen on validation as before)",
                       "MPNN": "width {32, 64, 128}; lr log-U[3e-4, 1e-2]; positive weight {none, sqrt, full}; "
                               "weight decay log-U[1e-5, 1e-2]; extra edge feature {no, yes}"}}


if __name__ == "__main__":
    g = draw()
    assert len(g["HGB"]) == len(g["MPNN"]) == N_NEW and draw() == g      # deterministic
    out = Path(__file__).resolve().parent / "s5_grids.json"
    out.write_text(json.dumps(g, indent=1))
    for fam in ("HGB", "MPNN"):
        for c in g[fam]:
            print(fam, c)
