"""
E5 (ANALYSIS_PLAN_v14_R6.md, R6-E): equal tuning of GAT-sum on the 13 E1 generators with the GAT's 16-configuration grid,
seeds, split and selection rule (e1_retune). Run with XRP_GEN=v2 from option_b/code.
Writes option_b/results/e5/<gen>/retuned_gatsum__GATSUM_preds.npy and _selection.json. Resumes (skips finished generators).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

J7 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(J7 / "rigorous_rerun"))
from common import CACHE, SEEDS  # noqa: E402

import e1_retune  # noqa: E402
import gat_sum  # noqa: E402,F401  (registers GATSUM)

OD = J7 / "option_b" / "results" / "e5"
# optional E5_GENS="g1,g2" runs a subset (parallel workers on disjoint subsets); results do not depend on the split
import os  # noqa: E402
GENS = [g for g in e1_retune.GENS if g in os.environ["E5_GENS"].split(",")] if os.environ.get("E5_GENS") else e1_retune.GENS


def main():
    from run_detection import SPECS
    from run_final import train_val_split
    builder = SPECS["gat__GAT"][1]                      # identical inputs to the GAT (and the MPNN)
    grid = e1_retune.GAT16
    assert len(grid) == 16
    t0 = time.time()
    for name in GENS:
        od = OD / name
        od.mkdir(parents=True, exist_ok=True)
        f = od / "retuned_gatsum__GATSUM_preds.npy"
        if f.exists():
            continue
        dev = {k: v for k, v in np.load(CACHE / f"s7_{name}_dev.npz").items()}
        test = {k: v for k, v in np.load(CACHE / f"s7_{name}_test.npz").items()}
        y = dev["y"]
        kd, kt = builder(dev), builder(test)
        preds, sel, fails = [], [], []
        for s in SEEDS:
            tr, va = train_val_split(dev["ctx"], y, s)
            d = {"Vtr": kd[1][tr], "Etr": kd[2][tr], "ytr": y[tr], "Vva": kd[1][va], "Eva": kd[2][va],
                 "yva": y[va], "Gte": {"test": (kt[1], kt[2])}}
            best, tried = e1_retune.select("GATSUM", d, s, grid, fails)
            preds.append(best[0]["test"].astype(np.float32) if best else np.full(len(test["y"]), np.nan, np.float32))
            sel.append({"seed": s, "selected": best[1] if best else None, "val_pr": best[2] if best else None, "tried": tried})
        np.save(f, np.array(preds, np.float32))
        (od / "retuned_gatsum__GATSUM_selection.json").write_text(json.dumps({"selection": sel, "failures": fails}, default=str, indent=1))
        ok = [p for p in preds if np.isfinite(p).all()]
        print(f"[{name}] GAT-sum PR {np.mean([average_precision_score(test['y'], p) for p in ok]) if ok else float('nan'):.3f} "
              f"failures {len(fails)} ({time.time() - t0:.0f}s)", flush=True)
    print("E5 DONE", flush=True)


if __name__ == "__main__":
    main()
