"""
Option B, E1 (ANALYSIS_PLAN_v11_E1.md): per-generator tuning of every tunable detector with a 16-config budget.
Run with XRP_GEN=v2 from option_b/code. Reads original corpora/results read-only; writes option_b/results/e1/.

New runs: proto__HGB, edgefeat__HGB, gat__GAT (16-config grid incl. weight decay).
Reused (same protocol, plan v10 D): mpnn__MPNN, edgefeatraw__HGB from rigorous_rerun/outputs_v2/stage12/retune.
Untunable: rule__RULE, proto__MLPpub -> plan-v7 predictions (rigorous_rerun/outputs_v2/stage7/<gen>/).
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

J7 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(J7 / "rigorous_rerun"))
from common import CACHE, OUT, SEEDS  # noqa: E402

OD = J7 / "option_b" / "results" / "e1"
GENS = ["d2", "cutoff", "fovfree", "and", "sat", "rand00", "rand02", "rand03", "rand04", "rand05", "rand06", "rand07", "rand11"]
GAT16 = [dict(h=h, lr=lr, pw=pw, wd=wd) for h in (32, 64) for lr in (1e-3, 3e-3) for pw in ("none", "sqrt") for wd in (1e-4, 1e-3)]
NEW = ["proto__HGB", "edgefeat__HGB", "gat__GAT"]


def specs():
    import baselines  # noqa: F401  (registers GAT/RULE)
    from models import GRIDS
    from run_detection import SPECS
    return {"proto__HGB": ("HGB", SPECS["proto__HGB"][1], GRIDS["HGB"]),
            "edgefeat__HGB": ("HGB", SPECS["edgefeat__HGB"][1], GRIDS["HGB"]),
            "gat__GAT": ("GAT", SPECS["gat__GAT"][1], GAT16)}


def select(fam, d, seed, grid, log):
    """models.select_and_predict, plus: failed configs are skipped and logged (plan v11 section 8)."""
    from models import FAMILIES
    best, tried = None, []
    for cfg in grid:
        dd = dict(d)
        try:
            pv, pt = FAMILIES[fam](dd, cfg, seed)
            ap = float(average_precision_score(d["yva"], pv))
            if not np.isfinite(ap):
                raise ValueError("non-finite validation AP")
        except Exception as e:  # noqa: BLE001
            log.append({"cfg": cfg, "error": repr(e), "trace": traceback.format_exc()[-400:]})
            continue
        n_iter = dd.get("_meta", {}).get("n_iter")
        tried.append({"cfg": cfg, "val_pr": ap, "n_iter": n_iter})
        if best is None or ap > best[2]:
            best = (pt, cfg, ap, n_iter)
    return best, tried


def main():
    from run_final import train_val_split
    S = specs()
    t0 = time.time()
    for name in GENS:
        dev = {k: v for k, v in np.load(CACHE / f"s7_{name}_dev.npz").items()}
        test = {k: v for k, v in np.load(CACHE / f"s7_{name}_test.npz").items()}
        y = dev["y"]
        od = OD / name
        od.mkdir(parents=True, exist_ok=True)
        for m in NEW:
            f = od / f"{m}_preds.npy"
            if f.exists():
                continue
            fam, builder, grid = S[m]
            kd, kt = builder(dev), builder(test)
            preds, sel, fails = [], [], []
            for s in SEEDS:
                tr, va = train_val_split(dev["ctx"], y, s)
                if kd[0] == "tab":
                    d = {"Xtr": kd[1][tr], "ytr": y[tr], "Xva": kd[1][va], "yva": y[va], "Xte": {"test": kt[1]}}
                else:
                    d = {"Vtr": kd[1][tr], "Etr": kd[2][tr], "ytr": y[tr], "Vva": kd[1][va], "Eva": kd[2][va],
                         "yva": y[va], "Gte": {"test": (kt[1], kt[2])}}
                best, tried = select(fam, d, s, grid, fails)
                if best is None:
                    preds.append(np.full(len(test["y"]), np.nan, np.float32))
                    sel.append({"seed": s, "selected": None, "tried": tried})
                    continue
                preds.append(best[0]["test"].astype(np.float32))
                sel.append({"seed": s, "selected": best[1], "val_pr": best[2], "n_iter": best[3], "tried": tried})
            np.save(f, np.array(preds, np.float32))
            (od / f"{m}_selection.json").write_text(json.dumps({"selection": sel, "failures": fails}, default=str, indent=1))
            ok = [p for p in preds if np.isfinite(p).all()]
            pr = np.mean([average_precision_score(test["y"], p) for p in ok]) if ok else float("nan")
            print(f"[{name}] {m:16s} PR {pr:.3f} failures {len(fails)} ({time.time() - t0:.0f}s)", flush=True)
    print("E1 DONE", flush=True)


if __name__ == "__main__":
    main()
