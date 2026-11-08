"""
Option B, E3 (ANALYSIS_PLAN_v13_REVIEW.md section 4; deviation D4): temporal-accumulation generators.
Run with XRP_GEN=v2 from option_b/code.
  python e3_temporal.py within     re-tune the five tunable detectors on dose_ramp / dose_intermittent (E1 protocol)
  python e3_temporal.py transfer   six fixed-configuration detectors trained on the v2 main corpus, scored on both
Inputs (read-only): rigorous_rerun/outputs_v2/cache/s7_dose_*_{dev,test}.npz, plan-v7 predictions in
rigorous_rerun/outputs_v2/stage7/dose_*/. Results: option_b/results/e3/. No existing file is changed.
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

OD = J7 / "option_b" / "results" / "e3"
OD.mkdir(parents=True, exist_ok=True)
GENS = ("dose_ramp", "dose_intermittent")


def within():
    import e1_retune
    from models import GRIDS
    from run_final import train_val_split
    from stage7 import model_specs
    S = e1_retune.specs()                                   # proto__HGB, edgefeat__HGB, gat__GAT (GAT16)
    S["mpnn__MPNN"] = ("MPNN", model_specs()["mpnn__MPNN"][1], GRIDS["MPNN"])
    S["edgefeatraw__HGB"] = ("HGB", model_specs()["edgefeatraw__HGB"][1], GRIDS["HGB"])
    assert all(len(g) == 16 for _, _, g in S.values()), {m: len(g) for m, (_, _, g) in S.items()}
    t0 = time.time()
    for name in GENS:
        dev = {k: x for k, x in np.load(CACHE / f"s7_{name}_dev.npz").items()}
        test = {k: x for k, x in np.load(CACHE / f"s7_{name}_test.npz").items()}
        od = OD / name
        od.mkdir(exist_ok=True)
        y = dev["y"]
        for m, (fam, builder, grid) in S.items():
            f = od / f"retuned_{m}_preds.npy"
            if f.exists():
                continue
            kd, kt = builder(dev), builder(test)
            preds, sel, fails = [], [], []
            for s in SEEDS:
                tr, va = train_val_split(dev["ctx"], y, s)
                if kd[0] == "tab":
                    d = {"Xtr": kd[1][tr], "ytr": y[tr], "Xva": kd[1][va], "yva": y[va], "Xte": {"test": kt[1]}}
                else:
                    d = {"Vtr": kd[1][tr], "Etr": kd[2][tr], "ytr": y[tr], "Vva": kd[1][va], "Eva": kd[2][va],
                         "yva": y[va], "Gte": {"test": (kt[1], kt[2])}}
                best, tried = e1_retune.select(fam, d, s, grid, fails)
                preds.append(best[0]["test"].astype(np.float32) if best else np.full(len(test["y"]), np.nan, np.float32))
                sel.append({"seed": s, "selected": best[1] if best else None, "val_pr": best[2] if best else None,
                            "n_iter": best[3] if best else None, "tried": tried})
            np.save(f, np.array(preds, np.float32))
            (od / f"retuned_{m}_selection.json").write_text(json.dumps({"selection": sel, "failures": fails}, default=str, indent=1))
            ok = [p for p in preds if np.isfinite(p).all()]
            print(f"[{name}] retuned {m:16s} PR {np.mean([average_precision_score(test['y'], p) for p in ok]) if ok else float('nan'):.3f} "
                  f"failures {len(fails)} ({time.time() - t0:.0f}s)", flush=True)
    print("E3 WITHIN DONE", flush=True)


def transfer():
    from stage7 import model_specs, train
    dev = {k: x for k, x in np.load(CACHE / "s4_main.npz").items()}
    tests = {g: {k: x for k, x in np.load(CACHE / f"s7_{g}_test.npz").items()} for g in GENS}
    od = OD / "transfer"
    od.mkdir(exist_ok=True)
    for m, (fam, builder, src) in model_specs().items():
        f = od / f"{m}.npz"
        if not f.exists():
            rs = train(fam, builder, src, dev, tests)
            np.savez_compressed(f, **{g: np.array([r["pred"][g] for r in rs], np.float32) for g in GENS})
        z = np.load(f)
        print("transfer", m, {g: round(float(np.mean([average_precision_score(tests[g]["y"], p) for p in z[g]])), 3) for g in GENS}, flush=True)
    print("E3 TRANSFER DONE", flush=True)


if __name__ == "__main__":
    {"within": within, "transfer": transfer}[sys.argv[1]]()
