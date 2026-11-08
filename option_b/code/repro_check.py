"""Phase 1 reproducibility spot-check: recompute reported PR-AUCs from saved predictions and labels
(no retraining) and compare with the stored summaries. Run with XRP_GEN=v2 from option_b/code.
-> option_b/results/phase1_repro.json"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

J7 = Path(__file__).resolve().parents[2]
RR = J7 / "rigorous_rerun"
sys.path.insert(0, str(RR))
from common import CACHE, OUT  # noqa: E402

res, worst = {}, 0.0
# 1. family generators: stored summary PR-AUC vs recomputed from *_preds.npy
for d in sorted((OUT / "stage7").iterdir()):
    s = d / "summary.json"
    if not s.exists() or d.name == "main":
        continue
    y = np.load(CACHE / f"s7_{d.name}_test.npz")["y"]
    summ = json.loads(s.read_text())["models"]
    for m, v in summ.items():
        p = np.load(d / f"{m}_preds.npy")
        r = float(np.mean([average_precision_score(y, x) for x in p]))
        res[f"stage7/{d.name}/{m}"] = (v["pr_auc"][0], r)
        worst = max(worst, abs(v["pr_auc"][0] - r))
# 2. re-tuning (stage12): bootstrap point estimates vs recomputed
rt = json.loads((OUT / "stage12" / "retune.json").read_text())
for n, v in rt.items():
    y = np.load(CACHE / f"s7_{n}_test.npz")["y"]
    for m, key in (("mpnn__MPNN", "mpnn_rt"), ("edgefeatraw__HGB", "hgb_rt")):
        p = np.load(OUT / "stage12" / "retune" / n / f"{m}_preds.npy")
        r = float(np.mean([average_precision_score(y, x) for x in p]))
        res[f"stage12/retune/{n}/{m}"] = (v["boot"]["models"][key]["pr_auc"][0], r)
        worst = max(worst, abs(v["boot"]["models"][key]["pr_auc"][0] - r))
# 3. lock-box 2
L2 = json.loads((OUT / "stage8" / "lockbox2.json").read_text())
y = np.load(CACHE / "s8_lockbox2.npz")["y"]
for m in ("mpnn__MPNN", "edgefeatraw__HGB"):
    p = np.load(OUT / "stage8" / f"{m}_lb2.npy")
    r = float(np.mean([average_precision_score(y, x) for x in p]))
    res[f"stage8/lb2/{m}"] = (L2["unsafe"]["models"][m]["pr_auc"][0], r)
    worst = max(worst, abs(L2["unsafe"]["models"][m]["pr_auc"][0] - r))
out = {"n_checked": len(res), "max_abs_diff": worst, "items": res}
(J7 / "option_b" / "results" / "phase1_repro.json").write_text(json.dumps(out, indent=1))
print(f"{len(res)} stored metrics recomputed from saved predictions; max |diff| = {worst:.2e}")
