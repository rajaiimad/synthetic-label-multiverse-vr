"""
r5, Phase 0 / item I (ANALYSIS_PLAN_v13_REVIEW.md): descriptive facts for the triage of the r4 review.
Run with XRP_GEN=v2 from option_b/code. Reads cached corpora and phase4.json read-only; writes
option_b/results/phase5/facts.json. No model is trained and no inferential statistic is computed here.

* Q3: unsafe frames and distinct tracks in each 15-cohort validation split (per training seed) and in the
  60-cohort test corpus, for the 13 E1 generators, the two E2 generators and the two temporal generators.
* Q5: transfer ratios above 1 with the within-generator PR-AUC, prevalence and ceiling of the cell.
* Temporal generators: specification, gain, prevalence, ceiling and plan-v7 fixed-configuration PR-AUC.
* Kendall tau granularity for n detectors.
"""
import json
import sys
from pathlib import Path

import numpy as np

J7 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(J7 / "rigorous_rerun"))
from common import CACHE, OUT, SEEDS  # noqa: E402

OB = J7 / "option_b"
OD = OB / "results" / "phase5"
OD.mkdir(parents=True, exist_ok=True)
GENS = ["d2", "cutoff", "fovfree", "and", "sat", "rand00", "rand02", "rand03", "rand04", "rand05", "rand06", "rand07", "rand11"]
DOSE = ["dose_ramp", "dose_intermittent"]


def track_keys(ctx):
    return ctx[:, 0] * 1000 + ctx[:, 1]


def counts(y, ctx, atk):
    tk = track_keys(ctx)
    return {"rows": int(len(y)), "positives": int(y.sum()), "prevalence": float(y.mean()),
            "tracks_with_positive": int(len(np.unique(tk[y == 1]))),
            "attacked_tracks": int(len(np.unique(tk[atk >= 0]))), "tracks": int(len(np.unique(tk)))}


def corpus_counts(dev_path, test_path):
    from run_final import train_val_split
    dev, test = np.load(dev_path), np.load(test_path)
    out = {"test": counts(test["y"], test["ctx"], test["atk_id"]), "validation": []}
    for s in SEEDS:
        _, va = train_val_split(dev["ctx"], dev["y"], s)
        c = counts(dev["y"][va], dev["ctx"][va], dev["atk_id"][va])
        c["seed"], c["cohorts"] = int(s), int(len(np.unique(dev["ctx"][va, 0])))
        out["validation"].append(c)
    v = out["validation"]
    out["validation_summary"] = {k: [int(min(x[k] for x in v)), float(np.mean([x[k] for x in v])), int(max(x[k] for x in v))]
                                 for k in ("positives", "tracks_with_positive", "attacked_tracks")}
    return out


def main():
    res = {"Q3": {}}
    for g in GENS + DOSE:
        res["Q3"][g] = corpus_counts(CACHE / f"s7_{g}_dev.npz", CACHE / f"s7_{g}_test.npz")
    for v in ("pair", "mix"):
        res["Q3"][f"e2_{v}"] = corpus_counts(OB / "cache" / f"e2_{v}_dev.npz", OB / "cache" / f"e2_{v}_test.npz")
    for g, d in res["Q3"].items():
        print(g, "test", d["test"]["positives"], d["test"]["tracks_with_positive"], "val [min,mean,max]",
              d["validation_summary"]["positives"], d["validation_summary"]["tracks_with_positive"], flush=True)

    R = json.loads((OB / "results" / "phase4" / "phase4.json").read_text())
    rows = []
    for g in GENS:
        s7 = json.loads((OUT / "stage7" / g / "summary.json").read_text())
        for m, r in R["transfer"]["ratio"][g].items():
            if r > 1:
                rows.append({"gen": g, "det": m, "ratio": r, "within_fixed": R["fixed"][g][m], "transfer": R["transfer"]["transfer_pr"][g][m],
                             "prevalence": s7["prevalence_test"], "ceiling": s7["ceiling"]["pr_auc"],
                             "within_over_prevalence": R["fixed"][g][m] / s7["prevalence_test"]})
    res["Q5_ratio_above_1"] = rows

    res["temporal_generators"] = {}
    for g in DOSE:
        s = json.loads((OUT / "stage7" / g / "summary.json").read_text())
        res["temporal_generators"][g] = {"spec": s["spec"], "gain": s["gain"], "prevalence_test": s["prevalence_test"],
                                         "ceiling": s["ceiling"], "fixed_pr_auc": {m: v["pr_auc"][0] for m, v in s["models"].items()}}
    res["tau_granularity"] = {n: 2.0 / (n * (n - 1) / 2) for n in (6, 7, 8, 9, 10, 11)}
    (OD / "facts.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("Q5_ratio_above_1", "temporal_generators", "tau_granularity")}, indent=1))


if __name__ == "__main__":
    main()
