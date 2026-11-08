"""
E4 (deviation D6, DEVIATIONS_v13.md): matched-input transfer control. Run with XRP_GEN=v2 from option_b/code.
  python e4_matched_transfer.py build   v2-generator labels on the E1 development cohorts (seeds 60000-60059);
                                         asserts that every input array equals s7_d2_dev.npz
  python e4_matched_transfer.py train   six fixed-configuration detectors trained on it, scored on 13 E1 + 2 E2 + 2 E3 tests
  python e4_matched_transfer.py analyse matched ratios, tau with intervals, decomposition -> results/e4/e4.json
Reads existing corpora and results read-only; writes option_b/cache/e4_v2_dev.npz and option_b/results/e4/.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

J7 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(J7 / "rigorous_rerun"))
from common import CACHE  # noqa: E402

OB = J7 / "option_b"
OBC = OB / "cache"
OD = OB / "results" / "e4"
OD.mkdir(parents=True, exist_ok=True)
DEV = OBC / "e4_v2_dev.npz"
GENS = ["d2", "cutoff", "fovfree", "and", "sat", "rand00", "rand02", "rand03", "rand04", "rand05", "rand06", "rand07", "rand11"]
E2 = ["pair", "mix"]
DOSE = ["dose_ramp", "dose_intermittent"]
INPUTS = ("ctx", "V_raw", "E_chan", "E_geo", "E_w", "V_bb", "E_bb", "own_prov", "atk_id")


def test_corpora():
    t = {g: CACHE / f"s7_{g}_test.npz" for g in GENS + DOSE}
    t.update({f"e2_{v}": OBC / f"e2_{v}_test.npz" for v in E2})
    return t


def build():
    from stage3 import S3, V2_GAIN, generate
    t0 = time.time()
    t = generate(S3["dev"][0], S3["dev"][1] - S3["dev"][0], {}, V2_GAIN)
    ref = np.load(CACHE / "s7_d2_dev.npz")
    # the frozen severity backbone is re-evaluated, so its float32 outputs can differ in the last digits; every other
    # input must be bit-identical. The E1 arrays are then used, so that the inputs are exactly those of E1.
    bb_err = {}
    for k in INPUTS:
        if k in ("V_bb", "E_bb"):
            bb_err[k] = float(np.abs(t[k].astype(np.float64) - ref[k]).max())
            assert bb_err[k] < 1e-4, (k, bb_err[k])
            t[k] = ref[k]
        else:
            assert np.array_equal(t[k], ref[k]), f"input {k} differs from the E1 development corpus"
    g4 = np.load(CACHE / "s3_g4_dev.npz")
    for k in INPUTS:
        assert k in ("V_bb", "E_bb") or np.array_equal(t[k], g4[k]), f"input {k} differs from s3_g4_dev"
    jac = float((t["y"] & g4["y"]).sum() / max(1, (t["y"] | g4["y"]).sum()))
    np.savez_compressed(DEV, **t)
    info = {"gain": float(V2_GAIN), "rows": int(len(t["y"])), "positives": int(t["y"].sum()), "prevalence": float(t["y"].mean()),
            "inputs_equal_e1_dev": True, "severity_backbone_max_abs_diff_before_alignment": bb_err,
            "jaccard_with_s3_g4_dev_labels": jac, "seconds": time.time() - t0}
    (OD / "build.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1), flush=True)


def train():
    from stage7 import model_specs, train as fit
    dev = {k: x for k, x in np.load(DEV).items()}
    tests = {g: {k: x for k, x in np.load(f).items()} for g, f in test_corpora().items()}
    t0 = time.time()
    for m, (fam, builder, src) in model_specs().items():
        f = OD / f"{m}.npz"
        if not f.exists():
            rs = fit(fam, builder, src, dev, tests)
            np.savez_compressed(f, **{g: np.array([r["pred"][g] for r in rs], np.float32) for g in tests})
        z = np.load(f)
        print(m, {g: round(float(np.mean([average_precision_score(tests[g]["y"], p) for p in z[g]])), 3) for g in tests},
              f"({time.time() - t0:.0f}s)", flush=True)
    print("E4 TRAIN DONE", flush=True)


def analyse():
    import phase4 as P
    from ob_stats import Sorted, cohort_ids, tau
    R = json.loads((P.OD / "phase4.json").read_text())
    REV = json.loads((OB / "results" / "phase5" / "review.json").read_text())
    ZF = np.load(P.OD / "boot_fixed_e2.npz")
    PT = np.load(OB / "results" / "phase5" / "perseed_transfer.npz")
    FIX6 = P.FIXED6
    B = len(P.W)
    out = {"build": json.loads((OD / "build.json").read_text()), "rows": [], "tau": {}}
    z = {m: np.load(OD / f"{m}.npz") for m in FIX6}
    cell = {(r["gen"], r["det"]): r for r in REV["E_transfer"]["rows"]}
    for g in GENS:
        y = np.load(CACHE / f"s7_{g}_test.npz")["y"]
        gi = cohort_ids(len(y))
        pts, reps = {}, {}
        for m in FIX6:
            S = [Sorted(p, y, gi) for p in z[m][g]]
            pts[m] = float(np.mean([s.ap() for s in S]))
            reps[m] = np.array([np.mean([s.ap(w) for s in S]) for w in P.W])
        ceil = json.loads((P.OUT / "stage7" / g / "summary.json").read_text())["ceiling"]["pr_auc"]
        for m in FIX6:
            c = cell[(g, m)]
            out["rows"].append({"gen": g, "det": m, "matched": pts[m], "original_transfer": c["transfer"],
                                "within_fixed": c["within_fixed"], "within_retuned": c["within_retuned"],
                                "ratio_matched_fixed": pts[m] / c["within_fixed"], "ratio_matched_retuned": pts[m] / c["within_retuned"],
                                "ratio_original_fixed": c["ratio_fixed"], "training_data_effect": c["transfer"] - pts[m],
                                "label_rule_effect": pts[m] - c["within_fixed"], "weak": c["weak"], "ceiling": ceil})
        fx = {m: R["fixed"][g][m] for m in FIX6}
        bt = [tau([ZF[f"fixed|{g}|{m}"][b] for m in FIX6], [reps[m][b] for m in FIX6]) for b in range(B)]
        ra = float(np.median([tau([pts[m] for m in FIX6], [reps[m][b] for m in FIX6]) for b in range(B)]))
        out["tau"][g] = {"matched": tau([fx[m] for m in FIX6], [pts[m] for m in FIX6]), "ci": np.percentile(bt, [2.5, 97.5]).tolist(),
                         "original": REV["D_tau"]["transfer_tau"][g]["tau"], "resampling_agreement_matched": ra,
                         "best_matched": max(FIX6, key=lambda m: pts[m]),
                         "best_original": max(FIX6, key=lambda m: cell[(g, m)]["transfer"])}
        # double check: original transfer points from the saved per-seed replicates
        assert all(abs(PT[f"full|{g}|{m}"].mean() - cell[(g, m)]["transfer"]) < 1e-12 for m in FIX6)
    # E2 and E3 (point estimates only)
    out["other"] = {}
    for g, f in test_corpora().items():
        if g in GENS:
            continue
        y = np.load(f)["y"]
        out["other"][g] = {m: float(np.mean([average_precision_score(y, p) for p in z[m][g]])) for m in FIX6}
    rows = out["rows"]
    t_m = [v["matched"] for v in out["tau"].values()]
    t_o = [v["original"] for v in out["tau"].values()]
    out["summary"] = {
        "n_cells": len(rows),
        "n_ratio_matched_fixed_gt1": sum(r["ratio_matched_fixed"] > 1 for r in rows),
        "n_ratio_matched_retuned_gt1": sum(r["ratio_matched_retuned"] > 1 for r in rows),
        "n_ratio_original_fixed_gt1": sum(r["ratio_original_fixed"] > 1 for r in rows),
        "median_tau_matched": float(np.median(t_m)), "min_tau_matched": float(min(t_m)), "max_tau_matched": float(max(t_m)),
        "median_tau_original": float(np.median(t_o)),
        "n_tau_ci_incl0_matched": sum(v["ci"][0] <= 0 <= v["ci"][1] for v in out["tau"].values()),
        "median_resampling_agreement_matched": float(np.median([v["resampling_agreement_matched"] for v in out["tau"].values()])),
        "mpnn_matched_le_070_retuned": [r["gen"] for r in rows if r["det"] == "mpnn__MPNN" and r["ratio_matched_retuned"] <= 0.70],
        "median_ratio_matched_retuned": {m: float(np.median([r["ratio_matched_retuned"] for r in rows if r["det"] == m])) for m in FIX6},
        "rule_best_matched": sum(v["best_matched"] == "rule__RULE" for v in out["tau"].values()),
        "mean_training_data_effect": {m: float(np.mean([r["training_data_effect"] for r in rows if r["det"] == m])) for m in FIX6},
        "mean_label_rule_effect": {m: float(np.mean([r["label_rule_effect"] for r in rows if r["det"] == m])) for m in FIX6},
    }
    (OD / "e4.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))
    print({g: (round(v["matched"], 2), round(v["original"], 2), [round(x, 2) for x in v["ci"]]) for g, v in out["tau"].items()})


if __name__ == "__main__":
    {"build": build, "train": train, "analyse": analyse}[sys.argv[1]]()
