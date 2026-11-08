"""
Option B, E2 (ANALYSIS_PLAN_v12_E2.md): group-level generators that break per-peer additivity.
Run with XRP_GEN=v2 from option_b/code.

  python e2_group.py build      labels for E2-pair / E2-mix on the v7 family seeds (inputs reused read-only)
  python e2_group.py within     7 detectors: v2 fixed configs and per-generator re-tuning (plan v11 protocol)
  python e2_group.py transfer   detectors trained on the v2 main corpus, scored on E2 test corpora
Corpora: option_b/cache/; results: option_b/results/e2/.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score

J7 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(J7 / "rigorous_rerun"))
from common import CACHE, OUT, PEER_HI, SEEDS  # noqa: E402

OBC = J7 / "option_b" / "cache"
OD = J7 / "option_b" / "results" / "e2"
for p in (OBC, OD):
    p.mkdir(parents=True, exist_ok=True)
TARGET, SIG, THR = 0.0138, 0.05, 0.30
V2_GAIN = 4.0056
VARIANTS = ("pair", "mix")


def exposures(t):
    """Per-edge v2 exposures c_j at gain 1, recomputed from stored inputs. Shape (rows, K)."""
    from stage3 import edge_contrib
    x = t["E_chan"] / PEER_HI
    return np.stack([edge_contrib(x[:, k], t["E_geo"][:, k, 0].astype(np.float64), t["E_geo"][:, k, 1].astype(np.float64), {})
                     for k in range(x.shape[1])], 1)


def group_terms(C):
    P, Q = C.sum(1), (C ** 2).sum(1)
    return P, (P ** 2 - Q) / 2.0


def score(variant, own, P, G, g):
    return own + g * G if variant == "pair" else own + (V2_GAIN / 2) * P + g * G


def calibrate(variant):
    from stage3 import S3, generate
    t = generate(S3["calib"][0], S3["calib"][1] - S3["calib"][0], {}, gain=1.0, with_backbone=False)
    P, G = group_terms(exposures(t))
    own = t["own_prov"].astype(np.float64)
    f = lambda g: (1 - norm.cdf((THR - score(variant, own, P, G, g)) / SIG)).mean() - TARGET
    return float(brentq(f, 1e-3, 1e7))


def build():
    meta = {}
    gains = {v: calibrate(v) for v in VARIANTS}
    # correctness check: own + 4.0 * sum_j c_j reproduces the stored g=4 oracle score (same seeds 61000-)
    ref = np.load(CACHE / "s3_g4_test.npz")
    src_test = np.load(CACHE / "s7_floor05_test.npz")
    assert np.array_equal(ref["E_chan"], src_test["E_chan"]) and np.array_equal(ref["E_geo"], src_test["E_geo"])
    err = float(np.abs(ref["own_prov"] + 4.0 * exposures(ref).sum(1) - ref["oracle_score"]).max())
    assert err < 1e-4, err
    meta["recompute_max_abs_err"] = err
    for which in ("dev", "test"):
        src = {k: v for k, v in np.load(CACHE / f"s7_floor05_{which}.npz").items()}
        eps = (src["s_noisy"] - src["oracle_score"]) / SIG          # unit noise draw, reused
        P, G = group_terms(exposures(src))
        own = src["own_prov"].astype(np.float64)
        for v in VARIANTS:
            s = score(v, own, P, G, gains[v])
            t = {k: src[k] for k in ("ctx", "V_raw", "V_bb", "E_src", "E_chan", "E_bb", "E_geo", "E_w", "atk_id", "atk_ch", "N", "T")}
            t["oracle_score"] = s.astype(np.float32)
            t["s_noisy"] = (s + SIG * eps).astype(np.float32)
            t["y"] = (t["s_noisy"] > THR).astype(np.int64)
            t["y_att"] = np.zeros_like(src["y_att"])                 # attribution not defined for E2
            np.savez_compressed(OBC / f"e2_{v}_{which}.npz", **t)
            ben = t["atk_id"] < 0
            meta[f"{v}_{which}"] = {"gain": gains[v], "prevalence": float(t["y"].mean()),
                                    "benign_share": float(t["y"][ben].sum() / max(1, t["y"].sum()))}
            if which == "test":                      # v2-form labels at the same seeds and noise, for agreement
                y_v2 = (own + V2_GAIN * P + SIG * eps > THR).astype(int)
                meta[f"{v}_test"]["jaccard_with_v2_form_labels"] = float((t["y"] & y_v2).sum() / max(1, (t["y"] | y_v2).sum()))
            if which == "test":
                pc = norm.cdf((s - THR) / SIG)
                meta[f"{v}_test"]["ceiling"] = {"pr_auc": float(average_precision_score(t["y"], pc)),
                                                "roc_auc": float(roc_auc_score(t["y"], pc))}
    (OD / "build.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def within():
    import e1_retune
    from run_final import train_val_split
    from stage7 import model_specs, train
    S = e1_retune.specs()
    from models import GRIDS
    S["mpnn__MPNN"] = ("MPNN", model_specs()["mpnn__MPNN"][1], GRIDS["MPNN"])
    S["edgefeatraw__HGB"] = ("HGB", model_specs()["edgefeatraw__HGB"][1], GRIDS["HGB"])
    t0 = time.time()
    for v in VARIANTS:
        dev = {k: x for k, x in np.load(OBC / f"e2_{v}_dev.npz").items()}
        test = {k: x for k, x in np.load(OBC / f"e2_{v}_test.npz").items()}
        od = OD / v
        od.mkdir(exist_ok=True)
        for m, (fam, builder, src) in model_specs().items():                  # fixed v2 configurations
            f = od / f"fixed_{m}_preds.npy"
            if not f.exists():
                np.save(f, np.array([r["pred"]["test"] for r in train(fam, builder, src, dev, {"test": test})], np.float32))
            print(f"[{v}] fixed {m:18s} PR {np.mean([average_precision_score(test['y'], p) for p in np.load(f)]):.3f} ({time.time() - t0:.0f}s)", flush=True)
        y = dev["y"]
        for m, (fam, builder, grid) in S.items():                            # re-tuned (plan v11 protocol)
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
                sel.append({"seed": s, "selected": best[1] if best else None, "val_pr": best[2] if best else None, "tried": tried})
            np.save(f, np.array(preds, np.float32))
            (od / f"retuned_{m}_selection.json").write_text(json.dumps({"selection": sel, "failures": fails}, default=str, indent=1))
            print(f"[{v}] retuned {m:16s} PR {np.nanmean([average_precision_score(test['y'], p) for p in preds if np.isfinite(p).all()]):.3f} "
                  f"failures {len(fails)} ({time.time() - t0:.0f}s)", flush=True)
    print("E2 WITHIN DONE", flush=True)


def transfer():
    from stage7 import model_specs, train
    dev = {k: x for k, x in np.load(CACHE / "s4_main.npz").items()}
    tests = {v: {k: x for k, x in np.load(OBC / f"e2_{v}_test.npz").items()} for v in VARIANTS}
    od = OD / "transfer"
    od.mkdir(exist_ok=True)
    for m, (fam, builder, src) in model_specs().items():
        f = od / f"{m}.npz"
        if not f.exists():
            rs = train(fam, builder, src, dev, tests)
            np.savez_compressed(f, **{v: np.array([r["pred"][v] for r in rs], np.float32) for v in VARIANTS})
        z = np.load(f)
        print("transfer", m, {v: round(float(np.mean([average_precision_score(tests[v]["y"], p) for p in z[v]])), 3) for v in VARIANTS}, flush=True)
    print("E2 TRANSFER DONE", flush=True)


if __name__ == "__main__":
    {"build": build, "within": within, "transfer": transfer}[sys.argv[1]]()
