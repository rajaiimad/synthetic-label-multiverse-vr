"""
rigorous_rerun.stage12
======================
PREREGISTRATION_v10 (round-6 external review). Run with XRP_GEN=v2.

  python stage12.py defence    A: clamping / personal-boundary defence on the 70000-seed attack corpora
  python stage12.py prox       B: proximity share, benign share by gain, 1/max(d,0.5) kernel variant
  python stage12.py symptom    C: victim symptom channel (k = 1, 3), RQ1 rerun
  python stage12.py retune     D: per-generator re-tuning of MPNN and raw-channel HGB
  python stage12.py events     E: exploratory event-level alarms on lock-box 1
Outputs: outputs_v2/stage12/<step>.json (+ prediction arrays). Each step resumes.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from common import CACHE, OUT, PEER_HI, SEEDS, jdump, thresholds_from_val
from stage3 import V2_GAIN, calibrate_gain, edge_contrib
from stage7 import RANDOM, STRUCT, T, ceiling, corpus7, model_specs, train

OD = OUT / "stage12"
OD.mkdir(parents=True, exist_ok=True)
load = lambda f: {k: v for k, v in np.load(CACHE / f).items()}


def boot(Y, P, pairs, rows_per_cohort=8 * 4 * T):
    """Cohort bootstrap (B as in stage3_ci) of seed-averaged PR-AUC and paired differences."""
    from stage3_ci import B, prep, wmetrics
    g = np.arange(len(Y)) // rows_per_cohort
    nc = g.max() + 1
    srt = {m: [prep(p, Y, g) for p in ps] for m, ps in P.items()}
    point = {m: np.mean([wmetrics(ys, np.ones_like(ys)) for ys, _ in L], 0) for m, L in srt.items()}
    rng = np.random.default_rng(0)
    bs = {m: [] for m in srt}
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, nc, nc), minlength=nc).astype(float)
        for m, L in srt.items():
            bs[m].append(np.mean([wmetrics(ys, cnt[gs]) for ys, gs in L], 0))
    bs = {m: np.array(b) for m, b in bs.items()}
    r = {"models": {m: {"pr_auc": [float(point[m][1]), *np.percentile(bs[m][:, 1], [2.5, 97.5]).tolist()]} for m in P},
         "diffs": {}}
    for a, b in pairs:
        d = bs[a][:, 1] - bs[b][:, 1]
        r["diffs"][f"{a} - {b}"] = {"diff": float(point[a][1] - point[b][1]), "ci": np.percentile(d, [2.5, 97.5]).tolist(),
                                    "p_le0": float(np.mean(d <= 0))}
    return r


def geo_score(t):
    return ((0.5 + 0.5 * t["E_geo"][..., 1]) / (t["E_geo"][..., 0] + 0.5)).sum(1)


# ------------------------------------------------------------------ A. defence
def defence():
    import stage5
    attacks = {"standard": {}, "near": {"near": True}, "near_benign": {"near": True, "near_clamp": False}}
    defences = {"none": {}, "clamp": {"clamp": True}, "dmin05": {"dmin": 0.5}, "dmin10": {"dmin": 1.0},
                "clamp_dmin10": {"clamp": True, "dmin": 1.0}}
    tests = {}
    for a, av in attacks.items():
        for d, dv in defences.items():
            tests[f"{a}|{d}"] = stage5.base(f"s12_{a}_{d}", {**av, **dv})
    ref = stage5.labels(tests["standard|none"])[0].sum()
    res = {"reference_harm": int(ref), "corpora": {}}
    dev = load("s4_main.npz")
    specs = model_specs()
    runs = {}
    for m in ("rule__RULE", "edgefeat__HGB", "edgefeatraw__HGB", "mpnn__MPNN"):
        f = OD / f"defence_{m}.npz"
        if not f.exists():
            fam, builder, src = specs[m]
            rs = train(fam, builder, src, dev, tests)
            np.savez_compressed(f, **{c.replace("|", "__"): np.array([r["pred"][c] for r in rs], np.float32) for c in tests},
                                thr=json.dumps([r["thr"] for r in rs]))
        runs[m] = np.load(f)
    for c, t in tests.items():
        y_ai, p_ai = stage5.labels(t)
        ben = t["atk_id"] < 0
        r = {"harm": int(y_ai.sum()), "harm_rel": float(y_ai.sum() / ref), "expected_harm": float(p_ai.sum()),
             "prevalence": float(t["y"].mean()), "benign_share": float(t["y"][ben].sum() / max(1, t["y"].sum())), "models": {}}
        for m, rr in runs.items():
            thr = json.loads(str(rr["thr"]))
            P_ = rr[c.replace("|", "__")]
            if y_ai.sum():
                ev = [stage5.evaluate(t, p, th) for p, th in zip(P_, thr)]
                r["models"][m] = {k: float(np.mean([e[k] for e in ev])) for k in ev[0]}
            else:                                   # no attacker-induced frame left
                r["models"][m] = {"pr_attacker": None, "pr_unsafe": float(np.mean([average_precision_score(t["y"], p) for p in P_]))}
        gs = geo_score(t)
        r["models"]["geometry_only"] = {"pr_attacker": float(average_precision_score(y_ai, gs)) if y_ai.sum() else None,
                                        "pr_unsafe": float(average_precision_score(t["y"], gs))}
        res["corpora"][c] = r
        print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "models"},
              {m: round(v["pr_attacker"], 3) if v.get("pr_attacker") is not None else None for m, v in r["models"].items()}, flush=True)
    jdump(res, OD / "defence.json")


# ------------------------------------------------------------------ B. proximity / calibration
def contributions(t, v=None):
    v = v or {}
    x = t["E_chan"] / PEER_HI
    return np.stack([edge_contrib(x[:, k], t["E_geo"][:, k, 0].astype(np.float64), t["E_geo"][:, k, 1].astype(np.float64), v)
                     for k in range(x.shape[1])], 1)


def prox():
    res = {}
    t = load("s4_main.npz")
    C = V2_GAIN * contributions(t)
    rec = t["own_prov"] + C.sum(1)
    assert np.abs(rec - t["oracle_score"]).max() < 1e-4, "contribution recompute mismatch"
    y, ben = t["y"] == 1, t["atk_id"] < 0
    near = (C * (t["E_geo"][..., 0] < 1.0)).sum(1)
    caused = y & (t["s_noisy"] - near <= 0.30)
    res["main_dev"] = {"unsafe": int(y.sum()), "benign_track_unsafe": int((y & ben).sum()),
                       "benign_share": float((y & ben).sum() / y.sum()),
                       "benign_unsafe_caused_by_peers_lt1m": float((caused & ben).sum() / max(1, (y & ben).sum())),
                       "all_unsafe_caused_by_peers_lt1m": float(caused.sum() / y.sum()),
                       "min_distance": float(t["E_geo"][..., 0].min())}
    res["gain_sweep_benign_share"] = {}
    for g in (2, 3, 4, 5, 6):
        tt = load(f"s3_g{g}_test.npz")
        yy = tt["y"] == 1
        res["gain_sweep_benign_share"][f"g{g}"] = {"prevalence": float(yy.mean()),
                                                   "benign_share": float((yy & (tt["atk_id"] < 0)).sum() / yy.sum())}
    print(json.dumps(res, indent=1), flush=True)
    # B3: 1/max(d, 0.5) kernel
    v = {"dfloor": 0.5}
    od = OD / "floor05"
    od.mkdir(exist_ok=True)
    gain = calibrate_gain(v)
    dev, test = corpus7("floor05", "dev", v, gain), corpus7("floor05", "test", v, gain)
    yt = test["y"]
    r = {"gain": gain, "prevalence_dev": float(dev["y"].mean()), "prevalence_test": float(yt.mean()),
         "benign_share_test": float(yt[test["atk_id"] < 0].sum() / yt.sum()), "models": {}}
    pc = ceiling(test, v)
    r["ceiling"] = {"roc_auc": float(roc_auc_score(yt, pc)), "pr_auc": float(average_precision_score(yt, pc))}
    P = {}
    for m, (fam, builder, src) in model_specs().items():
        f = od / f"{m}_preds.npy"
        if not f.exists():
            np.save(f, np.array([x["pred"]["test"] for x in train(fam, builder, src, dev, {"test": test})], np.float32))
        P[m] = np.load(f)
        r["models"][m] = {"pr_auc": float(np.mean([average_precision_score(yt, p) for p in P[m]]))}
        print("floor05", m, round(r["models"][m]["pr_auc"], 3), flush=True)
    P["ceiling"] = pc[None]
    r["ci"] = boot(yt, P, [("mpnn__MPNN", "edgefeatraw__HGB")] +
                   [(a, b) for a in ("mpnn__MPNN", "edgefeatraw__HGB") for b in ("rule__RULE", "edgefeat__HGB", "gat__GAT", "proto__MLPpub")])
    res["floor05"] = r
    jdump(res, OD / "prox.json")
    print("floor05 ceiling", round(r["ceiling"]["pr_auc"], 3), "benign share", round(r["benign_share_test"], 3),
          {k: (round(d["diff"], 3), [round(x, 3) for x in d["ci"]]) for k, d in r["ci"]["diffs"].items()}, flush=True)


# ------------------------------------------------------------------ C. symptom channel
def symptom():
    from run_detection import SPECS, tab
    from stage3 import causal_mean
    specs = model_specs()

    def victim_temporal(t):
        X = np.concatenate([t["V_bb"], t["V_raw"]], 1)
        return np.concatenate([X, causal_mean(np.abs(t["V_raw"]), T)], 1)

    models = {"victim_all__HGB": ("HGB", SPECS["victim_all__HGB"][1], "victim_all__HGB"),
              "victim_temporal__HGB": ("HGB", tab(victim_temporal), "victim_all__HGB"),
              "mpnn__MPNN": specs["mpnn__MPNN"], "edgefeatraw__HGB": specs["edgefeatraw__HGB"],
              "mpnn_raw__MPNN": ("MPNN", SPECS["mpnn_raw__MPNN"][1], "mpnn_raw__MPNN")}
    res = {}
    for name, k in (("symptom_weak", 1.0), ("symptom_strong", 3.0)):
        od = OD / name
        od.mkdir(exist_ok=True)
        v = {"symptom": k}
        dev, test = corpus7(name, "dev", v, V2_GAIN), corpus7(name, "test", v, V2_GAIN)
        yt = test["y"]
        P = {}
        for m, (fam, builder, src) in models.items():
            f = od / f"{m}_preds.npy"
            if not f.exists():
                np.save(f, np.array([x["pred"]["test"] for x in train(fam, builder, src, dev, {"test": test})], np.float32))
            P[m] = np.load(f)
            print(name, m, round(float(np.mean([average_precision_score(yt, p) for p in P[m]])), 3),
                  round(float(np.mean([roc_auc_score(yt, p) for p in P[m]])), 3), flush=True)
        r = {"prevalence_test": float(yt.mean()),
             "roc_auc": {m: float(np.mean([roc_auc_score(yt, p) for p in ps])) for m, ps in P.items()}}
        pairs = [(a, b) for a in ("mpnn__MPNN", "edgefeatraw__HGB", "mpnn_raw__MPNN") for b in ("victim_all__HGB", "victim_temporal__HGB")]
        pairs.append(("mpnn_raw__MPNN", "mpnn__MPNN"))
        r["ci"] = boot(yt, P, pairs)
        res[name] = r
        print(name, {k_: (round(d["diff"], 3), [round(x, 3) for x in d["ci"]]) for k_, d in r["ci"]["diffs"].items()}, flush=True)
    jdump(res, OD / "symptom.json")


# ------------------------------------------------------------------ D. re-tuning
DETECTABLE = list(STRUCT) + ["rand00", "rand02", "rand03", "rand04", "rand05", "rand06", "rand07", "rand11"]


def retune():
    from models import select_and_predict
    from run_final import train_val_split
    specs = model_specs()
    res = json.loads((OD / "retune.json").read_text()) if (OD / "retune.json").exists() else {}
    t0 = time.time()
    for name in DETECTABLE:
        if name in res:
            continue
        od = OD / "retune" / name
        od.mkdir(parents=True, exist_ok=True)
        dev, test = load(f"s7_{name}_dev.npz"), load(f"s7_{name}_test.npz")
        y, yt = dev["y"], test["y"]
        P, cfgs = {}, {}
        for m in ("mpnn__MPNN", "edgefeatraw__HGB"):
            fam, builder, _ = specs[m]
            f = od / f"{m}_preds.npy"
            kd, kt = builder(dev), builder(test)
            if not f.exists():
                preds, sel = [], []
                for s in SEEDS:
                    tr, va = train_val_split(dev["ctx"], y, s)
                    if kd[0] == "tab":
                        d = {"Xtr": kd[1][tr], "ytr": y[tr], "Xva": kd[1][va], "yva": y[va], "Xte": {"test": kt[1]}}
                    else:
                        d = {"Vtr": kd[1][tr], "Etr": kd[2][tr], "ytr": y[tr], "Vva": kd[1][va], "Eva": kd[2][va],
                             "yva": y[va], "Gte": {"test": (kt[1], kt[2])}}
                    _, pt, cfg, vap, _ = select_and_predict(fam, d, s)
                    preds.append(pt["test"]); sel.append({"cfg": cfg, "val_pr": vap})
                np.save(f, np.array(preds, np.float32))
                (od / f"{m}_selection.json").write_text(json.dumps(sel, default=str))
            P[m] = np.load(f)
            cfgs[m] = json.loads((od / f"{m}_selection.json").read_text())
            print(f"[{name}] {m} retuned PR {np.mean([average_precision_score(yt, p) for p in P[m]]):.3f} ({time.time() - t0:.0f}s)", flush=True)
        fixed = {m: np.load(OUT / "stage7" / name / f"{m}_preds.npy") for m in ("mpnn__MPNN", "edgefeatraw__HGB")}
        b = boot(yt, {"mpnn_rt": P["mpnn__MPNN"], "hgb_rt": P["edgefeatraw__HGB"],
                      "mpnn_fixed": fixed["mpnn__MPNN"], "hgb_fixed": fixed["edgefeatraw__HGB"]},
                 [("mpnn_rt", "hgb_rt"), ("mpnn_fixed", "hgb_fixed"), ("mpnn_rt", "mpnn_fixed"), ("hgb_rt", "hgb_fixed")])
        res[name] = {"boot": b, "selected": cfgs}
        jdump(res, OD / "retune.json")
        print(name, {k: (round(d["diff"], 3), [round(x, 3) for x in d["ci"]]) for k, d in b["diffs"].items()}, flush=True)


# ------------------------------------------------------------------ counts (B3 as 24th generator; D summary)
def counts():
    from stage8_extra import holm
    e8 = json.loads((OUT / "stage8" / "extra.json").read_text())
    pr = json.loads((OD / "prox.json").read_text())["floor05"]
    test = load("s7_floor05_test.npz")
    P = {m: np.load(OD / "floor05" / f"{m}_preds.npy") for m in ("mpnn__MPNN", "edgefeatraw__HGB")}
    d = boot(test["y"], P, [("mpnn__MPNN", "edgefeatraw__HGB")])["diffs"]["mpnn__MPNN - edgefeatraw__HGB"]
    rows = [(r["gen"], r["diff"], r["p"]) for r in e8["holm"]["rows"]]
    rows.append(("floor05", d["diff"], max(1e-3, 2 * min(d["p_le0"], 1 - d["p_le0"]))))
    adj = holm(np.array([r[2] for r in rows]))
    win = dict(e8["winners"])
    win["floor05"] = max(pr["models"], key=lambda m: pr["models"][m]["pr_auc"])
    res = {"n": len(rows), "floor05_diff": d,
           "raw_sig": [n for (n, df, _), a in zip(rows, adj) if a < 0.05 and df < 0],
           "mpnn_sig": [n for (n, df, _), a in zip(rows, adj) if a < 0.05 and df > 0],
           "raw_higher": sum(df < 0 for _, df, _ in rows), "n_full_state_wins": sum(w in ("mpnn__MPNN", "edgefeatraw__HGB") for w in win.values()),
           "winners": win}
    if (OD / "retune.json").exists():
        rt = json.loads((OD / "retune.json").read_text())
        rr = {}
        for n, v in rt.items():
            b = v["boot"]["diffs"]
            rr[n] = {"retuned": b["mpnn_rt - hgb_rt"], "fixed": b["mpnn_fixed - hgb_fixed"],
                     "mpnn_gain": b["mpnn_rt - mpnn_fixed"], "hgb_gain": b["hgb_rt - hgb_fixed"],
                     "pr": {m: v["boot"]["models"][m]["pr_auc"][0] for m in v["boot"]["models"]}}
        res["retune"] = {"n": len(rr), "rows": rr,
                         "sign_changes": [n for n, r in rr.items() if np.sign(r["retuned"]["diff"]) != np.sign(r["fixed"]["diff"])],
                         "hgb_higher_retuned": sum(r["retuned"]["diff"] < 0 for r in rr.values()),
                         "ci_excl0_hgb": [n for n, r in rr.items() if r["retuned"]["ci"][1] < 0],
                         "ci_excl0_mpnn": [n for n, r in rr.items() if r["retuned"]["ci"][0] > 0]}
    jdump(res, OD / "counts.json")
    print(json.dumps({k: v for k, v in res.items() if k not in ("winners",)}, indent=1, default=str)[:4000])


# ------------------------------------------------------------------ E. events (exploratory)
def events():
    t = load("s4_lockbox.npz")
    y = t["y"]
    nt = len(y) // T
    yv = y.reshape(nt, T)
    pos_track = yv.any(1)
    res = {"victim_tracks": int(nt), "tracks_with_unsafe": int(pos_track.sum())}
    for m in ("mpnn__MPNN", "edgefeatraw__HGB"):
        z = np.load(OUT / "stage7" / "main" / f"{m}.npz")
        thr = json.loads(str(z["thr"]))
        rr = []
        for p, th in zip(z["lockbox"], thr):
            a = (p >= th["fpr1"]).reshape(nt, T).astype(int)
            run = np.zeros_like(a)
            for tt in range(T):
                run[:, tt] = (run[:, tt - 1] + 1) * a[:, tt] if tt else a[:, tt]
            alarm = (run >= 3).any(1)
            frame_fpr = float(a.reshape(-1)[y == 0].mean())
            rr.append({"track_recall": float(alarm[pos_track].mean()), "track_false_alarm": float(alarm[~pos_track].mean()),
                       "frame_fpr": frame_fpr, "frame_false_alarms_per_min_60fps": frame_fpr * 3600})
        res[m] = {k: float(np.mean([r[k] for r in rr])) for k in rr[0]}
    jdump(res, OD / "events.json")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"defence": defence, "prox": prox, "symptom": symptom, "retune": retune, "events": events, "counts": counts}[sys.argv[1]]()
