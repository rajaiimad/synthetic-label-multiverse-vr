"""
Plan v15: equal-budget training of MPNN and Raw-HGB on one generator, for both targets (U: all unsafe; A: attack-caused).

  python v15/train_v15.py --stage X --gen d2             Stage X (exploratory): existing generator, cohorts and seeds
  python v15/train_v15.py --stage C --gen c117 [--r3]    Stage C (confirmatory): prior draw k = 117, new cohorts and seeds
  python v15/train_v15.py --stage S2                     base generator v2 (in-band) for the generator-shift outcome
  python v15/train_v15.py --stage smoke --gen d2         tiny run on separate seeds, to test the code only

For each target, detector and training seed: 45/15 cohort split of the development corpus (stratified on the TRAINING
target), 16 configurations, selection by validation PR-AUC on the same target, test corpus scored once; the selected
model is saved. Outputs: results_v15/<stage>/<gen>/ (preds_<target>_<det>.npy with shape (seeds, test rows),
selection_*.json, labels_test.npz, meta.json, models/). Resumes: finished (target, detector) pairs are skipped.
Run with XRP_GEN=v2 from the repository root.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("XRP_GEN", "v2")
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import gen_v15 as G  # noqa: E402
import baselines  # noqa: E402,F401  (registers RULE)
import models  # noqa: E402
from run_detection import SPECS  # noqa: E402
from run_final import train_val_split  # noqa: E402
from stage3 import V2_GAIN, calibrate_gain  # noqa: E402
from stage7 import STRUCT, sample_prior  # noqa: E402

OUT = ROOT / "results_v15"
DETS = {"mpnn": ("MPNN", "mpnn__MPNN"), "rawhgb": ("HGB", "edgefeatraw__HGB")}
TARGETS = {"U": "y", "A": "y_A"}
K_MC = 32


def ceiling_a_full(t):
    """D3 (plan P7): full-knowledge oracle for target A, a loose upper bound. Knows the benign-plan score s0 as well as
    s: P(Y_A = 1 | s, s0) = Phi((theta - s0)/sigma) - Phi((theta - s)/sigma) if s > s0, else 0 (attacked victims only)."""
    from scipy.stats import norm
    sg = float(t["sigma"])
    p = np.where(t["victim_of_attack"] == 1,
                 np.maximum(0.0, norm.cdf((G.THETA - t["score_b0"]) / sg) - norm.cdf((G.THETA - t["score64"]) / sg)), 0.0)
    return float(average_precision_score(t["y_A"], p)) if t["y_A"].any() else 0.0


# ---------------------------------------------------------------- generator specification
def spec(stage, gen):
    if stage in ("X", "smoke"):
        sx = json.loads((HERE / "stage_x_generators.json").read_text())
        v = STRUCT[gen] if gen in STRUCT else sample_prior(int(gen[4:]))
        cfg = {"v": v, "gain": sx[gen]["gain"], "train_seeds": G.TRAIN_SEEDS_X}
        if stage == "X":
            cfg.update(dev=(60_000, 60_060), test=(61_000, 61_060))
        else:   # code test only: separate seeds, tiny corpora, two configurations, one seed
            cfg.update(dev=(9_000_000, 9_000_008), test=(9_100_000, 9_100_004), train_seeds=(101,))
        return cfg
    if stage == "C":
        k = int(gen[1:])
        v = sample_prior(k)
        r = G.stage_c_ranges(k)
        return {"v": v, "gain": calibrate_gain(v), "train_seeds": G.TRAIN_SEEDS_C, "dev": r["dev"], "test": r["test"], "k": k}
    if stage == "S2":
        return {"v": {}, "gain": V2_GAIN, "train_seeds": G.TRAIN_SEEDS_C, "dev": G.BASE_S2_DEV, "test": None}
    raise ValueError(stage)


# ---------------------------------------------------------------- model saving and re-scoring
def save_model(path, fam, meta, cfg):
    import joblib
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    if fam == "MPNN":
        net = meta["net"]
        dv = net.rho[0].in_features - 2 * cfg["h"]          # rho input = [V, sum, max] (MPNN default aggregation)
        de = net.phi[0].in_features - dv                     # phi input = [V, E] (E after the optional log feature)
        torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "scalers": meta["scalers"], "cfg": cfg,
                    "dv": dv, "de": de}, str(path) + ".pt")
    else:
        joblib.dump({"model": meta["model"], "n_iter": meta["n_iter"], "cfg": cfg}, str(path) + ".joblib", compress=3)


def score_saved(path, fam, kind):
    """Predict with a saved model on a built input (kind = builder output)."""
    if fam == "MPNN":
        import torch
        b = torch.load(str(path) + ".pt", weights_only=False)
        cfg, (sv, se) = b["cfg"], b["scalers"]
        net = models.MPNN(b["dv"], b["de"], cfg["h"])
        net.load_state_dict(b["state_dict"])
        net.eval()
        V, E = kind[1], kind[2]
        out = []
        with torch.no_grad():
            for s in range(0, len(V), 8192):
                v_ = torch.tensor(sv(V[s:s + 8192]))
                e_ = torch.tensor(se(models._logd(E[s:s + 8192], cfg["logd"])))
                out.append(torch.sigmoid(net(v_, e_)).numpy())
        return np.concatenate(out).astype(np.float32)
    import joblib
    b = joblib.load(str(path) + ".joblib")
    for k, p in enumerate(b["model"].staged_predict_proba(kind[1]), 1):
        if k == b["n_iter"]:
            return p[:, 1].astype(np.float32)


# ---------------------------------------------------------------- training
def select(fam, d, seed, grid, fails):
    """As option_b/code/e1_retune.select (failed configurations skipped and logged), keeping the selected model."""
    best, tried = None, []
    for cfg in grid:
        dd = dict(d)
        try:
            pv, pt = models.FAMILIES[fam](dd, cfg, seed)
            ap = float(average_precision_score(d["yva"], pv))
            if not np.isfinite(ap):
                raise ValueError("non-finite validation AP")
        except Exception as e:  # noqa: BLE001
            fails.append({"cfg": cfg, "seed": seed, "error": repr(e)})
            continue
        tried.append({"cfg": cfg, "val_pr": ap, "n_iter": dd.get("_meta", {}).get("n_iter")})
        if best is None or ap > best["val_pr"]:
            best = {"pt": pt, "cfg": cfg, "val_pr": ap, "meta": dd.get("_meta", {})}
    return best, tried


def make_d(kind_dev, kind_test, y, tr, va):
    if kind_dev[0] == "tab":
        return {"Xtr": kind_dev[1][tr], "ytr": y[tr], "Xva": kind_dev[1][va], "yva": y[va],
                "Xte": {} if kind_test is None else {"test": kind_test[1]}}
    return {"Vtr": kind_dev[1][tr], "Etr": kind_dev[2][tr], "ytr": y[tr], "Vva": kind_dev[1][va], "Eva": kind_dev[2][va],
            "yva": y[va], "Gte": {} if kind_test is None else {"test": (kind_test[1], kind_test[2])}}


def train_one(od, tag, det, dev, test, ylab, seeds, grid_n=None, keep_rows=None):
    """Train detector `det` on labels `ylab` of the dev corpus; score the test corpus. Writes preds_<tag>_<det>.npy."""
    f = od / f"preds_{tag}_{det}.npy"
    if f.exists():
        return
    fam, specname = DETS[det]
    builder = SPECS[specname][1]
    grid = models.GRIDS[fam][:grid_n] if grid_n else models.GRIDS[fam]
    dsub = dev if keep_rows is None else {k: (v[keep_rows] if (hasattr(v, "shape") and v.shape and len(v) == len(dev["y"])) else v)
                                         for k, v in dev.items()}
    y = dsub[ylab]
    kd = builder(dsub)
    kt = builder(test) if test is not None else None
    preds, sel, fails = [], [], []
    for s in seeds:
        tr, va = train_val_split(dsub["ctx"], y, s)
        best, tried = select(fam, make_d(kd, kt, y, tr, va), s, grid, fails)
        if best is None:
            raise RuntimeError(f"no configuration trained for {tag}/{det}/seed {s}")
        if test is not None:
            preds.append(best["pt"]["test"].astype(np.float32))
        save_model(od / "models" / f"{tag}_{det}_seed{s}", fam, best["meta"], best["cfg"])
        sel.append({"seed": s, "selected": best["cfg"], "val_pr": best["val_pr"], "tried": tried,
                    "n_train_pos": int(y[tr].sum()), "n_val_pos": int(y[va].sum())})
    (od / f"selection_{tag}_{det}.json").write_text(json.dumps({"selection": sel, "failures": fails}, default=str, indent=1))
    if test is not None:
        np.save(f, np.array(preds, np.float32))
    else:
        (od / f"done_{tag}_{det}.flag").write_text("ok")


def train_s5(od, tag, det, dev, test, ylab, seeds, grid_n=None):
    """Deviation D2, secondary outcome S5 (tuning budget). Per seed, the 16 frozen extra configurations of
    v15/s5_grids.json are trained on the same split as the primary run; selection is over the union of the primary 16
    and these 16 (validation PR-AUC on the same target; a new configuration must be strictly better, so ties keep the
    primary choice, as the primary selection keeps the first best in grid order). The primary predictions and
    selection are reused, not recomputed, and are never modified. Writes preds_S5<tag>_<det>.npy."""
    f = od / f"preds_S5{tag}_{det}.npy"
    if f.exists():
        return
    fam, specname = DETS[det]
    grid = json.loads((HERE / "s5_grids.json").read_text())[fam]
    grid = grid[:grid_n] if grid_n else grid
    prim = json.loads((od / f"selection_{tag}_{det}.json").read_text())["selection"]
    prim_preds = np.load(od / f"preds_{tag}_{det}.npy")
    builder = SPECS[specname][1]
    y = dev[ylab]
    kd, kt = builder(dev), builder(test)
    preds, sel, fails = [], [], []
    for i, s in enumerate(seeds):
        assert prim[i]["seed"] == s, "primary selection out of seed order"
        tr, va = train_val_split(dev["ctx"], y, s)
        best, tried = select(fam, make_d(kd, kt, y, tr, va), s, grid, fails)
        use_new = best is not None and best["val_pr"] > prim[i]["val_pr"]
        if use_new:
            preds.append(best["pt"]["test"].astype(np.float32))
            save_model(od / "models" / f"S5{tag}_{det}_seed{s}", fam, best["meta"], best["cfg"])
        else:
            preds.append(prim_preds[i])
        sel.append({"seed": s, "source": "extra" if use_new else "primary", "primary_val_pr": prim[i]["val_pr"],
                    "best_extra": None if best is None else {"cfg": best["cfg"], "val_pr": best["val_pr"]}, "tried": tried})
    (od / f"selection_S5{tag}_{det}.json").write_text(json.dumps({"selection": sel, "failures": fails}, default=str, indent=1))
    np.save(f, np.array(preds, np.float32))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", required=True, choices=["X", "C", "S2", "smoke"])
    ap_.add_argument("--gen", default="v2")
    ap_.add_argument("--r3", action="store_true", help="also run robustness R3 (positive-count control)")
    ap_.add_argument("--s5", action="store_true", help="also run S5 (tuning budget, deviation D2)")
    a = ap_.parse_args()
    t0 = time.time()
    sp = spec(a.stage, a.gen)
    od = OUT / a.stage / a.gen
    od.mkdir(parents=True, exist_ok=True)
    grid_n = 2 if a.stage == "smoke" else None
    dev = G.generate_v15(sp["dev"][0], sp["dev"][1] - sp["dev"][0], sp["v"], sp["gain"], inband=True, k_mc=0)
    test = None
    if sp["test"] is not None:
        test = G.generate_v15(sp["test"][0], sp["test"][1] - sp["test"][0], sp["v"], sp["gain"], inband=True,
                              k_mc=4 if a.stage == "smoke" else K_MC)
        np.savez_compressed(od / "labels_test.npz", y=test["y"].astype(np.int8), y_A=test["y_A"].astype(np.int8),
                            y_A0=test["y_A0"].astype(np.int8), prevented=test["prevented"].astype(np.int8),
                            victim_of_attack=test["victim_of_attack"].astype(np.int8),
                            p_A_obs=test["p_A_obs"].astype(np.float32), score64=test["score64"],
                            score_b0=test["score_b0"],             # D3: for the full-knowledge A ceiling (plan P7)
                            ctx=test["ctx"].astype(np.int32), sigma=test["sigma"])
    meta = {"stage": a.stage, "gen": a.gen, "spec": json.loads(json.dumps(sp["v"])), "gain": sp["gain"],
            "dev_seeds": sp["dev"], "test_seeds": sp["test"], "train_seeds": list(sp["train_seeds"]),
            "dev_pos": {t: int(dev[l].sum()) for t, l in TARGETS.items()}, "dev_rows": int(len(dev["y"])),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "sklearn": __import__("sklearn").__version__, "torch": __import__("torch").__version__},
            "git_commit": os.environ.get("GITHUB_SHA", "local"), "inband": True, "k_mc": K_MC}
    if test is not None:
        meta.update(test_rows=int(len(test["y"])), test_prev={t: float(test[l].mean()) for t, l in TARGETS.items()},
                    test_prev_A0=float(test["y_A0"].mean()), prevented=int(test["prevented"].sum()),
                    ceilings=G.ceilings(test), ceiling_A_full=ceiling_a_full(test))
    (od / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    if a.stage in ("C", "S2"):
        G.assert_disjoint([tuple(sp["dev"])] + ([tuple(sp["test"])] if sp["test"] else []))
    print(f"[{a.stage}/{a.gen}] corpora ready ({time.time() - t0:.0f}s) {meta.get('ceilings')}", flush=True)
    # primary training: both targets, both detectors
    for tag, lab in TARGETS.items():
        for det in DETS:
            train_one(od, tag, det, dev, test, lab, sp["train_seeds"], grid_n)
            print(f"[{a.stage}/{a.gen}] {tag} {det} done ({time.time() - t0:.0f}s)", flush=True)
    if a.stage == "X":   # the rule needs no training: documents that excess-based detectors cannot see the in-band attack
        f = od / "preds_rule.npy"
        if not f.exists():
            kd, kt = SPECS["rule__RULE"][1](dev), SPECS["rule__RULE"][1](test)
            tr, va = train_val_split(dev["ctx"], dev["y"], 0)
            _, pt = models.FAMILIES["RULE"](make_d(kd, kt, dev["y"], tr, va), {}, 0)
            np.save(f, pt["test"][None].astype(np.float32))
    if a.stage == "C":
        # S2: base-generator models scored on this generator's test corpus (if the S2 models are present)
        base = OUT / "S2" / "v2" / "models"
        if base.exists():
            for tag in TARGETS:
                for det, (fam, specname) in DETS.items():
                    f = od / f"preds_S2{tag}_{det}.npy"
                    if f.exists():
                        continue
                    kt = SPECS[specname][1](test)
                    np.save(f, np.array([score_saved(base / f"{tag}_{det}_seed{s}", fam, kt) for s in sp["train_seeds"]], np.float32))
        if a.r3:   # R3: U-target models trained with unsafe positives subsampled to the A-positive count (rows removed)
            pos = np.where(dev["y"] == 1)[0]
            keep_pos = np.random.default_rng([sp["k"], 13]).choice(pos, size=int(dev["y_A"].sum()), replace=False)
            keep = np.sort(np.r_[np.where(dev["y"] == 0)[0], keep_pos])
            for det in DETS:
                train_one(od, "UR3", det, dev, test, "y", sp["train_seeds"], grid_n, keep_rows=keep)
    if a.s5 and a.stage in ("C", "smoke"):   # S5 (deviation D2): first 10 Stage C generators in draw order
        for tag, lab in TARGETS.items():
            for det in DETS:
                train_s5(od, tag, det, dev, test, lab, sp["train_seeds"], grid_n)
                print(f"[{a.stage}/{a.gen}] S5 {tag} {det} done ({time.time() - t0:.0f}s)", flush=True)
    print(f"[{a.stage}/{a.gen}] DONE ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
