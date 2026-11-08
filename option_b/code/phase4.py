"""
Option B, Phase 4: reanalysis (plans v11 and v12, plus descriptive analyses of existing results).
Run with XRP_GEN=v2 from option_b/code after e1_retune.py and e2_group.py. Writes option_b/results/phase4/.

All AP values use the tie-aware routine of ob_stats (sklearn semantics) for point estimates and bootstrap.
Shared cohort-bootstrap weights (B=1000, seed 0) are used for every generator and detector (shared test seeds).
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import norm

J7 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(J7 / "rigorous_rerun"))
from common import CACHE, OUT  # noqa: E402
from ob_stats import Sorted, boot_weights, cohort_ids, holm, p_two_sided, tau  # noqa: E402

OB = J7 / "option_b"
OD = OB / "results" / "phase4"
OD.mkdir(parents=True, exist_ok=True)
B = 1000
GENS = ["d2", "cutoff", "fovfree", "and", "sat", "rand00", "rand02", "rand03", "rand04", "rand05", "rand06", "rand07", "rand11"]
STRUCT = GENS[:5]
FIXED6 = ["rule__RULE", "proto__MLPpub", "edgefeat__HGB", "gat__GAT", "mpnn__MPNN", "edgefeatraw__HGB"]
RET7 = ["rule__RULE", "proto__MLPpub", "proto__HGB", "edgefeat__HGB", "gat__GAT", "mpnn__MPNN", "edgefeatraw__HGB"]
FULL = ("mpnn__MPNN", "edgefeatraw__HGB")
W = boot_weights(60, B, 0)


def evaluate(y, P):
    """P: (seeds, n). Returns point (mean over seeds) and bootstrap vector (B,)."""
    g = cohort_ids(len(y))
    S = [Sorted(p, y, g) for p in P]
    point = float(np.mean([s.ap() for s in S]))
    bs = np.array([np.mean([s.ap(w) for s in S]) for w in W])
    return point, bs


def fixed_path(gen, m):
    return OUT / "stage7" / gen / f"{m}_preds.npy"


def retuned_path(gen, m):
    if m in ("rule__RULE", "proto__MLPpub"):
        return fixed_path(gen, m)
    if m in FULL:
        return OUT / "stage12" / "retune" / gen / f"{m}_preds.npy"
    return OB / "results" / "e1" / gen / f"{m}_preds.npy"


def table_block(cond_paths, ys):
    """cond_paths: {gen: {m: path}} -> {gen: {m: (point, bs)}} (missing files -> None)."""
    out = {}
    for gen, paths in cond_paths.items():
        out[gen] = {}
        for m, p in paths.items():
            out[gen][m] = evaluate(ys[gen], np.load(p)) if Path(p).exists() else None
            print(gen, m, None if out[gen][m] is None else round(out[gen][m][0], 3), flush=True)
    return out


def pairwise(block, dets, label):
    rows = []
    for gen, d in block.items():
        for a, b in combinations(dets, 2):
            if d.get(a) is None or d.get(b) is None:
                continue
            diff = d[a][1] - d[b][1]
            sd = float(np.std(diff, ddof=1))
            rows.append({"set": label, "gen": gen, "a": a, "b": b, "diff": d[a][0] - d[b][0],
                         "ci": np.percentile(diff, [2.5, 97.5]).tolist(), "p": p_two_sided(diff, B),
                         # deviation D1 (DEVIATIONS.md): bootstrap-SE normal p-value, no 1/B floor
                         "p_wald": float(2 * norm.sf(abs(d[a][0] - d[b][0]) / sd)) if sd > 0 else 1.0})
    for key, out in (("p", "p_holm"), ("p_wald", "p_wald_holm")):
        for r, a in zip(rows, holm([r[key] for r in rows])):
            r[out] = float(a)
    return rows


def reversals(rows, key="p_holm"):
    sig = {}
    for r in rows:
        if r[key] < 0.05:
            win, lose = (r["a"], r["b"]) if r["diff"] > 0 else (r["b"], r["a"])
            sig.setdefault((win, lose), []).append(r["gen"])
    out = []
    for (w, l), gens in sig.items():
        if (l, w) in sig and w < l:
            out.append({"pair": [w, l], f"{w}_better_in": gens, f"{l}_better_in": sig[(l, w)]})
    return out


def taus(block, dets):
    gens = [g for g in block if all(block[g].get(m) is not None for m in dets)]
    pts, bss = {}, {}
    for g in gens:
        pts[g] = np.array([block[g][m][0] for m in dets])
        bss[g] = np.array([block[g][m][1] for m in dets])          # (dets, B)
    res = []
    for a, b in combinations(gens, 2):
        bt = [tau(bss[a][:, k], bss[b][:, k]) for k in range(B)]
        res.append({"a": a, "b": b, "tau": tau(pts[a], pts[b]), "ci": np.nanpercentile(bt, [2.5, 97.5]).tolist()})
    t = np.array([r["tau"] for r in res])
    return {"pairs": res, "median": float(np.median(t)), "min": float(t.min()), "max": float(t.max()),
            "n_negative": int((t < 0).sum()), "n_pairs": len(res)}


def winners(block, dets):
    return {g: max((m for m in dets if block[g].get(m) is not None), key=lambda m: block[g][m][0]) for g in block}


def e1_and_family():
    ys = {g: np.load(CACHE / f"s7_{g}_test.npz")["y"] for g in GENS}
    fixed = table_block({g: {m: fixed_path(g, m) for m in FIXED6} for g in GENS}, ys)
    ret = table_block({g: {m: retuned_path(g, m) for m in RET7} for g in GENS}, ys)
    res = {"fixed": {g: {m: (v[0] if v else None) for m, v in d.items()} for g, d in fixed.items()},
           "retuned": {g: {m: (v[0] if v else None) for m, v in d.items()} for g, d in ret.items()},
           "retuned_ci": {g: {m: (np.percentile(v[1], [2.5, 97.5]).tolist() if v else None) for m, v in d.items()} for g, d in ret.items()}}
    pw = pairwise(ret, RET7, "retuned")
    res["pairwise_retuned"] = pw
    res["H1_reversals"] = reversals(pw)
    res["H1_supported"] = bool(res["H1_reversals"])
    res["pairwise_fixed"] = pairwise(fixed, FIXED6, "fixed")
    res["reversals_fixed"] = reversals(res["pairwise_fixed"])
    # deviation D1: same definitions with p_wald_holm, reported beside (not instead of) the pre-specified test
    res["H1_D1_reversals"] = reversals(pw, "p_wald_holm")
    res["H1_D1_supported"] = bool(res["H1_D1_reversals"])
    res["reversals_fixed_D1"] = reversals(res["pairwise_fixed"], "p_wald_holm")
    res["n_sig_retuned"] = {"prespecified": sum(r["p_holm"] < 0.05 for r in pw), "D1": sum(r["p_wald_holm"] < 0.05 for r in pw)}
    res["min_attainable_holm_p_prespecified"] = min(1.0, len(pw) / B)
    res["tau_retuned"] = taus(ret, RET7)
    res["tau_fixed"] = taus(fixed, FIXED6)
    # fixed vs re-tuned within each generator (6 common detectors: tuning effect on the ranking)
    res["tau_fixed_vs_retuned"] = {g: tau([fixed[g][m][0] for m in FIXED6], [ret[g][m][0] for m in FIXED6])
                                   for g in GENS if all(fixed[g][m] and ret[g][m] for m in FIXED6)}
    res["winners_fixed"], res["winners_retuned"] = winners(fixed, FIXED6), winners(ret, RET7)
    res["full_state_best_retuned"] = sum(w in FULL for w in res["winners_retuned"].values())
    # tuning vs generator decomposition (descriptive)
    dec = {}
    for m in FIXED6:
        if not all(fixed[g][m] and ret[g][m] for g in GENS):
            continue
        rf = {g: sorted(FIXED6, key=lambda x: -fixed[g][x][0]).index(m) + 1 for g in GENS}
        rr = {g: sorted(FIXED6, key=lambda x: -ret[g][x][0]).index(m) + 1 for g in GENS}
        dpr = [ret[g][m][0] - fixed[g][m][0] for g in GENS]
        dec[m] = {"mean_abs_pr_change_tuning": float(np.mean(np.abs(dpr))), "mean_pr_change_tuning": float(np.mean(dpr)),
                  "mean_abs_rank_change_tuning": float(np.mean([abs(rr[g] - rf[g]) for g in GENS])),
                  "sd_pr_across_generators_retuned": float(np.std([ret[g][m][0] for g in GENS], ddof=1)),
                  "sd_rank_across_generators_retuned": float(np.std([rr[g] for g in GENS], ddof=1))}
    res["decomposition"] = dec
    # pre-specified MPNN rule (plan v7), fixed configurations, as originally scored
    s7 = {g: json.loads((OUT / "stage7" / g / "summary.json").read_text()) for g in GENS}
    best = {g: max(s7[g]["models"], key=lambda m: s7[g]["models"][m]["pr_auc"][0]) for g in GENS}
    res["mpnn_rule"] = {"structural_mpnn_best": sum(best[g] == "mpnn__MPNN" for g in STRUCT),
                        "random_detectable_mpnn_best": sum(best[g] == "mpnn__MPNN" for g in GENS[5:]), "best": best}
    return res, fixed, ret


def transfer_analysis(fixed):
    tr = {}
    for m in FIXED6:
        z = np.load(OUT / "stage7" / "main" / f"{m}.npz")
        for g in GENS:
            y = np.load(CACHE / f"s7_{g}_test.npz")["y"]
            gi = cohort_ids(len(y))
            tr.setdefault(g, {})[m] = float(np.mean([Sorted(p, y, gi).ap() for p in z[g]]))
    ratio = {g: {m: tr[g][m] / fixed[g][m][0] for m in FIXED6} for g in GENS}
    best_tr = {g: max(FIXED6, key=lambda m: tr[g][m]) for g in GENS}
    return {"transfer_pr": tr, "ratio": ratio, "best_transferred": best_tr,
            "mpnn_loses_30pct": [g for g in GENS if ratio[g]["mpnn__MPNN"] <= 0.70],
            "tau_within_vs_transfer": {g: tau([fixed[g][m][0] for m in FIXED6], [tr[g][m] for m in FIXED6]) for g in GENS}}


def lockbox2():
    import stage5
    t = {k: v for k, v in np.load(CACHE / "s8_lockbox2.npz").items()}
    y_ai, _ = stage5.labels(t)
    g = cohort_ids(len(t["y"]))
    Wl = boot_weights(120, B, 0)
    out = {}
    for tgt, yy in (("unsafe", t["y"]), ("attacker", y_ai)):
        pts, bss = {}, {}
        for m in ("mpnn__MPNN", "edgefeatraw__HGB"):
            S = [Sorted(p, yy, g) for p in np.load(OUT / "stage8" / f"{m}_lb2.npy")]
            pts[m] = float(np.mean([s.ap() for s in S]))
            bss[m] = np.array([np.mean([s.ap(w) for s in S]) for w in Wl])
        d = bss["mpnn__MPNN"] - bss["edgefeatraw__HGB"]
        out[tgt] = {"pr": pts, "diff": pts["mpnn__MPNN"] - pts["edgefeatraw__HGB"], "ci": np.percentile(d, [2.5, 97.5]).tolist()}
    return out


def v1_audit():
    r = {}
    for ver, base in (("v1", J7 / "rigorous_rerun" / "outputs"), ("v2", OUT)):
        r[ver] = {}
        for m in ("proto__MLPpub", "mpnn__MPNN", "ORACLE"):
            lb = json.loads((base / "final_external" / f"{m}.json").read_text())["lockbox"]["by_seed"]
            r[ver][m] = {"roc_auc": lb["roc_auc"][0], "pr_auc": lb["pr_auc"][0]}
        pc = base / "per_channel_detection_lockbox.json"
        if pc.exists():
            r[ver]["per_channel"] = json.loads(pc.read_text())
    return r


def e2_analysis(ret):
    from e2_group import VARIANTS, OBC
    res = {"build": json.loads((OB / "results" / "e2" / "build.json").read_text())}
    for v in VARIANTS:
        y = np.load(OBC / f"e2_{v}_test.npz")["y"]
        dv = OB / "results" / "e2" / v
        paths = {m: (dv / f"fixed_{m}_preds.npy" if m in ("rule__RULE", "proto__MLPpub") else dv / f"retuned_{m}_preds.npy") for m in RET7}
        blk = table_block({v: paths}, {v: y})
        fx = table_block({v: {m: dv / f"fixed_{m}_preds.npy" for m in FIXED6}}, {v: y})
        pw = pairwise(blk, RET7, f"e2_{v}")
        pts = np.array([blk[v][m][0] for m in RET7])
        res[v] = {"retuned": {m: blk[v][m][0] for m in RET7},
                  "retuned_ci": {m: np.percentile(blk[v][m][1], [2.5, 97.5]).tolist() for m in RET7},
                  "fixed": {m: fx[v][m][0] for m in FIXED6}, "pairwise": pw,
                  "winner_retuned": max(RET7, key=lambda m: blk[v][m][0]),
                  "tau_vs_family_retuned": {g: tau(pts, [ret[g][m][0] for m in RET7]) for g in GENS}}
        t = np.array(list(res[v]["tau_vs_family_retuned"].values()))
        res[v]["tau_vs_family_summary"] = {"median": float(np.median(t)), "min": float(t.min()), "max": float(t.max())}
        z = {m: np.load(OB / "results" / "e2" / "transfer" / f"{m}.npz")[v] for m in FIXED6}
        gi = cohort_ids(len(y))
        trn = {m: float(np.mean([Sorted(p, y, gi).ap() for p in z[m]])) for m in FIXED6}
        res[v]["transfer"] = trn
        res[v]["transfer_ratio"] = {m: trn[m] / fx[v][m][0] for m in FIXED6}
    return res


def main():
    res, fixed, ret = e1_and_family()
    res["transfer"] = transfer_analysis(fixed)
    res["lockbox2"] = lockbox2()
    res["v1_audit"] = v1_audit()
    if (OB / "results" / "e2" / "transfer").exists():
        res["e2"] = e2_analysis(ret)
    (OD / "phase4.json").write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed(OD / "boot_retuned.npz", **{f"{g}|{m}": v[1] for g, d in ret.items() for m, v in d.items() if v})
    print("H1 supported:", res["H1_supported"], res["H1_reversals"])
    print("H1 (D1) supported:", res["H1_D1_supported"], res["H1_D1_reversals"], res["n_sig_retuned"])
    print("tau retuned:", {k: res["tau_retuned"][k] for k in ("median", "min", "max", "n_negative", "n_pairs")})
    print("winners retuned:", res["winners_retuned"])


if __name__ == "__main__":
    main()
