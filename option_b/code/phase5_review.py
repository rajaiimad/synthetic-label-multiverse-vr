"""
r5 analyses of existing predictions (ANALYSIS_PLAN_v13_REVIEW.md section 3: A, B, C, D, E, G, H; E3 parts of A3/C/D
after e3_temporal.py). Run with XRP_GEN=v2 from option_b/code:
  python phase5_review.py perseed [workers]   per-seed cohort-bootstrap AP replicates (cached, B = 1000, seed 0)
  python phase5_review.py analyse             all analyses -> option_b/results/phase5/review.json
Reads predictions and phase4 outputs read-only. Double checks are asserted inline.
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

import phase4 as P
from ob_stats import Sorted, cohort_ids, holm, tau
from r5_stats import crossover, friedman_nemenyi, interaction

OD = P.OB / "results" / "phase5"
OD.mkdir(parents=True, exist_ok=True)
E3 = P.OB / "results" / "e3"
DOSE = ["dose_ramp", "dose_intermittent"]
B = len(P.W)
SIDX = np.random.default_rng(1).integers(0, 5, (B, 5))           # two-level: seed draws, shared by all detectors
CONTRASTS = [("mpnn__MPNN", "gat__GAT"), ("proto__HGB", "proto__MLPpub"), ("edgefeatraw__HGB", "edgefeat__HGB"),
             ("mpnn__MPNN", "edgefeatraw__HGB")]


def test_y(g):
    return np.load(P.CACHE / f"s7_{g}_test.npz")["y"]


def e3_path(g, m):
    if m in ("rule__RULE", "proto__MLPpub"):
        return P.fixed_path(g, m)
    return E3 / g / f"retuned_{m}_preds.npy"


def perseed_job(args):
    key, g, preds_file, member = args
    y = test_y(g)
    Pm = np.load(preds_file)[member] if member else np.load(preds_file)
    S = [Sorted(p, y, cohort_ids(len(y))) for p in Pm]
    full = np.array([s.ap() for s in S])
    reps = np.array([[s.ap(w) for w in P.W] for s in S])            # (5, B)
    return key, full, reps


def jobs(which):
    if which == "retuned":
        return [(f"{g}|{m}", g, str(P.retuned_path(g, m)), None) for g in P.GENS for m in P.RET7]
    if which == "transfer":
        return [(f"{g}|{m}", g, str(P.OUT / "stage7" / "main" / f"{m}.npz"), g) for g in P.GENS for m in P.FIXED6]
    if which == "e3":
        return [(f"{g}|{m}", g, str(e3_path(g, m)), None) for g in DOSE for m in P.RET7] + \
               [(f"fixed|{g}|{m}", g, str(P.fixed_path(g, m)), None) for g in DOSE for m in P.FIXED6] + \
               [(f"tr|{g}|{m}", g, str(E3 / "transfer" / f"{m}.npz"), g) for g in DOSE for m in P.FIXED6]
    raise ValueError(which)


def perseed(which, workers):
    t0 = time.time()
    full, reps = {}, {}
    with ProcessPoolExecutor(workers) as ex:
        for key, f, r in ex.map(perseed_job, jobs(which)):
            full[key], reps[key] = f, r
            print(which, key, np.round(f, 3), f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(OD / f"perseed_{which}.npz", **{f"full|{k}": v for k, v in full.items()},
                        **{f"reps|{k}": v for k, v in reps.items()})


def load_perseed(which):
    z = np.load(OD / f"perseed_{which}.npz")
    full = {k[5:]: z[k] for k in z.files if k.startswith("full|")}
    reps = {k[5:]: z[k] for k in z.files if k.startswith("reps|")}
    return full, reps


def two_level(reps_k):
    """reps_k: (5, B) per-seed replicates -> (B,) two-level replicate values."""
    return reps_k[SIDX, np.arange(B)[:, None]].mean(1)


def ci(x):
    return np.percentile(x, [2.5, 97.5]).tolist()


def interaction_family(points, reps, dets, gens, reps2=None):
    rows = []
    for a, b in combinations(dets, 2):
        d = np.array([points[g][a] - points[g][b] for g in gens])
        D = np.stack([reps[g][a] - reps[g][b] for g in gens], 1)
        r = {"a": a, "b": b, **interaction(d, D), "diffs": d.tolist()}
        D2 = np.stack([reps2[g][a] - reps2[g][b] for g in gens], 1) if reps2 is not None else None
        if reps2 is not None:
            r2 = interaction(d, D2)
            r["two_level"] = {"Q": r2["Q"], "p": r2["p"], "p_boot": r2["p_boot"], "cond": r2["cond"]}
        r["crossover"] = crossover(d, D)                                    # deviation D5
        if reps2 is not None:
            r["crossover_two_level"] = crossover(d, D2)
        rows.append(r)
    for r, adj in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = float(adj)
    for r, adj in zip(rows, holm([r["crossover"]["p"] for r in rows])):
        r["crossover"]["p_holm"] = float(adj)
    if reps2 is not None:
        for r, adj in zip(rows, holm([r["crossover_two_level"]["p"] for r in rows])):
            r["crossover_two_level"]["p_holm"] = float(adj)
    if reps2 is not None:
        for r, adj in zip(rows, holm([r["two_level"]["p"] for r in rows])):
            r["two_level"]["p_holm"] = float(adj)
    return {"G": len(gens), "n_pairs": len(rows), "n_sig_holm": int(sum(r["p_holm"] < 0.05 for r in rows)),
            "n_crossover_holm": int(sum(r["crossover"]["p_holm"] < 0.05 for r in rows)),
            "n_crossover_holm_two_level": int(sum(r["crossover_two_level"]["p_holm"] < 0.05 for r in rows)) if reps2 is not None else None,
            "gens": list(gens),
            "n_sig_holm_two_level": int(sum(r["two_level"]["p_holm"] < 0.05 for r in rows)) if reps2 is not None else None,
            "rows": rows}


def rank_agreement(points, reps, dets):
    """Median over replicates of tau(point ranking, replicate ranking)."""
    pt = [points[m] for m in dets]
    return float(np.median([tau(pt, [reps[m][b] for m in dets]) for b in range(B)]))


def analyse():
    R = json.loads((P.OD / "phase4.json").read_text())
    facts = json.loads((OD / "facts.json").read_text())
    saved_ret = np.load(P.OD / "boot_retuned.npz")
    saved_fx = np.load(P.OD / "boot_fixed_e2.npz")
    full, preps = load_perseed("retuned")
    out = {"B": B, "plan": "ANALYSIS_PLAN_v13_REVIEW.md"}

    # ---- double checks against phase4 outputs
    pts = {g: {m: float(full[f"{g}|{m}"].mean()) for m in P.RET7} for g in P.GENS}
    cond = {g: {m: preps[f"{g}|{m}"].mean(0) for m in P.RET7} for g in P.GENS}
    e_pt = max(abs(pts[g][m] - R["retuned"][g][m]) for g in P.GENS for m in P.RET7)
    e_bs = max(float(np.abs(cond[g][m] - saved_ret[f"{g}|{m}"]).max()) for g in P.GENS for m in P.RET7)
    assert e_pt < 1e-12 and e_bs < 1e-12, (e_pt, e_bs)
    two = {g: {m: two_level(preps[f"{g}|{m}"]) for m in P.RET7} for g in P.GENS}
    out["checks"] = {"point_vs_phase4_max_abs_err": e_pt, "replicates_vs_boot_retuned_max_abs_err": e_bs}

    # ---- A1 / A2: interaction
    out["A1_retuned"] = interaction_family(pts, cond, P.RET7, P.GENS, reps2=two)
    fx_pts = {g: {m: R["fixed"][g][m] for m in P.FIXED6} for g in P.GENS}
    fx_reps = {g: {m: saved_fx[f"fixed|{g}|{m}"] for m in P.FIXED6} for g in P.GENS}
    out["A2_fixed"] = interaction_family(fx_pts, fx_reps, P.FIXED6, P.GENS)

    # ---- B: Friedman / Nemenyi (descriptive)
    X = np.array([[pts[g][m] for m in P.RET7] for g in P.GENS])
    out["B_friedman"] = {"detectors": P.RET7, **friedman_nemenyi(X)}

    # ---- C: seed variability
    C = {"per_seed": {}, "pairs": {}}
    for g in P.GENS:
        C["per_seed"][g] = {m: full[f"{g}|{m}"].tolist() for m in ("mpnn__MPNN", "edgefeatraw__HGB", "gat__GAT")}
    for a, b in combinations(P.RET7, 2):
        rows = {}
        for g in P.GENS:
            ds = full[f"{g}|{a}"] - full[f"{g}|{b}"]
            dc, d2 = cond[g][a] - cond[g][b], two[g][a] - two[g][b]
            v_seed = float(np.var(ds, ddof=1) / 5)
            rows[g] = {"diff": float(ds.mean()), "sd_across_seeds": float(np.std(ds, ddof=1)),
                       "sd_cohort_boot": float(np.std(dc, ddof=1)), "sd_two_level": float(np.std(d2, ddof=1)),
                       "var_seed_mean": v_seed, "var_cohort": float(np.var(dc, ddof=1)), "var_two_level": float(np.var(d2, ddof=1)),
                       "seed_share": v_seed / float(np.var(d2, ddof=1)) if np.var(d2) > 0 else None,
                       "ci_cohort": ci(dc), "ci_two_level": ci(d2),
                       "two_level_excludes_0": bool(np.percentile(d2, 2.5) > 0 or np.percentile(d2, 97.5) < 0),
                       "cohort_excludes_0": bool(np.percentile(dc, 2.5) > 0 or np.percentile(dc, 97.5) < 0)}
        C["pairs"][f"{a}|{b}"] = rows
    top = C["pairs"]["mpnn__MPNN|edgefeatraw__HGB"]
    C["C4_rule"] = {"n_gens_seed_share_gt_half": int(sum(r["seed_share"] > 0.5 for r in top.values())),
                    "threshold": 7, "met": bool(sum(r["seed_share"] > 0.5 for r in top.values()) >= 7)}
    C["n_excl0"] = {k: {"cohort": int(sum(r["cohort_excludes_0"] for r in v.values())),
                        "two_level": int(sum(r["two_level_excludes_0"] for r in v.values()))} for k, v in C["pairs"].items()}
    out["C_seed"] = C

    # ---- D: tau uncertainty
    D = {"granularity": {"6": 2 / 15, "7": 2 / 21}}
    for key in ("tau_retuned", "tau_fixed"):
        prs = R[key]["pairs"]
        D[key] = {"median": R[key]["median"], "n_pairs": len(prs),
                  "ci_width_median": float(np.median([p["ci"][1] - p["ci"][0] for p in prs])),
                  "n_ci_includes_1": int(sum(p["ci"][1] >= 1 - 1e-12 for p in prs)),
                  "n_ci_below_0_upper": int(sum(p["ci"][1] < 0 for p in prs)),
                  "median_ci": [float(np.median([p["ci"][0] for p in prs])), float(np.median([p["ci"][1] for p in prs]))]}
    tfull, treps = load_perseed("transfer")
    tr_pts = {g: {m: float(tfull[f"{g}|{m}"].mean()) for m in P.FIXED6} for g in P.GENS}
    tr_reps = {g: {m: treps[f"{g}|{m}"].mean(0) for m in P.FIXED6} for g in P.GENS}
    e_tr = max(abs(tr_pts[g][m] - R["transfer"]["transfer_pr"][g][m]) for g in P.GENS for m in P.FIXED6)
    assert e_tr < 1e-12, e_tr
    out["checks"]["transfer_point_vs_phase4_max_abs_err"] = e_tr
    D["transfer_tau"] = {}
    for g in P.GENS:
        bt = [tau([fx_reps[g][m][b] for m in P.FIXED6], [tr_reps[g][m][b] for m in P.FIXED6]) for b in range(B)]
        t0_ = tau([fx_pts[g][m] for m in P.FIXED6], [tr_pts[g][m] for m in P.FIXED6])
        assert abs(t0_ - R["transfer"]["tau_within_vs_transfer"][g]) < 1e-12
        D["transfer_tau"][g] = {"tau": t0_, "ci": ci(np.array(bt))}
    D["resampling_agreement"] = {g: {"retuned7": rank_agreement(pts[g], cond[g], P.RET7),
                                     "fixed6": rank_agreement(fx_pts[g], fx_reps[g], P.FIXED6),
                                     "transfer6": rank_agreement(tr_pts[g], tr_reps[g], P.FIXED6)} for g in P.GENS}
    for k in ("retuned7", "fixed6", "transfer6"):
        D[f"resampling_agreement_median_{k}"] = float(np.median([v[k] for v in D["resampling_agreement"].values()]))
    D["e2_vs_family"] = {}
    for v in ("pair", "mix"):
        e2p = [R["e2"][v]["retuned"][m] for m in P.RET7]
        rows = {}
        for g in P.GENS:
            bt = [tau([saved_fx[f"e2|{v}|{m}"][b] for m in P.RET7], [cond[g][m][b] for m in P.RET7]) for b in range(B)]
            rows[g] = {"tau": tau(e2p, [pts[g][m] for m in P.RET7]), "ci": ci(np.array(bt))}
        D["e2_vs_family"][v] = rows
    out["D_tau"] = D

    # ---- E: transfer reworked
    E = {"rows": [], "training_positives": {"v2_main": 1539}}
    for g in P.GENS:
        s7 = json.loads((P.OUT / "stage7" / g / "summary.json").read_text())
        ceil = s7["ceiling"]["pr_auc"]
        E["training_positives"][g] = int(np.load(P.CACHE / f"s7_{g}_dev.npz")["y"].sum())
        for m in P.FIXED6:
            wf, wr, t = fx_pts[g][m], pts[g][m], tr_pts[g][m]
            E["rows"].append({"gen": g, "det": m, "transfer": t, "within_fixed": wf, "within_retuned": wr,
                              "ratio_fixed": t / wf, "ratio_retuned": t / wr, "loss_abs_fixed": wf - t,
                              "loss_ceiling_norm_retuned": (wr - t) / ceil, "within_frac_ceiling": wf / ceil,
                              "weak": bool(wf < 0.25 * ceil)})
    rows = E["rows"]
    E["n_cells"] = len(rows)
    E["n_ratio_fixed_gt1"] = int(sum(r["ratio_fixed"] > 1 for r in rows))
    E["n_ratio_retuned_gt1"] = int(sum(r["ratio_retuned"] > 1 for r in rows))
    E["n_weak"] = int(sum(r["weak"] for r in rows))
    E["n_ratio_fixed_gt1_weak"] = int(sum(r["ratio_fixed"] > 1 and r["weak"] for r in rows))
    E["spearman_ratio_vs_within_frac_ceiling"] = float(spearmanr([r["ratio_fixed"] for r in rows], [r["within_frac_ceiling"] for r in rows]).statistic)
    E["mpnn_ratio_retuned_le_070"] = [r["gen"] for r in rows if r["det"] == "mpnn__MPNN" and r["ratio_retuned"] <= 0.70]
    E["mpnn_ratio_fixed_le_070"] = [r["gen"] for r in rows if r["det"] == "mpnn__MPNN" and r["ratio_fixed"] <= 0.70]
    assert E["mpnn_ratio_fixed_le_070"] == R["transfer"]["mpnn_loses_30pct"]
    E["median_ratio_by_det"] = {m: {"fixed": float(np.median([r["ratio_fixed"] for r in rows if r["det"] == m])),
                                    "retuned": float(np.median([r["ratio_retuned"] for r in rows if r["det"] == m]))} for m in P.FIXED6}
    out["E_transfer"] = E

    # ---- G: controlled contrasts
    A1 = {(r["a"], r["b"]): r for r in out["A1_retuned"]["rows"]}
    G_ = {}
    for a, b in CONTRASTS:
        key = (a, b) if (a, b) in A1 else (b, a)
        r, sign = A1[key], (1 if key == (a, b) else -1)
        cp = C["pairs"][f"{key[0]}|{key[1]}"]
        d = sign * np.array([cp[g]["diff"] for g in P.GENS])
        G_[f"{a} - {b}"] = {"mean": float(d.mean()), "min": float(d.min()), "max": float(d.max()),
                            "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()),
                            "n_two_level_excl0": int(sum(cp[g]["two_level_excludes_0"] for g in P.GENS)),
                            "n_cohort_excl0": int(sum(cp[g]["cohort_excludes_0"] for g in P.GENS)),
                            "interaction_Q": r["Q"], "interaction_p_holm": r["p_holm"],
                            "interaction_two_level_p_holm": r["two_level"]["p_holm"]}
    out["G_contrasts"] = G_

    # ---- H: severity-output-only reference (no training)
    H = {}
    corp = {g: P.CACHE / f"s7_{g}_test.npz" for g in P.GENS + DOSE}
    corp.update({f"e2_{v}": P.OB / "cache" / f"e2_{v}_test.npz" for v in ("pair", "mix")})
    for g, f in corp.items():
        z = np.load(f)
        s = np.exp(z["V_bb"][:, 2]) + np.exp(z["V_bb"][:, 3])
        H[g] = {"pr_auc": Sorted(s, z["y"], cohort_ids(len(s))).ap(), "prevalence": float(z["y"].mean())}
    out["H_severity_only"] = H
    out["I_counts"] = facts["Q3"]

    # ---- E3 (if available)
    if (OD / "perseed_e3.npz").exists():
        out["E3"] = analyse_e3(pts, cond, two, full, R)
    (OD / "review.json").write_text(json.dumps(out, indent=1))
    summarise(out)


def analyse_e3(pts, cond, two, full, R):
    f3, r3 = load_perseed("e3")
    e = {"retuned": {}, "retuned_ci": {}, "fixed": {}, "transfer": {}}
    p3 = {g: {m: float(f3[f"{g}|{m}"].mean()) for m in P.RET7} for g in DOSE}
    c3 = {g: {m: r3[f"{g}|{m}"].mean(0) for m in P.RET7} for g in DOSE}
    t3 = {g: {m: two_level(r3[f"{g}|{m}"]) for m in P.RET7} for g in DOSE}
    for g in DOSE:
        e["retuned"][g] = p3[g]
        e["retuned_ci"][g] = {m: ci(c3[g][m]) for m in P.RET7}
        e["fixed"][g] = {m: float(f3[f"fixed|{g}|{m}"].mean()) for m in P.FIXED6}
        e["transfer"][g] = {m: float(f3[f"tr|{g}|{m}"].mean()) for m in P.FIXED6}
        s7 = json.loads((P.OUT / "stage7" / g / "summary.json").read_text())
        e.setdefault("ceiling", {})[g] = s7["ceiling"]["pr_auc"]
        e.setdefault("prevalence", {})[g] = s7["prevalence_test"]
        e.setdefault("temporal_variants_fixed", {})[g] = {m: v["pr_auc"][0] for m, v in s7["models"].items() if "_cm" in m or "_ema" in m}
        e.setdefault("winner", {})[g] = max(P.RET7, key=lambda m: p3[g][m])
        e.setdefault("ranking", {})[g] = sorted(P.RET7, key=lambda m: -p3[g][m])
        e.setdefault("tau_vs_e1", {})[g] = {}
        for h in P.GENS:
            bt = [tau([c3[g][m][b] for m in P.RET7], [cond[h][m][b] for m in P.RET7]) for b in range(B)]
            e["tau_vs_e1"][g][h] = {"tau": tau([p3[g][m] for m in P.RET7], [pts[h][m] for m in P.RET7]), "ci": ci(np.array(bt))}
        tv = [v["tau"] for v in e["tau_vs_e1"][g].values()]
        e.setdefault("tau_vs_e1_summary", {})[g] = {"median": float(np.median(tv)), "min": float(min(tv)), "max": float(max(tv))}
        # fixed-config check against plan-v7 summary
        for m in P.FIXED6:
            assert abs(e["fixed"][g][m] - s7["models"][m]["pr_auc"][0]) < 0.01, (g, m)
    # max-|t| over 2 x 21 comparisons (deviation D2 method)
    import phase4_d2 as D2
    blk = {g: {m: (p3[g][m], c3[g][m]) for m in P.RET7} for g in DOSE}
    rows, c = D2.max_t(blk, P.RET7, DOSE)
    e["maxt"] = D2.summary(rows, c, blk, P.RET7, DOSE)
    # E1 points vs E3: flips relative to E1 majority order
    allg = {**{g: pts[g] for g in P.GENS}, **p3}
    e["flips_15"] = D2.flips({g: {m: (allg[g][m], None) for m in P.RET7} for g in allg}, P.RET7, list(allg))
    # A3: interaction over 15 generators
    pts15 = {**pts, **p3}
    cond15 = {**cond, **c3}
    two15 = {**two, **t3}
    e["A3_interaction_15"] = interaction_family(pts15, cond15, P.RET7, P.GENS + DOSE, reps2=two15)
    # C on E3 for the top pair and MPNN-GAT
    e["C_seed"] = {}
    for a, b in (("mpnn__MPNN", "edgefeatraw__HGB"), ("mpnn__MPNN", "gat__GAT")):
        for g in DOSE:
            ds = f3[f"{g}|{a}"] - f3[f"{g}|{b}"]
            d2 = t3[g][a] - t3[g][b]
            dc = c3[g][a] - c3[g][b]
            e["C_seed"][f"{a}|{b}|{g}"] = {"diff": float(ds.mean()), "per_seed": ds.tolist(), "ci_cohort": ci(dc), "ci_two_level": ci(d2)}
    return e


def summarise(o):
    A1 = o["A1_retuned"]
    print("A1 sig (Holm, chi2):", A1["n_sig_holm"], "/", A1["n_pairs"], " two-level:", A1["n_sig_holm_two_level"])
    for r in sorted(A1["rows"], key=lambda r: r["p"]):
        print(f"  {r['a'][:10]:10s} {r['b'][:16]:16s} Q {r['Q']:7.1f} p {r['p']:.2e} holm {r['p_holm']:.3g} pboot {r['p_boot']:.3g} "
              f"2L holm {r['two_level']['p_holm']:.3g} cond {r['cond']:.1e} signs +{r['n_pos']}/-{r['n_neg']}")
    print("A1 crossover (D5) sig:", A1["n_crossover_holm"], "two-level:", A1["n_crossover_holm_two_level"])
    for r in sorted(A1["rows"], key=lambda r: r["crossover"]["p"])[:8]:
        print(f"  D5 {r['a'][:10]:10s} {r['b'][:16]:16s} p+ {r['crossover']['p_plus']:.3f} p- {r['crossover']['p_minus']:.3f} holm {r['crossover']['p_holm']:.3f} "
              f"2L holm {r['crossover_two_level']['p_holm']:.3f} zmax {r['crossover']['z_max']:.2f} zmin {r['crossover']['z_min']:.2f}")
    print("A2 sig:", o["A2_fixed"]["n_sig_holm"], "/", o["A2_fixed"]["n_pairs"], "crossover:", o["A2_fixed"]["n_crossover_holm"])
    print("B:", {k: o["B_friedman"][k] for k in ("avg_rank", "chi2", "p_chi2", "F", "p_F", "CD")})
    print("C4:", o["C_seed"]["C4_rule"])
    for g, r in o["C_seed"]["pairs"]["mpnn__MPNN|edgefeatraw__HGB"].items():
        print(f"  {g:8s} diff {r['diff']:+.3f} sd_seed {r['sd_across_seeds']:.3f} sd_boot {r['sd_cohort_boot']:.3f} sd_2L {r['sd_two_level']:.3f} "
              f"share {r['seed_share']:.2f} ci2L {np.round(r['ci_two_level'], 3)}")
    print("D:", {k: v for k, v in o["D_tau"].items() if k.startswith("resampling_agreement_median") or k in ("tau_retuned", "tau_fixed")})
    print("D transfer tau:", {g: (round(v["tau"], 2), np.round(v["ci"], 2).tolist()) for g, v in o["D_tau"]["transfer_tau"].items()})
    E = o["E_transfer"]
    print("E:", {k: E[k] for k in ("n_cells", "n_ratio_fixed_gt1", "n_ratio_retuned_gt1", "n_weak", "n_ratio_fixed_gt1_weak",
                                    "spearman_ratio_vs_within_frac_ceiling", "mpnn_ratio_retuned_le_070", "mpnn_ratio_fixed_le_070")})
    print("G:", json.dumps(o["G_contrasts"], indent=1))
    print("H:", {g: round(v["pr_auc"], 3) for g, v in o["H_severity_only"].items()})
    if "E3" in o:
        e = o["E3"]
        print("E3 retuned:", {g: {m: round(v, 3) for m, v in d.items()} for g, d in e["retuned"].items()})
        print("E3 ranking:", e["ranking"], "tau vs E1:", e["tau_vs_e1_summary"])
        print("E3 maxt sig", e["maxt"]["n_sig_maxt"], "reversals", len(e["maxt"]["reversals_maxt"]), "flips", len(e["maxt"]["point_estimate_flips"]))
        print("A3 sig", e["A3_interaction_15"]["n_sig_holm"], "two-level", e["A3_interaction_15"]["n_sig_holm_two_level"])


if __name__ == "__main__":
    if sys.argv[1] == "perseed":
        for w in sys.argv[3:] if len(sys.argv) > 3 else ("retuned", "transfer"):
            perseed(w, int(sys.argv[2]))
    else:
        analyse()
