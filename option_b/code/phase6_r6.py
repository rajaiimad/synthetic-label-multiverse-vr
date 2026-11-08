"""
r6 analyses of existing predictions (ANALYSIS_PLAN_v14_R6.md: R6-A, R6-B, R6-C, R6-D; R6-E after e5_gat_sum.py).
Run with XRP_GEN=v2 from option_b/code:
  python phase6_r6.py perseed [workers]   per-seed replicates for E1 fixed-configuration and E4 matched-transfer predictions
  python phase6_r6.py analyse              -> option_b/results/phase6/r6.json
Reads predictions and earlier outputs read-only.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np

import phase4 as P
from ob_stats import Sorted, cohort_ids, holm, tau
from phase5_review import SIDX, interaction_family, load_perseed, two_level
from r5_stats import crossover

OD = P.OB / "results" / "phase6"
OD.mkdir(parents=True, exist_ok=True)
E4D = P.OB / "results" / "e4"
E5D = P.OB / "results" / "e5"
B = len(P.W)
UNDET = ["rand01", "rand08", "rand09", "rand10"]


def job(args):
    key, g, f, member = args
    y = np.load(P.CACHE / f"s7_{g}_test.npz")["y"]
    Pm = np.load(f)[member] if member else np.load(f)
    S = [Sorted(p, y, cohort_ids(len(y))) for p in Pm]
    return key, np.array([s.ap() for s in S]), np.array([[s.ap(w) for w in P.W] for s in S])


def perseed(workers):
    jobs = [(f"fixed|{g}|{m}", g, str(P.fixed_path(g, m)), None) for g in P.GENS for m in P.FIXED6] + \
           [(f"matched|{g}|{m}", g, str(E4D / f"{m}.npz"), g) for g in P.GENS for m in P.FIXED6]
    if (E5D / "d2" / "retuned_gatsum__GATSUM_preds.npy").exists():
        jobs += [(f"gatsum|{g}|gatsum", g, str(E5D / g / "retuned_gatsum__GATSUM_preds.npy"), None) for g in P.GENS
                 if (E5D / g / "retuned_gatsum__GATSUM_preds.npy").exists()]
    full, reps = {}, {}
    with ProcessPoolExecutor(workers) as ex:
        for k, a, r in ex.map(job, jobs):
            full[k], reps[k] = a, r
            print(k, np.round(a, 3), flush=True)
    name = "perseed_r6_e5.npz" if any(k.startswith("gatsum") for k in full) else "perseed_r6.npz"
    np.savez_compressed(OD / name, **{f"full|{k}": v for k, v in full.items()}, **{f"reps|{k}": v for k, v in reps.items()})


def load(name):
    z = np.load(OD / name)
    return ({k[5:]: z[k] for k in z.files if k.startswith("full|")}, {k[5:]: z[k] for k in z.files if k.startswith("reps|")})


def power(d, D, alphas=(0.05 / 21, 0.05), n=100_000, seed=11):
    """R6-B: power of the intersection-union crossover test under N(mu, R)."""
    rng = np.random.default_rng(seed)
    s = D.std(0, ddof=1)
    Rm = np.corrcoef(D, rowvar=False)
    L = np.linalg.cholesky(Rm + 1e-10 * np.eye(len(d)))
    Z0 = rng.standard_normal((n, len(d))) @ L.T
    crit = {a: float(np.quantile(Z0.max(1), 1 - a)) for a in alphas}
    med = np.median(s)
    i, j = np.argsort(np.abs(s - med))[:2]
    out = {"crit": crit, "gens_two": [int(i), int(j)], "sd_two": [float(s[i]), float(s[j])], "sd_median": float(med)}

    def pw(mu, c):
        Z = Z0 + mu
        return float(np.mean((Z.max(1) > c) & ((-Z).max(1) > c)))
    deltas = np.round(np.arange(0.01, 0.41, 0.01), 2)
    scales = np.round(np.arange(0.25, 4.01, 0.25), 2)
    for a in alphas:
        c = crit[a]
        two = {}
        for dl in deltas:
            mu = np.zeros(len(d)); mu[i] = dl / s[i]; mu[j] = -dl / s[j]
            two[float(dl)] = pw(mu, c)
        sc = {float(k): pw(k * d / s, c) for k in scales}
        mde = next((dl for dl, v in two.items() if v >= 0.8), None)
        c80 = next((k for k, v in sc.items() if v >= 0.8), None)
        out[f"alpha_{a:.5f}"] = {"two_gen": two, "scaled": sc, "mde80_two_gen": mde, "scale80": c80, "power_observed": pw(d / s, c)}
    return out


def analyse():
    R = json.loads((P.OD / "phase4.json").read_text())
    REV = json.loads((P.OB / "results" / "phase5" / "review.json").read_text())
    E4 = json.loads((E4D / "e4.json").read_text())
    f1, r1 = load_perseed("retuned")
    ft, rt = load_perseed("transfer")
    f6, r6 = load("perseed_r6.npz")
    saved_fx = np.load(P.OD / "boot_fixed_e2.npz")
    FIX6 = P.FIXED6
    out = {}
    # ---- double checks
    e_fx = max(abs(f6[f"fixed|{g}|{m}"].mean() - R["fixed"][g][m]) for g in P.GENS for m in FIX6)
    e_fxr = max(float(np.abs(r6[f"fixed|{g}|{m}"].mean(0) - saved_fx[f"fixed|{g}|{m}"]).max()) for g in P.GENS for m in FIX6)
    e_m = max(abs(f6[f"matched|{r['gen']}|{r['det']}"].mean() - r["matched"]) for r in E4["rows"])
    assert e_fx < 1e-12 and e_fxr < 1e-12 and e_m < 1e-12, (e_fx, e_fxr, e_m)
    out["checks"] = {"fixed_points": e_fx, "fixed_replicates_vs_boot_fixed_e2": e_fxr, "matched_points_vs_e4": e_m}

    # ---- R6-A: undetectable generators
    A = {}
    for g in UNDET:
        s7 = json.loads((P.OUT / "stage7" / g / "summary.json").read_text())
        y = np.load(P.CACHE / f"s7_{g}_test.npz")["y"]
        prs = {m: float(np.mean([Sorted(p, y, cohort_ids(len(y))).ap() for p in np.load(P.fixed_path(g, m))])) for m in FIX6}
        A[g] = {"ceiling": s7["ceiling"]["pr_auc"], "prevalence": float(y.mean()), "pr_auc": prs,
                "lift": {m: v / float(y.mean()) for m, v in prs.items()}}
    out["A_undetectable"] = A

    # ---- R6-B: power of the crossover test for MPNN - Raw-HGB
    d = np.array([f1[f"{g}|mpnn__MPNN"].mean() - f1[f"{g}|edgefeatraw__HGB"].mean() for g in P.GENS])
    Dc = np.stack([r1[f"{g}|mpnn__MPNN"].mean(0) - r1[f"{g}|edgefeatraw__HGB"].mean(0) for g in P.GENS], 1)
    D2 = np.stack([two_level(r1[f"{g}|mpnn__MPNN"]) - two_level(r1[f"{g}|edgefeatraw__HGB"]) for g in P.GENS], 1)
    out["B_power"] = {"cohort": power(d, Dc), "two_level": power(d, D2), "gens": P.GENS}

    # ---- R6-C: transfer baselines and matched - original
    pts_fx = {g: [f6[f"fixed|{g}|{m}"].mean() for m in FIX6] for g in P.GENS}
    pts_tr = {g: [ft[f"{g}|{m}"].mean() for m in FIX6] for g in P.GENS}
    pts_mt = {g: [f6[f"matched|{g}|{m}"].mean() for m in FIX6] for g in P.GENS}
    cond = lambda src, key: src[key].mean(0)  # noqa: E731
    rep = {"cohort": {}, "two_level": {}}
    for g in P.GENS:
        for lvl, fn in (("cohort", lambda a: a.mean(0)), ("two_level", two_level)):
            rep[lvl][g] = {"fx": np.array([fn(r6[f"fixed|{g}|{m}"]) for m in FIX6]),
                           "tr": np.array([fn(rt[f"{g}|{m}"]) for m in FIX6]),
                           "mt": np.array([fn(r6[f"matched|{g}|{m}"]) for m in FIX6])}
    C = {}
    for lvl in ("cohort", "two_level"):
        ra = {k: float(np.median([np.median([tau(pts[g], rep[lvl][g][k][:, b]) for b in range(B)]) for g in P.GENS]))
              for k, pts in (("fx", pts_fx), ("tr", pts_tr), ("mt", pts_mt))}
        diff = []
        for b in range(B):
            t_o = [tau(rep[lvl][g]["fx"][:, b], rep[lvl][g]["tr"][:, b]) for g in P.GENS]
            t_m = [tau(rep[lvl][g]["fx"][:, b], rep[lvl][g]["mt"][:, b]) for g in P.GENS]
            diff.append(np.median(t_m) - np.median(t_o))
        t_o0 = np.median([tau(pts_fx[g], pts_tr[g]) for g in P.GENS])
        t_m0 = np.median([tau(pts_fx[g], pts_mt[g]) for g in P.GENS])
        C[lvl] = {"resampling_agreement": ra, "median_tau_original": float(t_o0), "median_tau_matched": float(t_m0),
                  "diff_point": float(t_m0 - t_o0), "diff_ci": np.percentile(diff, [2.5, 97.5]).tolist(),
                  "diff_ci_includes_0": bool(np.percentile(diff, 2.5) <= 0 <= np.percentile(diff, 97.5))}
    assert abs(C["cohort"]["median_tau_original"] - np.median([REV["D_tau"]["transfer_tau"][g]["tau"] for g in P.GENS])) < 1e-12
    assert abs(C["cohort"]["median_tau_matched"] - E4["summary"]["median_tau_matched"]) < 1e-12
    out["C_transfer"] = C

    # ---- R6-D: MPNN - GAT using only seeds where the MPNN did not select the extra feature
    Dd = {}
    for g in P.GENS:
        sel = json.loads((P.OUT / "stage12" / "retune" / g / "mpnn__MPNN_selection.json").read_text())
        off = [k for k, s in enumerate(sel) if not s["cfg"].get("logd")]
        dm = f1[f"{g}|mpnn__MPNN"] - f1[f"{g}|gat__GAT"]
        Dd[g] = {"n_seeds_without": len(off), "diff_all": float(dm.mean()),
                 "diff_without": float(dm[off].mean()) if off else None,
                 "diff_with": float(np.delete(dm, off).mean()) if len(off) < 5 else None}
    have = [v for v in Dd.values() if v["diff_without"] is not None]
    out["D_mpnn_gat_feature"] = {"per_gen": Dd, "n_gens_with_seed_without": len(have),
                                 "n_selections_with_feature": int(sum(5 - v["n_seeds_without"] for v in Dd.values())),
                                 "min_diff_without": float(min(v["diff_without"] for v in have)),
                                 "n_positive_without": int(sum(v["diff_without"] > 0 for v in have))}

    # ---- R6-E: GAT-sum (if available)
    if (OD / "perseed_r6_e5.npz").exists():
        fe, re_ = load("perseed_r6_e5.npz")
        gens = [g for g in P.GENS if f"gatsum|{g}|gatsum" in fe]
        pts = {g: {"mpnn": float(f1[f"{g}|mpnn__MPNN"].mean()), "gat": float(f1[f"{g}|gat__GAT"].mean()),
                   "gatsum": float(fe[f"gatsum|{g}|gatsum"].mean())} for g in gens}
        cnd = {g: {"mpnn": r1[f"{g}|mpnn__MPNN"].mean(0), "gat": r1[f"{g}|gat__GAT"].mean(0),
                   "gatsum": re_[f"gatsum|{g}|gatsum"].mean(0)} for g in gens}
        two = {g: {"mpnn": two_level(r1[f"{g}|mpnn__MPNN"]), "gat": two_level(r1[f"{g}|gat__GAT"]),
                   "gatsum": two_level(re_[f"gatsum|{g}|gatsum"])} for g in gens}
        E = {"gens": gens, "pr_auc": pts, "ci": {g: np.percentile(cnd[g]["gatsum"], [2.5, 97.5]).tolist() for g in gens}}
        for a, b in (("mpnn", "gatsum"), ("gatsum", "gat"), ("mpnn", "gat")):
            dv = np.array([pts[g][a] - pts[g][b] for g in gens])
            E[f"{a}-{b}"] = {"diffs": dv.tolist(), "mean": float(dv.mean()), "min": float(dv.min()), "max": float(dv.max()),
                             "n_pos": int((dv > 0).sum()),
                             "ci_two_level": {g: np.percentile(two[g][a] - two[g][b], [2.5, 97.5]).tolist() for g in gens},
                             "ci_cohort": {g: np.percentile(cnd[g][a] - cnd[g][b], [2.5, 97.5]).tolist() for g in gens}}
        gap_reduction = [(pts[g]["gatsum"] - pts[g]["gat"]) / (pts[g]["mpnn"] - pts[g]["gat"]) for g in gens]
        E["median_gap_reduction"] = float(np.median(gap_reduction))
        fam = interaction_family({g: pts[g] for g in gens}, cnd, ["mpnn", "gatsum", "gat"], gens, reps2=two)
        E["interaction"] = {f"{r['a']}-{r['b']}": {"Q": r["Q"], "p": r["p"], "p_two_level": r["two_level"]["p"],
                                                   "crossover_p": r["crossover"]["p"], "crossover_p_two_level": r["crossover_two_level"]["p"]}
                            for r in fam["rows"]}
        keys = ["mpnn-gatsum", "gatsum-gat"]
        for lvl, kk in (("p", "holm"), ("crossover_p", "crossover_holm"), ("p_two_level", "holm_two_level"),
                        ("crossover_p_two_level", "crossover_holm_two_level")):
            for k, adj in zip(keys, holm([E["interaction"][k][lvl] for k in keys])):
                E["interaction"][k][kk] = float(adj)
        rank = {g: 1 + sum(R["retuned"][g][m] > pts[g]["gatsum"] for m in P.RET7) for g in gens}
        E["rank_if_added"] = rank
        out["E_gatsum"] = E
    (OD / "r6.json").write_text(json.dumps(out, indent=1, default=float))
    summarise(out)


def summarise(o):
    print("checks", o["checks"])
    for g, v in o["A_undetectable"].items():
        print("A", g, "ceil %.3f prev %.4f" % (v["ceiling"], v["prevalence"]), {m[:6]: round(x, 3) for m, x in v["pr_auc"].items()},
              "lift %.2f-%.2f" % (min(v["lift"].values()), max(v["lift"].values())))
    for lvl in ("cohort", "two_level"):
        b = o["B_power"][lvl]
        for k, v in b.items():
            if k.startswith("alpha"):
                print("B", lvl, k, "crit", round(b["crit"][float(k.split('_')[1])] if False else 0, 2), "MDE80 two-gen", v["mde80_two_gen"],
                      "scale80", v["scale80"], "power at observed", round(v["power_observed"], 3))
        print("   two generators", [o["B_power"]["gens"][i] for i in b["gens_two"]], "sd", np.round(b["sd_two"], 3), "crit", b["crit"])
    for lvl, c in o["C_transfer"].items():
        print("C", lvl, c)
    print("D", {k: v for k, v in o["D_mpnn_gat_feature"].items() if k != "per_gen"})
    if "E_gatsum" in o:
        e = o["E_gatsum"]
        print("E gatsum", {g: round(v["gatsum"], 3) for g, v in e["pr_auc"].items()})
        print("E gap reduction median", round(e["median_gap_reduction"], 3), {k: (round(e[k]["mean"], 3), e[k]["n_pos"]) for k in ("mpnn-gatsum", "gatsum-gat", "mpnn-gat")})
        print("E interaction", e["interaction"], "rank", e["rank_if_added"])


if __name__ == "__main__":
    if sys.argv[1] == "perseed":
        perseed(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    else:
        analyse()
