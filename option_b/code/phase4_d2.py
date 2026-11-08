"""
Option B, deviation D2 (plans/DEVIATIONS.md): single-step max-|t| simultaneous test per comparison family,
point-estimate reversals, and diagnostics of the bootstrap difference distributions.
Run with XRP_GEN=v2 from option_b/code after phase4.py. Reads phase4.json and boot_retuned.npz; writes
results/phase4/d2.json and boot_fixed_e2.npz. Does not modify any existing output.
"""
import json
from itertools import combinations

import numpy as np
from scipy.stats import kurtosis, norm, skew

import phase4 as P

R = json.loads((P.OD / "phase4.json").read_text())


def max_t(block, dets, gens):
    """block[g][m] = (point, bs). Returns per-comparison rows and the family critical value."""
    rows = []
    for g in gens:
        for a, b in combinations(dets, 2):
            d = block[g][a][1] - block[g][b][1]
            rows.append({"gen": g, "a": a, "b": b, "diff": block[g][a][0] - block[g][b][0], "boot": d})
    D = np.array([r["boot"] for r in rows])
    pt = np.array([r["diff"] for r in rows])
    sd = D.std(1, ddof=1)
    tmax = np.abs((D - pt[:, None]) / sd[:, None]).max(0)
    c = float(np.quantile(tmax, 0.95))
    for r, s in zip(rows, sd):
        r["z"] = float(abs(r["diff"]) / s)
        r["sig_maxt"] = bool(r["z"] > c)
        r["p_maxt_adj"] = float(np.mean(tmax >= r["z"]))          # single-step adjusted p (floored at 0 -> report < 1/B)
        r["cross0"] = float(min(np.mean(r["boot"] <= 0), np.mean(r["boot"] >= 0)))
        r["skew"], r["exkurt"] = float(skew(r["boot"])), float(kurtosis(r["boot"]))
        del r["boot"]
    return rows, c


def reversals(rows, key):
    sig = {}
    for r in rows:
        if r[key]:
            w, l_ = (r["a"], r["b"]) if r["diff"] > 0 else (r["b"], r["a"])
            sig.setdefault((w, l_), []).append(r["gen"])
    return [{"pair": [w, l_], "better_in": g, "worse_in": sig[(l_, w)]} for (w, l_), g in sig.items() if (l_, w) in sig and w < l_]


def flips(block, dets, gens):
    out = []
    for a, b in combinations(dets, 2):
        s = {g: np.sign(block[g][a][0] - block[g][b][0]) for g in gens}
        if 1 in s.values() and -1 in s.values():
            out.append({"pair": [a, b], "a_better": [g for g in gens if s[g] > 0], "b_better": [g for g in gens if s[g] < 0]})
    return out


def summary(rows, c, block, dets, gens):
    sk = np.abs([r["skew"] for r in rows])
    ku = np.array([r["exkurt"] for r in rows])
    sigrows = [r for r in rows if r["sig_maxt"]]
    return {"n": len(rows), "critical_value": c, "bonferroni_z": float(norm.isf(0.025 / len(rows))),
            "n_sig_maxt": len(sigrows), "reversals_maxt": reversals(rows, "sig_maxt"),
            "point_estimate_flips": flips(block, dets, gens), "n_pairs": len(list(combinations(dets, 2))),
            "n_cross0_zero": int(sum(r["cross0"] == 0 for r in rows)),
            "skew_abs_median": float(np.median(sk)), "skew_abs_p95": float(np.percentile(sk, 95)), "skew_abs_max": float(sk.max()),
            "exkurt_median": float(np.median(ku)), "exkurt_p95": float(np.percentile(ku, 95)),
            "min_z_sig": float(min(r["z"] for r in sigrows)) if sigrows else None,
            "rows": rows}


def main():
    ys = {g: np.load(P.CACHE / f"s7_{g}_test.npz")["y"] for g in P.GENS}
    # re-tuned replicates: recompute and check against the saved ones
    ret = P.table_block({g: {m: P.retuned_path(g, m) for m in P.RET7} for g in P.GENS}, ys)
    saved = np.load(P.OD / "boot_retuned.npz")
    err = max(float(np.abs(ret[g][m][1] - saved[f"{g}|{m}"]).max()) for g in P.GENS for m in P.RET7)
    assert err < 1e-12, err
    fixed = P.table_block({g: {m: P.fixed_path(g, m) for m in P.FIXED6} for g in P.GENS}, ys)
    from e2_group import OBC, VARIANTS
    e2 = {}
    for v in VARIANTS:
        y = np.load(OBC / f"e2_{v}_test.npz")["y"]
        dv = P.OB / "results" / "e2" / v
        paths = {m: (dv / f"fixed_{m}_preds.npy" if m in ("rule__RULE", "proto__MLPpub") else dv / f"retuned_{m}_preds.npy") for m in P.RET7}
        e2[v] = P.table_block({v: paths}, {v: y})[v]
    np.savez_compressed(P.OD / "boot_fixed_e2.npz", **{f"fixed|{g}|{m}": fixed[g][m][1] for g in P.GENS for m in P.FIXED6},
                        **{f"e2|{v}|{m}": e2[v][m][1] for v in e2 for m in P.RET7})
    # consistency with phase4.json point estimates
    perr = max(abs(fixed[g][m][0] - R["fixed"][g][m]) for g in P.GENS for m in P.FIXED6)
    perr = max(perr, max(abs(e2[v][m][0] - R["e2"][v]["retuned"][m]) for v in e2 for m in P.RET7))
    assert perr < 1e-12, perr
    out = {"replicate_check_max_abs_err": err, "point_check_max_abs_err": perr, "B": len(P.W)}
    rr, c = max_t(ret, P.RET7, P.GENS)
    out["retuned"] = summary(rr, c, ret, P.RET7, P.GENS)
    rf, c = max_t(fixed, P.FIXED6, P.GENS)
    out["fixed"] = summary(rf, c, fixed, P.FIXED6, P.GENS)
    for v in e2:
        rv, c = max_t({v: e2[v]}, P.RET7, [v])
        out[f"e2_{v}"] = summary(rv, c, {v: e2[v]}, P.RET7, [v])
    # D1 and pre-specified p-values attached for side-by-side reporting
    pw = {(q["gen"], q["a"], q["b"]): q for q in R["pairwise_retuned"]}
    for r in out["retuned"]["rows"]:
        q = pw[(r["gen"], r["a"], r["b"])]
        r.update({"p_prespec": q["p"], "p_prespec_holm": q["p_holm"], "p_d1": q["p_wald"], "p_d1_holm": q["p_wald_holm"]})
    out["retuned"]["n_prespec_at_floor"] = int(sum(r["p_prespec"] <= 1 / len(P.W) for r in out["retuned"]["rows"]))
    (P.OD / "d2.json").write_text(json.dumps(out, indent=1))
    for k in ("retuned", "fixed", "e2_pair", "e2_mix"):
        s = out[k]
        print(k, "n", s["n"], "c", round(s["critical_value"], 3), "sig", s["n_sig_maxt"], "rev", len(s["reversals_maxt"]),
              "flips", len(s["point_estimate_flips"]), "/", s["n_pairs"], "cross0==0", s["n_cross0_zero"], "skew med", round(s["skew_abs_median"], 3))
    print("D2 DONE")


if __name__ == "__main__":
    main()
