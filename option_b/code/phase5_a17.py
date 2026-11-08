"""
Deviation D7 (DEVIATIONS_v13.md): plan-v13 interaction (A) and crossover (D5) tests over all 17 generators
(E1 13 + E2 2 + E3 2). Run with XRP_GEN=v2 from option_b/code after phase5_review.py (perseed retuned, e3).
Writes option_b/results/phase5/perseed_e2.npz and a17.json. Reads predictions read-only.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import phase4 as P
from ob_stats import Sorted, cohort_ids
from phase5_review import DOSE, OD, interaction_family, load_perseed, two_level

E2 = ["pair", "mix"]


def job(args):
    v, m = args
    y = np.load(P.OB / "cache" / f"e2_{v}_test.npz")["y"]
    kind = "fixed" if m in ("rule__RULE", "proto__MLPpub") else "retuned"
    Pm = np.load(P.OB / "results" / "e2" / v / f"{kind}_{m}_preds.npy")
    S = [Sorted(p, y, cohort_ids(len(y))) for p in Pm]
    return f"e2_{v}|{m}", np.array([s.ap() for s in S]), np.array([[s.ap(w) for w in P.W] for s in S])


def main():
    f = OD / "perseed_e2.npz"
    if not f.exists():
        full, reps = {}, {}
        with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as ex:
            for k, a, r in ex.map(job, [(v, m) for v in E2 for m in P.RET7]):
                full[k], reps[k] = a, r
                print(k, np.round(a, 3), flush=True)
        np.savez_compressed(f, **{f"full|{k}": v for k, v in full.items()}, **{f"reps|{k}": v for k, v in reps.items()})
    f2, r2 = load_perseed("e2")
    f1, r1 = load_perseed("retuned")
    f3, r3 = load_perseed("e3")
    R = json.loads((P.OD / "phase4.json").read_text())
    saved = np.load(P.OD / "boot_fixed_e2.npz")
    # double checks: E2 points and replicates reproduce phase4.json / boot_fixed_e2.npz exactly
    e_pt = max(abs(f2[f"e2_{v}|{m}"].mean() - R["e2"][v]["retuned"][m]) for v in E2 for m in P.RET7)
    e_bs = max(float(np.abs(r2[f"e2_{v}|{m}"].mean(0) - saved[f"e2|{v}|{m}"]).max()) for v in E2 for m in P.RET7)
    assert e_pt < 1e-12 and e_bs < 1e-12, (e_pt, e_bs)
    gens = P.GENS + [f"e2_{v}" for v in E2] + DOSE
    src = {**{g: (f1, r1) for g in P.GENS}, **{f"e2_{v}": (f2, r2) for v in E2}, **{g: (f3, r3) for g in DOSE}}
    pts = {g: {m: float(src[g][0][f"{g}|{m}"].mean()) for m in P.RET7} for g in gens}
    cond = {g: {m: src[g][1][f"{g}|{m}"].mean(0) for m in P.RET7} for g in gens}
    two = {g: {m: two_level(src[g][1][f"{g}|{m}"]) for m in P.RET7} for g in gens}
    A = interaction_family(pts, cond, P.RET7, gens, reps2=two)
    A["checks"] = {"e2_point_vs_phase4": e_pt, "e2_replicates_vs_boot_fixed_e2": e_bs}
    (OD / "a17.json").write_text(json.dumps(A, indent=1))
    mh = [r for r in A["rows"] if {r["a"], r["b"]} == {"mpnn__MPNN", "edgefeatraw__HGB"}][0]
    print("A17 G", A["G"], "interaction sig", A["n_sig_holm"], "two-level", A["n_sig_holm_two_level"],
          "| crossover sig", A["n_crossover_holm"], "two-level", A["n_crossover_holm_two_level"])
    print("MPNN-RawHGB crossover p+ %.3f p- %.3f holm %.2f 2L %.2f signs +%d/-%d" % (
        mh["crossover"]["p_plus"], mh["crossover"]["p_minus"], mh["crossover"]["p_holm"], mh["crossover_two_level"]["p_holm"], mh["n_pos"], mh["n_neg"]))
    print("min other crossover holm", min(r["crossover"]["p_holm"] for r in A["rows"] if r is not mh), "checks", A["checks"])


if __name__ == "__main__":
    main()
