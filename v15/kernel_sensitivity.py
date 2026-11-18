"""
Post-hoc sensitivity analysis of psi_G to the distance-kernel family of the prior (NOT pre-registered; labelled
exploratory in the paper; recorded in redesign_v15/DEVIATIONS_v15.md). It changes no pre-registered estimate and no
decision.

Why: the exploratory subgroups (plan 12.1) showed that the distance kernel nearly decides the winner (power-law kernel:
Raw-HGB wins almost always; cut-off kernel: the MPNN wins most of the time). psi_G is defined over the prior, and the
prior draws the kernel family with probability 1/2 each. This script asks how much of psi_G comes from mixing the two
kernel families:

  1. the pre-registered H1 machinery (DerSimonian-Laird, pi = Phi(mu/tau), psi_G = 2 pi (1 - pi), percentile bootstrap
     over generators with B = 4000) and the model-based noise reference psi_N, applied within each kernel family;
  2. the observed counts and the split-half noise share within each family;
  3. psi_G for a prior that draws the cut-off family with probability w: P(w) = w pi_cut + (1 - w) pi_pow,
     psi_G(w) = 2 P(w) (1 - P(w)), with the two families bootstrapped independently;
  4. the kernel family of every screened draw (regenerated from the prior seed) and the share that qualified.

The functions are those of analyse_v15.aggregate; on all M generators they reproduce analysis_<stage>.json (asserted).

  python v15/kernel_sensitivity.py --stage C      -> results_v15/kernel_sensitivity_C.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_v15 import B_ACROSS, OUT, ci, dl, pi_of, reml  # noqa: E402

W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def psiN(y, v):
    """Model-based noise reference, identical to analyse_v15.aggregate.psiN."""
    mu, t2 = dl(y, v)
    shr = mu + t2 / (t2 + v) * (y - mu)
    z = np.abs(shr) / np.sqrt(v)
    return float(np.mean(2 * stats.norm.cdf(z) * stats.norm.sf(z)))


def h1(y, v, BI):
    pi = pi_of(*dl(y, v))
    pib = np.array([pi_of(*dl(y[i], v[i])) for i in BI])
    pN = psiN(y, v)
    pNb = np.array([psiN(y[i], v[i]) for i in BI])
    mu, t2 = dl(y, v)
    mu_r, t2_r = reml(y, v)
    pr = pi_of(mu_r, t2_r)
    # minority wins (the detector that wins less often) that are clear: |lambda| beyond 1.96 SD, or beyond the S6 margin
    minority = (y < 0) if (y > 0).mean() >= 0.5 else (y > 0)
    z = np.abs(y) / np.sqrt(v)
    return {"n": len(y), "mu": mu, "tau": float(np.sqrt(t2)), "pi": pi, "pi_ci95": ci(pib),
            "psi_G": 2 * pi * (1 - pi), "psi_G_ci95": ci(2 * pib * (1 - pib)),
            "psi_N_model": pN, "psi_N_model_ci95": ci(pNb),
            "psi_G_minus_psi_N": 2 * pi * (1 - pi) - pN, "psi_G_minus_psi_N_ci95": ci(2 * pib * (1 - pib) - pNb),
            "reml_pi": pr, "reml_tau": float(np.sqrt(t2_r)), "reml_psi_G": 2 * pr * (1 - pr),
            "n_mpnn_wins": int((y > 0).sum()), "share_mpnn_wins": float((y > 0).mean()),
            "observed_psi_G": float(2 * (y > 0).mean() * (1 - (y > 0).mean())),
            "n_minority_wins": int(minority.sum()),
            "n_minority_wins_beyond_1p96_sd": int((minority & (z > 1.959963984540054)).sum()),
            "n_minority_wins_beyond_margin_1p10": int((minority & (np.abs(y) > np.log(1.10))).sum()),
            "n_minority_wins_beyond_both": int((minority & (z > 1.959963984540054) & (np.abs(y) > np.log(1.10))).sum())}, pib


def run(stage):
    an = json.loads((OUT / f"analysis_{stage}.json").read_text())
    gens = an["generators"]
    S = {g: json.loads((OUT / stage / g / "summary_v15.json").read_text()) for g in gens}
    cut = np.array(["dcut" in S[g]["meta"]["spec"] for g in gens])
    assert all(("dpow" in S[g]["meta"]["spec"]) != c for g, c in zip(gens, cut)), "every generator has exactly one kernel"
    M = len(gens)
    res = {"stage": stage, "M": M, "status": "post-hoc sensitivity analysis, not pre-registered",
           "prior_share_cutoff": 0.5, "n_cutoff": int(cut.sum()), "n_power": int((~cut).sum()), "targets": {}}
    rng = np.random.default_rng(15)
    BI_all = np.random.default_rng(15).integers(0, M, (B_ACROSS, M))       # the resamples of analyse_v15.aggregate
    for t in ("U", "A"):
        y = np.array([S[g]["frame"][t]["lam"] for g in gens])
        v = np.array([S[g]["frame"][t]["v_two"] for g in gens])
        full, _ = h1(y, v, BI_all)
        if t == "U":                                                       # same functions reproduce the analysis
            assert abs(full["psi_G"] - an["frame"]["H1_U"]["psi_G"]) < 1e-9
            assert abs(full["psi_N_model"] - an["frame"]["psi_N_model"]["est"]) < 1e-9
            assert np.allclose(full["psi_G_ci95"], an["frame"]["H1_U"]["psi_G_ci95"], atol=1e-9)
        r = {"all": full}
        pib = {}
        for fam, m in (("cutoff", cut), ("power", ~cut)):
            BI = rng.integers(0, int(m.sum()), (B_ACROSS, int(m.sum())))         # resample within the family
            r[fam], pib[fam] = h1(y[m], v[m], BI)
            sh = [S[g]["frame"][t].get("split_half") for g in np.array(gens)[m]]
            sh = [h for h in sh if h]
            r[fam]["split_half_share_differs"] = float(np.mean([np.sign(a) != np.sign(b) for a, b in sh]))
            r[fam]["split_half_n"] = len(sh)
        # psi_G for a prior that draws the cut-off family with probability w
        mix = []
        for w in sorted(set(W_GRID) | {float(cut.mean())}):
            P, Pb = w * r["cutoff"]["pi"] + (1 - w) * r["power"]["pi"], w * pib["cutoff"] + (1 - w) * pib["power"]
            Pr = w * r["cutoff"]["reml_pi"] + (1 - w) * r["power"]["reml_pi"]
            mix.append({"w_cutoff": w, "P_mpnn": P, "psi_G": 2 * P * (1 - P), "psi_G_ci95": ci(2 * Pb * (1 - Pb)),
                        "reml_psi_G": 2 * Pr * (1 - Pr),
                        "observed_mix": bool(abs(w - cut.mean()) < 1e-12)})
        r["mixture"] = mix
        r["n_clear_reversals_within_families"] = r["cutoff"]["n_minority_wins_beyond_both"] + r["power"]["n_minority_wins_beyond_both"]
        r["n_reversals_within_families"] = r["cutoff"]["n_minority_wins"] + r["power"]["n_minority_wins"]
        a, b = r["cutoff"]["pi"], r["power"]["pi"]
        r["w_at_max_psi_G"] = float((0.5 - b) / (a - b)) if a != b and 0 <= (0.5 - b) / (a - b) <= 1 else None
        res["targets"][t] = r
    # kernel family of every screened draw (the prior is deterministic in k)
    import gen_v15  # noqa: F401,E402  (puts the prior's module on the path, as in screen_v15.py)
    from stage7 import sample_prior  # noqa: E402
    draws = json.loads((OUT / "screen" / "stage_c_generators.json").read_text())["draws"]
    kc = np.array(["dcut" in sample_prior(d["k"]) for d in draws])
    q = np.array([d["qualifies"] for d in draws])
    used = set(gens)
    assert all(("dcut" in sample_prior(int(g[1:]))) == c for g, c in zip(gens, cut)), "kernel of used generators"
    res["screening"] = {"n_draws": len(draws), "n_cutoff_drawn": int(kc.sum()), "n_power_drawn": int((~kc).sum()),
                        "n_cutoff_qualified": int((kc & q).sum()), "n_power_qualified": int((~kc & q).sum()),
                        "share_cutoff_qualified": float((kc & q).sum() / kc.sum()),
                        "share_power_qualified": float((~kc & q).sum() / (~kc).sum()),
                        "n_used": len(used)}
    f = OUT / f"kernel_sensitivity_{stage}.json"
    f.write_text(json.dumps(res, indent=1))
    u = res["targets"]["U"]
    print(f)
    for fam in ("all", "cutoff", "power"):
        x = u[fam]
        print(f"U {fam:6s} n={x['n']:3d} wins={x['n_mpnn_wins']:3d} pi={x['pi']:.3f} {np.round(x['pi_ci95'], 3)} "
              f"psiG={x['psi_G']:.3f} {np.round(x['psi_G_ci95'], 3)} psiN={x['psi_N_model']:.3f} "
              f"G-N={x['psi_G_minus_psi_N']:.3f} {np.round(x['psi_G_minus_psi_N_ci95'], 3)} obs={x['observed_psi_G']:.3f}"
              + (f" split={x['split_half_share_differs']:.2f}" if fam != "all" else ""))
    for m in u["mixture"]:
        print(f"  w={m['w_cutoff']:.2f} P={m['P_mpnn']:.3f} psiG={m['psi_G']:.3f} {np.round(m['psi_G_ci95'], 3)}")
    print("w at max psi_G:", u["w_at_max_psi_G"], "| screening:", res["screening"])
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["C"], required=True)
    run(ap.parse_args().stage)
