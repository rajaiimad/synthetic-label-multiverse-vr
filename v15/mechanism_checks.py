"""
POST HOC checks of the kernel effect and its proposed mechanism (NOT pre-registered; labelled post hoc in the paper;
recorded in redesign_v15/DEVIATIONS_v15.md). Existing outputs only: no training, no new data. No pre-registered
estimate is changed and no generator is dropped or down-weighted.

  a. S2 (detectors trained on the base generator v2 only) by kernel family and target: does the within-generator
     winner survive when the detectors never saw the generator's kernel?
  b. Within the power-law family, rank correlation of lambda^U with the exponent p and with |p - 1| (Raw-HGB's
     engineered features weight channels by 1/(d + 0.1), close to p = 1); within the cut-off family, with the radius c.
     Spearman's rho with a percentile bootstrap interval over generators of the family (small samples: 41 and 59).
  c. Pooled numbers of clear wins per detector (beyond 1.96 bootstrap SE and beyond the 10% margin of S6), overall and
     by family, to compare with the clear within-family reversals.
  d. Within-family observed 2p(1 - p) with a bootstrap interval, beside the split-half noise share (one evidence
     standard for the within-family claim).

  python v15/mechanism_checks.py --stage C   -> results_v15/mechanism_checks_C.json, .md
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_v15 import B_ACROSS, OUT, ci  # noqa: E402

Z = 1.959963984540054
MARGIN = float(np.log(1.10))


def cp(k, n):
    return [0.0 if k == 0 else float(stats.beta.ppf(0.025, k, n - k + 1)),
            1.0 if k == n else float(stats.beta.ppf(0.975, k + 1, n - k))]


def spearman_boot(x, y, rng):
    r = float(stats.spearmanr(x, y).statistic)
    idx = rng.integers(0, len(x), (B_ACROSS, len(x)))
    bs = [stats.spearmanr(x[i], y[i]).statistic for i in idx]
    return {"n": int(len(x)), "rho": r, "boot95": ci(bs), "p_value": float(stats.spearmanr(x, y).pvalue)}


def run(stage):
    an = json.loads((OUT / f"analysis_{stage}.json").read_text())
    gens = an["generators"]
    S = {g: json.loads((OUT / stage / g / "summary_v15.json").read_text()) for g in gens}
    spec = {g: S[g]["meta"]["spec"] if "meta" in S[g] else None for g in gens}
    if any(v is None for v in spec.values()):                              # meta is stored in meta.json for some runs
        spec = {g: json.loads((OUT / stage / g / "meta.json").read_text())["spec"] for g in gens}
    fam = {"cutoff": np.array(["dcut" in spec[g] for g in gens])}
    fam["power"] = ~fam["cutoff"]
    lam = {t: np.array([S[g]["frame"][t]["lam"] for g in gens]) for t in "UA"}
    var = {t: np.array([S[g]["frame"][t]["v_two"] for g in gens]) for t in "UA"}
    s2 = {t: np.array([S[g][f"S2{t}"]["lam"] for g in gens]) for t in "UA"}
    rng = np.random.default_rng(15)
    res = {"stage": stage, "M": len(gens), "status": "POST HOC (not pre-registered); existing outputs only"}

    # a. S2 by family and target
    a = {}
    for t in "UA":
        same = np.sign(s2[t]) == np.sign(lam[t])
        assert abs(same.mean() - an["S2_shift"][t]["share_same_winner"]) < 1e-12      # reproduces the pre-registered S2
        a[t] = {"all": {"n": len(gens), "n_same_winner": int(same.sum()), "share_same_winner_cp95": cp(int(same.sum()), len(gens)),
                        "n_mpnn_wins_within": int((lam[t] > 0).sum()), "n_mpnn_wins_S2": int((s2[t] > 0).sum())}}
        for f, m in fam.items():
            k = int(same[m].sum()); n = int(m.sum())
            a[t][f] = {"n": n, "n_same_winner": k, "share_same_winner": k / n, "share_same_winner_cp95": cp(k, n),
                       "n_mpnn_wins_within": int((lam[t][m] > 0).sum()), "n_mpnn_wins_S2": int((s2[t][m] > 0).sum()),
                       "n_mpnn_within_and_S2": int(((lam[t] > 0) & (s2[t] > 0) & m).sum()),
                       "n_rawhgb_within_and_S2": int(((lam[t] < 0) & (s2[t] < 0) & m).sum()),
                       "median_lambda_S2": float(np.median(s2[t][m])), "median_lambda_within": float(np.median(lam[t][m]))}
    res["a_S2_by_family"] = a

    # b. lambda against the kernel parameter within each family
    p = np.array([spec[g].get("dpow", np.nan) for g in gens]); c = np.array([spec[g].get("dcut", np.nan) for g in gens])
    b = {}
    for t in "UA":
        mp, mc = fam["power"], fam["cutoff"]
        b[t] = {"power_lambda_vs_p": spearman_boot(p[mp], lam[t][mp], rng),
                "power_lambda_vs_abs_p_minus_1": spearman_boot(np.abs(p[mp] - 1), lam[t][mp], rng),
                "cutoff_lambda_vs_c": spearman_boot(c[mc], lam[t][mc], rng)}
    # descriptive: lambda^U by tercile of p and of c
    def terciles(x, y):
        q = np.quantile(x, [1 / 3, 2 / 3])
        bins = [x <= q[0], (x > q[0]) & (x <= q[1]), x > q[1]]
        return [{"range": [float(x[m].min()), float(x[m].max())], "n": int(m.sum()), "median_lambda": float(np.median(y[m])),
                 "n_mpnn_wins": int((y[m] > 0).sum())} for m in bins]
    b["U_power_by_p_tercile"] = terciles(p[fam["power"]], lam["U"][fam["power"]])
    b["U_cutoff_by_c_tercile"] = terciles(c[fam["cutoff"]], lam["U"][fam["cutoff"]])
    b["p_range"] = [float(np.nanmin(p)), float(np.nanmax(p))]; b["c_range"] = [float(np.nanmin(c)), float(np.nanmax(c))]
    res["b_kernel_parameter"] = b

    # c. pooled clear wins per detector
    cc = {}
    for t in "UA":
        clear = (np.abs(lam[t]) > Z * np.sqrt(var[t])) & (np.abs(lam[t]) > MARGIN)
        row = {"all": {"n_mpnn_clear": int((clear & (lam[t] > 0)).sum()), "n_rawhgb_clear": int((clear & (lam[t] < 0)).sum()),
                       "n_not_clear": int((~clear).sum())}}
        for f, m in fam.items():
            row[f] = {"n": int(m.sum()), "n_mpnn_clear": int((clear & (lam[t] > 0) & m).sum()),
                      "n_rawhgb_clear": int((clear & (lam[t] < 0) & m).sum()), "n_not_clear": int((~clear & m).sum())}
        cc[t] = row
    res["c_clear_wins"] = cc

    # d. within-family observed 2p(1 - p) and split-half noise, one evidence standard
    d = {}
    for t in "UA":
        sh = np.array([np.sign(S[g]["frame"][t]["split_half"][0]) != np.sign(S[g]["frame"][t]["split_half"][1]) for g in gens])
        d[t] = {}
        for f, m in fam.items():
            y = lam[t][m]; n = len(y)
            idx = rng.integers(0, n, (B_ACROSS, n))
            pb = (y[idx] > 0).mean(1)
            pw = (y > 0).mean()
            k = int(sh[m].sum())
            d[t][f] = {"n": n, "observed_psi_G": float(2 * pw * (1 - pw)), "observed_psi_G_boot95": ci(2 * pb * (1 - pb)),
                       "split_half_n": k, "split_half_share": k / n, "split_half_cp95": cp(k, n)}
    res["d_within_family_observed"] = d

    (OUT / f"mechanism_checks_{stage}.json").write_text(json.dumps(res, indent=1))
    (OUT / f"mechanism_checks_{stage}.md").write_text(summary(res), encoding="utf-8")
    print(OUT / f"mechanism_checks_{stage}.json")
    return res


def summary(r):
    iv = lambda x: f"[{x[0]:.2f}, {x[1]:.2f}]".replace("-", "−")   # noqa: E731
    L = [f"# Mechanism checks (POST HOC), Stage {r['stage']}, M = {r['M']}", "",
         "Existing outputs only; no pre-registered value changed. Exact intervals for counts; bootstrap otherwise.", "",
         "## a. S2 (trained on base generator v2 only): same winner as within-generator", "",
         "| Target | Family | n | Same winner [exact 95%] | MPNN wins within | MPNN wins S2 |", "|---|---|---|---|---|---|"]
    for t in "UA":
        for f in ("all", "cutoff", "power"):
            x = r["a_S2_by_family"][t][f]
            L.append(f"| {t} | {f} | {x['n']} | {x['n_same_winner']} {iv(x['share_same_winner_cp95'])} | "
                     f"{x['n_mpnn_wins_within']} | {x['n_mpnn_wins_S2']} |")
    L += ["", "## b. λ against the kernel parameter (Spearman ρ, bootstrap 95%; small samples)", "",
          "| Target | Relation | n | ρ [boot] | p |", "|---|---|---|---|---|"]
    for t in "UA":
        for k, x in r["b_kernel_parameter"][t].items():
            L.append(f"| {t} | {k} | {x['n']} | {x['rho']:.2f} {iv(x['boot95'])} | {x['p_value']:.3f} |".replace("-", "−"))
    L += ["", "λ^U by tercile of p (power law) and of c (cut-off):", "", "| Family | Range | n | Median λ^U | MPNN wins |", "|---|---|---|---|---|"]
    for nm, k in (("power (p)", "U_power_by_p_tercile"), ("cut-off (c)", "U_cutoff_by_c_tercile")):
        for x in r["b_kernel_parameter"][k]:
            L.append(f"| {nm} | {x['range'][0]:.2f}–{x['range'][1]:.2f} | {x['n']} | {x['median_lambda']:.3f} | {x['n_mpnn_wins']} |".replace("-", "−"))
    L += ["", "## c. Clear wins (beyond 1.96 SE and the 10% margin)", "", "| Target | Family | MPNN clear | Raw-HGB clear | Not clear |", "|---|---|---|---|---|"]
    for t in "UA":
        for f in ("all", "cutoff", "power"):
            x = r["c_clear_wins"][t][f]
            L.append(f"| {t} | {f} | {x['n_mpnn_clear']} | {x['n_rawhgb_clear']} | {x['n_not_clear']} |")
    L += ["", "## d. Within-family observed 2p(1 − p) beside split-half noise", "", "| Target | Family | n | Observed 2p(1 − p) [boot] | Split-half disagreements [exact 95%] |", "|---|---|---|---|---|"]
    for t in "UA":
        for f in ("cutoff", "power"):
            x = r["d_within_family_observed"][t][f]
            L.append(f"| {t} | {f} | {x['n']} | {x['observed_psi_G']:.2f} {iv(x['observed_psi_G_boot95'])} | "
                     f"{x['split_half_n']}/{x['n']} {iv(x['split_half_cp95'])} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["C"], required=True)
    run(ap.parse_args().stage)
