"""
Plan v15 results pipeline. Presentation only: it computes no new estimate, test or interval. Every number comes from
  results_v15/analysis_<stage>.json          (analyse_v15.py aggregate)
  results_v15/<stage>/<gen>/summary_v15.json (analyse_v15.py summarise)
  results_v15/<stage>/<gen>/labels_test.npz  (train_v15.py; descriptive counts of the three label classes)
  results_v15/<stage>/<gen>/meta.json        (ceilings)
  results_v15/screen/stage_c_generators.json (Stage C only: every screened draw)
The only arithmetic here is descriptive: counts, shares, medians, and the within-generator 95% interval
lambda +- 1.96 sqrt(v_two) drawn in the figures.

  python v15/report_v15.py build --stage C
      -> results_v15/report_C/numbers.json, tables/*.md + *.csv, figures/*.pdf + *.png
  python v15/report_v15.py render IN.md OUT.md --stage C
      -> replaces {{key}} or {{key:fmt}} with values from numbers.json. Fails on an unknown key. Lists every number
         typed by hand outside a macro, for review. fmt: a Python format spec (.2f, .3f, d), or p (p-value),
         pct / pct1 (share as a percentage), ci2 / ci3 (a two-element interval as [lo, hi]).
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results_v15"
DETS = ("mpnn", "rawhgb")
NAME = {"mpnn": "MPNN", "rawhgb": "Raw-HGB"}
Z = 1.959963984540054
C_U, C_A, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"   # dataviz reference palette (light)


# =============================================================== loading
def load(stage):
    an = json.loads((RES / f"analysis_{stage}.json").read_text())
    gens = an["generators"]
    S = {g: json.loads((RES / stage / g / "summary_v15.json").read_text()) for g in gens}
    return an, gens, S


def flat(o, p=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flat(v, f"{p}.{k}" if p else str(k)))
    elif isinstance(o, list):
        if o and all(isinstance(v, (int, float)) for v in o):
            out[p] = o                       # whole interval, for {{key:ci2}}
        for i, v in enumerate(o):
            out.update(flat(v, f"{p}.{i}"))
    else:
        out[p] = o
    return out


# =============================================================== derived descriptive numbers
def three_class(stage, gens):
    """Frames per class: attack-caused (y_A), unsafe regardless of the attack (y and not y_A), safe (not y);
    prevented = safe under the attack but unsafe under the benign plan (a subset of safe)."""
    rows = []
    for g in gens:
        L = np.load(RES / stage / g / "labels_test.npz")
        y, ya, pv = L["y"].astype(bool), L["y_A"].astype(bool), L["prevented"].astype(bool)
        rows.append({"gen": g, "frames": int(len(y)), "attack_caused": int(ya.sum()), "unsafe_regardless": int((y & ~ya).sum()),
                     "safe": int((~y).sum()), "prevented": int(pv.sum()), "victim_frames": int(L["victim_of_attack"].sum())})
    return rows


def screening(gens):
    f = RES / "screen" / "stage_c_generators.json"
    if not f.exists():
        return None
    sc = json.loads(f.read_text())
    draws = sc["draws"]
    q = [d for d in draws if d["qualifies"]]
    done = [x for x in sc["generators"] if (RES / "C" / x / "summary_v15.json").exists()]
    first_m = sc["generators"][:sc["M"]]
    return {"n_draws": len(draws), "k_first": draws[0]["k"], "k_last": draws[-1]["k"], "n_qualified": len(q),
            "n_excluded": len(draws) - len(q), "share_excluded": (len(draws) - len(q)) / len(draws),
            "n_excluded_U": sum(d["ceilings"]["U"] < sc["threshold"] for d in draws),
            "n_excluded_A": sum(d["ceilings"]["A"] < sc["threshold"] for d in draws),
            "n_agg_max_screened": sum(d["agg"] == "max" for d in draws), "n_agg_max_used": sum(
                d["agg"] == "max" for d in draws if d["name"] in gens),
            "M": sc["M"], "spares": sc["spares"], "n_completed": len(done),
            "failed_in_first_M": [x for x in first_m if x not in done],
            "spares_used": [x for x in gens if x not in first_m], "threshold": sc["threshold"]}


def derived(stage, an, gens, S):
    d = {"M": len(gens)}
    for metric in ("frame", "episode"):
        lu = np.array([S[g][metric]["U"]["lam"] for g in gens])
        la = np.array([S[g][metric]["A"]["lam"] for g in gens])
        d[metric] = {"n_mpnn_wins_U": int((lu > 0).sum()), "n_mpnn_wins_A": int((la > 0).sum()),
                     "n_sign_differs_UA": int((np.sign(lu) != np.sign(la)).sum()),
                     "n_mpnn_both": int(((lu > 0) & (la > 0)).sum()), "n_rawhgb_both": int(((lu < 0) & (la < 0)).sum()),
                     "n_mpnn_U_only": int(((lu > 0) & (la < 0)).sum()), "n_mpnn_A_only": int(((lu < 0) & (la > 0)).sum()),
                     "corr_lambdaU_delta": float(np.corrcoef(lu, la - lu)[0, 1]) if len(gens) > 2 else None}
    tc = three_class(stage, gens)
    tot = {k: sum(r[k] for r in tc) for k in tc[0] if k != "gen"}
    sh = np.array([r["attack_caused"] / max(1, r["attack_caused"] + r["unsafe_regardless"]) for r in tc])
    d["three_class"] = {**tot, "share_unsafe_attack_caused_pooled": tot["attack_caused"] / max(1, tot["attack_caused"] + tot["unsafe_regardless"]),
                        "share_unsafe_attack_caused_median": float(np.median(sh)),
                        "share_unsafe_attack_caused_min": float(sh.min()), "share_unsafe_attack_caused_max": float(sh.max()),
                        "prevented_per_attack_caused_pooled": tot["prevented"] / max(1, tot["attack_caused"])}
    ce = np.array([[S[g]["meta"]["ceilings"]["U"], S[g]["meta"]["ceilings"]["A"]] for g in gens])
    pv = np.array([[S[g]["prev"]["U"], S[g]["prev"]["A"]] for g in gens])
    d["ceiling_U_median"], d["ceiling_A_median"] = float(np.median(ce[:, 0])), float(np.median(ce[:, 1]))
    d["prev_U_median"], d["prev_A_median"] = float(np.median(pv[:, 0])), float(np.median(pv[:, 1]))
    # plan 12.1: prior components as exploratory subgroups (descriptive counts only, no test)
    lu = np.array([S[g]["frame"]["U"]["lam"] for g in gens]); la = np.array([S[g]["frame"]["A"]["lam"] for g in gens])
    spec = [S[g]["meta"]["spec"] for g in gens]
    groups = {**{f"agg_{a}": np.array([s.get("agg", "sum") == a for s in spec]) for a in ("sum", "max", "lse")},
              "kernel_power": np.array(["dpow" in s for s in spec]), "kernel_cutoff": np.array(["dcut" in s for s in spec])}
    d["subgroups"] = {k: {"n": int(m.sum()), "n_mpnn_wins_U": int((lu[m] > 0).sum()), "n_mpnn_wins_A": int((la[m] > 0).sum()),
                          "median_lambda_U": float(np.median(lu[m])) if m.any() else None,
                          "n_target_flips": int((np.sign(lu[m]) != np.sign(la[m])).sum())} for k, m in groups.items()}
    d["psi_G_over_psi_N"] = an["frame"]["H1_U"]["psi_G"] / an["frame"]["psi_N_model"]["est"]
    d["df"] = len(gens) - 1                                            # H2 t-test degrees of freedom
    d["bound_pi_upper"] = 1 - an["frame"]["checks"]["bounds_share_rawhgb_sig"]   # plan 8.5: k-/M bounds 1 - pi_U
    lo, hi = an["frame"]["H2"]["ci975"]                                # the H2 bound as a multiplicative change
    d["h2_ratio_lo"], d["h2_ratio_hi"] = float(np.exp(lo)), float(np.exp(hi))
    sc = screening(gens) if stage == "C" else None
    if sc:
        d["screening"] = sc
    # post hoc (model adequacy): how the inverse-variance weights of the DL fit are spread over the generators
    from scipy import stats
    if len(gens) <= 20:                                                # Stage X: too few generators for this description
        return d, tc
    vu = np.array([S[g]["frame"]["U"]["v_two"] for g in gens])
    w = 1.0 / vu
    top = np.argsort(-w)[:20]
    rest = np.setdiff1d(np.arange(len(gens)), top)
    d["dl_weights"] = {"n_top": 20, "weight_share_top": float(w[top].sum() / w.sum()),
                       "median_abs_lambda_top": float(np.median(np.abs(lu[top]))),
                       "median_abs_lambda_rest": float(np.median(np.abs(lu[rest]))),
                       "spearman_v_abs_lambda": float(stats.spearmanr(vu, np.abs(lu)).statistic)}
    return d, tc


# =============================================================== tables
def fmt(v, spec=".2f"):
    if v is None:
        return "–"
    if spec == "p":
        return "< 0.001" if v < 0.001 else f"{v:.3f}"
    if spec == "pe":                                                   # for running text: "p {{x:pe}}"
        return "< 0.001" if v < 0.001 else f"= {v:.3f}"
    if spec in ("pct", "pct1"):
        n = 0 if spec == "pct" else 1
        return f"{100 * v:.{n + 1 if 0 < abs(v) < 0.01 else n}f}%"
    if spec == "x1":                                                   # 2.4 × 10^−7^ (pandoc superscript)
        e = int(np.floor(np.log10(abs(v)))) if v else 0
        return f"{v / 10 ** e:.1f} × 10^{e}^".replace("-", "−")
    if spec in ("r2", "r3"):                                          # for abstracts: 0.36–0.50, or −0.18 to 0.12
        n = int(spec[-1])
        sep = " to " if v[0] < 0 else "–"
        return f"{v[0]:.{n}f}{sep}{v[1]:.{n}f}".replace("-", "−")
    if spec in ("ci2", "ci3"):
        n = int(spec[-1])
        return f"[{v[0]:.{n}f}, {v[1]:.{n}f}]".replace("-", "−")
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int) and spec in ("", "d"):
        return str(v)
    return format(v, spec).replace("-", "−")


def write_table(out, name, header, rows, caption):
    (out / f"{name}.csv").write_text("\n".join(",".join(f'"{c}"' for c in r) for r in [header] + rows) + "\n", encoding="utf-8")
    md = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    (out / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def g(an, path, default=None):
    o = an
    for k in path.split("."):
        if not isinstance(o, dict) or k not in o:
            return default
        o = o[k]
    return o


def tables_mechanism(an, out):
    """Tables of v15/mechanism_checks.py (POST HOC): the kernel effect, S2 by family, clear wins."""
    X = an["mech"]
    fam = {"all": "Both", "cutoff": "Cut-off", "power": "Power law"}
    rows = []
    for t in "UA":
        for f in ("all", "cutoff", "power"):
            x = X["a_S2_by_family"][t][f]
            rows.append([t, fam[f], x["n"], f"{x['n_same_winner']} {fmt(x['share_same_winner_cp95'], 'ci2')}",
                         x["n_mpnn_wins_within"], x["n_mpnn_wins_S2"]])
    write_table(out, "T17_mech_S2", ["Target", "Kernel family", "Generators", "Same winner as within-generator [exact 95%]",
                                     "MPNN wins, within-generator", "MPNN wins, trained on v2 only (S2)"], rows,
                "Post hoc: S2 by kernel family.")
    lab = {"power_lambda_vs_p": "Power law: λ against exponent p", "power_lambda_vs_abs_p_minus_1": "Power law: λ against the distance of p from 1",
           "cutoff_lambda_vs_c": "Cut-off: λ against radius c (m)"}
    rows = [[t, lab[k], x["n"], fmt(x["rho"], ".2f") + " " + fmt(x["boot95"], "ci2")]
            for t in "UA" for k, x in X["b_kernel_parameter"][t].items()]
    write_table(out, "T18_mech_param", ["Target", "Relation", "Generators", "Spearman ρ [bootstrap 95%]"], rows,
                "Post hoc: λ against the kernel parameter within each family.")
    rows = []
    for nm, k, unit in (("Power law", "U_power_by_p_tercile", "p"), ("Cut-off", "U_cutoff_by_c_tercile", "c")):
        for x in X["b_kernel_parameter"][k]:
            rows.append([nm, f"{unit} {x['range'][0]:.2f}–{x['range'][1]:.2f}", x["n"], fmt(x["median_lambda"], ".3f"), x["n_mpnn_wins"]])
    write_table(out, "T19_mech_terciles", ["Kernel family", "Tercile of the parameter", "Generators", "Median λ^U", "MPNN wins (U)"],
                rows, "Post hoc: λ^U by tercile of the kernel parameter.")
    rows = []
    for t in "UA":
        for f in ("all", "cutoff", "power"):
            x = X["c_clear_wins"][t][f]
            row = [t, fam[f], x["n_mpnn_clear"], x["n_rawhgb_clear"], x["n_not_clear"]]
            if f == "all":
                row += ["–", "–"]
            else:
                d = X["d_within_family_observed"][t][f]
                row += [fmt(d["observed_psi_G"], ".2f") + " " + fmt(d["observed_psi_G_boot95"], "ci2"),
                        f"{d['split_half_n']}/{d['n']} {fmt(d['split_half_cp95'], 'ci2')}"]
            rows.append(row)
    write_table(out, "T20_mech_clear", ["Target", "Kernel family", "Clear MPNN wins", "Clear Raw-HGB wins", "Not clear",
                                        "Observed 2p(1 − p) [boot]", "Split-half disagreements [exact 95%]"], rows,
                "Post hoc: clear wins and within-family reversals beside split-half noise.")


def tables_adequacy(an, out):
    """Tables of v15/model_adequacy.py (post hoc). Pre-registered values are shown unchanged beside the checks."""
    A = an["adequacy"]
    a, b, c, dd, e, f, g, h = (A[k] for k in ("a_sign_shares", "b_target_flips", "c_S6", "d_between_generator_spread",
                                              "e_heavy_tails", "f_episode_level", "g_stage_x_cross_platform", "h_t_random_effects_U"))
    M = A["M"]
    pv = lambda est, iv: fmt(est, ".2f") + " " + fmt(iv, "ci2")                                  # noqa: E731
    cnt = lambda k, iv: f"{k}/{M} {fmt(iv, 'ci2')}"                                            # noqa: E731
    t4 = h["t"]["4"]
    rows = [["π_U (share of generators on which the MPNN wins, U)", pv(a["U"]["preregistered_pi_DL"], a["U"]["preregistered_pi_DL_ci95"]),
             "Observed: " + cnt(a["U"]["n_mpnn_wins"], a["U"]["share_cp95"]) + f"; REML {fmt(a['U']['pi_REML'], '.2f')}; t (ν = 4) "
             + pv(t4["pi"], t4["pi_ci95"])],
            ["π_A (the same under A)", pv(a["A"]["preregistered_pi_DL"], a["A"]["preregistered_pi_DL_ci95"]),
             "Observed: " + cnt(a["A"]["n_mpnn_wins"], a["A"]["share_cp95"]) + f"; REML {fmt(a['A']['pi_REML'], '.2f')}"],
            ["ψ_G (generator-flip probability, U)", pv(a["U"]["preregistered_psi_G_DL"], a["U"]["preregistered_psi_G_DL_ci95"]),
             "Observed 2p(1 − p): " + pv(a["U"]["observed_psi_G"], a["U"]["observed_psi_G_boot95"]) + f"; t (ν = 4) {fmt(t4['psi_G'], '.2f')}"],
            ["Target-flip probability", "ψ_T = " + pv(b["preregistered_psi_T"], b["preregistered_psi_T_ci95"]),
             "Observed target flips: " + cnt(b["n_target_flips"], b["target_flips_cp95"])],
            ["Noise-flip probability", f"ψ_N = {fmt(b['preregistered_psi_N_model'], '.2f')}",
             "Split-half: U " + cnt(b["split_half_U"]["n"], b["split_half_U"]["cp95"]) + ", A " + cnt(b["split_half_A"]["n"], b["split_half_A"]["cp95"])],
            ["Target flips minus noise", "ψ_T − ψ_N = " + pv(b["preregistered_psi_T_minus_psi_N"]["est"], b["preregistered_psi_T_minus_psi_N"]["ci95"]),
             "Observed minus split-half: U " + pv(b["flips_minus_split_half_U"]["est"], b["flips_minus_split_half_U"]["paired_boot95"])
             + ", A " + pv(b["flips_minus_split_half_A"]["est"], b["flips_minus_split_half_A"]["paired_boot95"])],
            ["Between-generator SD of λ^U", f"τ (DL) = {dd['U']['DL']['tau']:.3f}", f"τ (REML) = {dd['U']['REML']['tau']:.3f}"],
            ["H1 decision across τ = 0.10–0.60 (π with τ fixed)", "–",
             "Same for every τ: " + ("yes" if dd["U"]["H1_decision_same_for_every_tau"] and dd["A"]["H1_decision_same_for_every_tau"] else "no")]]
    write_table(out, "T9_adequacy_overview", ["Quantity", "Pre-registered (model-based)", "Post hoc (observed or alternative model)"],
                rows, "Model adequacy: pre-registered model-based estimates beside model-free checks.")
    rows = []
    for m, x in c["margins"].items():
        rows.append([f"{float(m) - 1:.0%}", pv(x["observed_psi_G_c"], x["observed_psi_G_c_boot95"]), fmt(x["preregistered_model_psi_G_c"], ".2f"),
                     fmt(x["REML_based_psi_G_c"], ".2f"), cnt(x["observed_meaningful_target_flips"], x["observed_meaningful_target_flips_cp95"]),
                     fmt(x["preregistered_model_psi_T_c"], ".2f"), fmt(x["preregistered_model_psi_N_c"], ".2f")])
    write_table(out, "T10_adequacy_S6", ["Margin", "Observed ψ_G(c) [boot]", "Model ψ_G(c), DL", "ψ_G(c), REML-based",
                                         "Observed meaningful target flips", "Model ψ_T(c), DL", "Model ψ_N(c), DL"], rows,
                "S6 at each margin: observed against model.")
    rows = [[f"{x['tau']:.2f}", fmt(x["mu_at_tau"], ".3f"), pv(x["pi"], x["pi_ci95"]), fmt(x["psi_G"], ".2f"),
             "yes" if x["pi_ci95_inside_0.10_0.90"] else "no"] for x in dd["U"]["tau_grid"]]
    write_table(out, "T11_adequacy_tau", ["τ (fixed)", "μ at τ", "π_U [95%]", "ψ_G", "Interval inside (0.10, 0.90)"], rows,
                "π_U with the between-generator SD held fixed.")
    rows = []
    for t in "UA":
        x = e[t]
        rows += [[t, y["generator"], fmt(y["lambda"], ".2f"), f"{y['bootstrap_variance']:.3f}", y["kernel"], y["aggregation"],
                  y["n_A_positive_test_frames"]] for y in x["top5_abs_lambda"]]
    write_table(out, "T12_adequacy_tails", ["Target", "Generator", "λ", "Bootstrap variance", "Kernel", "Aggregation",
                                            "A-positive test frames"], rows,
                f"Heavy tails: the five largest |λ| per target (excess kurtosis {e['U']['excess_kurtosis']:.1f} under U and "
                f"{e['A']['excess_kurtosis']:.1f} under A).")
    rows = [[{"log": "Log ratio (pre-registered)", "raw": "Raw difference"}[k], fmt(f[k]["mean"], ".3f"), fmt(f[k]["t_p"], "p"),
             fmt(f[k]["wilcoxon_p"], "p"), f"{f[k]['n_negative']} / {f[k]['n_positive']}", fmt(f[k]["sign_test_p"], "p"),
             fmt(f[k]["loo_t_p_range"], "ci3")] for k in ("log", "raw")]
    write_table(out, "T13_adequacy_episode", ["Scale of δ", "Mean δ", "t-test p", "Wilcoxon p", "Negative / positive",
                                              "Sign-test p", "Leave-one-out range of t-test p"], rows,
                "Episode-level target shift (S1) on two scales.")
    rows = [[y["generator"], fmt(y["delta_log"], ".2f"), fmt(y["delta_raw"], ".3f"), fmt(y["mean_episode_pr_auc"], ".2f"),
             y["n_A_positive_test_frames"], fmt(y["t_p_without"], ".4f")] for y in f["most_negative_log_delta"]]
    write_table(out, "T14_adequacy_episode_top", ["Generator", "δ, log ratio", "δ, raw", "Mean episode PR-AUC",
                                                  "A-positive test frames", "t-test p without it"], rows,
                "The five most negative episode-level δ on the log scale.")
    rows = []
    for r in g["per_generator"]:
        rows.append([r["generator"], fmt(r["frame.U.lam_laptop"], ".3f"), fmt(r["frame.U.lam_github"], ".3f"),
                     fmt(r["frame.A.lam_laptop"], ".3f"), fmt(r["frame.A.lam_github"], ".3f"),
                     fmt(max(r[f"{m}.{t}.lam_diff_in_SE"] for m in ("frame", "episode") for t in "UA"), ".2f"),
                     "/".join("yes" if r[f"frame.{t}.same_winner"] else "no" for t in "UA"),
                     "/".join("yes" if r[f"episode.{t}.same_winner"] else "no" for t in "UA")])
    write_table(out, "T15_adequacy_stagex", ["Generator", "λ^U laptop", "λ^U GitHub", "λ^A laptop", "λ^A GitHub",
                                             "Largest difference in λ (in SE)", "Same frame winner (U/A)", "Same episode winner (U/A)"],
                rows, "Stage X on the GitHub runners against the laptop.")
    rows = [["Normal (ML)", fmt(h["normal_ML"]["mu"], ".3f"), fmt(h["normal_ML"]["tau"], ".3f"), fmt(h["normal_ML"]["pi"], ".2f"),
             fmt(h["normal_ML"]["psi_G"], ".2f")]]
    rows += [[f"t, ν = {nu}", fmt(x["mu"], ".3f"), fmt(x["tau"], ".3f"),
              fmt(x["pi"], ".2f") + (" " + fmt(x["pi_ci95"], "ci2") if "pi_ci95" in x else ""), fmt(x["psi_G"], ".2f")]
             for nu, x in h["t"].items()]
    write_table(out, "T16_adequacy_t", ["Random-effects distribution", "μ", "τ (scale)", "π_U", "ψ_G"], rows,
                "Side analysis: random effects with t-distributed generator effects, target U.")


def tables(stage, an, gens, S, d, tc, out):
    # T1: primary and key secondary estimands, frame level (primary) and episode level (S1)
    rows = []
    spec = [("π_U (share of generators on which MPNN wins, target U)", "H1_U.pi", "H1_U.pi_ci95", ".2f"),
            ("ψ_G (generator-flip probability, U)", "H1_U.psi_G", "H1_U.psi_G_ci95", ".2f"),
            ("μ_U (mean λ^U)", "H1_U.mu", None, ".3f"), ("τ_U (between-generator SD of λ^U)", "H1_U.tau", None, ".3f"),
            ("H1 decision (95% interval of π_U inside (0.10, 0.90))", "H1_U.reject_IUT", None, ""),
            ("π_A (target A)", "H1_A.pi", "H1_A.pi_ci95", ".2f"), ("ψ_G^A", "H1_A.psi_G", "H1_A.psi_G_ci95", ".2f"),
            ("μ_δ (mean target shift, log scale), 97.5% t-interval", "H2.mean_delta", "H2.ci975", ".3f"),
            ("exp(μ_δ) (multiplicative change of the PR-AUC ratio)", "H2.ratio_change", None, ".3f"),
            ("H2 t-test p (α = 0.025)", "H2.p", None, "p"), ("H2 decision", "H2.reject", None, ""),
            ("Standardised μ_δ / SD(δ)", "H2.standardised", None, ".2f"), ("Wilcoxon p (H2 without the normality assumption)", "H2.wilcoxon_p", None, "p"),
            ("μ_δ on the raw PR-AUC difference scale", "H2.raw_scale.mean_delta", "H2.raw_scale.ci95", ".3f"),
            ("ψ_T (target-flip probability)", "psi_T.est", "psi_T.ci95", ".2f"),
            ("Observed share with opposite signs of λ^U and λ^A", "psi_T.naive_share_sign_differs", None, ".2f"),
            ("ψ_N (noise-flip probability, model-based)", "psi_N_model.est", "psi_N_model.ci95", ".2f"),
            ("ψ_N split-half, U (observed share)", "psi_N_split_half_U.share_differs", None, ".2f"),
            ("ψ_N split-half, A (observed share)", "psi_N_split_half_A.share_differs", None, ".2f"),
            ("ψ_G − ψ_N", "psi_G_minus_psi_N.est", "psi_G_minus_psi_N.ci95", ".2f"),
            ("ψ_T − ψ_N", "psi_T_minus_psi_N.est", "psi_T_minus_psi_N.ci95", ".2f")]
    for lab, k, kci, f in spec:
        r = [lab]
        for metric in ("frame", "episode"):
            v, c = g(an, f"{metric}.{k}"), g(an, f"{metric}.{kci}") if kci else None
            r.append(fmt(v, f or ".2f") + (" " + fmt(c, "ci3" if f == ".3f" else "ci2") if c else ""))
        rows.append(r)
    write_table(out, "T1_primary", ["Estimand", "Frame level (primary)", "Episode level (S1)"], rows,
                f"Table 1. Primary hypotheses and key secondary estimands (Stage {stage}, M = {d['M']} generators). "
                "Brackets: 95% intervals unless stated.")
    # T2: secondary outcomes S2-S5, R3 and model checks (frame level)
    rows = []
    s2 = an.get("S2_shift")
    if s2:
        for t in ("U", "A"):
            for det in DETS:
                rows.append([f"S2 median retention PR(shift)/PR(within), {NAME[det]}, target {t}",
                             fmt(s2[t]["median_retention"][det]) + " " + fmt(s2[t]["median_retention_ci95"][det], "ci2")])
            rows.append([f"S2 share with the same winner as within-generator, target {t}",
                         fmt(s2[t]["share_same_winner"]) + " " + fmt(s2[t]["share_same_winner_ci95"], "ci2")])
    s3 = g(an, "frame.S3_A0")
    if s3:
        rows += [["S3 mean shift with A0 (attacker absent) instead of A", fmt(s3["mean_delta"], ".3f") + " " + fmt(s3.get("ci95_t"), "ci3")],
                 ["S3 t-test p", fmt(s3.get("p"), "p")],
                 ["S3 ψ_T with A0", fmt(g(s3, "psi_T.est")) + " " + fmt(g(s3, "psi_T.ci95"), "ci2") if g(s3, "psi_T.est") is not None else "–"]]
    s4 = g(an, "frame.S4_mismatch")
    if s4:
        rows.append(["S4 share of generators where U-trained models give the matched-A winner",
                     fmt(s4["share_same_winner_as_matched_A"]) + " " + fmt(s4.get("ci95"), "ci2")])
    r3 = an.get("R3_positive_count")
    if r3:
        rows.append([f"R3 mean change of λ^U with positives subsampled (n = {r3['n']})",
                     fmt(r3["mean_change_lambda_U"], ".3f") + " " + fmt(r3["ci95"], "ci3")])
    s5 = an.get("S5_tuning_budget")
    if s5:
        for t in ("U", "A"):
            v = s5[t]
            rows.append([f"S5 same winner with 32 configurations, target {t}", f"{v['n_same_winner']} of {s5['n']}"])
            rows.append([f"S5 mean change of λ, target {t}", fmt(v["mean_change_lambda"], ".3f") + " " + fmt(v["mean_change_lambda_ci95"], "ci3")])
            for det in DETS:
                rows.append([f"S5 mean log gain in PR-AUC, {NAME[det]}, target {t}",
                             fmt(v["mean_log_gain"][det], ".3f") + " " + fmt(v["mean_log_gain_ci95"][det], "ci3")])
            rows.append([f"S5 primary λ within 1.96 SE of zero (context), target {t}",
                         f"{v['n_primary_lambda_within_1p96_sd_of_zero']} of {s5['n']}"])
        if "sensitive_to_tuning_budget" in s5:
            rows.append(["S5 pre-specified reading: sensitive to the tuning budget", fmt(s5["sensitive_to_tuning_budget"], "")])
    s6 = an.get("S6")
    if s6:
        for k in ("1.10", "1.05", "1.25"):
            r6, tag = s6[k], f"S6 margin {k}" + (" (primary)" if k == s6["primary"] else "")
            o, m = r6["observed"], r6["model"]
            rows += [[f"{tag}: generators with MPNN / Raw-HGB better by the margin / near tie, U",
                      f"{o['U']['n_mpnn']} / {o['U']['n_rawhgb']} / {o['U']['n_near_tie']}"],
                     [f"{tag}: the same, A", f"{o['A']['n_mpnn']} / {o['A']['n_rawhgb']} / {o['A']['n_near_tie']}"],
                     [f"{tag}: target flips, all / with both gaps above the margin",
                      f"{o['n_target_flips']} / {o['n_meaningful_target_flips']}"],
                     [f"{tag}: ψ_G(c) − ψ_N(c)", fmt(m["psi_G_minus_psi_N"]["est"]) + " " + fmt(m["psi_G_minus_psi_N"]["ci95"], "ci2")],
                     [f"{tag}: ψ_T(c) − ψ_N(c)", fmt(m["psi_T_minus_psi_N"]["est"]) + " " + fmt(m["psi_T_minus_psi_N"]["ci95"], "ci2")]]
    ch = g(an, "frame.checks", {})
    rows += [["Shapiro–Wilk p of λ^U (descriptive)", fmt(ch.get("shapiro_lambda_U_p"), "p")],
             ["Leave-one-generator-out range of π̂_U", fmt(ch.get("loo_pi_range"), "ci2")],
             ["Leave-one-generator-out range of μ̂_δ", fmt(ch.get("loo_mean_delta_range"), "ci3")],
             ["Share with λ^U significantly > 0 (Holm), lower bound for π_U", fmt(ch.get("bounds_share_mpnn_sig"))],
             ["Share with λ^U significantly < 0 (Holm), lower bound for 1 − π_U", fmt(ch.get("bounds_share_rawhgb_sig"))]]
    write_table(out, "T2_secondary", ["Outcome", "Estimate [95% interval]"], rows,
                f"Table 2. Secondary outcomes and model checks, frame level (Stage {stage}).")
    # T3: per generator (supplement)
    rows = []
    for x in gens:
        s, m = S[x], S[x]["meta"]
        fu, fa = s["frame"]["U"], s["frame"]["A"]
        rows.append([x, m["spec"].get("agg", "sum"), fmt(m["ceilings"]["U"]), fmt(m["ceilings"]["A"]),
                     fmt(m.get("ceiling_A_full")), f"{100 * s['prev']['U']:.2f}", f"{100 * s['prev']['A']:.2f}",
                     fmt(fu["pr_mpnn"], ".3f"), fmt(fu["pr_rawhgb"], ".3f"), fmt(fa["pr_mpnn"], ".3f"), fmt(fa["pr_rawhgb"], ".3f"),
                     fmt(fu["lam"], ".3f") + " ± " + fmt(Z * np.sqrt(fu["v_two"]), ".3f"),
                     fmt(fa["lam"], ".3f") + " ± " + fmt(Z * np.sqrt(fa["v_two"]), ".3f"), fmt(fa["lam"] - fu["lam"], ".3f")])
    write_table(out, "T3_per_generator",
                ["Generator", "Agg.", "Ceiling U", "Ceiling A", "Ceiling A (full)", "Prev. U %", "Prev. A %", "PR U MPNN", "PR U Raw-HGB",
                 "PR A MPNN", "PR A Raw-HGB", "λ^U ± 1.96 SE", "λ^A ± 1.96 SE", "δ"], rows,
                f"Table S1. Per-generator results, frame level (Stage {stage}). SE from the two-level bootstrap.")
    # T4: three label classes
    t = d["three_class"]
    rows = [["All test frames", f"{t['frames']:,}", "100%"],
            ["Unsafe, attack-caused (target A positive)", f"{t['attack_caused']:,}", fmt(t["attack_caused"] / t["frames"], "pct1")],
            ["Unsafe regardless of the attack (U positive, A negative)", f"{t['unsafe_regardless']:,}", fmt(t["unsafe_regardless"] / t["frames"], "pct1")],
            ["Safe", f"{t['safe']:,}", fmt(t["safe"] / t["frames"], "pct1")],
            ["  of which prevented by the attack (unsafe under the benign plan)", f"{t['prevented']:,}", fmt(t["prevented"] / t["frames"], "pct1")],
            ["Share of unsafe frames that are attack-caused: pooled; median [min, max] over generators",
             fmt(t["share_unsafe_attack_caused_pooled"], "pct1"),
             fmt(t["share_unsafe_attack_caused_median"], "pct1") + f" [{fmt(t['share_unsafe_attack_caused_min'], 'pct1')}, "
             f"{fmt(t['share_unsafe_attack_caused_max'], 'pct1')}]"]]
    write_table(out, "T4_label_classes", ["Class", "Frames", "Share"], rows,
                f"Table 3. Test frames by label class, pooled over the {d['M']} generators (Stage {stage}).")
    lab = {"agg_sum": "Aggregation: sum", "agg_max": "Aggregation: maximum", "agg_lse": "Aggregation: soft maximum",
           "kernel_power": "Distance kernel: power law", "kernel_cutoff": "Distance kernel: cut-off"}
    rows = [[lab[k], v["n"], v["n_mpnn_wins_U"], v["n_mpnn_wins_A"], fmt(v["median_lambda_U"], ".3f"), v["n_target_flips"]]
            for k, v in d["subgroups"].items()]
    write_table(out, "T6_subgroups", ["Prior component", "Generators", "MPNN wins (U)", "MPNN wins (A)", "Median λ^U",
                                      "Target flips"], rows,
                f"Table S3. Exploratory subgroups by prior component (Stage {stage}; descriptive, no test).")
    if "kernel" in an:                                                  # post-hoc kernel sensitivity (exploratory)
        K = an["kernel"]["targets"]
        fam = {"all": "Both (as analysed)", "cutoff": "Cut-off only", "power": "Power law only"}
        rows = []
        for t in ("U", "A"):
            for f in ("all", "cutoff", "power"):
                x = K[t][f]
                sh = (x.get("split_half_share_differs") if f != "all"
                      else g(an, f"frame.psi_N_split_half_{t}.share_differs"))
                rows.append([t, fam[f], x["n"], x["n_mpnn_wins"],
                             "–" if f == "all" else f"{x['n_minority_wins']} ({x['n_minority_wins_beyond_both']})",
                             fmt(x["psi_G"], ".2f") + " " + fmt(x["psi_G_ci95"], "ci2"), fmt(x["reml_psi_G"], ".2f"),
                             fmt(x["psi_N_model"], ".2f") + " " + fmt(x["psi_N_model_ci95"], "ci2"), fmt(sh, ".2f")])
        write_table(out, "T7_kernel_families", ["Target", "Kernel family", "Generators", "MPNN wins",
                                                "Wins of the less frequent winner (clear)", "ψ_G, DL [95% CI]",
                                                "ψ_G, REML", "ψ_N, model [95% CI]", "ψ_N, split-half"], rows,
                    f"Table S4. Post-hoc kernel sensitivity: generator-flip probability within each distance-kernel family "
                    f"(Stage {stage}).")
        lab_w = {0.5: " (prior)", round(an["kernel"]["n_cutoff"] / an["kernel"]["M"], 6): " (analysed set)"}
        rows = []
        for mu_, ma in zip(K["U"]["mixture"], K["A"]["mixture"]):
            w = mu_["w_cutoff"]
            rows.append([f"{w:.2f}{lab_w.get(round(w, 6), '')}",
                         fmt(mu_["psi_G"], ".2f") + " " + fmt(mu_["psi_G_ci95"], "ci2"), fmt(mu_["reml_psi_G"], ".2f"),
                         fmt(ma["psi_G"], ".2f") + " " + fmt(ma["psi_G_ci95"], "ci2"), fmt(ma["reml_psi_G"], ".2f")])
        write_table(out, "T8_kernel_mixture", ["Share of cut-off kernels in the prior", "ψ_G (U), DL [95% CI]",
                                               "ψ_G (U), REML", "ψ_G (A), DL [95% CI]", "ψ_G (A), REML"], rows,
                    f"Table S5. Post-hoc: ψ_G implied for priors with other shares of the two kernel families (Stage {stage}).")
    if "adequacy" in an:                                                # post-hoc model-adequacy checks (Task 1b)
        tables_adequacy(an, out)
    if "mech" in an:                                                    # post-hoc checks of the kernel effect (P3)
        tables_mechanism(an, out)
    if "screening" in d:
        sc = d["screening"]
        rows = [["Draws screened (k = {}–{})".format(sc["k_first"], sc["k_last"]), sc["n_draws"]],
                ["Qualified (both ceilings ≥ {})".format(sc["threshold"]), sc["n_qualified"]],
                ["Excluded", f"{sc['n_excluded']} ({fmt(sc['share_excluded'], 'pct')})"],
                ["  ceiling U below threshold", sc["n_excluded_U"]], ["  ceiling A below threshold", sc["n_excluded_A"]],
                ["Generators trained (M + spares)", sc["M"] + sc["spares"]], ["Completed", sc["n_completed"]],
                ["Failed among the first M", len(sc["failed_in_first_M"])], ["Spares used", len(sc["spares_used"])],
                ["Analysed", d["M"]]]
        write_table(out, "T5_screening", ["Step", "Count"], rows, "Table S2. Screening of Stage C generators (plan 6.2, 8.6).")


# =============================================================== figures
def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight"})
    return plt


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")


def figures(stage, an, gens, S, d, out):
    plt = setup()
    from scipy import stats
    lu = np.array([S[x]["frame"]["U"]["lam"] for x in gens]); su = Z * np.sqrt([S[x]["frame"]["U"]["v_two"] for x in gens])
    la = np.array([S[x]["frame"]["A"]["lam"] for x in gens]); sa = Z * np.sqrt([S[x]["frame"]["A"]["v_two"] for x in gens])
    M = len(gens)
    # Display range: heavy-tailed lambda (generators with very few A positives) would compress every other generator.
    # Values beyond +-LIM are drawn at the edge as triangles and counted in the panel; exact values are in Table S12.
    LIM = 2.5 if np.abs(np.r_[lu, la]).max() > 2.5 else max(1.08 * np.abs(np.r_[lu, la]).max(), 0.1)
    # F1 forest plot: lambda^U and lambda^A per generator with within-generator 95% intervals, sorted by lambda^U
    o = np.argsort(lu)
    fig, axs = plt.subplots(1, 2, figsize=(7, max(2.5, 0.09 * M + 0.8)), sharey=True)
    for ax, lam, se, c, t in ((axs[0], lu, su, C_U, "U (all unsafe)"), (axs[1], la, sa, C_A, "A (attack-caused)")):
        y = np.arange(M)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.hlines(y, np.clip(lam - se, -LIM, LIM)[o], np.clip(lam + se, -LIM, LIM)[o], color=c, lw=0.8, alpha=0.6)
        inside = np.abs(lam[o]) <= LIM
        ax.plot(lam[o][inside], y[inside], "o", ms=2.6, color=c, mec="white", mew=0.4)
        for side, mk in ((-1, "<"), (1, ">")):
            off = side * lam[o] > LIM
            ax.plot(np.full(off.sum(), side * LIM), y[off], mk, ms=3.2, color=c, mec="white", mew=0.4)
        n_off = int((np.abs(lam) > LIM).sum())
        if n_off:
            ax.text(0.0, -0.06, f"{n_off} estimate(s) beyond ±{LIM:g} shown at the edge", transform=ax.transAxes,
                    ha="left", va="top", color=INK2, fontsize=6)
        ax.set_xlim(-LIM * 1.04, LIM * 1.04)
        mu, tau = g(an, f"frame.H1_{t[0]}.mu"), g(an, f"frame.H1_{t[0]}.tau")
        ax.axvspan(mu - tau, mu + tau, color=GRID, zorder=0, lw=0)
        ax.axvline(mu, color=INK, lw=0.8, ls="--")
        ax.set_title(f"Target {t}", fontsize=8, color=INK, loc="left")
        ax.set_xlabel(r"$\lambda$ = log(PR-AUC MPNN / PR-AUC Raw-HGB)")
        ax.text(1.0, 1.01, "MPNN better →", transform=ax.transAxes, ha="right", va="bottom", color=INK2, fontsize=7)
    axs[0].set_yticks(np.arange(M)); axs[0].set_yticklabels(np.array(gens)[o], fontsize=4 if M > 40 else 6)
    axs[0].set_ylabel(rf"Generator (sorted by $\lambda^U$), M = {M}")
    save(fig, out, "F1_forest"); plt.close(fig)
    # F2 lambda^U vs lambda^A: off-diagonal quadrants are generators where the target choice changes the winner
    fig, ax = plt.subplots(figsize=(3.5, 3.4))
    lim = LIM * 1.04
    ax.axhline(0, color=INK2, lw=0.6); ax.axvline(0, color=INK2, lw=0.6)
    ax.plot([-lim, lim], [-lim, lim], color=GRID, lw=0.8, zorder=0)
    inside = (np.abs(lu) <= LIM) & (np.abs(la) <= LIM)
    ax.errorbar(lu[inside], la[inside], xerr=su[inside], yerr=sa[inside], fmt="none", ecolor=GRID, elinewidth=0.6, zorder=1)
    ax.plot(lu[inside], la[inside], "o", ms=3.2, color=C_U, mec="white", mew=0.4, zorder=2)
    ax.plot(np.clip(lu[~inside], -LIM, LIM), np.clip(la[~inside], -LIM, LIM), "D", ms=3.2, color=C_U, mec="white", mew=0.4,
            zorder=2)
    if (~inside).any():
        ax.text(0.5, -0.2, f"Diamonds: {int((~inside).sum())} generator(s) beyond ±{LIM:g}, drawn at the edge",
                transform=ax.transAxes, ha="center", va="top", fontsize=6, color=INK2)
    q = d["frame"]
    for (x, y, ha, va, k) in ((0.98, 0.98, "right", "top", "n_mpnn_both"), (0.02, 0.02, "left", "bottom", "n_rawhgb_both"),
                              (0.98, 0.02, "right", "bottom", "n_mpnn_U_only"), (0.02, 0.98, "left", "top", "n_mpnn_A_only")):
        ax.text(x, y, f"n = {q[k]}", transform=ax.transAxes, ha=ha, va=va, fontsize=7, color=INK2)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel(r"$\lambda^U$ (target U)"); ax.set_ylabel(r"$\lambda^A$ (target A)")
    ax.set_title(f"Same generator, two targets (M = {M})", fontsize=8, color=INK, loc="left")
    save(fig, out, "F2_lambdaU_vs_lambdaA"); plt.close(fig)
    # F3 distribution of delta with the H2 estimate and its 97.5% t-interval
    dl_ = la - lu
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    DL = 1.5 if np.abs(dl_).max() > 1.5 else None                      # display range, as LIM above
    ax.hist(np.clip(dl_, -DL, DL) if DL else dl_, bins=np.linspace(-DL, DL, 31) if DL else min(30, max(8, M // 4)),
            color=C_A, alpha=0.85, edgecolor="white", lw=0.4)
    if DL:
        ax.text(0.0, -0.3, f"{int((dl_ < -DL).sum())} value(s) below −{DL:g} and {int((dl_ > DL).sum())} above {DL:g} "
                "are counted in the edge bins", transform=ax.transAxes, ha="left", va="top", fontsize=6, color=INK2)
    ax.axvline(0, color=INK2, lw=0.6)
    h2 = an["frame"]["H2"]
    ax.axvline(h2["mean_delta"], color=INK, lw=0.9, ls="--")
    ax.axvspan(h2["ci975"][0], h2["ci975"][1], color=GRID, zorder=0, lw=0)
    ax.set_xlabel(r"$\delta = \lambda^A - \lambda^U$"); ax.set_ylabel("Generators")
    ax.set_title(f"Target shift (mean {fmt(h2['mean_delta'], '.3f')}, 97.5% CI {fmt(h2['ci975'], 'ci3')})", fontsize=8, color=INK, loc="left")
    save(fig, out, "F3_delta"); plt.close(fig)
    # F4 flip probabilities with 95% intervals, frame (primary) and episode (S1) level
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.0), sharey=True)
    rows = [(r"$\psi_G$ (generator, U)", "H1_U.psi_G", "H1_U.psi_G_ci95"), (r"$\psi_G$ (generator, A)", "H1_A.psi_G", "H1_A.psi_G_ci95"),
            (r"$\psi_T$ (target)", "psi_T.est", "psi_T.ci95"), (r"$\psi_T$ with A0 (S3)", "S3_A0.psi_T.est", "S3_A0.psi_T.ci95"),
            (r"$\psi_N$ (noise, model)", "psi_N_model.est", "psi_N_model.ci95")]
    hi = max([0.5] + [g(an, f"{m}.{kc}")[1] for m in ("frame", "episode") for _, _, kc in rows if g(an, f"{m}.{kc}")])
    for ax, metric in zip(axs, ("frame", "episode")):
        for i, (lab, k, kc) in enumerate(rows):
            v, c = g(an, f"{metric}.{k}"), g(an, f"{metric}.{kc}")
            if v is None:
                continue
            col = INK2 if "noise" in lab else C_U
            ax.hlines(i, c[0], c[1], color=col, lw=1.0); ax.plot(v, i, "o", ms=4, color=col, mec="white", mew=0.5)
        ax.set_xlim(0, hi + 0.02); ax.axvline(0.5, color=GRID, lw=0.6)
        ax.set_title("Frame level (primary)" if metric == "frame" else "Episode level (S1)", fontsize=8, color=INK, loc="left")
        ax.set_xlabel("Probability that the winner flips")
    axs[0].set_yticks(range(len(rows))); axs[0].set_yticklabels([r[0] for r in rows]); axs[0].invert_yaxis()
    save(fig, out, "F4_flip_probabilities"); plt.close(fig)
    # F5 normal Q-Q plot of lambda^U (model check, plan 8.5)
    fig, ax = plt.subplots(figsize=(2.8, 2.6))
    (qx, qy), (sl, ic, _) = stats.probplot(lu, dist="norm")
    ax.plot(qx, qy, "o", ms=2.8, color=C_U, mec="white", mew=0.4); ax.plot(qx, sl * qx + ic, color=INK2, lw=0.8)
    ax.set_xlabel("Normal quantile"); ax.set_ylabel(r"Ordered $\lambda^U$")
    ax.set_title(f"Q-Q plot (Shapiro–Wilk p {fmt(g(an, 'frame.checks.shapiro_lambda_U_p'), 'p')})", fontsize=8, color=INK, loc="left")
    save(fig, out, "F5_qq_lambdaU"); plt.close(fig)


# =============================================================== text rendering
MACRO = re.compile(r"\{\{\s*([A-Za-z0-9_.\-]+)(?::([A-Za-z0-9.,]+))?\s*\}\}")
NUM = re.compile(r"(?<![\w.{])[-−]?\d+(?:\.\d+)?%?(?![\w}])")


def render(src, dst, numbers):
    text = Path(src).read_text(encoding="utf-8")
    missing = [m.group(1) for m in MACRO.finditer(text) if m.group(1) not in numbers]
    if missing:
        sys.exit(f"unknown keys: {sorted(set(missing))}")
    out = MACRO.sub(lambda m: fmt(numbers[m.group(1)], m.group(2) or (".2f" if isinstance(numbers[m.group(1)], float) else "")), text)
    Path(dst).write_text(out, encoding="utf-8")
    bare = [(i + 1, n.group(0)) for i, line in enumerate(MACRO.sub("", text).splitlines())
            for n in NUM.finditer(line) if "." in n.group(0) or "%" in n.group(0)]
    print(f"rendered {len(MACRO.findall(text))} macros into {dst}")
    if bare:
        print(f"{len(bare)} numbers typed by hand (check each is not a result):")
        for ln, n in bare:
            print(f"  line {ln}: {n}")


# =============================================================== main
def build(stage):
    an, gens, S = load(stage)
    out = RES / f"report_{stage}"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    d, tc = derived(stage, an, gens, S)
    s6f = RES / f"s6_meaningful_gap_{stage}.json"                     # deviation D5 (secondary outcome S6), if run
    if s6f.exists():
        an = {**an, "S6": json.loads(s6f.read_text())}
        o = an["S6"]["1.10"]["observed"]["U"]                          # descriptive analogue of psi_G(c), primary margin
        d["s6_empirical_psiG_110"] = 2 * o["n_mpnn"] * o["n_rawhgb"] / len(gens) ** 2
    dgf = RES / f"psi_T_diagnostics_{stage}.json"                     # post-hoc diagnostic (exploratory), if run
    if dgf.exists():
        an = {**an, "psiT_diag": json.loads(dgf.read_text())}
    ksf = RES / f"kernel_sensitivity_{stage}.json"                    # post-hoc kernel sensitivity (exploratory), if run
    if ksf.exists():
        an = {**an, "kernel": json.loads(ksf.read_text())}
    pvf = RES.parent / "redesign_v15" / "provenance" / "stage9_near_benign.json"   # earlier analysis quoted in Section 3
    if pvf.exists():
        an = {**an, "earlier_stage9": json.loads(pvf.read_text())}
    mcf = RES / f"mechanism_checks_{stage}.json"                      # post-hoc checks of the kernel effect, if run
    if mcf.exists():
        an = {**an, "mech": json.loads(mcf.read_text())}
    adf = RES / f"model_adequacy_{stage}.json"                        # post-hoc model-adequacy checks, if run
    if adf.exists():
        ad = json.loads(adf.read_text())
        ad["f_episode_level"] = {k: v for k, v in ad["f_episode_level"].items() if k != "per_generator"}   # in the JSON only
        wc = ad["g_stage_x_cross_platform"]["winner_changes"]
        ad["g_stage_x_cross_platform"]["n_winner_changes"] = len(wc)
        if wc:
            ad["g_stage_x_cross_platform"]["max_abs_lambda_at_winner_change"] = max(max(abs(x["lam_laptop"]), abs(x["lam_github"])) for x in wc)
            ad["g_stage_x_cross_platform"]["max_diff_in_SE_at_winner_change"] = max(x["diff_in_SE"] for x in wc)
        mx = ad["g_stage_x_cross_platform"]["max_abs_diff"]
        ad["g_stage_x_cross_platform"]["max_lam_diff_in_SE"] = max(v for k, v in mx.items() if k.endswith("lam_diff_in_SE"))
        ad["g_stage_x_cross_platform"]["max_pr_mpnn_diff_frame"] = max(mx["frame.U.pr_mpnn"], mx["frame.A.pr_mpnn"])
        an = {**an, "adequacy": ad}
    imf =RES / "github_import_report.json"                             # deviation D6: verification of the runner outputs
    if stage == "C" and imf.exists():
        an = {**an, "import": json.loads(imf.read_text())}
    numbers = {**flat({k: v for k, v in an.items() if k != "generators"}), **flat(d, "derived")}
    (out / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    tables(stage, an, gens, S, d, tc, out / "tables")
    figures(stage, an, gens, S, d, out / "figures")
    print(f"{len(numbers)} numbers, tables and figures in {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "render"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--stage", required=True, choices=["X", "C"])
    a = ap.parse_args()
    if a.step == "build":
        build(a.stage)
    else:
        render(a.files[0], a.files[1], json.loads((RES / f"report_{a.stage}" / "numbers.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
