"""
Post-hoc model-adequacy checks of the Stage C analysis (NOT pre-registered; labelled post hoc in the paper; recorded in
redesign_v15/DEVIATIONS_v15.md). Computed from existing outputs only: no training, no new data.

Rules: every pre-registered estimate is read from analysis_<stage>.json (or s6_meaningful_gap_<stage>.json) and reported
unchanged beside the new checks; no model, estimator, test, generator set or threshold is changed; no generator is dropped
or down-weighted in any reported estimate (leave-one-out appears only as an influence diagnostic, as in plan 8.5). The
heavy-tail sensitivity analysis (h) down-weights extreme generators by construction; it is a labelled side analysis and
never replaces a pre-registered result.

  a. observed sign shares with exact (Clopper-Pearson) intervals; observed psi_G = 2p(1 - p) with a bootstrap interval
  b. observed target flips against the split-half noise shares (paired bootstrap of the difference)
  c. S6 at each margin: observed counts against the model (DL, pre-registered) and a REML-based value
  d. DerSimonian-Laird against REML; pi = Phi(mu/tau) across tau = 0.10 ... 0.60
  e. heavy tails: kurtosis, skewness, Shapiro-Wilk, Q-Q plots of lambda^U and lambda^A; the 5 largest |lambda|
  f. episode level: per-generator delta on the log and raw scales; which generators drive the log-scale test
  g. Stage X on GitHub against the laptop, per generator
  h. random effects with t-distributed generator effects (nu = 3, 4, 5, 10) and the normal ML fit

  python v15/model_adequacy.py --stage C   -> results_v15/model_adequacy_C.json, .md, _qq.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_v15 import B_ACROSS, OUT, ci, dl, pi_of, reml  # noqa: E402

GH = OUT.parent / "results_v15_github" / "43888871250"
TAU_GRID = [round(x, 2) for x in np.arange(0.10, 0.601, 0.05)]
MARGINS = ("1.05", "1.10", "1.25")
NUS = (3, 4, 5, 10)
B_T = 1000                                                    # bootstrap size for the t random-effects fit (h)
Z = 1.959963984540054


def cp(k, n):
    """Exact (Clopper-Pearson) 95% interval for k successes out of n."""
    return [0.0 if k == 0 else float(stats.beta.ppf(0.025, k, n - k + 1)),
            1.0 if k == n else float(stats.beta.ppf(0.975, k + 1, n - k))]


def kernel(spec):
    return "cut-off" if "dcut" in spec else "power law"


def run(stage):
    an = json.loads((OUT / f"analysis_{stage}.json").read_text())
    gens = an["generators"]
    S = {g: json.loads((OUT / stage / g / "summary_v15.json").read_text()) for g in gens}
    M = len(gens)
    BI = np.random.default_rng(15).integers(0, M, (B_ACROSS, M))           # the resamples of analyse_v15.aggregate
    arr = lambda m, t, k: np.array([S[g][m][t][k] for g in gens])           # noqa: E731
    lam = {t: arr("frame", t, "lam") for t in "UA"}
    var = {t: arr("frame", t, "v_two") for t in "UA"}
    spec = {g: S[g]["meta"]["spec"] for g in gens}
    info = lambda i: {"generator": gens[i], "kernel": kernel(spec[gens[i]]), "aggregation": spec[gens[i]].get("agg", "sum")}  # noqa: E731
    npos = np.array([int(np.load(OUT / stage / g / "labels_test.npz")["y_A"].sum()) for g in gens])
    res = {"stage": stage, "M": M, "status": "post-hoc model-adequacy checks, not pre-registered; pre-registered values unchanged"}

    # ------------------------------------------------------------------ a. sign shares and observed psi_G
    a = {}
    for t in "UA":
        y, h = lam[t], an["frame"][f"H1_{t}"]
        assert abs(pi_of(*dl(y, var[t])) - h["pi"]) < 1e-12
        k = int((y > 0).sum()); p = k / M
        pb = (y[BI] > 0).mean(1)
        a[t] = {"n_mpnn_wins": k, "share_mpnn_wins": p, "share_cp95": cp(k, M),
                "share_cp95_inside_0.10_0.90": bool(cp(k, M)[0] > 0.10 and cp(k, M)[1] < 0.90),
                "observed_psi_G": 2 * p * (1 - p), "observed_psi_G_boot95": ci(2 * pb * (1 - pb)),
                "preregistered_pi_DL": h["pi"], "preregistered_pi_DL_ci95": h["pi_ci95"],
                "preregistered_psi_G_DL": h["psi_G"], "preregistered_psi_G_DL_ci95": h["psi_G_ci95"],
                "pi_REML": h["reml"]["pi"]}
    a["U"]["preregistered_bounds_share_mpnn_sig"] = an["frame"]["checks"]["bounds_share_mpnn_sig"]
    a["U"]["preregistered_bounds_share_rawhgb_sig"] = an["frame"]["checks"]["bounds_share_rawhgb_sig"]
    res["a_sign_shares"] = a

    # ------------------------------------------------------------------ b. target flips against split-half noise
    flips = np.sign(lam["U"]) != np.sign(lam["A"])
    assert abs(flips.mean() - an["frame"]["psi_T"]["naive_share_sign_differs"]) < 1e-12
    sh = {t: np.array([np.sign(S[g]["frame"][t]["split_half"][0]) != np.sign(S[g]["frame"][t]["split_half"][1])
                       for g in gens]) for t in "UA"}
    b = {"n_target_flips": int(flips.sum()), "share_target_flips": float(flips.mean()), "target_flips_cp95": cp(int(flips.sum()), M),
         "target_flip_generators": [info(i) for i in np.flatnonzero(flips)]}
    for t in "UA":
        assert abs(sh[t].mean() - an["frame"][f"psi_N_split_half_{t}"]["share_differs"]) < 1e-12
        dif = flips[BI].mean(1) - sh[t][BI].mean(1)
        b[f"split_half_{t}"] = {"n": int(sh[t].sum()), "share": float(sh[t].mean()), "cp95": cp(int(sh[t].sum()), M)}
        b[f"flips_minus_split_half_{t}"] = {"est": float(flips.mean() - sh[t].mean()), "paired_boot95": ci(dif)}
    b["flips_exceed_noise_on_observed_data"] = bool(all(b[f"flips_minus_split_half_{t}"]["paired_boot95"][0] > 0 for t in "UA"))
    b["preregistered_psi_T"] = an["frame"]["psi_T"]["est"]; b["preregistered_psi_T_ci95"] = an["frame"]["psi_T"]["ci95"]
    b["preregistered_psi_N_model"] = an["frame"]["psi_N_model"]["est"]
    b["preregistered_psi_T_minus_psi_N"] = an["frame"]["psi_T_minus_psi_N"]
    res["b_target_flips"] = b

    # ------------------------------------------------------------------ c. S6 observed against model, each margin
    s6 = json.loads((OUT / f"s6_meaningful_gap_{stage}.json").read_text())
    mu_r, t2_r = reml(lam["U"], var["U"])
    c_out = {}
    for m in MARGINS:
        c = s6["margins"][m]
        o, mo = s6[m]["observed"], s6[m]["model"]
        y = lam["U"]
        assert int((y >= c).sum()) == o["U"]["n_mpnn"] and int((y <= -c).sum()) == o["U"]["n_rawhgb"]
        sp, sm = (y >= c).mean(), (y <= -c).mean()
        bp, bm = (y[BI] >= c).mean(1), (y[BI] <= -c).mean(1)
        rp = float(stats.norm.cdf((mu_r - c) / np.sqrt(t2_r))); rm = float(stats.norm.cdf((-c - mu_r) / np.sqrt(t2_r)))
        mf = o["n_meaningful_target_flips"]
        c_out[m] = {"log_margin": c,
                    "observed_share_mpnn_beyond": float(sp), "observed_share_rawhgb_beyond": float(sm),
                    "observed_psi_G_c": float(2 * sp * sm), "observed_psi_G_c_boot95": ci(2 * bp * bm),
                    "preregistered_model_pi_plus": mo["pi_plus"], "preregistered_model_pi_minus": mo["pi_minus"],
                    "preregistered_model_psi_G_c": mo["psi_G"], "preregistered_model_psi_G_c_ci95": mo["psi_G_ci95"],
                    "REML_based_pi_plus": rp, "REML_based_pi_minus": rm, "REML_based_psi_G_c": 2 * rp * rm,
                    "observed_meaningful_target_flips": mf, "observed_meaningful_target_flips_share": mf / M,
                    "observed_meaningful_target_flips_cp95": cp(mf, M),
                    "preregistered_model_psi_T_c": mo["psi_T"], "preregistered_model_psi_N_c": mo["psi_N"],
                    "direction_psi_G_c": "model below observed" if mo["psi_G"] < 2 * sp * sm else "model above observed",
                    "direction_psi_T_c": "model above observed" if mo["psi_T"] > mf / M else "model below observed"}
    res["c_S6"] = {"DL_tau_U": an["frame"]["H1_U"]["tau"], "REML_tau_U": float(np.sqrt(t2_r)), "margins": c_out}

    # ------------------------------------------------------------------ d. DL against REML; pi across tau
    d = {}
    for t in "UA":
        y, v, h = lam[t], var[t], an["frame"][f"H1_{t}"]
        Y, V = y[BI], v[BI]
        grid = []
        for tau in TAU_GRID:
            w = 1.0 / (v + tau ** 2); mu_t = float((w * y).sum() / w.sum())
            W = 1.0 / (V + tau ** 2); pib = stats.norm.cdf((W * Y).sum(1) / W.sum(1) / tau)
            pci = ci(pib)
            grid.append({"tau": tau, "pi_at_DL_mu": float(stats.norm.cdf(h["mu"] / tau)), "mu_at_tau": mu_t,
                         "pi": float(stats.norm.cdf(mu_t / tau)), "pi_ci95": pci,
                         "psi_G": float(2 * stats.norm.cdf(mu_t / tau) * (1 - stats.norm.cdf(mu_t / tau))),
                         "pi_ci95_inside_0.10_0.90": bool(pci[0] > 0.10 and pci[1] < 0.90)})
        d[t] = {"DL": {"mu": h["mu"], "tau": h["tau"], "pi": h["pi"], "psi_G": h["psi_G"]},
                "REML": {"mu": h["reml"]["mu"], "tau": h["reml"]["tau"], "pi": h["reml"]["pi"],
                         "psi_G": 2 * h["reml"]["pi"] * (1 - h["reml"]["pi"])},
                "tau_grid": grid,
                "H1_decision_same_for_every_tau": bool(all(r["pi_ci95_inside_0.10_0.90"] for r in grid))}
    res["d_between_generator_spread"] = d

    # ------------------------------------------------------------------ e. heavy tails
    e = {}
    for t in "UA":
        y = lam[t]
        sw = stats.shapiro(y)
        top = np.argsort(-np.abs(y))[:5]
        e[t] = {"excess_kurtosis": float(stats.kurtosis(y)), "skewness": float(stats.skew(y)),
                "shapiro_W": float(sw.statistic), "shapiro_p": float(sw.pvalue),
                "median_abs_lambda": float(np.median(np.abs(y))),
                "n_abs_lambda_above_1": int((np.abs(y) > 1).sum()),
                "top5_abs_lambda": [{**info(i), "lambda": float(y[i]), "bootstrap_variance": float(var[t][i]),
                                     "lambda_other_target": float(lam["A" if t == "U" else "U"][i]),
                                     "n_A_positive_test_frames": int(npos[i])} for i in top]}
    assert abs(e["U"]["shapiro_p"] - an["frame"]["checks"]["shapiro_lambda_U_p"]) < 1e-12
    res["e_heavy_tails"] = e
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
    for k, t in enumerate("UA"):
        (osm, osr), (sl, ic, _) = stats.probplot(lam[t], dist="norm")
        ax[k].scatter(osm, osr, s=12, color="#2a6f97")
        ax[k].plot(osm, sl * osm + ic, color="#888888", lw=1)
        ax[k].set_title(f"λ^{t} (excess kurtosis {e[t]['excess_kurtosis']:.1f})".replace("-", "−"), fontsize=10)
        ax[k].set_xlabel("Normal quantile"); ax[k].set_ylabel("Ordered λ")
        for s in ("top", "right"):
            ax[k].spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"model_adequacy_{stage}_qq.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------ f. episode level: log against raw scale
    pr = {(t, det): arr("episode", t, f"pr_{det}") for t in "UA" for det in ("mpnn", "rawhgb")}
    dlog = arr("episode", "A", "lam") - arr("episode", "U", "lam")
    draw = (pr["A", "mpnn"] - pr["A", "rawhgb"]) - (pr["U", "mpnn"] - pr["U", "rawhgb"])
    assert abs(stats.ttest_1samp(dlog, 0).pvalue - an["episode"]["H2"]["p"]) < 1e-12
    assert abs(stats.ttest_1samp(draw, 0).pvalue - an["episode"]["H2"]["raw_scale"]["p"]) < 1e-12
    level = np.mean([pr[k] for k in pr], axis=0)                              # mean episode PR-AUC of the four models
    f = {}
    for nm, x in (("log", dlog), ("raw", draw)):
        tt = stats.ttest_1samp(x, 0)
        f[nm] = {"mean": float(x.mean()), "median": float(np.median(x)), "t_p": float(tt.pvalue),
                 "wilcoxon_p": float(stats.wilcoxon(x).pvalue), "n_negative": int((x < 0).sum()),
                 "n_positive": int((x > 0).sum()), "sign_test_p": float(stats.binomtest(int((x < 0).sum()), int((x != 0).sum())).pvalue),
                 "loo_t_p_range": [float(min(stats.ttest_1samp(np.delete(x, i), 0).pvalue for i in range(M))),
                                   float(max(stats.ttest_1samp(np.delete(x, i), 0).pvalue for i in range(M)))]}
    neg = np.argsort(dlog)[:5]
    q = level <= np.quantile(level, 0.25)
    f["n_sign_disagreements_log_vs_raw"] = int((np.sign(dlog) != np.sign(draw)).sum())
    f["spearman_log_raw"] = float(stats.spearmanr(dlog, draw).statistic)
    f["spearman_abs_log_vs_level"] = float(stats.spearmanr(np.abs(dlog), level).statistic)
    f["share_of_negative_log_sum_from_lowest_quartile_level"] = float(dlog[q & (dlog < 0)].sum() / dlog[dlog < 0].sum())
    f["share_of_negative_raw_sum_from_lowest_quartile_level"] = float(draw[q & (draw < 0)].sum() / draw[draw < 0].sum())
    f["median_level_lowest_quartile"] = float(np.median(level[q])); f["median_level_rest"] = float(np.median(level[~q]))
    f["most_negative_log_delta"] = [{**info(i), "delta_log": float(dlog[i]), "delta_raw": float(draw[i]),
                                     "mean_episode_pr_auc": float(level[i]), "pr_A_mpnn": float(pr["A", "mpnn"][i]),
                                     "pr_A_rawhgb": float(pr["A", "rawhgb"][i]), "n_A_positive_test_frames": int(npos[i]),
                                     "t_p_without": float(stats.ttest_1samp(np.delete(dlog, i), 0).pvalue)} for i in neg]
    f["per_generator"] = [{"generator": g, "delta_log": float(dlog[i]), "delta_raw": float(draw[i]),
                           "mean_episode_pr_auc": float(level[i])} for i, g in enumerate(gens)]
    f["preregistered_episode_H2_p"] = an["episode"]["H2"]["p"]
    res["f_episode_level"] = f

    # ------------------------------------------------------------------ g. Stage X: GitHub runners against the laptop
    imp = json.loads((OUT / "github_import_report.json").read_text())
    rows, mx = [], {}
    for g in sorted(p.name for p in (OUT / "X").iterdir() if (p / "summary_v15.json").exists()):
        lo = json.loads((OUT / "X" / g / "summary_v15.json").read_text())
        gh = json.loads((GH / f"X-{g}" / "summary_v15.json").read_text())
        r = {"generator": g}
        for m in ("frame", "episode"):
            for t in "UA":
                for k in ("pr_mpnn", "pr_rawhgb", "lam"):
                    dd = gh[m][t][k] - lo[m][t][k]
                    r[f"{m}.{t}.{k}.github_minus_laptop"] = dd
                    mx[f"{m}.{t}.{k}"] = max(mx.get(f"{m}.{t}.{k}", 0.0), abs(dd))
                r[f"{m}.{t}.same_winner"] = bool(np.sign(gh[m][t]["lam"]) == np.sign(lo[m][t]["lam"]))
                zd = abs(gh[m][t]["lam"] - lo[m][t]["lam"]) / np.sqrt(lo[m][t]["v_two"])   # in units of the laptop SE of lambda
                r[f"{m}.{t}.lam_diff_in_SE"] = float(zd)
                mx[f"{m}.{t}.lam_diff_in_SE"] = max(mx.get(f"{m}.{t}.lam_diff_in_SE", 0.0), float(zd))
                r[f"{m}.{t}.lam_laptop"], r[f"{m}.{t}.lam_github"] = lo[m][t]["lam"], gh[m][t]["lam"]
        rows.append(r)
    sx = imp["stage_x"]
    flipped = [{"generator": r["generator"], "metric_target": f"{m}.{t}", "lam_laptop": r[f"{m}.{t}.lam_laptop"],
                "lam_github": r[f"{m}.{t}.lam_github"], "diff_in_SE": r[f"{m}.{t}.lam_diff_in_SE"]}
               for r in rows for m in ("frame", "episode") for t in "UA" if not r[f"{m}.{t}.same_winner"]]
    res["g_stage_x_cross_platform"] = {
        "winner_changes": flipped,
        "n_generators": len(rows), "max_abs_diff": mx,
        "n_same_winner": {f"{m}.{t}": int(sum(r[f"{m}.{t}.same_winner"] for r in rows)) for m in ("frame", "episode") for t in "UA"},
        "M_laptop": sx["local_v15b"], "M_github": sx["runner_v15b"], "G1_laptop": sx["G1_local"], "G1_github": sx["G1_runner"],
        "power_H2_M100_laptop": sx["power_H2_M100_local"], "power_H2_M100_github": sx["power_H2_M100_runner"],
        "stage_c_aggregate_reproduced_locally": bool(imp["checks"]["local re-run of the aggregate reproduces the runner's analysis_C.json"]["ok"]),
        "per_generator": rows}

    # ------------------------------------------------------------------ h. t-distributed random effects (side analysis)
    y, v = lam["U"], var["U"]
    step = min(0.002, float(np.sqrt(v.min())) / 4)
    th = np.arange(-8.0, 8.0 + step, step)
    K = stats.norm.pdf(y[:, None], th[None, :], np.sqrt(v)[:, None])       # N(y_g; theta, v_g) on the theta grid

    def fit(rows, nu):
        Kr = K[rows]

        def nll(p):
            mu, tau = p[0], np.exp(p[1])
            dens = (stats.norm.pdf(th, mu, tau) if nu is None else stats.t.pdf((th - mu) / tau, nu) / tau)
            L = Kr @ dens * step
            return -np.sum(np.log(np.maximum(L, 1e-300)))
        r = optimize.minimize(nll, [float(np.median(y[rows])), np.log(0.3)], method="Nelder-Mead",
                              options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
        mu, tau = float(r.x[0]), float(np.exp(r.x[1]))
        pi = float(stats.norm.cdf(mu / tau) if nu is None else stats.t.cdf(mu / tau, nu))
        return {"mu": mu, "tau": tau, "pi": pi, "psi_G": 2 * pi * (1 - pi), "loglik": -float(r.fun), "converged": bool(r.success)}
    allr = np.arange(M)
    h = {"note": "side analysis; down-weights extreme generators by construction; does not replace the pre-registered DL estimate",
         "normal_ML": fit(allr, None), "t": {}}
    for nu in NUS:
        h["t"][str(nu)] = fit(allr, nu)
    boot = [fit(BI[i], 4) for i in range(B_T)]
    h["t"]["4"]["pi_ci95"] = ci([x["pi"] for x in boot]); h["t"]["4"]["psi_G_ci95"] = ci([x["psi_G"] for x in boot])
    h["t"]["4"]["n_boot"] = B_T
    h["t"]["4"]["pi_ci95_inside_0.10_0.90"] = bool(h["t"]["4"]["pi_ci95"][0] > 0.10 and h["t"]["4"]["pi_ci95"][1] < 0.90)
    res["h_t_random_effects_U"] = h

    (OUT / f"model_adequacy_{stage}.json").write_text(json.dumps(res, indent=1))
    (OUT / f"model_adequacy_{stage}.md").write_text(summary(res), encoding="utf-8")
    print(OUT / f"model_adequacy_{stage}.json")
    return res


def summary(r):
    f2 = lambda x: f"{x:.2f}".replace("-", "−")                                      # noqa: E731
    iv = lambda x, n=2: f"[{x[0]:.{n}f}, {x[1]:.{n}f}]".replace("-", "−")          # noqa: E731
    a, b, c, d, e, f, g, h = (r[k] for k in ("a_sign_shares", "b_target_flips", "c_S6", "d_between_generator_spread",
                                             "e_heavy_tails", "f_episode_level", "g_stage_x_cross_platform", "h_t_random_effects_U"))
    L = [f"# Model adequacy (post hoc), Stage {r['stage']}, M = {r['M']}", "",
         "Post-hoc checks computed from existing outputs (`v15/model_adequacy.py`). Pre-registered values are unchanged and "
         "shown beside the checks. Square brackets: 95% intervals (exact Clopper–Pearson for counts; percentile bootstrap "
         "over generators otherwise).", "",
         "## a. Sign shares (model-free) beside the model", "",
         "| Target | MPNN wins | Share [exact 95%] | Observed 2p(1−p) [boot] | Pre-registered π (DL) | Pre-registered ψ_G (DL) | π (REML) |",
         "|---|---|---|---|---|---|---|"]
    for t in "UA":
        x = a[t]
        L.append(f"| {t} | {x['n_mpnn_wins']} | {f2(x['share_mpnn_wins'])} {iv(x['share_cp95'])} | {f2(x['observed_psi_G'])} "
                 f"{iv(x['observed_psi_G_boot95'])} | {f2(x['preregistered_pi_DL'])} {iv(x['preregistered_pi_DL_ci95'])} | "
                 f"{f2(x['preregistered_psi_G_DL'])} {iv(x['preregistered_psi_G_DL_ci95'])} | {f2(x['pi_REML'])} |")
    L += ["", f"The exact interval of the observed share lies inside (0.10, 0.90) under U: {a['U']['share_cp95_inside_0.10_0.90']}. "
          "The observed 2p(1−p) includes winner changes caused by noise and is descriptive.", "",
          "## b. Target flips against noise (observed data)", "",
          f"Target flips: {b['n_target_flips']}/{r['M']} {iv(b['target_flips_cp95'], 3)}. Split-half noise: U "
          f"{b['split_half_U']['n']}/{r['M']}, A {b['split_half_A']['n']}/{r['M']}. Flips minus noise (paired bootstrap): U "
          f"{f2(b['flips_minus_split_half_U']['est'])} {iv(b['flips_minus_split_half_U']['paired_boot95'])}, A "
          f"{f2(b['flips_minus_split_half_A']['est'])} {iv(b['flips_minus_split_half_A']['paired_boot95'])}. "
          f"**Target flips exceed noise on the observed data: {'yes' if b['flips_exceed_noise_on_observed_data'] else 'no'}.** "
          "The split-half reference uses half the data per replication and therefore overstates full-size noise, which "
          "makes this comparison conservative for a claim that flips exceed noise. "
          f"Pre-registered model-based ψ_T = {f2(b['preregistered_psi_T'])} {iv(b['preregistered_psi_T_ci95'])} and "
          f"ψ_T − ψ_N = {f2(b['preregistered_psi_T_minus_psi_N']['est'])} {iv(b['preregistered_psi_T_minus_psi_N']['ci95'])} "
          "(unchanged; the model overstates the observed flips).", "",
          "## c. S6 (meaningful gap): observed against model", "",
          f"τ_U: DL {c['DL_tau_U']:.3f}, REML {c['REML_tau_U']:.3f}.", "",
          "| Margin | Observed ψ_G(c) [boot] | Model ψ_G(c), DL (pre-registered) | ψ_G(c), REML-based | Observed meaningful target flips | Model ψ_T(c) (pre-registered) | Model ψ_N(c) |",
          "|---|---|---|---|---|---|---|"]
    for m, x in c["margins"].items():
        L.append(f"| {m} | {f2(x['observed_psi_G_c'])} {iv(x['observed_psi_G_c_boot95'])} | {f2(x['preregistered_model_psi_G_c'])} | "
                 f"{f2(x['REML_based_psi_G_c'])} | {x['observed_meaningful_target_flips']}/{r['M']} | "
                 f"{f2(x['preregistered_model_psi_T_c'])} | {f2(x['preregistered_model_psi_N_c'])} |")
    L += ["", "The DL model understates the generator flips beyond the margin (its τ is small) and overstates the target "
          "flips beyond the margin; the REML-based ψ_G(c) lies on the other side of the observed value.", "",
          "## d. Between-generator spread", "",
          "| Target | DL μ, τ, π | REML μ, τ, π |", "|---|---|---|"]
    for t in "UA":
        L.append(f"| {t} | {d[t]['DL']['mu']:.3f}, {d[t]['DL']['tau']:.3f}, {f2(d[t]['DL']['pi'])} | "
                 f"{d[t]['REML']['mu']:.3f}, {d[t]['REML']['tau']:.3f}, {f2(d[t]['REML']['pi'])} |".replace("-", "−"))
    L += ["", "π with τ fixed (μ re-estimated at each τ; bootstrap interval), target U:", "",
          "| τ | μ at τ | π [95%] | Inside (0.10, 0.90) |", "|---|---|---|---|"]
    for x in d["U"]["tau_grid"]:
        L.append(f"| {x['tau']:.2f} | {x['mu_at_tau']:.3f} | {f2(x['pi'])} {iv(x['pi_ci95'])} | {x['pi_ci95_inside_0.10_0.90']} |".replace("-", "−"))
    L += ["", f"H1 decision the same for every τ in the grid: U {d['U']['H1_decision_same_for_every_tau']}, "
          f"A {d['A']['H1_decision_same_for_every_tau']}.", "",
          "## e. Heavy tails", "",
          "| Target | Excess kurtosis | Skewness | Shapiro–Wilk p | Generators with abs(λ) > 1 |", "|---|---|---|---|---|"]
    for t in "UA":
        L.append(f"| {t} | {e[t]['excess_kurtosis']:.1f} | {e[t]['skewness']:.2f} | {e[t]['shapiro_p']:.1e} | {e[t]['n_abs_lambda_above_1']} |".replace("-", "−"))
    for t in "UA":
        L += ["", f"Largest abs(λ^{t}):", "", "| Generator | λ | Bootstrap variance | Kernel | Aggregation | A-positive test frames |",
              "|---|---|---|---|---|---|"]
        L += [f"| {x['generator']} | {f2(x['lambda'])} | {x['bootstrap_variance']:.3f} | {x['kernel']} | {x['aggregation']} | "
              f"{x['n_A_positive_test_frames']} |" for x in e[t]["top5_abs_lambda"]]
    L += ["", f"Q-Q plots: `model_adequacy_{r['stage']}_qq.png`.", "",
          "## f. Episode level: log against raw scale", "",
          "| Scale | Mean δ | t-test p | Wilcoxon p | Negative / positive | Sign test p | Leave-one-out t p range |",
          "|---|---|---|---|---|---|---|"]
    for nm in ("log", "raw"):
        x = f[nm]
        L.append(f"| {nm} | {x['mean']:.3f} | {x['t_p']:.4f} | {x['wilcoxon_p']:.4f} | {x['n_negative']} / {x['n_positive']} | "
                 f"{x['sign_test_p']:.3f} | {iv(x['loo_t_p_range'], 4)} |".replace("-", "−"))
    L += ["", f"Signs disagree between the scales on {f['n_sign_disagreements_log_vs_raw']} generators (Spearman ρ = "
          f"{f['spearman_log_raw']:.2f}). The log ratio weights a PR-AUC difference by roughly the inverse of the PR-AUC level: "
          f"the quarter of generators with the lowest episode PR-AUC (median {f['median_level_lowest_quartile']:.2f} against "
          f"{f['median_level_rest']:.2f}) contribute {f['share_of_negative_log_sum_from_lowest_quartile_level']:.0%} of the negative "
          f"log-scale sum and {f['share_of_negative_raw_sum_from_lowest_quartile_level']:.0%} of the negative raw-scale sum.", "",
          "Most negative log-scale δ:", "", "| Generator | δ log | δ raw | Mean episode PR-AUC | A-positive test frames | t p without it |",
          "|---|---|---|---|---|---|"]
    L += [f"| {x['generator']} | {f2(x['delta_log'])} | {x['delta_raw']:.3f} | {x['mean_episode_pr_auc']:.2f} | "
          f"{x['n_A_positive_test_frames']} | {x['t_p_without']:.4f} |".replace("-", "−") for x in f["most_negative_log_delta"]]
    L += ["", "## g. Stage X: GitHub runners against the laptop", "",
          f"{g['n_generators']} generators. M: laptop {g['M_laptop']}, GitHub {g['M_github']}; G1: {g['G1_laptop']} and "
          f"{g['G1_github']}. Same winner (frame U, frame A, episode U, episode A): {g['n_same_winner']['frame.U']}, "
          f"{g['n_same_winner']['frame.A']}, {g['n_same_winner']['episode.U']}, {g['n_same_winner']['episode.A']} of "
          f"{g['n_generators']}. Stage C aggregate reproduced locally from the runner outputs: {g['stage_c_aggregate_reproduced_locally']}.", "",
          "| Quantity | Max abs(GitHub − laptop) |", "|---|---|"]
    L += [f"| {k} | {v:.4f} |" for k, v in g["max_abs_diff"].items()]
    L += ["", "Rows ending in `lam_diff_in_SE` give the largest difference in λ in units of the laptop's bootstrap SE of λ.", "",
          "Winner changes between the platforms:", ""]
    L += ["- " + f"{x['generator']} ({x['metric_target']}): λ laptop {x['lam_laptop']:.3f}, GitHub {x['lam_github']:.3f}, "
          f"difference {x['diff_in_SE']:.2f} SE".replace("-", "−") for x in g["winner_changes"]] or ["- none"]
    L += ["", "## h. t-distributed random effects, target U (side analysis; does not replace the pre-registered estimate)", "",
          "| Model | μ | τ | π | ψ_G |", "|---|---|---|---|---|",
          f"| Normal (ML) | {h['normal_ML']['mu']:.3f} | {h['normal_ML']['tau']:.3f} | {f2(h['normal_ML']['pi'])} | {f2(h['normal_ML']['psi_G'])} |".replace("-", "−")]
    for nu, x in h["t"].items():
        extra = f" {iv(x['pi_ci95'])}" if "pi_ci95" in x else ""
        L.append(f"| t, ν = {nu} | {x['mu']:.3f} | {x['tau']:.3f} | {f2(x['pi'])}{extra} | {f2(x['psi_G'])} |".replace("-", "−"))
    L += ["", f"Pre-registered DL: π = {f2(a['U']['preregistered_pi_DL'])} {iv(a['U']['preregistered_pi_DL_ci95'])}, "
          f"ψ_G = {f2(a['U']['preregistered_psi_G_DL'])}.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["C"], required=True)
    run(ap.parse_args().stage)
