"""
Plan v15 analysis (section 8). Two steps:

  python v15/analyse_v15.py summarise --stage C --gen c117   per-generator statistics -> <gen>/summary_v15.json
  python v15/analyse_v15.py aggregate --stage C              across-generator analysis -> results_v15/analysis_C.json/.md
  python v15/analyse_v15.py aggregate --stage X              exploratory summary, gates G1-G2, re-estimated M (v15b)

Per generator: tie-aware PR-AUC per training seed; lambda_t = log PR_MPNN - log PR_RawHGB (PR = mean over seeds);
two-level bootstrap (B = 2000: seeds with replacement, then cohorts) for the within-generator variances and the U-A
covariance; split-half replications; episode-level metric (S1); S2, S3, S4 and R3 quantities when present.
Across generators: H1 (DerSimonian-Laird, pi = Phi(mu/tau), 95% percentile bootstrap over generators, IUT inside
(0.10, 0.90)); H2 (two-sided t-test of delta = lambda_A - lambda_U, alpha 0.025); secondary estimands and checks.
Generators are independent units (separate cohorts and seeds), so across-generator inference needs no covariance model.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XRP_GEN", "v2")
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "option_b" / "code"))
from ob_stats import Sorted, holm  # noqa: E402

OUT = ROOT / "results_v15"
DETS = ("mpnn", "rawhgb")
B_WITHIN, B_ACROSS = 2000, 4000
ROWS_PER_COHORT = 8 * 4 * 48


# =============================================================== per generator
def episodes(ctx):
    """Episode = (cohort, track, victim). Returns episode index per row, number of episodes, cohort id per episode."""
    key = ctx[:, 0].astype(np.int64) * 1000 + ctx[:, 1] * 10 + ctx[:, 3]
    u, inv = np.unique(key, return_inverse=True)
    return inv, len(u), (u // 1000).astype(np.int64)


def ep_reduce(x, inv, n):
    out = np.full(n, -np.inf)
    np.maximum.at(out, inv, x)
    return out


def ap_matrix(P, y, g, W):
    """P: (S, n) predictions; W: (B, n_cohorts) weights. Returns point AP per seed (S,) and AP per seed x replicate (S, B)."""
    pts, reps = [], []
    for p in P:
        s = Sorted(p, y, g)
        pts.append(s.ap())
        reps.append([s.ap(w) for w in W])
    return np.array(pts), np.array(reps)


def fcov(a, b, corr=False):
    """Covariance (or correlation) over the bootstrap replicates where both values are finite. Identical to np.cov /
    np.corrcoef when every replicate is finite; a replicate without positives (possible only in tiny corpora) gives a
    non-finite AP and is skipped, as np.nanvar skips it for the variances."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float((np.corrcoef if corr else np.cov)(a[m], b[m])[0, 1])


def summarise(stage, gen, b=B_WITHIN):
    od = OUT / stage / gen
    L = np.load(od / "labels_test.npz")
    ctx = L["ctx"]
    g = ctx[:, 0].astype(np.int64)
    nc = int(g.max()) + 1
    inv, ne, ge = episodes(ctx)
    rng = np.random.default_rng(sum(map(ord, gen)) * 7919 + 21)          # deterministic per generator name
    W = np.stack([np.bincount(rng.integers(0, nc, nc), minlength=nc).astype(float) for _ in range(b)])
    labels = {"U": L["y"].astype(int), "A": L["y_A"].astype(int), "A0": L["y_A0"].astype(int)}
    P = {f.stem[6:]: np.load(f) for f in od.glob("preds_*.npy")}       # keys like U_mpnn, A_rawhgb, S2U_mpnn, UR3_mpnn
    S = P["U_mpnn"].shape[0]
    sidx = rng.integers(0, S, (b, S))
    res = {"stage": stage, "gen": gen, "n_cohorts": nc, "n_seeds": S, "prev": {t: float(y.mean()) for t, y in labels.items()},
           "meta": json.loads((od / "meta.json").read_text())}

    def block(cond, lab, metric):
        """PR per detector: point (mean over seeds), per-seed, and two-level replicates."""
        y = labels[lab]
        out = {}
        for d in DETS:
            Pm = P[f"{cond}_{d}"]
            if metric == "episode":
                yy = ep_reduce(y.astype(float), inv, ne).astype(int)
                Pm = np.stack([ep_reduce(p.astype(np.float64), inv, ne) for p in Pm])
                gg = ge
            else:
                yy, gg = y, g
            pts, reps = ap_matrix(Pm, yy, gg, W)
            two = reps[sidx, np.arange(b)[:, None]].mean(1)                # (b,)
            coh = reps.mean(0)                                             # cohort-only replicates
            out[d] = {"pts": pts, "two": two, "coh": coh}
        return out

    for metric in ("frame", "episode"):
        R = {}
        for name, cond, lab in (("U", "U", "U"), ("A", "A", "A"), ("A0", "A", "A0"), ("UonA", "U", "A")):
            if f"{cond}_mpnn" not in P:
                continue
            blk = block(cond, lab, metric)
            pm, pr = blk["mpnn"]["pts"].mean(), blk["rawhgb"]["pts"].mean()
            lam_two = np.log(blk["mpnn"]["two"]) - np.log(blk["rawhgb"]["two"])
            lam_coh = np.log(blk["mpnn"]["coh"]) - np.log(blk["rawhgb"]["coh"])
            lam_seed = np.log(blk["mpnn"]["pts"]) - np.log(blk["rawhgb"]["pts"])
            R[name] = {"pr_mpnn": float(pm), "pr_rawhgb": float(pr), "lam": float(np.log(pm) - np.log(pr)),
                       "pr_seed": {"mpnn": blk["mpnn"]["pts"].tolist(), "rawhgb": blk["rawhgb"]["pts"].tolist()},
                       "v_two": float(np.nanvar(lam_two, ddof=1)), "sd_cohort": float(np.nanstd(lam_coh, ddof=1)),
                       "sd_seed": float(np.std(lam_seed, ddof=1)) if S > 1 else None, "_two": lam_two}
            # split-half replication: seeds [0, 1] with cohorts in the first half vs seeds [2, 3] with the second half
            if S >= 4 and metric == "frame":
                yy = labels[lab]
                half = {}
                for h, (ss, sel) in enumerate((((0, 1), g < nc // 2), ((2, 3), g >= nc // 2))):
                    w = np.zeros(nc); w[np.unique(g[sel])] = 1.0
                    prs = {d: np.mean([Sorted(P[f"{cond}_{d}"][s], yy, g).ap(w) for s in ss]) for d in DETS}
                    half[h] = float(np.log(prs["mpnn"]) - np.log(prs["rawhgb"]))
                R[name]["split_half"] = [half[0], half[1]]
        if "U" in R and "A" in R:
            R["cov_UA_two"] = fcov(R["U"]["_two"], R["A"]["_two"])
            R["corr_UA_two"] = fcov(R["U"]["_two"], R["A"]["_two"], corr=True)
        if "U" in R and "A0" in R:   # D3: within-generator covariance for the S3 recomputation of psi_T
            R["cov_UA0_two"] = fcov(R["U"]["_two"], R["A0"]["_two"])
        for v in R.values():
            if isinstance(v, dict):
                v.pop("_two", None)
        res[metric] = R
    # S2 (generator shift) and R3 (positive count): point PR-AUCs per detector, frame level
    for cond, lab in (("S2U", "U"), ("S2A", "A"), ("UR3", "U"), ("S5U", "U"), ("S5A", "A")):   # S5: deviation D2
        if f"{cond}_mpnn" in P:
            y = labels[lab]
            res[cond] = {d: float(np.mean([Sorted(p, y, g).ap() for p in P[f"{cond}_{d}"]])) for d in DETS}
            res[cond]["lam"] = float(np.log(res[cond]["mpnn"]) - np.log(res[cond]["rawhgb"]))
    if "rule" in P:   # Stage X only: the rule (cannot see an in-band attacker)
        res["rule"] = {t: float(Sorted(P["rule"][0], y, g).ap()) for t, y in labels.items()}
    (od / "summary_v15.json").write_text(json.dumps(res, indent=1, default=float))
    return res


# =============================================================== across generators
def dl(y, v):
    w = 1.0 / v
    sw = w.sum()
    mu_fe = (w * y).sum() / sw
    Q = (w * (y - mu_fe) ** 2).sum()
    c = sw - (w ** 2).sum() / sw
    tau2 = max(0.0, (Q - (len(y) - 1)) / c)
    ws = 1.0 / (v + tau2)
    return float((ws * y).sum() / ws.sum()), float(tau2)


def reml(y, v):
    def nll(t2):
        w = 1.0 / (v + t2)
        mu = (w * y).sum() / w.sum()
        return 0.5 * (np.sum(np.log(v + t2)) + np.log(w.sum()) + np.sum(w * (y - mu) ** 2))
    t2 = float(minimize_scalar(nll, bounds=(0.0, 10 * np.var(y) + 1e-9), method="bounded").x)
    w = 1.0 / (v + t2)
    return float((w * y).sum() / w.sum()), t2


def pi_of(mu, tau2):
    return float(stats.norm.cdf(mu / np.sqrt(tau2))) if tau2 > 0 else float(mu > 0)


def psi_T(X, Wm):
    """Bivariate random effects (method of moments): P(sign of the U and A effects differ)."""
    mu = X.mean(0)
    Sg = np.cov(X, rowvar=False) - Wm
    ev, evec = np.linalg.eigh(Sg)
    Sg = evec @ np.diag(np.maximum(ev, 1e-8)) @ evec.T          # project onto positive definite (floor 1e-8)
    s = np.sqrt(np.diag(Sg))
    F00 = stats.multivariate_normal(mean=mu, cov=Sg, allow_singular=True).cdf(np.zeros(2))
    p1, p2 = stats.norm.cdf(-mu[0] / s[0]), stats.norm.cdf(-mu[1] / s[1])
    return float((p1 - F00) + (p2 - F00))


def ci(x, lo=2.5, hi=97.5):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]


def tci(x, level=0.95):
    x = np.asarray(x, float)
    if len(x) < 2:
        return None
    h = stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean() - h), float(x.mean() + h)]


def s5_summary(S, gens):
    """Deviation D2, secondary outcome S5 (tuning budget; frame metric; no test, estimates only).
    Per target: generators whose winner (sign of lambda) is unchanged when selection is over 32 instead of 16
    configurations; mean change in lambda with a 95% t-interval; each detector's mean log gain in PR-AUC. Pre-specified
    reading: 'sensitive to the tuning budget' if the winner changes on 3 or more of the 10 generators for either target.
    The number of generators whose primary lambda lies within 1.96 within-generator SDs of zero is reported as context."""
    s5 = [x for x in gens if "S5U" in S[x] and "S5A" in S[x]]
    if not s5:
        return None
    out = {"n": len(s5), "generators": s5}
    for t in ("U", "A"):
        lam0 = np.array([S[x]["frame"][t]["lam"] for x in s5])
        lam1 = np.array([S[x][f"S5{t}"]["lam"] for x in s5])
        sd0 = np.sqrt([S[x]["frame"][t]["v_two"] for x in s5])
        same = np.sign(lam0) == np.sign(lam1)
        gain = {d: np.log([S[x][f"S5{t}"][d] for x in s5]) - np.log([S[x]["frame"][t][f"pr_{d}"] for x in s5]) for d in DETS}
        out[t] = {"n_same_winner": int(same.sum()), "share_same_winner": float(same.mean()),
                  "mean_change_lambda": float((lam1 - lam0).mean()), "mean_change_lambda_ci95": tci(lam1 - lam0),
                  "mean_log_gain": {d: float(gain[d].mean()) for d in DETS},
                  "mean_log_gain_ci95": {d: tci(gain[d]) for d in DETS},
                  "n_primary_lambda_within_1p96_sd_of_zero": int((np.abs(lam0) < 1.96 * sd0).sum())}
    out["sensitive_to_tuning_budget"] = bool(any(out[t]["n_same_winner"] <= len(s5) - 3 for t in ("U", "A")))
    return out


def aggregate(stage, gens, out_name=None):
    S = {gname: json.loads((OUT / stage / gname / "summary_v15.json").read_text()) for gname in gens}
    gens = [x for x in gens if x in S]
    M = len(gens)
    rng = np.random.default_rng(15)
    BI = rng.integers(0, M, (B_ACROSS, M))
    res = {"stage": stage, "M": M, "generators": gens}
    for metric in ("frame", "episode"):
        lu = np.array([S[x][metric]["U"]["lam"] for x in gens])
        vu = np.array([S[x][metric]["U"]["v_two"] for x in gens])
        la = np.array([S[x][metric]["A"]["lam"] for x in gens])
        va = np.array([S[x][metric]["A"]["v_two"] for x in gens])
        cua = np.array([S[x][metric].get("cov_UA_two", 0.0) for x in gens])
        r = {}
        # ---------------- H1 (target U) and the same under A
        for t, y, v in (("U", lu, vu), ("A", la, va)):
            mu, t2 = dl(y, v)
            mu_r, t2_r = reml(y, v)
            pib = np.array([pi_of(*dl(y[i], v[i])) for i in BI])
            pci = ci(pib)
            psib = 2 * pib * (1 - pib)
            pi_hat = pi_of(mu, t2)
            r[f"H1_{t}"] = {"mu": mu, "tau": float(np.sqrt(t2)), "pi": pi_hat, "pi_ci95": pci,
                            "psi_G": 2 * pi_hat * (1 - pi_hat), "psi_G_ci95": ci(psib),
                            "reject_IUT": bool(pci[0] > 0.10 and pci[1] < 0.90),
                            "reml": {"mu": mu_r, "tau": float(np.sqrt(t2_r)), "pi": pi_of(mu_r, t2_r)},
                            "naive_share_positive": float((y > 0).mean())}
        # ---------------- H2 (shift by the positive class)
        dlt = la - lu
        tt = stats.ttest_1samp(dlt, 0.0)
        md, sd = float(dlt.mean()), float(dlt.std(ddof=1))
        r["H2"] = {"mean_delta": md, "sd_delta": sd, "t": float(tt.statistic), "p": float(tt.pvalue), "reject": bool(tt.pvalue < 0.025),
                   "ci95": [md - stats.t.ppf(0.975, M - 1) * sd / np.sqrt(M), md + stats.t.ppf(0.975, M - 1) * sd / np.sqrt(M)],
                   "ci975": [md - stats.t.ppf(0.9875, M - 1) * sd / np.sqrt(M), md + stats.t.ppf(0.9875, M - 1) * sd / np.sqrt(M)],
                   "ratio_change": float(np.exp(md)), "standardised": md / sd if sd > 0 else None,
                   "wilcoxon_p": float(stats.wilcoxon(dlt).pvalue) if M >= 6 else None}
        # D3 (plan 8.2 and P8): the same comparison on the raw PR-AUC difference scale (robustness, no decision)
        raw = np.array([(S[x][metric]["A"]["pr_mpnn"] - S[x][metric]["A"]["pr_rawhgb"])
                        - (S[x][metric]["U"]["pr_mpnn"] - S[x][metric]["U"]["pr_rawhgb"]) for x in gens])
        r["H2"]["raw_scale"] = {"mean_delta": float(raw.mean()), "sd_delta": float(raw.std(ddof=1)) if M > 1 else None,
                                "p": float(stats.ttest_1samp(raw, 0.0).pvalue) if M > 1 else None, "ci95": tci(raw)}
        # ---------------- psi_T, psi_N and comparisons
        X = np.c_[lu, la]
        Wm = np.array([[vu.mean(), cua.mean()], [cua.mean(), va.mean()]])
        pT = psi_T(X, Wm)
        pTb = np.array([psi_T(X[i], np.array([[vu[i].mean(), cua[i].mean()], [cua[i].mean(), va[i].mean()]])) for i in BI])

        def psiN(y, v, idx=slice(None)):
            y, v = y[idx], v[idx]
            mu, t2 = dl(y, v)
            shr = mu + t2 / (t2 + v) * (y - mu)
            z = np.abs(shr) / np.sqrt(v)
            return float(np.mean(2 * stats.norm.cdf(z) * stats.norm.sf(z)))
        pN = psiN(lu, vu)
        pNb = np.array([psiN(lu, vu, i) for i in BI])
        pGb = np.array([2 * pi_of(*dl(lu[i], vu[i])) * (1 - pi_of(*dl(lu[i], vu[i]))) for i in BI])
        r["psi_T"] = {"est": pT, "ci95": ci(pTb), "naive_share_sign_differs": float((np.sign(lu) != np.sign(la)).mean())}
        r["psi_N_model"] = {"est": pN, "ci95": ci(pNb)}
        r["psi_G_minus_psi_N"] = {"est": r["H1_U"]["psi_G"] - pN, "ci95": ci(pGb - pNb)}
        r["psi_T_minus_psi_N"] = {"est": pT - pN, "ci95": ci(pTb - pNb)}
        if metric == "frame":
            sh = [S[x]["frame"]["U"].get("split_half") for x in gens]
            sh = [h for h in sh if h]
            if sh:
                r["psi_N_split_half_U"] = {"share_differs": float(np.mean([np.sign(a) != np.sign(b) for a, b in sh])), "n": len(sh)}
            sha = [S[x]["frame"]["A"].get("split_half") for x in gens]
            sha = [h for h in sha if h]
            if sha:
                r["psi_N_split_half_A"] = {"share_differs": float(np.mean([np.sign(a) != np.sign(b) for a, b in sha])), "n": len(sha)}
        # ---------------- model checks (frame metric reported in the paper)
        r["checks"] = {"shapiro_lambda_U_p": float(stats.shapiro(lu).pvalue) if M >= 3 else None,
                       "loo_pi_range": [float(min(pi_of(*dl(np.delete(lu, i), np.delete(vu, i))) for i in range(M))),
                                        float(max(pi_of(*dl(np.delete(lu, i), np.delete(vu, i))) for i in range(M)))],
                       "loo_mean_delta_range": [float(min(np.delete(dlt, i).mean() for i in range(M))),
                                                float(max(np.delete(dlt, i).mean() for i in range(M)))]}
        z = lu / np.sqrt(vu)
        kp = int((holm(stats.norm.sf(z)) < 0.05).sum())
        km = int((holm(stats.norm.cdf(z)) < 0.05).sum())
        r["checks"]["bounds_share_mpnn_sig"] = kp / M
        r["checks"]["bounds_share_rawhgb_sig"] = km / M
        # ---------------- secondary outcomes
        if all("A0" in S[x][metric] for x in gens):
            d0 = np.array([S[x][metric]["A0"]["lam"] for x in gens]) - lu
            r["S3_A0"] = {"mean_delta": float(d0.mean()), "ci95": ci([d0[i].mean() for i in BI]),
                          "share_sign_differs_vs_U": float((np.sign(lu) != np.sign(lu + d0)).mean())}
            # D3 (plan 8.4 S3: "H2 and psi_T are recomputed with these lambda"): the H2 t-test and the model-based psi_T
            if M > 1:
                t0 = stats.ttest_1samp(d0, 0.0)
                r["S3_A0"].update(t=float(t0.statistic), p=float(t0.pvalue), ci95_t=tci(d0))
            l0 = lu + d0
            v0 = np.array([S[x][metric]["A0"]["v_two"] for x in gens])
            c0 = [S[x][metric].get("cov_UA0_two") for x in gens]
            if M > 2 and all(c is not None for c in c0):
                c0 = np.array(c0, float)
                X0 = np.c_[lu, l0]
                pT0 = psi_T(X0, np.array([[vu.mean(), c0.mean()], [c0.mean(), v0.mean()]]))
                pT0b = np.array([psi_T(X0[i], np.array([[vu[i].mean(), c0[i].mean()], [c0[i].mean(), v0[i].mean()]]))
                                 for i in BI])
                r["S3_A0"]["psi_T"] = {"est": pT0, "ci95": ci(pT0b)}
        if all("UonA" in S[x][metric] for x in gens):
            lm = np.array([S[x][metric]["UonA"]["lam"] for x in gens])
            same4 = np.sign(lm) == np.sign(la)
            r["S4_mismatch"] = {"share_same_winner_as_matched_A": float(same4.mean()),
                                "ci95": ci([same4[i].mean() for i in BI])}             # D3: bootstrap interval
        res[metric] = r
    # S2 generator shift (frame metric)
    if all("S2U" in S[x] for x in gens):
        s2 = {}
        for t, key in (("U", "S2U"), ("A", "S2A")):
            ret = {d: np.array([S[x][key][d] / S[x]["frame"][t][f"pr_{d}"] for x in gens]) for d in DETS}
            same = np.array([np.sign(S[x][key]["lam"]) == np.sign(S[x]["frame"][t]["lam"]) for x in gens])
            s2[t] = {"median_retention": {d: float(np.median(ret[d])) for d in DETS},
                     "median_retention_ci95": {d: ci([np.median(ret[d][i]) for i in BI]) for d in DETS},
                     "share_same_winner": float(same.mean()), "share_same_winner_ci95": ci([same[i].mean() for i in BI])}
        res["S2_shift"] = s2
    r3 = [x for x in gens if "UR3" in S[x]]
    if r3:
        dd = np.array([S[x]["UR3"]["lam"] - S[x]["frame"]["U"]["lam"] for x in r3])
        res["R3_positive_count"] = {"n": len(r3), "mean_change_lambda_U": float(dd.mean()),
                                    "ci95": [float(dd.mean() - stats.t.ppf(0.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))),
                                             float(dd.mean() + stats.t.ppf(0.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd)))]
                                    if len(dd) > 1 else None}
    s5 = s5_summary(S, gens)
    if s5:
        res["S5_tuning_budget"] = s5
    # descriptive context
    res["context"] = {x: {"agg": S[x]["meta"]["spec"].get("agg", "sum"), "ceilings": S[x]["meta"].get("ceilings"),
                          "prev": S[x]["prev"], "prevented": S[x]["meta"].get("prevented"),
                          "lam_U": S[x]["frame"]["U"]["lam"], "lam_A": S[x]["frame"]["A"]["lam"],
                          "pr_U": [S[x]["frame"]["U"]["pr_mpnn"], S[x]["frame"]["U"]["pr_rawhgb"]],
                          "pr_A": [S[x]["frame"]["A"]["pr_mpnn"], S[x]["frame"]["A"]["pr_rawhgb"]],
                          "rule": S[x].get("rule")} for x in gens}
    (OUT / f"{out_name or 'analysis_' + stage}.json").write_text(json.dumps(res, indent=1, default=float))
    return res, S


# =============================================================== Stage X: gates and sample-size re-estimation (v15b)
def simulate_power(muU, tauU, mud, taud, rhoG, rhoE, fA, pairs, M, n=240, S=4, n_base=60, S_base=5, R=1000, seed=0):
    """As redesign_v15/power_sim_v15.py (H2 part), on the log scale with Stage X noise components."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pairs), (R, M))
    sdc, sds = pairs[idx, 0], pairs[idx, 1]
    seU2 = sdc ** 2 * n_base / n + sds ** 2 / S
    seA2 = seU2 * fA ** 2
    z1, z2 = rng.standard_normal((2, R, M))
    DU = muU + tauU * z1
    dtrue = mud + taud * (rhoG * z1 + np.sqrt(1 - rhoG ** 2) * z2)
    e1, e2 = rng.standard_normal((2, R, M))
    eU = np.sqrt(seU2) * e1
    eA = np.sqrt(seA2) * (rhoE * e1 + np.sqrt(1 - rhoE ** 2) * e2)
    d = (DU + dtrue + eA) - (DU + eU)
    t = d.mean(1) / (d.std(1, ddof=1) / np.sqrt(M))
    return float((2 * stats.t.sf(np.abs(t), M - 1) < 0.025).mean())


def stage_x(res, S):
    gens = res["generators"]
    prior = [x for x in gens if x.startswith("rand")]
    ce = {x: S[x]["meta"]["ceilings"] for x in gens}
    g1 = sum(ce[x]["U"] >= 0.15 and ce[x]["A"] >= 0.15 for x in prior)
    lifts = {d: [S[x]["frame"]["A"][f"pr_{d}"] / S[x]["prev"]["A"] for x in gens] for d in DETS}
    g2 = {d: float(np.median(lifts[d])) for d in DETS}
    out = {"G1_prior_detectable_both": g1, "G1_pass": g1 >= 6, "G2_median_lift_A": g2, "G2_pass": all(v >= 3 for v in g2.values()),
           "rule_lift": {x: {t: (S[x]["rule"][t] / S[x]["prev"][t] if S[x].get("rule") and S[x]["prev"][t] > 0 else None)
                             for t in ("U", "A")} for x in gens}}
    # re-estimation from the prior-drawn generators only (plan section 7)
    fr = {x: S[x]["frame"] for x in prior}
    lu = np.array([fr[x]["U"]["lam"] for x in prior]); vu = np.array([fr[x]["U"]["v_two"] for x in prior])
    la = np.array([fr[x]["A"]["lam"] for x in prior]); va = np.array([fr[x]["A"]["v_two"] for x in prior])
    cua = np.array([fr[x].get("cov_UA_two", 0.0) for x in prior])
    muU, t2U = dl(lu, vu)
    d = la - lu
    vd = vu + va - 2 * cua
    taud = float(np.sqrt(max(1e-6, d.var(ddof=1) - vd.mean())))
    rhoG = float(np.corrcoef(lu, d)[0, 1]) if len(prior) > 2 else 0.0
    rhoE = float(np.median(cua / np.sqrt(vu * va)))
    fA = float(np.median(np.sqrt(va / vu)))
    pairs = np.array([[fr[x]["U"]["sd_cohort"], fr[x]["U"]["sd_seed"] or 0.0] for x in prior], float)
    mud_plan = 0.75 * abs(float(d.mean()))
    powers = {M: simulate_power(muU, np.sqrt(max(t2U, 1e-6)), mud_plan, taud, rhoG, rhoE, fA, pairs, M) for M in (60, 70, 80, 90, 100)}
    chosen = next((M for M in (60, 70, 80, 90, 100) if powers[M] >= 0.80), 100)
    out["reestimation"] = {"muU": muU, "tauU": float(np.sqrt(t2U)), "mean_delta": float(d.mean()), "mud_plan_shrunk": mud_plan,
                           "taud": taud, "rhoG": rhoG, "rhoE": rhoE, "fA": fA, "power_H2_by_M": powers, "M_chosen": chosen,
                           "note": "only M is fixed from Stage X (plan section 7); hypotheses and analyses are unchanged"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["summarise", "aggregate"])
    ap.add_argument("--stage", required=True, choices=["X", "C", "smoke"])
    ap.add_argument("--gen")
    a = ap.parse_args()
    if a.step == "summarise":
        r = summarise(a.stage, a.gen)
        print(a.gen, {k: round(r["frame"][k]["lam"], 3) for k in ("U", "A") if k in r["frame"]})
        return
    if a.stage == "C":
        # plan 8.6: the first M qualifying draws that completed, in draw order (spares replace failed generators only)
        sc = json.loads((OUT / "screen" / "stage_c_generators.json").read_text())
        done = [x for x in sc["generators"] if (OUT / "C" / x / "summary_v15.json").exists()]
        failed = [x for x in sc["generators"][:sc["M"]] if x not in done]
        gens = done[:sc["M"]]
        print(f"Stage C: {len(gens)} of M = {sc['M']} used; failed {failed}; spares used {[x for x in gens if x not in sc['generators'][:sc['M']]]}")
    else:
        gens = sorted(p.name for p in (OUT / a.stage).iterdir() if (p / "summary_v15.json").exists())
    res, S = aggregate(a.stage, gens)
    if a.stage == "X":
        res["stage_x"] = stage_x(res, S)
        (OUT / "analysis_X.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res["frame"][k] for k in ("H1_U", "H2", "psi_T", "psi_N_model")}, indent=1, default=float))
    if a.stage == "X":
        print(json.dumps(res["stage_x"], indent=1, default=float))


if __name__ == "__main__":
    main()
