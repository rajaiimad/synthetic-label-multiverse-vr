"""
Post-hoc diagnostic of the model-based psi_T (NOT pre-registered; labelled exploratory in the paper; recorded in
redesign_v15/DEVIATIONS_v15.md). It changes no pre-registered estimate and no decision.

Why: in Stage C the model-based psi_T (plan 8.3, bivariate random effects by the method of moments) was 0.25, while the
observed share of generators with opposite signs of lambda^U and lambda^A was 0.08. The plan's model checks (8.5) use
leave-one-generator-out influence for pi_U and mu_delta; this script applies the same check to psi_T (is one generator
responsible?) and reports the quantities that describe the fit of the bivariate normal model: the implied
between-generator correlation, the rank correlation, the correlation a zero-mean normal model would need to reproduce
the observed share, normality of lambda^A, and the number of A-positive test frames of the most influential generators.

  python v15/psi_T_diagnostics.py --stage C      -> results_v15/psi_T_diagnostics_C.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_v15 import B_ACROSS, OUT, ci, psi_T  # noqa: E402


def W(vu, va, cua):
    return np.array([[vu.mean(), cua.mean()], [cua.mean(), va.mean()]])


def rho_G(X, Wm):
    """Between-generator correlation implied by the moment estimator (same steps as analyse_v15.psi_T)."""
    Sg = np.cov(X, rowvar=False) - Wm
    ev, evec = np.linalg.eigh(Sg)
    Sg = evec @ np.diag(np.maximum(ev, 1e-8)) @ evec.T
    return float(Sg[0, 1] / np.sqrt(Sg[0, 0] * Sg[1, 1]))


def diagnose(stage):
    an = json.loads((OUT / f"analysis_{stage}.json").read_text())
    gens = an["generators"]
    S = {g: json.loads((OUT / stage / g / "summary_v15.json").read_text())["frame"] for g in gens}
    lu = np.array([S[g]["U"]["lam"] for g in gens]); vu = np.array([S[g]["U"]["v_two"] for g in gens])
    la = np.array([S[g]["A"]["lam"] for g in gens]); va = np.array([S[g]["A"]["v_two"] for g in gens])
    cua = np.array([S[g].get("cov_UA_two", 0.0) for g in gens])
    M = len(gens)
    X = np.c_[lu, la]
    est = psi_T(X, W(vu, va, cua))
    assert abs(est - an["frame"]["psi_T"]["est"]) < 1e-9, "psi_T does not reproduce analysis_{stage}.json"
    loo = np.array([psi_T(np.delete(X, i, 0), W(np.delete(vu, i), np.delete(va, i), np.delete(cua, i))) for i in range(M)])
    order = np.argsort(-np.abs(loo - est))
    BI = np.random.default_rng(15).integers(0, M, (B_ACROSS, M))                # the resamples of analyse_v15.aggregate
    flips = np.sign(lu) != np.sign(la)
    z_a = np.abs(la) / np.sqrt(va)
    npos = np.array([int(np.load(OUT / stage / x / "labels_test.npz")["y_A"].sum()) for x in gens])
    res = {"stage": stage, "M": M, "status": "post-hoc diagnostic, not pre-registered",
           "psi_T_model": est, "rho_G_model": rho_G(X, W(vu, va, cua)),
           "observed_share_sign_differs": float(flips.mean()),
           "observed_share_sign_differs_ci95": ci([flips[i].mean() for i in BI]),
           "pearson_lambdaU_lambdaA": float(np.corrcoef(lu, la)[0, 1]),
           "spearman_lambdaU_lambdaA": float(stats.spearmanr(lu, la).statistic),
           # normal-copula correlation from the rank correlation, and the correlation at which a zero-mean bivariate
           # normal gives the observed share of opposite signs (P = arccos(rho) / pi)
           "rho_from_spearman": float(2 * np.sin(np.pi * stats.spearmanr(lu, la).statistic / 6)),
           "rho_matching_observed_share_zero_mean": float(np.cos(np.pi * flips.mean())),
           "shapiro_lambda_A_p": float(stats.shapiro(la).pvalue),
           "loo_psi_T_range": [float(loo.min()), float(loo.max())],
           "most_influential": [{"generator": gens[i], "psi_T_without": float(loo[i]), "lambda_U": float(lu[i]),
                                 "lambda_A": float(la[i]), "v_A": float(va[i]),
                                 "n_A_positive_test_frames": int(npos[i])} for i in order[:5]],
           "min_n_A_positive": int(npos.min()), "median_n_A_positive": float(np.median(npos)),
           "n_generators_below_50_A_positives": int((npos < 50).sum()),
           "median_v_A": float(np.median(va)), "max_v_A": float(va.max()),
           "n_lambda_A_within_1p96_sd_of_zero": int((z_a < 1.959963984540054).sum())}
    f = OUT / f"psi_T_diagnostics_{stage}.json"
    f.write_text(json.dumps(res, indent=1))
    print(f, json.dumps({k: res[k] for k in ("psi_T_model", "rho_G_model", "observed_share_sign_differs",
                                             "loo_psi_T_range", "spearman_lambdaU_lambdaA")}))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["X", "C"], required=True)
    diagnose(ap.parse_args().stage)
