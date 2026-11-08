"""
Statistics added for r5 (ANALYSIS_PLAN_v13_REVIEW.md). Self-check: python r5_stats.py
* interaction: Wald test of a constant paired difference across generators, using a bootstrap covariance.
* crossover: deviation D5 intersection-union crossover test.
* friedman_nemenyi: Friedman chi2, Iman-Davenport F, Nemenyi critical difference (Demsar 2006).
"""
import numpy as np
from scipy.stats import chi2, f as fdist, rankdata


Q05_NEMENYI = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def interaction(d, D):
    """d: (G,) point differences; D: (B, G) bootstrap replicate differences.
    H0: d_g constant over g. Returns Q, df, chi2 p, bootstrap-calibrated p (floored 1/B), condition number."""
    d, D = np.asarray(d, float), np.asarray(D, float)
    G = len(d)
    C = np.eye(G)[:-1] - np.eye(G, k=1)[:-1]                           # consecutive differences, (G-1, G)
    S = np.cov(D, rowvar=False)
    M = C @ S @ C.T
    cond = float(np.linalg.cond(M))
    Mi = np.linalg.pinv(M) if cond > 1e8 else np.linalg.inv(M)
    cd = C @ d
    Q = float(cd @ Mi @ cd)
    Dc = (D - D.mean(0)) @ C.T
    Qb = np.einsum("bi,ij,bj->b", Dc, Mi, Dc)
    return {"Q": Q, "df": G - 1, "p": float(chi2.sf(Q, G - 1)), "p_boot": float(max(1.0 / len(D), np.mean(Qb >= Q))),
            "cond": cond, "pinv": cond > 1e8, "sd_across": float(np.std(d, ddof=1)), "min": float(d.min()), "max": float(d.max()),
            "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum())}


def crossover(d, D):
    """Deviation D5: intersection-union test of a qualitative (crossover) interaction for one detector pair.
    d: (G,) point differences; D: (B, G) bootstrap replicates. One-sided max-|t| over generators in each direction;
    p = max(p_plus, p_minus), each floored at 1/B."""
    d, D = np.asarray(d, float), np.asarray(D, float)
    s = D.std(0, ddof=1)
    z = d / s
    Z = (D - d) / s
    floor = 1.0 / len(D)
    p_plus = max(floor, float(np.mean(Z.max(1) >= z.max())))
    p_minus = max(floor, float(np.mean((-Z).max(1) >= (-z).max())))
    return {"p": max(p_plus, p_minus), "p_plus": p_plus, "p_minus": p_minus, "z_max": float(z.max()), "z_min": float(z.min()),
            "g_max": int(z.argmax()), "g_min": int(z.argmin())}


def friedman_nemenyi(X):
    """X: (N blocks, k treatments), higher = better. Rank 1 = best."""
    N, k = X.shape
    R = np.vstack([rankdata(-x) for x in X])
    Rm = R.mean(0)
    chi = 12 * N / (k * (k + 1)) * (np.sum(Rm ** 2) - k * (k + 1) ** 2 / 4)
    Ff = (N - 1) * chi / (N * (k - 1) - chi)
    cd = Q05_NEMENYI[k] * np.sqrt(k * (k + 1) / (6 * N))
    return {"avg_rank": Rm.tolist(), "chi2": float(chi), "p_chi2": float(chi2.sf(chi, k - 1)), "F": float(Ff),
            "p_F": float(fdist.sf(Ff, k - 1, (k - 1) * (N - 1))), "CD": float(cd), "N": N, "k": k}


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    # interaction: constant shift -> small Q on average; heterogeneous -> large Q
    G, B = 6, 2000
    base = rng.normal(0, 0.02, (B, G))
    r0 = interaction(np.full(G, 0.1), 0.1 + base)
    r1 = interaction(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.3]), 0.1 + base)
    assert r0["Q"] < 1e-9 and r1["p"] < 1e-6, (r0, r1)
    # crossover: one generator strongly positive and one strongly negative -> small p; all positive -> p = 1-ish
    Dn = rng.normal(0, 0.02, (2000, 6))
    c1 = crossover(np.array([0.1, -0.1, 0, 0, 0, 0]), np.array([0.1, -0.1, 0, 0, 0, 0]) + Dn)
    c0 = crossover(np.array([0.1, 0.05, 0.02, 0.1, 0.1, 0.1]), np.array([0.1, 0.05, 0.02, 0.1, 0.1, 0.1]) + Dn)
    assert c1["p"] < 0.01 and c0["p"] > 0.5, (c1, c0)
    # Friedman: Demsar 2006 worked example structure (all ranks identical -> max chi2)
    fr = friedman_nemenyi(np.tile(np.arange(5)[::-1], (10, 1)).astype(float))
    assert abs(fr["chi2"] - 40.0) < 1e-9 and fr["avg_rank"] == [1, 2, 3, 4, 5], fr
    print("r5_stats self-check passed (interaction; crossover; Friedman)")
