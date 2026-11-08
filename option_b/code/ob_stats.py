"""Option B statistics (plan v11 sections 5-6).

* Tie-aware weighted average precision (scikit-learn semantics: tied scores form one threshold step).
* Cohort bootstrap with one shared set of resampled cohort indices for all generators and detectors.
* Holm adjustment, two-sided bootstrap p-values, Kendall's tau-b.
Self-check: python ob_stats.py  (asserts agreement with sklearn on random data with ties).
"""
import numpy as np
from scipy.stats import kendalltau


class Sorted:
    """Pre-sorted view of one prediction vector: labels, cohort ids and threshold-group ends."""
    def __init__(self, p, y, g):
        o = np.argsort(-p, kind="stable")
        ps = p[o]
        self.y = y[o].astype(float)
        self.g = g[o]
        self.ends = np.r_[np.nonzero(np.diff(ps))[0], len(ps) - 1]

    def ap(self, w=None):
        ww = np.ones_like(self.y) if w is None else w[self.g]
        tp = np.cumsum(ww * self.y)[self.ends]
        fp = np.cumsum(ww * (1 - self.y))[self.ends]
        P = tp[-1]
        if P <= 0:
            return np.nan
        dtp = np.diff(np.r_[0.0, tp])
        return float(np.sum(dtp * tp / np.maximum(tp + fp, 1e-12)) / P)


def cohort_ids(n_rows, rows_per_cohort=8 * 4 * 48):
    return np.arange(n_rows) // rows_per_cohort


def boot_weights(n_cohorts, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    return [np.bincount(rng.integers(0, n_cohorts, n_cohorts), minlength=n_cohorts).astype(float) for _ in range(B)]


def p_two_sided(d, B):
    d = np.asarray(d)
    return max(1.0 / B, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        adj[i] = run
    return adj


def tau(a, b):
    """Kendall's tau-b between two score vectors over the same detectors (higher = better)."""
    return float(kendalltau(a, b).statistic)


if __name__ == "__main__":
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(1)
    for trial in range(20):
        n = 5000
        y = (rng.random(n) < 0.05).astype(int)
        p = np.round(rng.random(n) + y * rng.random(n), rng.integers(1, 3))   # heavy ties
        s = Sorted(p, y, np.zeros(n, int))
        assert abs(s.ap() - average_precision_score(y, p)) < 1e-12, (s.ap(), average_precision_score(y, p))
        w = rng.integers(0, 3, 1).astype(float) + 1                           # uniform weight = unweighted
        assert abs(s.ap(np.array([w[0]])) - average_precision_score(y, p)) < 1e-12
    # integer weights equal row duplication
    g = rng.integers(0, 10, n)
    cnt = rng.integers(0, 3, 10).astype(float)
    rep = np.repeat(np.arange(n), cnt[g].astype(int))
    assert abs(Sorted(p, y, g).ap(cnt) - average_precision_score(y[rep], p[rep])) < 1e-12
    print("ob_stats self-check passed (tie-aware AP == sklearn, weights == duplication)")
