"""
E5 (ANALYSIS_PLAN_v14_R6.md, R6-E): GAT-sum, the r5 GAT with one change. Attention logits pass through a sigmoid
gate per peer and head instead of a softmax over peers, and the gated messages are summed over peers (unnormalised
aggregation). Inputs, message function (linear in peer features), readout, training loop and grid are those of the GAT
(rigorous_rerun/baselines.py). Importing this module registers the family "GATSUM".
Self-check: python gat_sum.py  (shapes, peer-permutation invariance, and identity with the GAT except the aggregation).
"""
import sys
from pathlib import Path

J7 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(J7 / "rigorous_rerun"))
import baselines  # noqa: E402,F401  (registers GAT; provides _fit_graph, GEO)
import models  # noqa: E402
from baselines import GEO, _fit_graph, _torch  # noqa: E402


def _net():
    torch = _torch()
    import torch.nn as nn

    class GATSum(nn.Module):
        """As baselines.GAT, but alpha = sigmoid(logit) per peer and head (no softmax over peers) and sum pooling."""
        def __init__(self, dv, dn, h, heads=4):
            super().__init__()
            self.heads, self.hd = heads, max(4, h // heads)
            H = heads * self.hd
            self.w_v = nn.Linear(dv, H); self.w_n = nn.Linear(dn, H); self.w_e = nn.Linear(3, H)
            self.att = nn.Parameter(torch.randn(heads, self.hd) * 0.1)
            self.w_msg = nn.Linear(dn, H)
            self.rho = nn.Sequential(nn.Linear(dv + H, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1))

        def forward(self, V, E):
            B, K, _ = E.shape
            xn, ge = E[..., :15], E[..., GEO]
            z = self.w_v(V).unsqueeze(1) + self.w_n(xn) + self.w_e(ge)
            z = torch.nn.functional.leaky_relu(z, 0.2).view(B, K, self.heads, self.hd)
            alpha = torch.sigmoid((z * self.att).sum(-1))                       # (B,K,heads), unnormalised
            m = self.w_msg(xn).view(B, K, self.heads, self.hd)
            agg = (alpha.unsqueeze(-1) * m).sum(1).reshape(B, -1)
            return self.rho(torch.cat([V, torch.relu(agg)], -1)).squeeze(-1)
    return GATSum


def fit_GATSUM(d, cfg, seed):
    GATSum = _net()
    return _fit_graph(d, cfg, seed, lambda dv: GATSum(dv, 15, cfg["h"]))


models.FAMILIES["GATSUM"] = fit_GATSUM


if __name__ == "__main__":
    import numpy as np
    torch = _torch()
    rng = np.random.default_rng(0)
    E = torch.tensor(rng.random((7, 3, 18)).astype(np.float32))
    V = torch.tensor(rng.random((7, 5)).astype(np.float32))
    GATSum = _net()
    _, GAT = baselines._nets()
    torch.manual_seed(1); a = GATSum(5, 15, 32).eval()
    torch.manual_seed(1); b = GAT(5, 15, 32).eval()
    # identical parameters (same construction order and seed)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    out = a(V, E)
    assert out.shape == (7,)
    perm = E[:, [2, 0, 1]]
    assert torch.allclose(a(V, perm), out, atol=1e-6), "not permutation invariant"
    # the only difference is the aggregation: replacing sigmoid+sum by softmax reproduces the GAT exactly
    import types

    def softmax_forward(self, V, E):
        B, K, _ = E.shape
        xn, ge = E[..., :15], E[..., GEO]
        z = self.w_v(V).unsqueeze(1) + self.w_n(xn) + self.w_e(ge)
        z = torch.nn.functional.leaky_relu(z, 0.2).view(B, K, self.heads, self.hd)
        alpha = torch.softmax((z * self.att).sum(-1), dim=1)
        m = self.w_msg(xn).view(B, K, self.heads, self.hd)
        agg = (alpha.unsqueeze(-1) * m).sum(1).reshape(B, -1)
        return self.rho(torch.cat([V, torch.relu(agg)], -1)).squeeze(-1)
    a.forward = types.MethodType(softmax_forward, a)
    assert torch.allclose(a(V, E), b(V, E), atol=1e-6)
    # unnormalised aggregation scales with the number of exposing peers; normalised does not
    print("gat_sum self-check passed (same parameters as GAT; differs only in aggregation; permutation invariant)")
