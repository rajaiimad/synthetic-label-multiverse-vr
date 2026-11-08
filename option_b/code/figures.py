"""Option B figures from results/phase4/phase4.json -> option_b/figures/ (PDF + PNG). Run after phase4.py."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

OB = Path(__file__).resolve().parents[1]
FD = OB / "figures"
FD.mkdir(exist_ok=True)
R = json.loads((OB / "results" / "phase4" / "phase4.json").read_text())
GENS = list(R["retuned"])
RET7 = ["rule__RULE", "proto__MLPpub", "proto__HGB", "edgefeat__HGB", "gat__GAT", "mpnn__MPNN", "edgefeatraw__HGB"]
FIXED6 = [m for m in RET7 if m != "proto__HGB"]
LBL = {"rule__RULE": "Rule", "proto__MLPpub": "Proto-MLP", "proto__HGB": "Proto-HGB", "edgefeat__HGB": "Excess-HGB",
       "gat__GAT": "GAT", "mpnn__MPNN": "MPNN", "edgefeatraw__HGB": "Raw-HGB"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#b2182b", "#f0efec", "#0d366b"])
INK, MUTED, SURF = "#1f1f1f", "#6b6b6b", "#fcfcfb"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FD / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def heat(M, rows, cols, cmap, norm, labels, name, cbar):
    fig, ax = plt.subplots(figsize=(3.4 + 0.12 * len(cols), 0.24 * len(rows) + 0.9))
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                v = norm(M[i, j]) if norm else M[i, j]
                ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=6,
                        color="white" if abs(v - (0.5 if cmap is DIV else 0)) > 0.45 else INK)
    ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=cbar).outline.set_visible(False)
    save(fig, name)


def rank_heatmap(key, dets, name):
    M = np.array([[R[key][g][m] for m in dets] for g in GENS], float)
    rk = (-M).argsort(1).argsort(1) + 1
    labels = [[f"{M[i, j]:.2f}\n#{rk[i, j]}" for j in range(len(dets))] for i in range(len(GENS))]
    heat(M, GENS, [LBL[m] for m in dets], SEQ, matplotlib.colors.Normalize(0, M.max()), labels, name, "Test PR-AUC")


def tau_heatmap():
    n = len(GENS)
    M = np.eye(n)
    for p in R["tau_retuned"]["pairs"]:
        i, j = GENS.index(p["a"]), GENS.index(p["b"])
        M[i, j] = M[j, i] = p["tau"]
    M[np.triu_indices(n, 1)] = np.nan
    labels = [[f"{M[i, j]:.2f}" for j in range(n)] for i in range(n)]
    heat(M, GENS, GENS, DIV, TwoSlopeNorm(0, -1, 1), labels, "fig_tau_retuned", "Kendall's tau-b (re-tuned)")


def tuning_effect():
    dets = [m for m in FIXED6 if m not in ("rule__RULE", "proto__MLPpub")]
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    ax.axvline(0, color=MUTED, lw=0.8)
    for k, m in enumerate(dets):
        d = [R["retuned"][g][m] - R["fixed"][g][m] for g in GENS]
        ax.scatter(d, np.full(len(d), k) + np.linspace(-0.18, 0.18, len(d)), s=10, color="#0d366b",
                   edgecolor=SURF, linewidth=0.5, zorder=3)
        ax.plot([np.mean(d)] * 2, [k - 0.3, k + 0.3], color=INK, lw=1.6)
    ax.set_yticks(range(len(dets)), [LBL[m] for m in dets])
    ax.set_xlabel("PR-AUC change, re-tuned minus fixed")
    ax.grid(axis="x", color="#e6e5e1", lw=0.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig_tuning_effect")


if __name__ == "__main__":
    rank_heatmap("fixed", FIXED6, "fig_rank_fixed")
    rank_heatmap("retuned", RET7, "fig_rank_retuned")
    tau_heatmap()
    tuning_effect()
    print("figures:", sorted(p.name for p in FD.glob("*.pdf")))
