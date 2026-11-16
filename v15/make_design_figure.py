"""Figure 1 of the v15 paper: study design schematic (static; no data). Writes results_v15/figures_static/F0_design.{pdf,png}.
  python v15/make_design_figure.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results_v15" / "figures_static"
INK, INK2, FILL, C_U, C_A = "#0b0b0b", "#52514e", "#f3f2ef", "#2a78d6", "#eb6834"

COLS = [  # (title, lines); a line is (text, colour); about 30 characters per line at most; {X} = math symbol (TEX)
    ("1  Generators",
     [("prior over harm generators", INK), ("(aggregation rule, kernel, ...)", INK2), ("", INK),
      ("draws k = 100, 101, {...}", INK), ("screen: oracle ceilings", INK), ("for U and A both {>=} 0.15", INK), ("", INK),
      ("M = 100 generators", INK), ("(+ 4 spares, used only", INK2), ("if a run fails)", INK2)]),
    ("2  One generator g",
     [("scene: N = 4 users", INK), ("in-band attacker: close,", INK), ("{>=} 1 m, channel at the top", INK2),
      ("of the benign band", INK2), ("", INK), ("own cohorts:", INK), ("60 dev, 240 test", INK), ("", INK),
      ("U: unsafe", C_U), ("A: unsafe and attack-caused", C_A)]),
    ("3  Detectors and targets",
     [("MPNN and Raw-HGB,", INK), ("each trained on U and on A", INK), ("16 configurations, 4 seeds", INK2), ("", INK),
      ("PR-AUC on test cohorts", INK), ("", INK),
      ("{LU} = log PR ratio, target U", C_U), ("{LA} = log PR ratio, target A", C_A), ("{DG} = {LA} {-} {LU}", INK),
      ("variance: two-level bootstrap", INK2)]),
    ("4  Across generators",
     [("H1: {PU} = P({LU} > 0)", INK), ("random effects; 95% CI", INK2), ("inside (0.10, 0.90)?", INK2), ("", INK),
      ("H2: mean {DG} = 0?", INK), ("t-test, {alpha} = 0.025", INK2), ("", INK),
      ("flip probabilities", INK), ("{PG} (generator), {PT} (target)", INK), ("vs {PN} (noise alone)", INK)]),
]
TEX = {"LU": r"$\lambda^U_g$", "LA": r"$\lambda^A_g$", "DG": r"$\delta_g$", "PU": r"$\pi_U$", "PG": r"$\psi_G$",
       "PT": r"$\psi_T$", "PN": r"$\psi_N$", ">=": "≥", "alpha": r"$\alpha$", "-": "−", "...": "…"}


def tex(t):
    for a, b in TEX.items():
        t = t.replace("{" + a + "}", b)
    return t


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.set_xlim(0, 4); ax.set_ylim(0, 1); ax.axis("off")
    w, gap, top, bot = 0.94, 0.06, 0.9, 0.04
    for i, (title, lines) in enumerate(COLS):
        x0 = i * (w + gap) + gap / 2
        ax.add_patch(FancyBboxPatch((x0, bot), w, top - bot, boxstyle="round,pad=0.0,rounding_size=0.03",
                                    fc=FILL, ec=INK2, lw=0.6))
        ax.text(x0 + 0.04, top + 0.03, title, ha="left", va="bottom", fontsize=6.6, weight="bold", color=INK)
        for j, (t, c) in enumerate(lines):
            ax.text(x0 + 0.05, top - 0.07 - j * 0.078, tex(t),
                    ha="left", va="center", color=c, fontsize=5.7)
        if i < len(COLS) - 1:
            ax.add_patch(FancyArrowPatch((x0 + w + 0.005, 0.47), (x0 + w + gap - 0.005, 0.47), arrowstyle="-|>",
                                         mutation_scale=8, color=INK2, lw=0.8))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F0_design.{ext}", dpi=300, bbox_inches="tight")
    print(OUT / "F0_design.pdf")


if __name__ == "__main__":
    main()
