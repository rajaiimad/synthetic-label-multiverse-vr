"""Plan v15: extra scenario taken directly from the six prior-drawn proxy generators (mu_d 0.028, tau_d 0.07, rho_G -0.76).
Uses the same simulation and analysis as power_sim_v15.py. Writes power_sim_v15_proxyrandom.json."""
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.argv = ["x"]
src = (HERE / "power_sim_v15.py").read_text()
ns = {"__file__": str(HERE / "power_sim_v15.py")}
exec(compile(src.split("# proxy-based scenarios")[0], "power_sim_v15", "exec"), ns)
sc = dict(muU=0.03, tauU=0.065, mud=0.028, taud=0.07, rhoG=-0.76, rhoE=0.3, fA=1.3)
res = []
for (M, n, S) in [(40, 240, 6), (50, 240, 6), (60, 240, 6), (60, 480, 6), (80, 240, 4)]:
    r = ns["simulate"](sc, M, n, S, R=600, seed=M * 1000 + n + S)
    res.append({"M": M, "n": n, "S": S, **r})
    print(M, n, S, "H1 %.2f H2 %.2f psiG± %.2f muD± %.3f cpu %.0f" % (r["powerH1"], r["powerH2"], r["psiG_ci95_halfwidth"], r["mu_delta_ci95_halfwidth"], r["cost_cpu_h"]))
(HERE / "power_sim_v15_proxyrandom.json").write_text(json.dumps({"scenario": sc, "results": res}, indent=1))
