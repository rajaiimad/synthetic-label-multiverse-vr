"""Plan v15: final design comparison at four seeds (split-half needs an even number). Same simulation and analysis as
power_sim_v15.py. Scenarios: base, proxy_random (six prior-drawn proxy generators), weak; and the H1/H2 null checks.
Writes power_sim_v15_design.json."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
src = (HERE / "power_sim_v15.py").read_text()
ns = {"__file__": str(HERE / "power_sim_v15.py")}
exec(compile(src.split("# proxy-based scenarios")[0], "power_sim_v15", "exec"), ns)
from scipy.stats import norm  # noqa: E402
base = dict(muU=0.03, tauU=0.065, mud=0.04, taud=0.06, rhoG=-0.5, rhoE=0.3, fA=1.3)
SC = {"base": base,
      "proxy_random": dict(muU=0.03, tauU=0.065, mud=0.028, taud=0.07, rhoG=-0.76, rhoE=0.3, fA=1.3),
      "weak": {**base, "tauU": 0.035, "mud": 0.02},
      "H1_boundary_pi0.90": {**base, "muU": norm.ppf(0.90) * 0.065},
      "H2_null": {**base, "mud": 0.0}}
res = []
for name, sc in SC.items():
    for (M, n, S) in [(60, 240, 4), (70, 240, 4), (80, 120, 4), (80, 240, 4), (100, 240, 4)]:
        r = ns["simulate"](sc, M, n, S, R=600, B=300, seed=abs(hash((name, M, n, S))) % 2 ** 31)
        res.append({"scenario": name, "M": M, "n": n, "S": S, **r})
        print(f"{name:20s} M {M:3d} n {n:3d} S {S}  H1 {r['powerH1']:.2f}  H2 {r['powerH2']:.2f}  cov90 {r['coverage_pi90']:.2f}  "
              f"psiG± {r['psiG_ci95_halfwidth']:.2f}  muD± {r['mu_delta_ci95_halfwidth']:.3f}  psiN {r['psiN_design']:.2f}  cpu {r['cost_cpu_h']:.0f}", flush=True)
(HERE / "power_sim_v15_design.json").write_text(json.dumps({"scenarios": SC, "results": res}, indent=1))
