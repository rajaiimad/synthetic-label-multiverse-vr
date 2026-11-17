"""
Deviation D6: import the artifacts of the GitHub run of the pre-registered chain and verify them.

  python v15/import_github_run.py 43888871250

1. Copies (never moves) results_v15_github/<run>/ artifacts into the layout the analysis code reads:
   C-<gen> -> results_v15/C/<gen>, screen -> results_v15/screen, S2 -> results_v15/S2/v2,
   C-analysis -> results_v15/analysis_C_github.json. The downloaded artifacts stay untouched.
2. Checks: every screened generator has a complete output; every meta.json records the run's commit; the first M
   completed generators in draw order are the ones the runner analysed.
3. Re-runs the pre-registered aggregate locally (analyse_v15.aggregate) on the downloaded per-generator summaries and
   compares every number with the runner's analysis_C.json.
4. Compares the runner's Stage X analysis (X-analysis) with the local Stage X analysis (deviation D1).
Writes results_v15/github_import_report.json and prints a summary. Exit code 1 if a check fails.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results_v15"
TOL = 1e-9


def flat(o, p=""):
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            out.update(flat(v, f"{p}.{k}" if p else str(k)))
        return out
    if isinstance(o, list):
        out = {}
        for i, v in enumerate(o):
            out.update(flat(v, f"{p}.{i}"))
        return out
    return {p: o}


def compare(a, b):
    """Numbers that differ by more than TOL (absolute) between two analysis dicts, and keys present in only one."""
    fa, fb = flat(a), flat(b)
    diff = []
    for k in sorted(set(fa) | set(fb)):
        x, y = fa.get(k), fb.get(k)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            if abs(float(x) - float(y)) > TOL and not (x != x and y != y):
                diff.append((k, x, y))
        elif x != y:
            diff.append((k, x, y))
    return diff


def main(run):
    src = ROOT / "results_v15_github" / run
    arts = sorted(p.name for p in src.iterdir() if p.is_dir())
    rep = {"run": run, "n_artifacts": len(arts), "checks": {}}
    ok = True

    def check(name, cond, info=None):
        nonlocal ok
        rep["checks"][name] = {"ok": bool(cond), "info": info}
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name + (f"  {info}" if info is not None else ""))

    # ---- 1. arrange (copy)
    (RES / "C").mkdir(parents=True, exist_ok=True)
    # Saved models are not read by the analysis and their copied paths exceed the Windows 260-character limit in the
    # authors' folder; they stay (archived) in results_v15_github/<run>/.
    skip = shutil.ignore_patterns("models")
    for a in arts:
        if a.startswith("C-") and a != "C-analysis":
            shutil.copytree(src / a, RES / "C" / a[2:], dirs_exist_ok=True, ignore=skip)
    shutil.copytree(src / "screen", RES / "screen", dirs_exist_ok=True)
    shutil.copytree(src / "S2", RES / "S2" / "v2", dirs_exist_ok=True, ignore=skip)
    shutil.copy(src / "C-analysis" / "analysis_C.json", RES / "analysis_C_github.json")

    # ---- 2. completeness and provenance
    sc = json.loads((RES / "screen" / "stage_c_generators.json").read_text())
    gens = sc["generators"]
    check("screening lists M + 4 generators", len(gens) == sc["M"] + sc["spares"], f"{len(gens)} = {sc['M']} + {sc['spares']}")
    check("screening M equals the v15b value 100", sc["M"] == 100, sc["M"])
    old = RES / "superseded_local_stageC_2026-11-10" / "screen" / "stage_c_generators.json"
    if sys.platform == "win32":   # this path exceeds the 260-character Windows limit in the authors' folder
        old = Path("\\\\?\\" + str(old.resolve()))
    if old.exists():   # the runner must reproduce the local screening of 2026-11-10 (same draws and decisions); gains and
        so = json.loads(old.read_text())   # ceilings may differ in the last digits between Linux and Windows
        same = [a["name"] for a in so["draws"]] == [b["name"] for b in sc["draws"]] and \
            all(a["qualifies"] == b["qualifies"] and a["agg"] == b["agg"] for a, b in zip(so["draws"], sc["draws"]))
        dmax = {k: max(abs(a["ceilings"][k] - b["ceilings"][k]) for a, b in zip(so["draws"], sc["draws"])) for k in ("U", "A", "A0")}
        dmax["gain"] = max(abs(a["gain"] - b["gain"]) for a, b in zip(so["draws"], sc["draws"]))
        margin = min(min(abs(b["ceilings"]["U"] - sc["threshold"]), abs(b["ceilings"]["A"] - sc["threshold"])) for b in sc["draws"])
        rep["screening_vs_local"] = {"max_abs_diff": dmax, "closest_ceiling_to_threshold": margin}
        check("runner screening reproduces the local screening of 2026-11-10 (same draws, decisions and qualified list; "
              "ceilings within 1e-4)", same and so["generators"] == gens and max(dmax["U"], dmax["A"]) < 1e-4,
              f"{len(sc['draws'])} draws, {len(gens)} qualified; max differences {dmax}; closest ceiling to threshold {margin:.2g}")
    need = ("summary_v15.json", "labels_test.npz", "meta.json", "preds_U_mpnn.npy", "preds_U_rawhgb.npy",
            "preds_A_mpnn.npy", "preds_A_rawhgb.npy", "preds_S2U_mpnn.npy", "preds_S2A_rawhgb.npy")
    missing = {g: [f for f in need if not (RES / "C" / g / f).exists()] for g in gens}
    missing = {g: m for g, m in missing.items() if m}
    check("every screened generator has complete outputs", not missing, missing or f"{len(gens)} complete")
    commits = sorted({json.loads((RES / "C" / g / "meta.json").read_text())["git_commit"] for g in gens})
    rep["commits"] = commits
    check("all Stage C outputs record one commit", len(commits) == 1, commits)
    r3 = [g for g in gens if (RES / "C" / g / "preds_UR3_mpnn.npy").exists()]
    s5 = [g for g in gens if (RES / "C" / g / "preds_S5U_mpnn.npy").exists()]
    check("R3 ran on the first 20 generators in draw order", r3 == gens[:20], len(r3))
    check("S5 ran on the first 10 generators in draw order", s5 == gens[:10], len(s5))

    # ---- 3. reproduce the aggregate locally and compare
    gh = json.loads((RES / "analysis_C_github.json").read_text())
    check("runner analysed the first M generators in draw order", gh["generators"] == gens[:sc["M"]], len(gh["generators"]))
    subprocess.run([sys.executable, str(ROOT / "v15" / "analyse_v15.py"), "aggregate", "--stage", "C"], check=True,
                   capture_output=True)
    loc = json.loads((RES / "analysis_C.json").read_text())
    d = compare(gh, loc)
    rep["aggregate_differences"] = [list(x) for x in d[:50]]
    check("local re-run of the aggregate reproduces the runner's analysis_C.json", not d,
          f"{len(d)} differing numbers (tolerance {TOL})")

    # ---- 4. Stage X: runner vs local
    xg = json.loads((src / "X-analysis" / "analysis_X.json").read_text())
    xl = json.loads((RES / "analysis_X.json").read_text())
    dx = compare(xg, xl)
    big = [x for x in dx if isinstance(x[1], (int, float)) and isinstance(x[2], (int, float)) and abs(x[1] - x[2]) > 1e-6]
    rep["stage_x"] = {"runner_v15b": xg["stage_x"]["reestimation"]["M_chosen"], "local_v15b": xl["stage_x"]["reestimation"]["M_chosen"],
                      "runner_gates": [xg["stage_x"]["G1_pass"], xg["stage_x"]["G2_pass"]],
                      "n_numbers_differing": len(dx), "n_differing_above_1e-6": len(big),
                      "largest": sorted(([k, a, b] for k, a, b in big), key=lambda t: -abs(t[1] - t[2]))[:10]}
    gx = xl["generators"]
    lam = {(w, t): [a["context"][x][f"lam_{t}"] for x in gx] for w, a in (("r", xg), ("l", xl)) for t in "UA"}
    rx, lx = xg["stage_x"], xl["stage_x"]
    rep["stage_x"].update({
        "n_generators": len(gx), "G1_runner": rx["G1_prior_detectable_both"], "G1_local": lx["G1_prior_detectable_both"],
        "G2_lift_runner": rx["G2_median_lift_A"], "G2_lift_local": lx["G2_median_lift_A"],
        "power_H2_M100_runner": rx["reestimation"]["power_H2_by_M"]["100"],
        "power_H2_M100_local": lx["reestimation"]["power_H2_by_M"]["100"],
        "taud_runner": rx["reestimation"]["taud"], "taud_local": lx["reestimation"]["taud"],
        **{f"n_winner_differs_{t}": sum((a > 0) != (b > 0) for a, b in zip(lam[("r", t)], lam[("l", t)])) for t in "UA"},
        **{f"max_abs_diff_lambda_{t}": max(abs(a - b) for a, b in zip(lam[("r", t)], lam[("l", t)])) for t in "UA"}})
    check("runner Stage X gives the same gates and M as the local Stage X",
          rep["stage_x"]["runner_v15b"] == rep["stage_x"]["local_v15b"] and all(rep["stage_x"]["runner_gates"]),
          {k: rep["stage_x"][k] for k in ("runner_v15b", "local_v15b", "runner_gates", "n_numbers_differing", "n_differing_above_1e-6")})
    rep["passed"] = bool(ok)
    (RES / "github_import_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print("IMPORT", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
