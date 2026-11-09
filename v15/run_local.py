"""
Plan v15, local execution of the pre-registered chain (deviation D1 in redesign_v15/DEVIATIONS_v15.md).

It runs the same commands, in the same order and with the same decision logic, as .github/workflows/v15.yml with
stage = all:
  G0 -> Stage X (13 generators) -> aggregate X: gates G1/G2 and M by the plan's rule (v15b, no human decision)
     -> if both gates pass: screening (M + 4 spares) -> S2 -> Stage C (R3 for the first 20 in draw order)
     -> aggregate C.
As in the workflow:
  - Stage X stops the chain if any generator fails;
  - Stage C needs S2;
  - the C aggregate uses the completed generators.

Resumable: a step whose done-marker exists is skipped, and train_v15 itself skips finished (target, detector)
pairs. After an interruption, run the same command again.

  python v15/run_local.py [--workers 3]

Logs: results_v15/local_run/ (run_local.log, and one log per job).
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results_v15"
LOG = OUT / "local_run"
PY = sys.executable
STAGE_X = ["d2", "cutoff", "fovfree", "and", "sat", "rand00", "rand02", "rand03", "rand04", "rand05", "rand06",
           "rand07", "rand11"]                               # = the workflow's stage_x matrix
SPARES = 4                                                   # = the workflow's env SPARES
LEGACY_CACHE = ROOT.parent / "rigorous_rerun" / "outputs_v2" / "cache"   # stored v2 near corpus, for check G0.6
STOP = threading.Event()                                     # set on Ctrl-C: no new job starts


def say(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG / "run_local.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def make_env(threads):
    e = dict(os.environ, XRP_GEN="v2", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
             CUDA_VISIBLE_DEVICES="-1",                      # CPU only, as on the GitHub runners
             OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
             LOKY_MAX_CPU_COUNT=str(threads))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    e["GITHUB_SHA"] = head + ("-dirty" if dirty else "")    # train_v15 records this variable as the code commit
    if LEGACY_CACHE.exists():
        e["V15_LEGACY_CACHE"] = str(LEGACY_CACHE)
    return e


def run(name, args, env):
    """Runs one command with its own log; returns True on exit code 0."""
    t0 = time.time()
    with open(LOG / f"{name}.log", "a", encoding="utf-8") as f:
        f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(args)}\n")
        f.flush()
        rc = subprocess.run([PY] + args, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    status = "ok" if rc == 0 else "interrupted (Ctrl-C), not a failure" if STOP.is_set() else f"FAILED (exit {rc})"
    say(f"{name}: {status} in {(time.time() - t0) / 60:.1f} min")
    return rc == 0


def step(name, cmds, env):
    """A step = commands run in order; skipped if its done-marker exists."""
    done = LOG / f"{name}.done"
    if done.exists():
        return True
    for args in cmds:
        if STOP.is_set() or not run(name, args, env):
            return False
    done.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    return True


def parallel(jobs, env, workers):
    """jobs: {name: [cmd, ...]}; returns the names that completed."""
    ex = ThreadPoolExecutor(workers)
    futs = {n: ex.submit(step, n, jobs[n], env) for n in jobs}
    try:
        ok = {n: f.result() for n, f in futs.items()}
    except KeyboardInterrupt:
        STOP.set()
        ex.shutdown(wait=True, cancel_futures=True)
        raise
    ex.shutdown()
    return [n for n in jobs if ok[n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--stop-after", choices=["X"], help="stop after Stage X and its aggregate (gates and M)")
    a = ap.parse_args()
    LOG.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":   # keep the computer from idle sleep while this runs (closing the lid still sleeps it)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    threads = max(1, (os.cpu_count() or 4) // a.workers)
    env = make_env(threads)
    say(f"start: commit {env['GITHUB_SHA']}, {a.workers} workers x {threads} threads, python {sys.version.split()[0]}")

    # ---------------- G0
    if not step("g0", [["v15/g0_checks.py"]], env):
        say("STOP: gate G0 failed (see results_v15/local_run/g0.log and v15/g0_report.json)")
        return 1

    # ---------------- Stage X (exploratory)
    jobs = {f"X_{g}": [["v15/train_v15.py", "--stage", "X", "--gen", g],
                       ["v15/analyse_v15.py", "summarise", "--stage", "X", "--gen", g]] for g in STAGE_X}
    done = parallel(jobs, env, a.workers)
    if len(done) < len(STAGE_X):
        say(f"STOP: Stage X incomplete ({len(done)}/{len(STAGE_X)}); fix and run again")
        return 1
    if not step("x_aggregate", [["v15/analyse_v15.py", "aggregate", "--stage", "X"]], env):
        say("STOP: Stage X aggregate failed")
        return 1
    x = json.loads((OUT / "analysis_X.json").read_text())["stage_x"]
    gates, M = bool(x["G1_pass"] and x["G2_pass"]), int(x["reestimation"]["M_chosen"])
    say(f"v15b: G1_pass={x['G1_pass']} G2_pass={x['G2_pass']} M={M}")
    if not gates:
        say("STOP: a feasibility gate failed; Stage C is not run (plan 6.1)")
        return 0
    if a.stop_after == "X":
        say("STOPPED after Stage X, as requested (--stop-after X). Run again without it to continue with Stage C.")
        return 0

    # ---------------- Stage C (confirmatory)
    if not step("screen", [["v15/screen_v15.py", "--M", str(M), "--spares", str(SPARES)]], env):
        say("STOP: screening failed")
        return 1
    if not step("S2", [["v15/train_v15.py", "--stage", "S2"]], env):
        say("STOP: S2 failed, so Stage C is not run (as in the workflow)")
        return 1
    gens = json.loads((OUT / "screen" / "stage_c_generators.json").read_text())["generators"]
    jobs = {f"C_{g}": [["v15/train_v15.py", "--stage", "C", "--gen", g] + (["--r3"] if i < 20 else []),
                       ["v15/analyse_v15.py", "summarise", "--stage", "C", "--gen", g]] for i, g in enumerate(gens)}
    done = parallel(jobs, env, a.workers)
    say(f"Stage C: {len(done)}/{len(gens)} generators completed (M = {M} needed)")
    if not step("c_aggregate", [["v15/analyse_v15.py", "aggregate", "--stage", "C"]], env):
        say("STOP: Stage C aggregate failed")
        return 1
    say("FINISHED: results_v15/analysis_C.json")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        STOP.set()
        say("PAUSED by the user (Ctrl-C). Finished work is kept; run the same command to resume.")
        sys.exit(130)
