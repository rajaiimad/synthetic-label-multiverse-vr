"""
Plan v15, Stage C screening (section 6.2): draw generators k = 100, 101, ... from the plan-v7 prior, in order, until M
qualify. A draw qualifies if the oracle ceilings for target U and the attainable ceiling for target A (Monte Carlo over
the benign plan, K = 32) are both >= 0.15 on its own 20-cohort screening corpus (in-band attack). Uses no detector.
Writes results_v15/screen/stage_c_generators.json (qualifying names in draw order, and the log of every draw).

  python v15/screen_v15.py --M 80
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XRP_GEN", "v2")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_v15 as G  # noqa: E402
from stage3 import calibrate_gain  # noqa: E402
from stage7 import sample_prior  # noqa: E402

THRESH = 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=80)
    ap.add_argument("--spares", type=int, default=4, help="extra qualifying draws, used only to replace failed generators")
    ap.add_argument("--kmax", type=int, default=998)   # D4: the plan says "until M qualify"; 998 is the largest k with disjoint ranges
    a = ap.parse_args()
    od = HERE.parent / "results_v15" / "screen"
    od.mkdir(parents=True, exist_ok=True)
    log, chosen, t0 = [], [], time.time()
    total = a.M + a.spares
    for k in range(100, a.kmax + 1):
        v = sample_prior(k)
        r = G.stage_c_ranges(k)
        G.assert_disjoint([r["dev"], r["test"], r["screen"]])
        gain = calibrate_gain(v)
        t = G.generate_v15(r["screen"][0], r["screen"][1] - r["screen"][0], v, gain, inband=True, k_mc=G_K)
        c = G.ceilings(t)
        ok = c["U"] >= THRESH and c["A"] >= THRESH
        log.append({"k": k, "name": f"c{k}", "agg": v.get("agg", "sum"), "gain": gain, "ceilings": c,
                    "prev_U": float(t["y"].mean()), "prev_A": float(t["y_A"].mean()), "qualifies": ok})
        if ok:
            chosen.append(f"c{k}")
        print(f"k {k} agg {v.get('agg', 'sum'):4s} ceil U {c['U']:.3f} A {c['A']:.3f} -> {'IN' if ok else 'out'} "
              f"({len(chosen)}/{total}, {time.time() - t0:.0f}s)", flush=True)
        if len(chosen) == total:
            break
    if len(chosen) < total:
        raise SystemExit(f"only {len(chosen)} qualifying generators up to k = {a.kmax}")
    (od / "stage_c_generators.json").write_text(json.dumps({"M": a.M, "spares": a.spares, "threshold": THRESH, "generators": chosen,
                                                            "draws": log}, indent=1))
    print(json.dumps(chosen))


G_K = 32

if __name__ == "__main__":
    main()
