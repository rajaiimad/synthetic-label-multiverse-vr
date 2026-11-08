"""
Deviation D3 (DEVIATIONS_v13.md; plan v13 section 3F): plan v11's pre-specified percentile test on the E1 re-tuned
family recomputed with B = 20,000 cohort-bootstrap replicates. Run with XRP_GEN=v2 from option_b/code.
Writes option_b/results/phase5/b20k.json. Reads predictions read-only.
Double check built in: the first 1,000 replicates of boot_weights(60, 20000, 0) equal boot_weights(60, 1000, 0), so the
B = 1000 replicate vectors saved by phase4.py are reproduced exactly from the first 1,000 columns.
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np

import phase4 as P
from ob_stats import Sorted, boot_weights, cohort_ids, holm, p_two_sided

B = 20000


def one(args):
    g, m = args
    y = np.load(P.CACHE / f"s7_{g}_test.npz")["y"]
    W = np.array(boot_weights(60, B, 0))
    S = [Sorted(p, y, cohort_ids(len(y))) for p in np.load(P.retuned_path(g, m))]
    return g, m, float(np.mean([s.ap() for s in S])), np.mean([[s.ap(w) for w in W] for s in S], 0)


def main():
    t0 = time.time()
    W = np.array(boot_weights(60, B, 0))
    assert np.array_equal(W[:1000], np.array(boot_weights(60, 1000, 0)))
    jobs = [(g, m) for g in P.GENS for m in P.RET7]
    blk = {}
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as ex:
        for g, m, pt, bs in ex.map(one, jobs):
            blk.setdefault(g, {})[m] = (pt, bs)
            print(g, m, round(pt, 4), f"{time.time() - t0:.0f}s", flush=True)
    saved = np.load(P.OD / "boot_retuned.npz")
    err = max(float(np.abs(blk[g][m][1][:1000] - saved[f"{g}|{m}"]).max()) for g in P.GENS for m in P.RET7)
    assert err < 1e-12, err
    rows = []
    for g in P.GENS:
        for a, b in combinations(P.RET7, 2):
            d = blk[g][a][1] - blk[g][b][1]
            rows.append({"gen": g, "a": a, "b": b, "diff": blk[g][a][0] - blk[g][b][0], "p": p_two_sided(d, B)})
    for r, adj in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = float(adj)
    rev = P.reversals(rows)
    out = {"B": B, "n": len(rows), "min_attainable_holm_p": len(rows) / B, "replicate_check_max_abs_err_vs_B1000": err,
           "n_sig": int(sum(r["p_holm"] < 0.05 for r in rows)), "n_at_floor": int(sum(r["p"] <= 1 / B for r in rows)),
           "reversals": rev, "rows": rows, "seconds": time.time() - t0}
    (P.OD.parent / "phase5").mkdir(exist_ok=True)
    (P.OD.parent / "phase5" / "b20k.json").write_text(json.dumps(out, indent=1))
    print("B20K DONE n_sig", out["n_sig"], "reversals", len(rev), "at floor", out["n_at_floor"], flush=True)


if __name__ == "__main__":
    main()
