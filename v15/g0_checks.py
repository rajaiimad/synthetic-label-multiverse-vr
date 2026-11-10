"""
Plan v15, gate G0 (section 4.6): code validation before any training. Writes v15/g0_report.json; exit code 1 on failure.
Run with XRP_GEN=v2 from the repository root:  python v15/g0_checks.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("XRP_GEN", "v2")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_v15 as G  # noqa: E402
from stage3 import V2_GAIN, calibrate_gain, generate  # noqa: E402
from stage7 import STRUCT, sample_prior  # noqa: E402
from common import CACHE  # noqa: E402

REP = {"checks": [], "passed": True}
SX = json.loads((HERE / "stage_x_generators.json").read_text())


def chk(name, ok, info=None):
    REP["checks"].append({"check": name, "ok": bool(ok), "info": info})
    REP["passed"] &= bool(ok)
    print(("PASS " if ok else "FAIL ") + name + ("" if info is None else f"  {info}"), flush=True)


STAGE3_FIELDS = ["ctx", "V_raw", "E_src", "E_chan", "E_geo", "E_w", "y_att", "sev", "oracle_score", "peer_term", "own_prov",
                 "atk_id", "atk_ch", "atk_term", "s_noisy", "y", "V_bb", "E_bb", "N", "T"]


def same(a, b):
    bad = [k for k in STAGE3_FIELDS if not np.array_equal(a[k], b[k])]
    return not bad, bad


# generators covering every aggregator: v2, one structural, and the first prior draws with sum, max and soft maximum
by_agg = {}
for k in range(12):
    by_agg.setdefault(sample_prior(k).get("agg", "sum"), k)
CASES = [("v2", {}, V2_GAIN), ("d2", STRUCT["d2"], SX["d2"]["gain"])] + \
        [(f"rand{k:02d}", sample_prior(k), SX.get(f"rand{k:02d}", {}).get("gain", 3.0)) for k in sorted(by_agg.values())]
print("cases:", [(c[0], c[1].get("agg", "sum")) for c in CASES], flush=True)

SEED = 60_000
for name, v, gain in CASES:
    # 1. without the in-band attacker: identical to stage3.generate
    a = generate(SEED, 2, v, gain)
    b = G.generate_v15(SEED, 2, v, gain, inband=False)
    ok, bad = same(a, b)
    chk(f"G0.1 {name}: identical to stage3 (standard attack)", ok, bad or None)
    # 2. with the in-band attacker: identical to stage3.generate with near=True
    a = generate(SEED, 2, {**v, "near": True}, gain)
    b = G.generate_v15(SEED, 2, v, gain, inband=True, k_mc=8)
    ok, bad = same(a, b)
    chk(f"G0.2 {name}: identical to stage3 (in-band attack)", ok, bad or None)
    # 3. definitions
    chk(f"G0.3 {name}: y_A <= y and y_A only on victims of an attack",
        bool(np.all(b["y_A"] <= b["y"]) and np.all(b["y_A"] <= b["victim_of_attack"])),
        {"y": int(b["y"].sum()), "y_A": int(b["y_A"].sum()), "y_A0": int(b["y_A0"].sum()), "prevented": int(b["prevented"].sum())})
    chk(f"G0.3 {name}: score64 equals stored oracle score and noise reproduces s_noisy",
        bool(np.array_equal(b["score64"].astype(np.float32), b["oracle_score"]) and
             np.array_equal((b["score64"] + b["noise"]).astype(np.float32), b["s_noisy"])))
    chk(f"G0.3 {name}: removing the attacker never raises the score",
        bool(np.all(b["score_abs"] <= b["score64"] + 1e-9)))
    if v.get("agg", "sum") == "sum":
        # attacker-absent score equals score minus the stored attacker term (Stage-4 definition) on attacked victims
        m = b["victim_of_attack"] == 1
        err = float(np.abs((b["score64"] - b["score_abs"])[m] - b["atk_term"][m].astype(np.float64)).max())
        chk(f"G0.4 {name}: attacker-absent recomputation equals the stored attacker term", err < 1e-5, err)
        s, at = b["s_noisy"].astype(np.float64), b["atk_term"].astype(np.float64)
        stage4 = ((s > 0.30) & (s - at <= 0.30) & m).astype(int)
        mism = int((stage4 != b["y_A0"]).sum())
        chk(f"G0.4 {name}: y_A0 equals the Stage-4 attacker-induced label (float32 rounding only)", mism <= 2, mism)
    # 5. determinism
    c = G.generate_v15(SEED, 2, v, gain, inband=True, k_mc=8)
    chk(f"G0.5 {name}: deterministic", all(np.array_equal(b[k], c[k]) for k in b), None)
    print(f"   {name} ceilings (2 cohorts, indicative only): {G.ceilings(b)}", flush=True)

# 6. the existing v2 near corpus (cohorts 70000-70001), when available locally
f = Path(os.environ.get("V15_LEGACY_CACHE", str(CACHE))) / "s7_near_near.npz"
if f.exists():
    z = np.load(f)
    b = G.generate_v15(70_000, 2, {}, V2_GAIN, inband=True)
    n = len(b["y"])
    # severity-model outputs are floating-point products: allowed to differ in the last digits (BLAS summation order)
    bb = [k for k in ("V_bb", "E_bb")]
    bad = [k for k in STAGE3_FIELDS if k not in ("N", "T", *bb) and not np.array_equal(z[k][:n], b[k])]
    dmax = max(float(np.abs(z[k][:n] - b[k]).max()) for k in bb)
    chk("G0.6 reproduces the stored v2 in-band corpus s7_near_near.npz (first 2 cohorts; backbone outputs within 1e-4)",
        not bad and dmax < 1e-4, {"different_fields": bad, "backbone_max_abs_diff": dmax})
else:
    chk("G0.6 stored v2 in-band corpus not available here (checked by G0.2 instead)", True, "skipped")

# 7. gains of the Stage X generators are reproduced by the label-free calibration
for name, v in (("d2", STRUCT["d2"]), ("rand00", sample_prior(0))):
    g = calibrate_gain(v)
    chk(f"G0.7 {name}: recalibrated gain equals the stored E1 gain", abs(g - SX[name]["gain"]) < 1e-9, (g, SX[name]["gain"]))

# 8. Stage X specifications equal the code's generator definitions
norm_ = lambda d: json.loads(json.dumps(d))  # noqa: E731  (tuples -> lists)
for name in SX:
    spec = STRUCT[name] if name in STRUCT else sample_prior(int(name[4:]))
    chk(f"G0.8 {name}: stored specification equals the code", norm_(spec) == SX[name]["spec"])

# 9. seed ranges of Stage C (k = 100..998, deviation D4), the base generator and the training seeds
rngs = [r for k in range(100, 999) for r in G.stage_c_ranges(k).values()] + [G.BASE_S2_DEV]
try:
    G.assert_disjoint(rngs)
    chk("G0.9 Stage C, base-generator and earlier seed ranges are disjoint", True)
except AssertionError as e:
    chk("G0.9 seed ranges disjoint", False, str(e))
chk("G0.9 Stage C training seeds are new", not set(G.TRAIN_SEEDS_C) & set(G.TRAIN_SEEDS_X), G.TRAIN_SEEDS_C)

(HERE / "g0_report.json").write_text(json.dumps(REP, indent=1, default=str))
print("G0", "PASSED" if REP["passed"] else "FAILED", f"({sum(c['ok'] for c in REP['checks'])}/{len(REP['checks'])})")
sys.exit(0 if REP["passed"] else 1)
