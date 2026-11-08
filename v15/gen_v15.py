"""
Plan v15: corpus generation with the in-band attacker and the counterfactual targets.

`generate_v15` reproduces `rigorous_rerun.stage3.generate` exactly (same random-number consumption, same fields; checked
by g0_checks.py) and adds, for every victim-frame:

  y_A        attack-caused: unsafe under the attack and safe had the attacker followed its benign plan b0
             (original pose, benign draw for the attack channel; same noise, same other peers)        [plan 4.2]
  y_A0       attacker-absent sensitivity target: unsafe, and safe without the attacker's contribution [plan 4.5]
  prevented  safe under the attack but unsafe under b0                                               [plan 4.3 P4]
  score_b0, score_abs   deterministic scores under b0 / without the attacker
  p_A_obs    attainable posterior of y_A: the full actual state is known, b0 is integrated out by K Monte Carlo draws
             from its distribution                                                                      [plan 4.3 P7]

The counterfactuals are computed for every aggregator (sum, max, soft maximum) by recomputing the aggregator with the
attacker's argument replaced (b0) or removed (absent). Dose generators and moving avatars are not supported (asserted).
Extra random streams never touch the generator's main stream:  b0 channel [seed, track, 9] (as plan v9);
Monte Carlo draws [seed, track, 30, k] (pose) and [seed, track, 31, k] (channel).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rigorous_rerun"))
from common import PEER_HI, PEER_NAMES, USER_CH, get_backbone  # noqa: E402
from coper_vr.config import PEER_CHANNELS  # noqa: E402
from coper_vr.data.gen_param import GenParams, _peer_state_param  # noqa: E402
from coper_vr.data.xrperil_multi import _fov_overlap, _sample_avatar_pose, _sim_user_track  # noqa: E402
from stage3 import BEN_HI, _near_pose, _rot, edge_contrib  # noqa: E402

THETA = 0.30

# ---------------------------------------------------------------- seed ranges (plan 6.3); every new range is checked
USED_RANGES = {
    "main": (42, 102), "calib_backbone": (10_000, 10_060), "lockbox1": (20_000, 20_120),
    "sizes": (30_000, 33_060), "shifts": (40_000, 45_040), "adaptive": (50_000, 54_040),
    "e1_dev": (60_000, 60_060), "e1_test": (61_000, 61_060), "near_v2": (70_000, 70_060), "lockbox2": (80_000, 80_120),
}
TRAIN_SEEDS_X = (0, 1, 2, 42, 2024)
TRAIN_SEEDS_C = (101, 211, 307, 401)


def stage_c_ranges(k):
    """Cohort seed ranges of Stage C generator k (prior index k >= 100)."""
    assert 100 <= k < 999
    return {"dev": (1_000_000 + 1000 * k, 1_000_000 + 1000 * k + 60),
            "test": (2_000_000 + 1000 * k, 2_000_000 + 1000 * k + 240),
            "screen": (3_000_000 + 1000 * k, 3_000_000 + 1000 * k + 20)}


BASE_S2_DEV = (4_000_000, 4_000_060)


def assert_disjoint(ranges):
    allr = list(USED_RANGES.values()) + list(ranges)
    for i, (a0, a1) in enumerate(allr):
        for b0, b1 in allr[i + 1:]:
            assert a1 <= b0 or b1 <= a0, f"seed ranges overlap: {(a0, a1)} {(b0, b1)}"


# ---------------------------------------------------------------- aggregation (identical formulas to stage3)
def aggregate(C, js, agg, v):
    """C: (N, T) float32 contributions at gain 1 (row of the victim is zero); js: the peers included."""
    if agg == "max":
        Z = np.zeros_like(C)
        Z[js] = C[js]
        return Z.max(0)
    if agg == "lse":
        tmp = v["lse_t"]
        Cj = C[js].astype(np.float64)
        m = Cj.max(0)
        return m + tmp * np.log(np.exp((Cj - m) / tmp).sum(0))
    Z = np.zeros_like(C)
    Z[js] = C[js]
    return Z.sum(0)


def generate_v15(seed0, n_cohorts, v, gain, inband=True, k_mc=0, N=4, tracks=8, T=48, with_backbone=True):
    """As stage3.generate(seed0, n_cohorts, {**v, 'near': True} if inband else v, gain), plus the v15 fields."""
    v = dict(v)
    if inband:
        v["near"] = True                                  # within-band attacker: close, in view, channel at band top
    agg, dose, spike_p, moving = v.get("agg", "sum"), v.get("dose"), v.get("spike_p", 0.0), v.get("moving", False)
    assert not dose and not moving and not v.get("dmin") and not v.get("clamp") and not v.get("oov") \
        and not v.get("symptom"), "unsupported variant"
    assert v.get("near_clamp", True), "v15 uses the top-of-band in-band attacker"
    params = GenParams(temporal_profile=v.get("profile", "ramp"))
    sigma = v.get("sigma", 0.05)
    rows = n_cohorts * tracks * N * T
    K = N - 1
    out = dict(ctx=np.zeros((rows, 4), np.int64), V_raw=np.zeros((rows, 20), np.float32),
               E_src=np.zeros((rows, K), np.int64), E_chan=np.zeros((rows, K, 5), np.float32),
               E_geo=np.zeros((rows, K, 3), np.float32), E_w=np.zeros((rows, K), np.float32),
               y_att=np.zeros((rows, K), np.int64), sev=np.zeros(rows, np.int64),
               oracle_score=np.zeros(rows, np.float32), peer_term=np.zeros(rows, np.float32),
               own_prov=np.zeros(rows, np.float32), atk_id=np.zeros(rows, np.int64), atk_ch=np.zeros(rows, np.int64),
               atk_term=np.zeros(rows, np.float32), s_noisy=np.zeros(rows, np.float32))
    ext = dict(y_A=np.zeros(rows, np.int64), y_A0=np.zeros(rows, np.int64), prevented=np.zeros(rows, np.int64),
               score_b0=np.zeros(rows, np.float64), score_abs=np.zeros(rows, np.float64),
               score64=np.zeros(rows, np.float64), noise=np.zeros(rows, np.float64),
               p_A_obs=np.zeros(rows, np.float64), victim_of_attack=np.zeros(rows, np.int64))
    user_rows = np.zeros((rows, 20), np.float32)
    r = 0
    for cid in range(n_cohorts):
        rng = np.random.default_rng(seed0 + cid)
        hmds = rng.integers(0, 4, N)
        for tk in range(tracks):
            poses = [_sample_avatar_pose(rng, 8.0) for _ in range(N)]
            udf = [_sim_user_track(rng, T, int(hmds[u])) for u in range(N)]
            is_atk = rng.random() < 0.5
            aid = int(rng.integers(0, N)) if is_atk else -1
            ach = str(rng.choice(PEER_NAMES)) if is_atk else ""
            pose0 = poses[aid] if is_atk else None        # the attacker's benign plan: its originally sampled pose
            if v.get("near") and is_atk:
                poses[aid] = _near_pose(np.random.default_rng([seed0 + cid, tk, 8]), poses, aid)
            pdf = [_peer_state_param(rng, T, u == aid, ach if u == aid else "", params) for u in range(N)]
            PC = np.stack([p[PEER_NAMES].values.astype(np.float32) for p in pdf])
            if spike_p > 0:
                for u in range(N):
                    if u == aid:
                        continue
                    m = rng.random(T) < spike_p
                    ks = rng.integers(0, 5, T)
                    for t in np.where(m)[0]:
                        k = ks[t]
                        PC[u, t, k] = rng.uniform(BEN_HI[k], BEN_HI[k] + 0.5 * (PEER_HI[k] - BEN_HI[k]))
            PC_b0 = None
            if is_atk:
                k = PEER_NAMES.index(ach)
                PC_b0 = PC[aid].copy()                    # benign plan: attack channel drawn from its benign range
                PC_b0[:, k] = np.random.default_rng([seed0 + cid, tk, 9]).uniform(PEER_CHANNELS[k].benign_lo, BEN_HI[k], T)
            if v.get("near") and v.get("near_clamp", True) and is_atk:
                PC[aid, :, k] = BEN_HI[k]
            pos = np.zeros((N, T, 3)); ori = np.zeros((N, T, 3)); fov = np.array([p[2] for p in poses])
            for u in range(N):
                vel = np.zeros(3)
                yaw = 0.0
                for t in range(T):
                    pos[u, t] = np.asarray(poses[u][0]) + vel * t
                    ori[u, t] = _rot(np.asarray(poses[u][1]), yaw * t)
            nrm = PC / PEER_HI
            D = np.zeros((N, N, T)); OV = np.zeros((N, N, T))
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    for t in range(T):
                        D[i, j, t] = np.linalg.norm(pos[j, t] - pos[i, t])
                        OV[i, j, t] = _fov_overlap(pos[i, t], ori[i, t], pos[j, t], fov[i])
            for i in range(N):
                js = [j for j in range(N) if j != i]
                own = (udf[i]["optic_flow"].values / 50.0 + np.abs(udf[i]["yaw_rate"].values) / 300.0
                       + udf[i]["latency_ms"].values / 200.0) / 3.0
                C = np.zeros((N, T), np.float32)
                for j in js:
                    C[j] = edge_contrib(nrm[j], D[i, j], OV[i, j], v)
                if agg == "max":
                    peer = C.max(0)
                elif agg == "lse":
                    tmp = v["lse_t"]
                    Cj = C[js].astype(np.float64)
                    m = Cj.max(0)
                    peer = m + tmp * np.log(np.exp((Cj - m) / tmp).sum(0))
                else:
                    peer = C.sum(0)
                g = gain
                score = own + g * peer
                noise = rng.normal(0, sigma, T)
                s = score + noise
                sev = np.zeros(T, np.int64); sev[s > 0.15] = 1; sev[s > 0.30] = 2; sev[s > 0.55] = 3
                attributed = is_atk and i != aid and (g * C[aid]).mean() > 0.02
                sl = slice(r, r + T)
                o = out
                o["ctx"][sl] = np.c_[np.full(T, cid), np.full(T, tk), np.arange(T), np.full(T, i)]
                user_rows[sl] = udf[i][USER_CH].values
                o["E_src"][sl] = js
                o["E_chan"][sl] = PC[js].transpose(1, 0, 2)
                d3 = np.stack([D[i, js].T, OV[i, js].T, np.clip(1 - D[i, js].T / 6.0, 0, 1)], 2)
                o["E_geo"][sl] = d3
                o["E_w"][sl] = 1 / (1 + np.exp(-(1.5 * np.exp(-d3[..., 0] / 3) + d3[..., 1] + 0.75 * d3[..., 2])))
                o["y_att"][sl] = np.array([[1 if (attributed and j == aid) else 0 for j in js]] * T)
                o["sev"][sl] = sev; o["oracle_score"][sl] = score; o["peer_term"][sl] = peer
                o["own_prov"][sl] = own; o["atk_id"][sl] = aid; o["s_noisy"][sl] = s
                if is_atk and i != aid and agg == "sum":
                    o["atk_term"][sl] = g * C[aid]
                o["atk_ch"][sl] = 0 if not ach else PEER_NAMES.index(ach) + 1
                # ---------------- v15 extensions (no draw from the main stream)
                ext["score64"][sl] = score
                ext["noise"][sl] = noise
                ext["score_b0"][sl] = score
                ext["score_abs"][sl] = score
                if is_atk and i != aid:
                    p0 = np.asarray(pose0[0])
                    D0 = np.array([np.linalg.norm(p0 - pos[i, t]) for t in range(T)])
                    OV0 = np.array([_fov_overlap(pos[i, t], ori[i, t], p0, fov[i]) for t in range(T)])
                    Cb = C.copy()
                    Cb[aid] = edge_contrib(PC_b0 / PEER_HI, D0, OV0, v)
                    score_b0 = own + g * aggregate(Cb, js, agg, v)
                    js_wo = [j for j in js if j != aid]
                    score_abs = own + g * aggregate(C, js_wo, agg, v)
                    unsafe = s > THETA
                    ext["y_A"][sl] = (unsafe & (score_b0 + noise <= THETA)).astype(np.int64)
                    ext["prevented"][sl] = (~unsafe & (score_b0 + noise > THETA)).astype(np.int64)
                    ext["y_A0"][sl] = (unsafe & (score_abs + noise <= THETA)).astype(np.int64)
                    ext["score_b0"][sl] = score_b0
                    ext["score_abs"][sl] = score_abs
                    ext["victim_of_attack"][sl] = 1
                    if k_mc:
                        acc = np.zeros(T)
                        hi = norm.cdf((THETA - score) / sigma)
                        for kk in range(k_mc):
                            pk = np.asarray(_sample_avatar_pose(np.random.default_rng([seed0 + cid, tk, 30, kk]), 8.0)[0])
                            Dk = np.array([np.linalg.norm(pk - pos[i, t]) for t in range(T)])
                            OVk = np.array([_fov_overlap(pos[i, t], ori[i, t], pk, fov[i]) for t in range(T)])
                            PCk = PC_b0.copy()
                            PCk[:, k] = np.random.default_rng([seed0 + cid, tk, 31, kk]).uniform(
                                PEER_CHANNELS[k].benign_lo, BEN_HI[k], T)
                            Ck = C.copy()
                            Ck[aid] = edge_contrib(PCk / PEER_HI, Dk, OVk, v)
                            sk = own + g * aggregate(Ck, js, agg, v)
                            acc += np.maximum(0.0, norm.cdf((THETA - sk) / sigma) - hi)
                        ext["p_A_obs"][sl] = acc / k_mc
                r += T
    out["V_raw"] = user_rows
    out["y"] = (out["sev"] >= 2).astype(np.int64)
    if with_backbone:
        proba = get_backbone().predict_proba(user_rows)
        bb = np.concatenate([np.log(np.clip(proba, 1e-9, 1)), (1 - proba.max(1))[:, None]], 1).astype(np.float32)
        out["V_bb"] = bb
        idx = {tuple(c): n for n, c in enumerate(out["ctx"][:, [0, 1, 2, 3]].tolist())}
        E_bb = np.zeros((rows, K, 5), np.float32)
        for n, (c, tk, t, i) in enumerate(out["ctx"].tolist()):
            for kk, j in enumerate(out["E_src"][n]):
                E_bb[n, kk] = bb[idx[(c, tk, t, int(j))]]
        out["E_bb"] = E_bb
    out["N"], out["T"] = np.array(N), np.array(T)
    out["sigma"] = np.array(sigma)
    out.update(ext)
    return out


def ceilings(t):
    """Oracle PR-AUCs: U from the exact posterior given the deterministic score; A from p_A_obs (needs k_mc > 0);
    A0 from the exact posterior given the attack and attacker-absent scores."""
    from sklearn.metrics import average_precision_score as ap
    sg = float(t["sigma"])
    pU = norm.cdf((t["score64"] - THETA) / sg)
    pA0 = np.where(t["victim_of_attack"] == 1,
                   np.maximum(0.0, norm.cdf((THETA - t["score_abs"]) / sg) - norm.cdf((THETA - t["score64"]) / sg)), 0.0)
    res = {"U": float(ap(t["y"], pU)), "A0": float(ap(t["y_A0"], pA0)) if t["y_A0"].any() else 0.0}
    res["A"] = float(ap(t["y_A"], t["p_A_obs"])) if t["y_A"].any() and t["p_A_obs"].any() else 0.0
    return res
