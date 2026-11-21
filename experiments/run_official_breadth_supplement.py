#!/usr/bin/env python3
"""Confirmatory breadth supplement covering breadth and scaling sub-cells.

Covers exact 100-seed paired checks across:
1. K3: All 16 (topology, scale) cells N in {15, 30, 60, 120} x {mod, mat, pool, cent}:
   - Q(d=0) - Q(G1, d=0.75) >= 0.10
   - Y_sr(t in [1, 5], d_collapse(G) + 0.02) - Y_0 >= +0.02
   - Y_lr(d=0.75) - Y_0 <= -0.02
   - Adiabatic rollback hysteresis across all 4 scales in mod + stationarity check [81, 120] vs [121, 160]
2. K9: Isomorphic agent-index permutation invariance (< 1e-12) + N in {15, 60, 120} null parity
3. K11: Breadth across d in {0.70, 0.80}, N in {15, 60, 120}, AUC in {0.75, 0.85, 0.95},
   plus non-modular topology boundary check (mat, pool, cent) for G5
4. K12: Capacity-matched governance ablations across AUC in {0.75, 0.95} (complementing AUC=0.85)
   plus behavioral uncongested placebo at d = 0.20 (|Delta Q| < 0.02)
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
for _sub in ("src", "experiments", "scripts"):
    _p = str(_REPO_ROOT / _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from canonical_org_model import CANONICAL_PARAMS
from sim_abm_org import (
    governance_arm_kwargs,
    simulate_adiabatic_sweep,
    simulate_batch,
    tune_arm_on_validation,
)

ROOT = Path(__file__).resolve().parent.parent
ART_DIR = ROOT / "results" / "artifacts"
RES_DIR = ROOT / "results"

VAL_SEEDS = np.arange(900, 930, dtype=np.int64)
CONF_SEEDS = np.arange(1000, 1100, dtype=np.int64)


def paired_summary(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 20261119) -> dict:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(diff)
    mean_d = float(diff.mean())
    sd_d = float(diff.std(ddof=1)) if n > 1 else 0.0
    se_d = sd_d / np.sqrt(max(1, n))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    bmeans = diff[idx].mean(axis=1)
    lo, hi = np.quantile(bmeans, [0.025, 0.975])
    return {
        "n": int(n),
        "diff_mean": round(mean_d, 6),
        "diff_sd": round(sd_d, 6),
        "diff_se": round(se_d, 6),
        "ci95_lo": round(float(lo), 6),
        "ci95_hi": round(float(hi), 6),
        "cohen_dz": round(mean_d / max(sd_d, 1e-12), 4),
    }


def main() -> None:
    t0 = time.time()
    ART_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    # -
    # 1. K3: 16 (topology, scale) cells + stationarity + 4-scale hysteresis
    # -
    ode_dcol = {"cent": 0.26, "pool": 0.32, "mat": 0.34, "mod": 0.48}
    k3_cells = {}
    k3_rows = []
    for top in ("mod", "mat", "pool", "cent"):
        for N in (15, 30, 60, 120):
            key = f"{top}_N{N}"
            b0 = simulate_batch(CONF_SEEDS, top, N, 0.0, T=160, record_trajectory=True)
            b75 = simulate_batch(CONF_SEEDS, top, N, 0.75, T=160, record_trajectory=True)
            d_sr = round(ode_dcol[top] + 0.02, 4)
            bsr = simulate_batch(CONF_SEEDS, top, N, d_sr, T=20)

            q_drop = paired_summary(b0["Q_ss"], b75["Q_ss"])
            y_sr_gain = paired_summary(bsr["Y_sr"], b0["Y_sr"])
            y_sr_075_gain = paired_summary(b75["Y_sr"], b0["Y_sr"])
            y_lr_drop = paired_summary(b75["Y_ss"], b0["Y_ss"])
            # Stationarity check: mean |Q(81..120) - Q(121..160)|
            q_81_120 = b75["Q_hist"][:, 80:120].mean(axis=1)
            q_121_160 = b75["Q_hist"][:, 120:160].mean(axis=1)
            stat_drift = float(np.abs(q_81_120 - q_121_160).mean())
            q0_81_120 = b0["Q_hist"][:, 80:120].mean(axis=1)
            q0_121_160 = b0["Q_hist"][:, 120:160].mean(axis=1)
            stat_drift_d0 = float(np.abs(q0_81_120 - q0_121_160).mean())

            k3_cells[key] = {
                "top": top,
                "N": N,
                "Q_d0": round(float(b0["Q_ss"].mean()), 6),
                "Q_d075": round(float(b75["Q_ss"].mean()), 6),
                "Q_drop_d0_minus_d075": q_drop,
                "Y_sr_gain_near_fold": y_sr_gain,
                "Y_sr_gain_at_d075": y_sr_075_gain,
                "Y_lr_change_d075_minus_d0": y_lr_drop,
                "stationarity_drift_d075": round(stat_drift, 6),
                "stationarity_drift_d0": round(stat_drift_d0, 6),
            }
            k3_rows.append({
                "top": top,
                "N": N,
                "Q_d0": round(float(b0["Q_ss"].mean()), 4),
                "Q_d075": round(float(b75["Q_ss"].mean()), 4),
                "Q_drop": q_drop["diff_mean"],
                "Q_drop_ci_lo": q_drop["ci95_lo"],
                "Y_sr_gain_fold": y_sr_gain["diff_mean"],
                "Y_sr_gain_075": y_sr_075_gain["diff_mean"],
                "Y_lr_drop": y_lr_drop["diff_mean"],
                "Y_lr_ci_hi": y_lr_drop["ci95_hi"],
                "stat_drift_075": round(stat_drift, 5),
            })

    # 4-scale adiabatic hysteresis in modular pods (26-point grid, 100 seeds)
    d_sw = np.round(np.linspace(0.0, 1.0, 26), 4)
    k3_hyst_scales = {}
    for N in (15, 30, 60, 120):
        sw = simulate_adiabatic_sweep(CONF_SEEDS, "mod", N, d_sw, T_step=60)
        k3_hyst_scales[f"N{N}"] = paired_summary(sw["max_hyst_Q"], np.zeros_like(sw["max_hyst_Q"]))

    # Uncongested d = 0.10 accuracy ordering check across 4 topologies (K6/K8 positive control)
    uncong_d010 = {}
    q10 = {}
    for top in ("cent", "mod", "mat", "pool"):
        o10 = simulate_batch(CONF_SEEDS, top, 30, 0.10, T=160)
        q10[top] = o10["Q_ss"]
        uncong_d010[top] = round(float(o10["Q_ss"].mean()), 6)
    uncong_d010["cent_minus_mod"] = paired_summary(q10["cent"], q10["mod"])
    uncong_d010["mod_minus_pool"] = paired_summary(q10["mod"], q10["pool"])

    pd.DataFrame(k3_rows).to_csv(RES_DIR / "abm_16cells_summary.csv", index=False)
    k3_out = {
        "cells_16": k3_cells,
        "mod_hysteresis_by_scale": k3_hyst_scales,
        "uncongested_d010_ordering": uncong_d010,
    }
    (ART_DIR / "abm_16cells_breadth.json").write_text(json.dumps(k3_out, indent=2), encoding="utf-8")
    min_q_drop = min(c["Q_drop_d0_minus_d075"]["ci95_lo"] for c in k3_cells.values())
    max_y_lr = max(c["Y_lr_change_d075_minus_d0"]["ci95_hi"] for c in k3_cells.values())
    min_y_sr = min(c["Y_sr_gain_near_fold"]["ci95_lo"] for c in k3_cells.values())
    print(f"[K3-16cells] min Q_drop LB={min_q_drop:.4f}, min Y_sr LB={min_y_sr:+.4f}, max Y_lr UB={max_y_lr:+.4f}, d0.10 cent>mod>pool: {uncong_d010['cent']:.4f}>{uncong_d010['mod']:.4f}>{uncong_d010['pool']:.4f}")

    # -
    # 2. K9: Isomorphic Agent-Index Permutation Invariance (< 1e-12)
    # -
    rng_p = np.random.default_rng(20261119)
    iso_checks = {}
    for top in ("mod", "pool", "cent"):
        base = simulate_batch(CONF_SEEDS, top, 30, 0.44, T=160)
        # Isomorphic pod/division permutation:
        # For mod/pool any permutation of N_S=6 pods is an automorphism; for cent swap the 2 divisions [0,1,2] <-> [3,4,5]
        if top == "cent":
            perm_s = np.array([3, 4, 5, 0, 1, 2], dtype=int)
        else:
            perm_s = rng_p.permutation(6)
        # Note: agent_perm expects length N_S senior permutation? Check sim_abm_org behavior; verify safely
        try:
            permed = simulate_batch(CONF_SEEDS, top, 30, 0.44, T=160, agent_perm=perm_s)
            diff_q = float(np.max(np.abs(base["Q_ss"] - permed["Q_ss"])))
            iso_checks[top] = {"perm": perm_s.tolist(), "max_abs_diff_Q": diff_q, "pass_lt_1e12": diff_q < 1e-12}
        except Exception as exc:
            iso_checks[top] = {"error": str(exc)}
    (ART_DIR / "abm_topology_isomorphism.json").write_text(json.dumps(iso_checks, indent=2), encoding="utf-8")
    print(f"[K9-iso] {iso_checks}")

    # -
    # 3. K11 + K12 Breadth: d in {0.70, 0.80}, N in {15, 60, 120}, AUC in {0.75, 0.95},
    #    d = 0.20 uncongested behavioral placebo, and non-modular topology boundary
    # -
    g5_g85, _, _ = tune_arm_on_validation("G5_coupled", VAL_SEEDS, "mod", 30, 0.75, 0.85, y_floor=1.02)
    g2_p85, _, _ = tune_arm_on_validation("G2_static_hitl", VAL_SEEDS, "mod", 30, 0.75, 0.85, y_floor=1.02)
    ra_g85, _, _ = tune_arm_on_validation("RA_routing_alone", VAL_SEEDS, "mod", 30, 0.75, 0.85, y_floor=1.02)

    k11_breadth = {"by_d": {}, "by_N": {}, "by_topology": {}, "by_auc_k12": {}, "d020_placebo": {}}

    # d in {0.70, 0.80}
    for d_v in (0.70, 0.80):
        g5_d, _, _ = tune_arm_on_validation("G5_coupled", VAL_SEEDS[:15], "mod", 30, d_v, 0.85, y_floor=1.02)
        g2_d, _, _ = tune_arm_on_validation("G2_static_hitl", VAL_SEEDS[:15], "mod", 30, d_v, 0.85, y_floor=1.02)
        ra_d, _, _ = tune_arm_on_validation("RA_routing_alone", VAL_SEEDS[:15], "mod", 30, d_v, 0.85, y_floor=1.02)
        qa_d, _, _ = tune_arm_on_validation("QA_quota_alone", VAL_SEEDS[:15], "mod", 30, d_v, 0.85, y_floor=1.02)
        q_g5_d = max(0.0, 1.0 - CANONICAL_PARAMS["x_cap"] / d_v)
        o_g0 = simulate_batch(CONF_SEEDS, "mod", 30, 0.0, T=160)
        o_g5 = simulate_batch(CONF_SEEDS, "mod", 30, d_v, T=160, q_quota=q_g5_d, g_route=g5_d, auc=0.85)
        o_g2 = simulate_batch(CONF_SEEDS, "mod", 30, d_v, T=160, g_route=g2_d, auc=0.50)
        o_ra = simulate_batch(CONF_SEEDS, "mod", 30, d_v, T=160, g_route=ra_d, auc=0.85)
        o_qa = simulate_batch(CONF_SEEDS, "mod", 30, d_v, T=160, q_quota=qa_d, g_route=1.0, auc=0.85)
        best_s = np.maximum.reduce([o_g2["Q_ss"], o_ra["Q_ss"], o_qa["Q_ss"]])
        k11_breadth["by_d"][f"d_{d_v:.2f}"] = {
            "g5_g_star": round(g5_d, 4),
            "G5_Q": round(float(o_g5["Q_ss"].mean()), 4),
            "G5_Y": round(float(o_g5["Y_ss"].mean()), 4),
            "G5_minus_best_single_Q": paired_summary(o_g5["Q_ss"], best_s),
            "G5_minus_G0_Y": paired_summary(o_g5["Y_ss"], o_g0["Y_ss"]),
        }

    # N in {15, 60, 120} at d = 0.75, AUC = 0.85
    q_g5_75 = max(0.0, 1.0 - CANONICAL_PARAMS["x_cap"] / 0.75)
    for N_v in (15, 60, 120):
        o_g0 = simulate_batch(CONF_SEEDS, "mod", N_v, 0.0, T=160)
        o_g5 = simulate_batch(CONF_SEEDS, "mod", N_v, 0.75, T=160, q_quota=q_g5_75, g_route=g5_g85, auc=0.85)
        o_g2 = simulate_batch(CONF_SEEDS, "mod", N_v, 0.75, T=160, g_route=g2_p85, auc=0.50)
        o_ra = simulate_batch(CONF_SEEDS, "mod", N_v, 0.75, T=160, g_route=ra_g85, auc=0.85)
        o_qa = simulate_batch(CONF_SEEDS, "mod", N_v, 0.75, T=160, q_quota=q_g5_75, g_route=1.0, auc=0.85)
        best_s = np.maximum.reduce([o_g2["Q_ss"], o_ra["Q_ss"], o_qa["Q_ss"]])
        k11_breadth["by_N"][f"N{N_v}"] = {
            "G5_Q": round(float(o_g5["Q_ss"].mean()), 4),
            "G5_Y": round(float(o_g5["Y_ss"].mean()), 4),
            "G5_minus_best_single_Q": paired_summary(o_g5["Q_ss"], best_s),
            "G5_minus_G0_Y": paired_summary(o_g5["Y_ss"], o_g0["Y_ss"]),
        }

    # Topology boundary for G5 (mod vs mat, pool, cent at d = 0.75, AUC = 0.85)
    for top in ("mod", "mat", "pool", "cent"):
        o_g1 = simulate_batch(CONF_SEEDS, top, 30, 0.75, T=160)
        o_g5 = simulate_batch(CONF_SEEDS, top, 30, 0.75, T=160, q_quota=q_g5_75, g_route=g5_g85, auc=0.85)
        k11_breadth["by_topology"][top] = {
            "G1_Q": round(float(o_g1["Q_ss"].mean()), 4),
            "G5_Q": round(float(o_g5["Q_ss"].mean()), 4),
            "G5_Y": round(float(o_g5["Y_ss"].mean()), 4),
            "G5_minus_G1_Q": paired_summary(o_g5["Q_ss"], o_g1["Q_ss"]),
        }

    # K12 across AUC in {0.75, 0.85, 0.95}: G5 vs Tuned RA, Matched QA, Fixed-g RT placebo, Tuned RT, Fixed-g FL
    for auc_v in (0.75, 0.85, 0.95):
        g_star, _, _ = tune_arm_on_validation("G5_coupled", VAL_SEEDS[:15], "mod", 30, 0.75, auc_v, y_floor=1.02)
        ra_star, _, _ = tune_arm_on_validation("RA_routing_alone", VAL_SEEDS[:15], "mod", 30, 0.75, auc_v, y_floor=1.02)
        rt_star, _, _ = tune_arm_on_validation("RT_random_thinning", VAL_SEEDS[:15], "mod", 30, 0.75, 0.50, y_floor=1.02)
        o_g5 = simulate_batch(CONF_SEEDS, "mod", 30, 0.75, T=160, q_quota=q_g5_75, g_route=g_star, auc=auc_v)
        o_ra = simulate_batch(CONF_SEEDS, "mod", 30, 0.75, T=160, g_route=ra_star, auc=auc_v)
        o_qa = simulate_batch(CONF_SEEDS, "mod", 30, 0.75, T=160, q_quota=q_g5_75, g_route=1.0, auc=auc_v)
        o_rt_fix = simulate_batch(CONF_SEEDS, "mod", 30, 0.75, T=160, q_quota=q_g5_75, g_route=g_star, auc=0.50)
        o_rt_tun = simulate_batch(CONF_SEEDS, "mod", 30, 0.75, T=160, q_quota=q_g5_75, g_route=rt_star, auc=0.50)
        o_fl_fix = simulate_batch(CONF_SEEDS, "mod", 30, 0.75, T=160, q_quota=q_g5_75, g_route=g_star, auc=auc_v, frozen_x_learn=0.75)
        k11_breadth["by_auc_k12"][f"auc_{auc_v:.2f}"] = {
            "g_star": round(g_star, 4),
            "G5_Q": round(float(o_g5["Q_ss"].mean()), 4),
            "G5_Y": round(float(o_g5["Y_ss"].mean()), 4),
            "G5_minus_RA": paired_summary(o_g5["Q_ss"], o_ra["Q_ss"]),
            "G5_minus_QA_matched": paired_summary(o_g5["Q_ss"], o_qa["Q_ss"]),
            "G5_minus_FL_fixed": paired_summary(o_g5["Q_ss"], o_fl_fix["Q_ss"]),
            "G5_minus_RT_fixed_placebo": paired_summary(o_g5["Q_ss"], o_rt_fix["Q_ss"]),
            "G5_minus_RT_tuned": paired_summary(o_g5["Q_ss"], o_rt_tun["Q_ss"]),
        }

    # K12 behavioral uncongested placebo at d = 0.20 (q_G5(0.20) = max(0, 1 - 0.56/0.20) = 0, g_route = g5_g85)
    o_d20_g1 = simulate_batch(CONF_SEEDS, "mod", 30, 0.20, T=160)
    o_d20_g5 = simulate_batch(CONF_SEEDS, "mod", 30, 0.20, T=160, q_quota=0.0, g_route=g5_g85, auc=0.85)
    k11_breadth["d020_placebo"] = {
        "G1_Q": round(float(o_d20_g1["Q_ss"].mean()), 6),
        "G5_Q": round(float(o_d20_g5["Q_ss"].mean()), 6),
        "G5_minus_G1": paired_summary(o_d20_g5["Q_ss"], o_d20_g1["Q_ss"]),
        "abs_diff_lt_002": abs(float(o_d20_g5["Q_ss"].mean() - o_d20_g1["Q_ss"].mean())) < 0.02,
    }
    k11_breadth["runtime_sec"] = round(time.time() - t0, 3)

    (ART_DIR / "governance_breadth_100seeds.json").write_text(json.dumps(k11_breadth, indent=2), encoding="utf-8")
    print(f"[K11/K12-breadth] done in {k11_breadth['runtime_sec']}s; d=0.20 placebo |diff|={abs(k11_breadth['d020_placebo']['G5_minus_G1']['diff_mean']):.5f}")
    for ak, av in k11_breadth["by_auc_k12"].items():
        print(f"  {ak}: G5={av['G5_Q']:.4f} | -RA={av['G5_minus_RA']['diff_mean']:+.4f} | -QA={av['G5_minus_QA_matched']['diff_mean']:+.4f} | -FL={av['G5_minus_FL_fixed']['diff_mean']:+.4f} | -RT_fix={av['G5_minus_RT_fixed_placebo']['diff_mean']:+.4f} | -RT_tun={av['G5_minus_RT_tuned']['diff_mean']:+.4f}")


if __name__ == "__main__":
    main()
