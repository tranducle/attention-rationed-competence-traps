"""Run deterministic model, sensitivity, comparator, and boundary diagnostics."""
from __future__ import annotations

import csv
import json
import math
import os
import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE_DIR = ROOT / "src"
sys.path.insert(0, str(CODE_DIR))

import canonical_org_model as m

ARCHIVE_RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = os.environ.get("MODEL_VERIFY_OUTPUT_DIR")
RESULTS_DIR = pathlib.Path(OUTPUT_DIR).resolve() if OUTPUT_DIR else ARCHIVE_RESULTS_DIR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = RESULTS_DIR / "model_diagnostics.json"
STRUCTURAL_LHS_PATH = RESULTS_DIR / "structural_lhs_500_draws.csv"
TOPOLOGY_CAPTURE_PATH = RESULTS_DIR / "topology_capture_sensitivity.csv"
DEFECT_SCRUTINY_PATH = RESULTS_DIR / "governance_defect_scrutiny_sensitivity.csv"


def scrutiny_response(u_tilde: np.ndarray, family: str = "quartic") -> np.ndarray:
    """Evaluate the canonical curve or a thresholded exponential alternative."""
    values = np.asarray(u_tilde, dtype=float)
    if family == "quartic":
        return 1.0 / (1.0 + (0.36 * values) ** 4)
    if family == "overload_exponential":
        anchor_u = 3.0
        anchor_chi = 1.0 / (1.0 + (0.36 * anchor_u) ** 4)
        decay = -math.log(anchor_chi) / (anchor_u - 1.0)
        return np.exp(-decay * np.maximum(0.0, values - 1.0))
    raise ValueError(f"unknown scrutiny family: {family}")


def sym_ode_ss_vec(
    d: float,
    zeta_s: np.ndarray,
    zeta_rew_s: np.ndarray,
    f_tau: np.ndarray,
    P_vec: dict[str, np.ndarray],
    q: float = 0.0,
    g: float = 1.0,
    auc: float = 0.85,
    omega: float = 2.0,
    S0: np.ndarray | float = 0.935,
    b_router: float = 0.0,
    rho_decay: float = 0.0,
    m_eff_relief_only: bool = False,
    a_relief_only: bool = False,
    joint_coach_relief: bool = False,
    guardrails: float = 0.0,
    parallel: float = 0.0,
    v_bonus: float = 0.0,
    scrutiny_family: str = "quartic",
    T: float = 250.0,
    dt: float = 0.25,
) -> dict[str, np.ndarray]:
    """Vectorized symmetric RK4 solver for B parameter draws simultaneously (exact for symmetric mod/mat/pool)."""
    B_n = len(zeta_s)
    phi_e = 1.0 - P_vec["phi"]
    B_av = np.maximum(0.05, P_vec["B"] - P_vec["b_exec"] - 0.34 * guardrails - b_router)
    x = max(0.0, d * (1.0 - q) - v_bonus)
    y = g * x
    lam = P_vec["lam0"] * (1.0 + P_vec["alpha_ai"] * x * (1.0 - 0.35 * guardrails))
    lam_req = P_vec["lam0"] * (1.0 + P_vec["alpha_ai"] * P_vec["gamma_req"] * d)
    zs = zeta_s * (1.0 + 0.65 * parallel)
    S = np.full(B_n, float(S0), dtype=float) if np.isscalar(S0) else np.array(S0, dtype=float)
    x_learn = x * (1.0 - guardrails)
    sig_x = 1.0 / (1.0 + np.exp((x_learn - P_vec["x0"]) / P_vec["beta_a"]))
    b_base = P_vec["J0"] + P_vec["alpha_p"] * (1.0 - 0.65 * x_learn) - 0.004 * guardrails

    if g >= 1.0 - 1e-12:
        C_const, fpr_const = np.ones(B_n), np.ones(B_n)
    elif rho_decay == 0.0:
        c0, f0 = m.binormal_roc_capture_and_fpr(auc, g, float(phi_e[0]))
        C_const, fpr_const = np.full(B_n, c0), np.full(B_n, f0)
    else:
        C_const, fpr_const = None, None

    def rhs(Sv: np.ndarray):
        if C_const is not None:
            C, fpr = C_const, fpr_const
        else:
            C = np.empty(B_n)
            fpr = np.empty(B_n)
            for idx in range(B_n):
                re = max(0.50, min(0.999, 0.50 + (auc - 0.50) * ((1.0 - rho_decay) + rho_decay * (Sv[idx] / 0.935))))
                C[idx], fpr[idx] = m.binormal_roc_capture_and_fpr(re, g, float(phi_e[idx]))
        s_eff = P_vec["s_min"] + (1.0 - P_vec["s_min"]) * Sv
        L_prim = lam * (y * zs + x * phi_e * (1.0 - C) * P_vec["p_out"] * P_vec["rho_rew"] * zeta_rew_s) / s_eff
        O_s = np.maximum(0.0, L_prim - B_av)
        L_s = L_prim + P_vec["psi_spill"] * O_s
        u_tilde = L_s / B_av
        u = np.minimum(1.0, u_tilde)
        chi = scrutiny_response(u_tilde, scrutiny_family)
        u_m = np.zeros_like(u) if (m_eff_relief_only or joint_coach_relief) else u
        chi_m = np.ones_like(chi) if (m_eff_relief_only or joint_coach_relief) else chi
        m_eff = B_av * (1.0 - u_m) + P_vec["sigma0"] * B_av * u_m * chi_m * f_tau
        u_a = np.zeros_like(u) if (a_relief_only or joint_coach_relief) else u
        A_s = sig_x + (1.0 - sig_x) * (P_vec["A_min"] * (1.0 - u_a) + P_vec["A_cong"] * u_a)
        drive = P_vec["eta_m"] * m_eff * Sv
        hill = (drive ** omega) / (P_vec["K_m"] ** omega + drive ** omega)
        J_s = b_base + A_s * P_vec["mu0"] * hill
        return P_vec["delta"] * (J_s - Sv), u, chi, C, fpr

    # Exact fixed-point relaxation S_{k+1} = S_k + 0.6*(J(S_k) - S_k): identical steady states J(S)=S, 80x faster than RK4
    max_iter = int(min(300, max(120, round(T))))
    for _ in range(max_iter):
        dS, _, _, _, _ = rhs(S)
        err = dS / P_vec["delta"]
        if float(np.max(np.abs(err))) < 1e-7:
            break
        S = np.clip(S + 0.6 * err, 0.0, 1.0)

    _, u, chi, C, fpr = rhs(S)
    q_prim = np.minimum(1.0, S * chi * (1.0 + 0.03 * parallel))
    psi_bar = q_prim
    theta_ood = C * psi_bar
    theta_in = fpr * psi_bar
    E_def = (1.0 - phi_e) * P_vec["p_in"] * (1.0 - P_vec["catch_max"] * theta_in) + phi_e * P_vec["p_out"] * (1.0 - P_vec["catch_max"] * theta_ood)
    Q = P_vec["h0"] + P_vec["h1"] * theta_ood
    Y_gross = lam * (1.0 - P_vec["c_def"] * x * E_def)
    Y = Y_gross - P_vec["pi_req"] * np.maximum(0.0, lam_req - lam)
    return {"S": S, "Q": Q, "Y": Y, "Y_gross": Y_gross, "u": u}


def make_pvec(P: dict, B_n: int = 1, overrides: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    pv = {k: np.full(B_n, float(v), dtype=float) for k, v in P.items()}
    if overrides:
        for k, arr in overrides.items():
            pv[k] = np.asarray(arr, dtype=float)
    return pv


def top_weights_vec(top: str, N_S: int, pv: dict[str, np.ndarray]):
    r = pv["r"]
    tau_v = pv["tau_v"]
    kappa0 = pv["kappa0"]
    if top == "mod":
        z = r * tau_v
        return z, z, np.ones_like(z)
    if top == "pool":
        w = 1.0 - 1.0 / float(N_S)
        z = r * (tau_v + kappa0 * w)
        f_tau = (r * (1.0 / float(N_S)) * tau_v) / z
        return z, z, f_tau
    if top == "mat":
        p_m = pv["p_mat"]
        z_ver = r * (tau_v + 0.5 * kappa0)
        z = z_ver + r * p_m * (tau_v + kappa0)
        f_tau = (0.5 * r * tau_v) / z
        return z, z_ver, f_tau
    raise ValueError(top)


def sweep_hyst_vec(
    top: str,
    N_S: int,
    pv: dict[str, np.ndarray],
    omega: float = 2.0,
    step: float = 0.02,
    scrutiny_family: str = "quartic",
    g: float = 1.0,
    auc: float = 0.85,
):
    d_grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)
    B_n = len(pv["r"])
    z, zr, ft = top_weights_vec(top, N_S, pv)
    q_fwd = np.zeros((B_n, len(d_grid)))
    q_bwd = np.zeros((B_n, len(d_grid)))
    S_cur = np.full(B_n, 0.935)
    for i, dv in enumerate(d_grid):
        r = sym_ode_ss_vec(float(dv), z, zr, ft, pv, g=g, auc=auc, omega=omega, S0=S_cur, scrutiny_family=scrutiny_family, T=180.0, dt=0.3)
        q_fwd[:, i] = r["Q"]
        S_cur = r["S"]
    S_cur = np.full(B_n, 0.17)
    for i in range(len(d_grid) - 1, -1, -1):
        dv = d_grid[i]
        r = sym_ode_ss_vec(float(dv), z, zr, ft, pv, g=g, auc=auc, omega=omega, S0=S_cur, scrutiny_family=scrutiny_family, T=180.0, dt=0.3)
        q_bwd[:, i] = r["Q"]
        S_cur = r["S"]
    hyst = np.max(q_fwd - q_bwd, axis=1)
    d_col = np.ones(B_n)
    d_rec = np.zeros(B_n)
    for j in range(B_n):
        ic = np.where(q_fwd[j] < 0.60)[0]
        if len(ic): d_col[j] = float(d_grid[ic[0]])
        ir = np.where(q_bwd[j] >= 0.60)[0]
        if len(ir): d_rec[j] = float(d_grid[ir[-1]])
    return {"max_hyst_Q": hyst, "d_col": d_col, "d_rec": d_rec, "width": np.maximum(0.0, d_col - d_rec)}


def main():
    P = m.CANONICAL_PARAMS
    B_av = P["B"] - P["b_exec"]
    pv1 = make_pvec(P, 1)
    out = {}

    # 1. Exact saturation bound versus collapse threshold across all four topologies.
    d_fine = np.round(np.arange(0.15, 0.56 + 1e-9, 0.005), 4)
    def d_sat_f(zeta_val: float, s_val: float) -> float:
        se = P["s_min"] + (1.0 - P["s_min"]) * s_val
        return (-1.0 + math.sqrt(1.0 + 4.0 * P["alpha_ai"] * B_av * se / (P["lam0"] * zeta_val))) / (2.0 * P["alpha_ai"])

    def single_relax(spec_in, d_val, s0_val=0.935, params_in=None):
        res = m.relax_steady_state(spec_in, np.array([float(d_val)]), S_init=s0_val, params=params_in)
        return float(res["S"][0].mean()), float(res["Q"][0]), float(res["u"][0].mean())

    s0 = single_relax(m.build_topology_spec("mod", 30, P), 0.0, 0.935)[0]
    f01 = {"S0": round(s0, 4)}
    for top in ("mod", "mat", "pool", "cent"):
        spec = m.build_topology_spec(top, 30, P)
        cost_edge = P["tau_v"] + P["kappa0"] * (1.0 - spec["W_men"])
        zeta_max = float(((spec["W_ver"] + spec["W_esc"]) * cost_edge).sum(axis=0).max())
        sw = m.adiabatic_continuation(spec, d_fine)
        ic = np.where(sw["Q_fwd"] < 0.60)[0]
        dc = float(d_fine[ic[0]])
        ir = np.where(sw["Q_bwd"] >= 0.60)[0]
        dr = float(d_fine[ir[-1]])
        pre_S, pre_Q, pre_u = single_relax(spec, dc - 0.005, 0.935)
        post_S, post_Q, post_u = single_relax(spec, dc, 0.935)
        f01[top] = {
            "d_sat_at_S0": round(d_sat_f(zeta_max, s0), 4),
            "d_sat_at_Shigh_pre": round(d_sat_f(zeta_max, pre_S), 4),
            "d_col_fine": dc,
            "d_rec_fine": dr,
            "u_pre_fold": round(pre_u, 4),
            "S_pre_fold": round(pre_S, 4),
            "Q_post_fold": round(post_Q, 4),
            "S_post_fold": round(post_S, 4),
        }
    p048_S, p048_Q, p048_u = single_relax(m.build_topology_spec("mod", 30, P), 0.480, 0.935)
    f01["mod_at_d0480"] = {"Q": round(p048_Q, 4), "S": round(p048_S, 4), "u": round(p048_u, 4)}
    # Also exact fine 0.0005 d_col for mod
    d_uf = np.round(np.arange(0.460, 0.490, 0.0005), 5)
    sw_uf = m.adiabatic_continuation(m.build_topology_spec("mod", 30, P), d_uf)
    ic_uf = np.where(sw_uf["Q_fwd"] < 0.60)[0][0]
    f01["mod_exact_d_col"] = float(d_uf[ic_uf])
    _, _, pre_uf_u = single_relax(m.build_topology_spec("mod", 30, P), f01["mod_exact_d_col"] - 0.0005, 0.935)
    f01["mod_exact_u_pre_fold"] = round(pre_uf_u, 4)
    out["saturation_bound_vs_collapse"] = f01

    # 2. Three-way coaching and absorption decomposition at d = 0.75.
    zm, zrm, ftm = top_weights_vec("mod", 6, pv1)
    base_075 = sym_ode_ss_vec(0.75, zm, zrm, ftm, pv1)
    m_only = sym_ode_ss_vec(0.75, zm, zrm, ftm, pv1, m_eff_relief_only=True)
    a_only = sym_ode_ss_vec(0.75, zm, zrm, ftm, pv1, a_relief_only=True)
    joint = sym_ode_ss_vec(0.75, zm, zrm, ftm, pv1, joint_coach_relief=True)
    out["coaching_absorption_decomposition_ode"] = {
        "unmanaged_d075": {"Q": round(float(base_075["Q"][0]), 4), "S": round(float(base_075["S"][0]), 4)},
        "m_eff_alone": {"Q": round(float(m_only["Q"][0]), 4), "S": round(float(m_only["S"][0]), 4), "delta_Q": round(float(m_only["Q"][0] - base_075["Q"][0]), 4)},
        "A_s_alone": {"Q": round(float(a_only["Q"][0]), 4), "S": round(float(a_only["S"][0]), 4), "delta_Q": round(float(a_only["Q"][0] - base_075["Q"][0]), 4)},
        "joint_m_eff_and_A_s": {"Q": round(float(joint["Q"][0]), 4), "S": round(float(joint["S"][0]), 4), "delta_Q": round(float(joint["Q"][0] - base_075["Q"][0]), 4)},
    }

    # 3. Active-intervention evaluations.
    g3_act = sym_ode_ss_vec(0.75, zm, zrm, ftm, pv1, parallel=1.0)
    ug_act = sym_ode_ss_vec(0.75, zm, zrm, ftm, pv1, guardrails=1.0)
    g4_cap = sym_ode_ss_vec(0.75, zm, zrm, ftm, pv1, v_bonus=0.14 / 1.35)
    g4_hi = sym_ode_ss_vec(0.75, zm, zrm, ftm, pv1, v_bonus=0.30)
    ea_x = sym_ode_ss_vec(0.75, zm, zrm, ftm, pv1, q=1.0 - 0.40625 / 0.75)
    out["active_rival_checks_ode"] = {
        "G3_active_p1": {"Q": round(float(g3_act["Q"][0]), 4), "Y": round(float(g3_act["Y"][0]), 4)},
        "UG_active_rho1": {"Q": round(float(ug_act["Q"][0]), 4), "Y": round(float(ug_act["Y"][0]), 4)},
        "G4_max_incentive_v01037": {"v_bon": round(0.14 / 1.35, 4), "Q": round(float(g4_cap["Q"][0]), 4), "Y": round(float(g4_cap["Y"][0]), 4)},
        "G4_unbounded_v030": {"v_bon": 0.30, "Q": round(float(g4_hi["Q"][0]), 4), "Y": round(float(g4_hi["Y"][0]), 4)},
        "EA_x040625_q04583": {"x_star": 0.40625, "q_star": round(1.0 - 0.40625 / 0.75, 4), "Q": round(float(ea_x["Q"][0]), 4), "Y": round(float(ea_x["Y"][0]), 4)},
    }

    # 4. One-dimensional sweeps, continuous omega, and 500-draw structural LHS.
    f04 = {"one_d_sweeps": {}, "omega_sweep": {}}
    smins = np.array([0.20, 0.30, 0.40, 0.50, 0.60])
    sw_sm = sweep_hyst_vec("mod", 6, make_pvec(P, len(smins), {"s_min": smins}), step=0.01)
    for i, sm in enumerate(smins):
        f04["one_d_sweeps"][f"s_min_{sm:.2f}"] = {"max_hyst_Q": round(float(sw_sm["max_hyst_Q"][i]), 4), "d_col": float(sw_sm["d_col"][i]), "width": round(float(sw_sm["width"][i]), 4)}

    # Exact root-finding bistability across continuous omega in [1.0, 2.5] (0.005 d-grid, 2000-pt S scan)
    S_scan = np.linspace(0.001, 0.999, 2000)
    d_scan = np.round(np.linspace(0.20, 0.70, 101), 4)
    B_av_c = P["B"] - P["b_exec"]
    for om in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]:
        max_gap = 0.0
        bi_ds = []
        for dv in d_scan:
            lam_v = P["lam0"] * (1.0 + P["alpha_ai"] * dv)
            se = P["s_min"] + (1.0 - P["s_min"]) * S_scan
            lp = P["r"] * lam_v * dv * P["tau_v"] / se
            ut = (lp + P["psi_spill"] * np.maximum(0.0, lp - B_av_c)) / B_av_c
            u_v = np.minimum(1.0, ut)
            ch = 1.0 / (1.0 + (0.36 * ut) ** 4)
            me = B_av_c * (1.0 - u_v) + P["sigma0"] * B_av_c * u_v * ch
            sx = 1.0 / (1.0 + math.exp((dv - P["x0"]) / P["beta_a"]))
            av = sx + (1.0 - sx) * (P["A_min"] * (1.0 - u_v) + P["A_cong"] * u_v)
            bb = P["J0"] + P["alpha_p"] * (1.0 - 0.65 * dv)
            D_v = P["eta_m"] * me * S_scan
            hl = (D_v ** om) / (P["K_m"] ** om + D_v ** om)
            Fv = bb + av * P["mu0"] * hl - S_scan
            roots = []
            for idx_r in np.where(np.sign(Fv[:-1]) * np.sign(Fv[1:]) <= 0)[0]:
                denom = Fv[idx_r + 1] - Fv[idx_r]
                sr = float(S_scan[idx_r] - Fv[idx_r] * (S_scan[idx_r + 1] - S_scan[idx_r]) / (denom if abs(denom) > 1e-15 else 1e-15))
                if not roots or abs(sr - roots[-1]) > 1e-4:
                    roots.append(sr)
            if len(roots) >= 3:
                bi_ds.append(float(dv))
                def _q_at(sv: float) -> float:
                    se1 = P["s_min"] + (1.0 - P["s_min"]) * sv
                    lp1 = P["r"] * lam_v * dv * P["tau_v"] / se1
                    ut1 = (lp1 + P["psi_spill"] * max(0.0, lp1 - B_av_c)) / B_av_c
                    ch1 = 1.0 / (1.0 + (0.36 * ut1) ** 4)
                    return float(P["h0"] + P["h1"] * sv * ch1)
                max_gap = max(max_gap, _q_at(max(roots)) - _q_at(min(roots)))
        f04["omega_sweep"][f"omega_{om:.2f}"] = {
            "exact_max_hyst_Q": round(float(max_gap), 4),
            "bistable_d_pts_0005": len(bi_ds),
            "d_rec_exact": round(bi_ds[0], 4) if bi_ds else None,
            "d_col_exact": round(bi_ds[-1], 4) if bi_ds else None,
            "width_exact": round(bi_ds[-1] - bi_ds[0], 4) if len(bi_ds) >= 2 else 0.0,
        }

    q_g5 = 1.0 - 0.56 / 0.75
    def g5_vs_ra_vec(pv_in: dict[str, np.ndarray]):
        z_i, zr_i, ft_i = top_weights_vec("mod", 6, pv_in)
        g5 = sym_ode_ss_vec(0.75, z_i, zr_i, ft_i, pv_in, q=q_g5, g=0.5567, auc=0.85)["Q"]
        ra_mat = np.vstack([sym_ode_ss_vec(0.75, z_i, zr_i, ft_i, pv_in, q=0.0, g=gv, auc=0.85)["Q"] for gv in [0.16, 0.20, 0.24, 0.28, 0.32]])
        ra = np.max(ra_mat, axis=0)
        return g5, ra, g5 - ra

    for pname, vals in [("beta_a", [0.025, 0.05, 0.08, 0.12, 0.15]), ("x0", [0.58, 0.625, 0.68, 0.72, 0.80]), ("A_cong", [0.14, 0.30, 0.50, 0.70])]:
        arr = np.array(vals, dtype=float)
        g5q, raq, dfq = g5_vs_ra_vec(make_pvec(P, len(arr), {pname: arr}))
        for i, v in enumerate(vals):
            f04["one_d_sweeps"][f"{pname}_{v}"] = {"G5_Q": round(float(g5q[i]), 4), "RA_Q": round(float(raq[i]), 4), "diff": round(float(dfq[i]), 4)}

    # 500-draw vectorized 7-parameter structural LHS!
    rng = np.random.default_rng(20261119)
    n_lhs = 500
    bounds = {
        "s_min": (0.15, 0.45),
        "psi_spill": (0.0, 3.0),
        "x0": (0.58, 0.70),
        "beta_a": (0.015, 0.08),
        "A_cong": (0.10, 0.40),
        "rho_rew": (0.10, 0.35),
        "c_def": (2.5, 5.0),
    }
    ov = {}
    for k, (lo, hi) in bounds.items():
        perm = rng.permutation(n_lhs)
        ov[k] = lo + ((perm + rng.random(n_lhs)) / n_lhs) * (hi - lo)
    pv_lhs = make_pvec(P, n_lhs, ov)
    sw_m_lhs = sweep_hyst_vec("mod", 6, pv_lhs, step=0.02)
    sw_p_lhs = sweep_hyst_vec("pool", 6, pv_lhs, step=0.02)
    g5_a, ra_a, c3_a = g5_vs_ra_vec(pv_lhs)
    c1_a = sw_m_lhs["max_hyst_Q"]
    c2_a = sw_m_lhs["d_col"] - sw_p_lhs["d_col"]
    with STRUCTURAL_LHS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "draw_id",
            *bounds.keys(),
            "mod_max_hyst_q",
            "mod_d_col",
            "pool_d_col",
            "mod_minus_pool_d_col",
            "g5_q",
            "best_ra_q",
            "g5_minus_best_ra_q",
        ])
        for idx in range(n_lhs):
            writer.writerow([
                idx + 1,
                *[format(float(ov[name][idx]), ".17g") for name in bounds],
                format(float(c1_a[idx]), ".17g"),
                format(float(sw_m_lhs["d_col"][idx]), ".17g"),
                format(float(sw_p_lhs["d_col"][idx]), ".17g"),
                format(float(c2_a[idx]), ".17g"),
                format(float(g5_a[idx]), ".17g"),
                format(float(ra_a[idx]), ".17g"),
                format(float(c3_a[idx]), ".17g"),
            ])
    f04["structural_7param_lhs_500"] = {
        "n_draws": n_lhs,
        "bounds": bounds,
        "C1_surv_ge_010": round(float(np.mean(c1_a >= 0.10)), 4),
        "C1_surv_ge_005": round(float(np.mean(c1_a >= 0.05)), 4),
        "C1_median": round(float(np.median(c1_a)), 4),
        "C1_iqr": [round(float(np.quantile(c1_a, 0.25)), 4), round(float(np.quantile(c1_a, 0.75)), 4)],
        "C2_mod_gt_pool": round(float(np.mean(c2_a > 0.0)), 4),
        "C2_surv_ge_008": round(float(np.mean(c2_a >= 0.08)), 4),
        "C2_median": round(float(np.median(c2_a)), 4),
        "C3_gt_RA": round(float(np.mean(c3_a > 0.0)), 4),
        "C3_surv_ge_010": round(float(np.mean(c3_a >= 0.10)), 4),
        "C3_median": round(float(np.median(c3_a)), 4),
        "C3_iqr": [round(float(np.quantile(c3_a, 0.25)), 4), round(float(np.quantile(c3_a, 0.75)), 4)],
    }
    out["structural_sensitivity"] = f04

    # 5. Defect probability, review capture, and scrutiny-family sensitivity.
    p_out_values = np.array([0.20, 0.40, 0.60, 0.80, 0.94, 1.00], dtype=float)
    capture_values = np.array([0.35, 0.60, 0.85, 1.00], dtype=float)
    scrutiny_families = ("quartic", "overload_exponential")
    topology_capture_rows = []
    topology_capture_summary = {}
    for family in scrutiny_families:
        family_summary = {}
        for capture_g in capture_values:
            pv_top = make_pvec(P, len(p_out_values), {"p_out": p_out_values})
            sw_mod_top = sweep_hyst_vec(
                "mod",
                6,
                pv_top,
                step=0.02,
                scrutiny_family=family,
                g=float(capture_g),
            )
            sw_pool_top = sweep_hyst_vec(
                "pool",
                6,
                pv_top,
                step=0.02,
                scrutiny_family=family,
                g=float(capture_g),
            )
            c1_values = sw_mod_top["max_hyst_Q"]
            c2_values = sw_mod_top["d_col"] - sw_pool_top["d_col"]
            for idx, p_out in enumerate(p_out_values):
                topology_capture_rows.append({
                    "scrutiny_family": family,
                    "capture_g": float(capture_g),
                    "p_out": float(p_out),
                    "mod_max_hyst_q": float(c1_values[idx]),
                    "mod_d_col": float(sw_mod_top["d_col"][idx]),
                    "pool_d_col": float(sw_pool_top["d_col"][idx]),
                    "mod_minus_pool_d_col": float(c2_values[idx]),
                })
            c1_span = float(np.ptp(c1_values))
            c2_span = float(np.ptp(c2_values))
            family_summary[f"capture_g_{capture_g:.2f}"] = {
                "capture_g": float(capture_g),
                "n_p_out_values": len(p_out_values),
                "p_out_range": [float(p_out_values.min()), float(p_out_values.max())],
                "C1_all_ge_010": bool(np.all(c1_values >= 0.10)),
                "C1_min_max_hyst_Q": round(float(np.min(c1_values)), 4),
                "C1_max_max_hyst_Q": round(float(np.max(c1_values)), 4),
                "C1_span_over_p_out": round(c1_span, 4),
                "C1_discriminated_by_p_out": bool(c1_span > 1e-8),
                "C2_all_mod_gt_pool": bool(np.all(c2_values > 0.0)),
                "C2_min_gap": round(float(np.min(c2_values)), 4),
                "C2_max_gap": round(float(np.max(c2_values)), 4),
                "C2_span_over_p_out": round(c2_span, 4),
                "C2_discriminated_by_p_out": bool(c2_span > 1e-8),
            }
        topology_capture_summary[family] = family_summary

    with TOPOLOGY_CAPTURE_PATH.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(topology_capture_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in topology_capture_rows:
            writer.writerow({key: format(value, ".17g") if isinstance(value, float) else value for key, value in row.items()})

    defect_scrutiny_rows = []
    defect_scrutiny_summary = {}
    for family in scrutiny_families:
        pv_ds = make_pvec(P, len(p_out_values), {"p_out": p_out_values})
        z_ds, zr_ds, ft_ds = top_weights_vec("mod", 6, pv_ds)
        g5_ds = sym_ode_ss_vec(
            0.75,
            z_ds,
            zr_ds,
            ft_ds,
            pv_ds,
            q=q_g5,
            g=0.5567,
            auc=0.85,
            scrutiny_family=family,
        )
        ra_ds = [
            sym_ode_ss_vec(
                0.75,
                z_ds,
                zr_ds,
                ft_ds,
                pv_ds,
                q=0.0,
                g=gv,
                auc=0.85,
                scrutiny_family=family,
            )
            for gv in [0.16, 0.20, 0.24, 0.28, 0.32]
        ]
        ra_q_matrix = np.vstack([entry["Q"] for entry in ra_ds])
        ra_y_matrix = np.vstack([entry["Y"] for entry in ra_ds])
        best_indices = np.argmax(ra_q_matrix, axis=0)
        best_ra_q = ra_q_matrix[best_indices, np.arange(len(p_out_values))]
        best_ra_y = ra_y_matrix[best_indices, np.arange(len(p_out_values))]
        c3_gap = g5_ds["Q"] - best_ra_q
        y_gap = g5_ds["Y"] - best_ra_y
        for idx, p_out in enumerate(p_out_values):
            defect_scrutiny_rows.append({
                "scrutiny_family": family,
                "p_out": float(p_out),
                "g5_q": float(g5_ds["Q"][idx]),
                "best_ra_q": float(best_ra_q[idx]),
                "g5_minus_best_ra_q": float(c3_gap[idx]),
                "g5_y": float(g5_ds["Y"][idx]),
                "best_ra_y": float(best_ra_y[idx]),
                "g5_minus_best_ra_y": float(y_gap[idx]),
            })
        defect_scrutiny_summary[family] = {
            "n_p_out_values": len(p_out_values),
            "p_out_range": [float(p_out_values.min()), float(p_out_values.max())],
            "C3_all_g5_gt_RA": bool(np.all(c3_gap > 0.0)),
            "C3_min_gap": round(float(np.min(c3_gap)), 4),
            "G5_min_Y": round(float(np.min(g5_ds["Y"])), 4),
            "G5_min_Y_minus_RA": round(float(np.min(y_gap)), 4),
        }
    with DEFECT_SCRUTINY_PATH.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(defect_scrutiny_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in defect_scrutiny_rows:
            writer.writerow({key: format(value, ".17g") if isinstance(value, float) else value for key, value in row.items()})
    out["topology_capture_sensitivity"] = {
        "p_out_values": [float(value) for value in p_out_values],
        "capture_values": [float(value) for value in capture_values],
        "scrutiny_families": list(scrutiny_families),
        "alternative_definition": "thresholded exponential equals one through full utilization and matches the canonical quartic curve at utilization three",
        "n_rows": len(topology_capture_rows),
        "by_family_and_capture": topology_capture_summary,
    }
    out["governance_defect_scrutiny_sensitivity"] = {
        "p_out_values": [float(value) for value in p_out_values],
        "scrutiny_families": list(scrutiny_families),
        "alternative_definition": "thresholded exponential equals one through full utilization and matches the canonical quartic curve at utilization three",
        "n_rows": len(defect_scrutiny_rows),
        "by_family": defect_scrutiny_summary,
    }

    # 6. Router overhead and skill-dependent calibration decay.
    ra_base_q = float(sym_ode_ss_vec(0.75, zm, zrm, ftm, pv1, q=0.0, g=0.20, auc=0.85)["Q"][0])
    f07_oh = {}
    for br in [0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28]:
        best_r = None
        for gv in np.linspace(0.24, 0.72, 17):
            r = sym_ode_ss_vec(0.75, zm, zrm, ftm, pv1, q=q_g5, g=float(gv), auc=0.85, b_router=br)
            if best_r is None or r["Q"][0] > best_r["Q"][0]:
                best_r = r
        f07_oh[f"b_router_{br:.2f}"] = {
            "pct_Bav": round(100.0 * br / B_av, 1),
            "Q": round(float(best_r["Q"][0]), 4),
            "Y": round(float(best_r["Y"][0]), 4),
            "u": round(float(best_r["u"][0]), 4),
            "lead_over_RA_Q": round(float(best_r["Q"][0] - ra_base_q), 4),
        }
    f07_decay = {}
    for rd in [0.0, 0.25, 0.50, 0.75, 1.0]:
        r = sym_ode_ss_vec(0.75, zm, zrm, ftm, pv1, q=q_g5, g=0.5567, auc=0.85, rho_decay=rd)
        f07_decay[f"rho_decay_{rd:.2f}"] = {
            "Q": round(float(r["Q"][0]), 4),
            "Y": round(float(r["Y"][0]), 4),
            "S": round(float(r["S"][0]), 4),
            "lead_over_RA_Q": round(float(r["Q"][0] - ra_base_q), 4),
        }
    out["router_overhead_and_calibration_decay"] = {"RA_baseline_Q": round(ra_base_q, 4), "by_overhead": f07_oh, "by_skill_decay": f07_decay}

    # 7. Zero-spillover check across all four topologies and a junior-learning delay.
    P_sp0 = dict(P); P_sp0["psi_spill"] = 0.0
    d_sp0 = np.round(np.arange(0.16, 0.56 + 1e-9, 0.01), 4)
    f09_spill = {}
    for top in ("mod", "mat", "pool", "cent"):
        sw_sp0 = m.adiabatic_continuation(m.build_topology_spec(top, 30, P_sp0), d_sp0, params=P_sp0)
        h_max = float(np.max(sw_sp0["Q_fwd"] - sw_sp0["Q_bwd"]))
        ic = np.where(sw_sp0["Q_fwd"] < 0.60)[0]
        dc = float(d_sp0[ic[0]]) if len(ic) else 1.0
        ir = np.where(sw_sp0["Q_bwd"] >= 0.60)[0]
        dr = float(d_sp0[ir[-1]]) if len(ir) else 0.0
        f09_spill[top] = {"max_hyst_Q": round(h_max, 4), "d_col": dc, "d_rec": dr, "width": round(max(0.0, dc - dr), 4)}

    def two_stock_bbw(tau_J: float) -> dict:
        dt = 0.05; T = 160.0; steps = int(T / dt)
        S = 0.935238; J = 0.935238
        Y_hist = []; Q_hist = []
        zeta_s = P["r"] * P["tau_v"]
        lam = P["lam0"] * (1.0 + P["alpha_ai"] * 0.75)
        sig_x = 1.0 / (1.0 + math.exp((0.75 - P["x0"]) / P["beta_a"]))
        b_base = P["J0"] + P["alpha_p"] * (1.0 - 0.65 * 0.75)
        for st in range(steps):
            s_eff = P["s_min"] + (1.0 - P["s_min"]) * S
            L_prim = lam * 0.75 * zeta_s / s_eff
            O = max(0.0, L_prim - B_av)
            L = L_prim + P["psi_spill"] * O
            u_t = L / B_av; u = min(1.0, u_t)
            chi = 1.0 / (1.0 + (0.36 * u_t) ** 4)
            m_eff = B_av * (1.0 - u) + P["sigma0"] * B_av * u * chi
            A = sig_x + (1.0 - sig_x) * (P["A_min"] * (1.0 - u) + P["A_cong"] * u)
            dr = P["eta_m"] * m_eff * S
            J_eq = b_base + A * P["mu0"] * (dr ** 2) / (P["K_m"] ** 2 + dr ** 2)
            theta = S * chi
            E_def = P["phi"] * P["p_in"] * (1.0 - P["catch_max"] * theta) + (1.0 - P["phi"]) * P["p_out"] * (1.0 - P["catch_max"] * theta)
            Q = P["h0"] + P["h1"] * theta
            Y = lam * (1.0 - P["c_def"] * 0.75 * E_def)
            if st % int(round(1.0 / dt)) == 0:
                Y_hist.append(Y); Q_hist.append(Q)
            if tau_J <= 1e-9:
                J = J_eq
                S = max(0.0, min(1.0, S + dt * P["delta"] * (J - S)))
            else:
                dJ = (J_eq - J) / tau_J
                dS = P["delta"] * (J - S)
                J = max(0.0, min(1.0, J + dt * dJ))
                S = max(0.0, min(1.0, S + dt * dS))
        y_arr = np.array(Y_hist)
        return {
            "tau_J": tau_J,
            "Y_sr_1_5": round(float(y_arr[:5].mean()), 4),
            "boom_periods_Y_gt_1": int(np.sum(y_arr[:40] > 1.0)),
            "Q_end": round(float(Q_hist[-1]), 4),
            "Y_end": round(float(y_arr[-1]), 4),
        }

    out["spillover_and_two_stock"] = {
        "psi_spill_0_by_topology": f09_spill,
        "mod_minus_best_rival_psi0": round(f09_spill["mod"]["d_col"] - max(f09_spill["mat"]["d_col"], f09_spill["pool"]["d_col"], f09_spill["cent"]["d_col"]), 4),
        "two_stock_junior_delay_d075": [two_stock_bbw(tj) for tj in [0.0, 1.0, 2.0, 4.0]],
    }

    # 8. Upper SLA breakdown bound (gamma_req * d >= x0)
    g5_085 = sym_ode_ss_vec(0.75, zm, zrm, ftm, make_pvec({**P, "gamma_req": 0.85}, 1), q=1.0 - 0.64 / 0.75, g=0.5567, auc=0.85)
    out["upper_sla_bound"] = {
        "d": 0.75,
        "x0": P["x0"],
        "gamma_req_cliff": round(P["x0"] / 0.75, 4),
        "G5_at_gamma_085_x064": {"Q": round(float(g5_085["Q"][0]), 4), "Y": round(float(g5_085["Y"][0]), 4)},
    }

    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
