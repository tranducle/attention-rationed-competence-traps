#!/usr/bin/env python3
"""Canonical two-tier networked organizational model for Stage 4 and Stage 5.

Every equation and parameter in this module matches
the manuscript formal specification (Section 2 and Appendix B) in an exact 1-to-1
correspondence with zero hidden branches, smooth proportional throughput
commitment `lam_req(d)`, genuine quadratic m = 2 mentoring cooperativity,
and pod-local first-pass verification in divisional sign-off hierarchies.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from scipy.stats import norm

CANONICAL_PARAMS: Dict[str, float] = {
    "r": 4.0,              # conserved junior-to-senior ratio N_J / N_S = 4:1
    "lam0": 1.0,           # baseline task arrival rate per junior per period
    "alpha_ai": 0.90,      # AI drafting throughput expansion on (0, inf)
    "phi": 0.80,           # routine inside-frontier share (phi_e = 0.20 OOD tail)
    "B": 1.0,              # conserved total attention budget per senior
    "b_exec": 0.22,        # senior direct execution share (B_av = 0.78)
    "tau_v": 0.174,        # base AI verification time per junior stream
    "kappa0": 0.058,       # cross-pod context-reconstruction verification cost
    "sigma0": 0.30,        # co-located review-as-mentoring recovery fraction
    "p_esc": 0.50,         # mandatory divisional principal sign-off escalation
    "p_mat": 0.11,         # secondary cross-functional review share in matrix
    "psi_spill": 2.85,     # sub-unitary divisional rework spillover (< 3.0)
    "rho_rew": 0.22,       # downstream rework load per unscreened OOD defect
    "s_min": 0.20,         # minimum eroded senior review speed at S_s = 0
    "J0": 0.12,            # unmentored novice floor (> 0)
    "alpha_p": 0.14,       # unassisted learning-by-doing gain (> 0)
    "mu0": 0.71,           # max mentoring transfer (J0 + alpha_p + mu0 = 0.97 <= 1)
    "eta_m": 4.35,         # mentoring input efficacy scale
    "K_m": 0.72,           # two-step mentoring half-saturation constant
    "A_min": 0.94,         # uncongested novice absorptive floor at high x
    "A_cong": 0.14,        # congested novice absorptive floor under queue overload
    "x0": 0.625,           # novice AI-crutch onset share
    "beta_a": 0.025,       # smooth logistic transition width for A_s(x, u_s)
    "delta": 0.18,         # demographic cohort turnover rate per period
    "nu": 0.10,            # pod-local promotion retention share
    "h0": 0.46,            # baseline OOD accuracy floor
    "h1": 0.35,            # senior-mastery-mediated OOD accuracy span (Q in [0.46, 0.81])
    "xi_esc": 0.64,        # second-pass independent residual defect catch factor
    "catch_max": 0.96,     # maximum defect interception fraction at full scrutiny
    "p_in": 0.012,         # AI defect rate on routine inside-frontier tasks
    "p_out": 0.94,         # AI hallucination/defect rate on outside-frontier tasks
    "c_def": 3.85,         # downstream liability per uncaptured task defect
    "gamma_req": 0.70,     # proportional throughput commitment fraction of d
    "pi_req": 2.40,        # contractual SLA backlog cost per unit throughput shortfall
    "x_cap": 0.56,         # G5 absorptive practice cap (q_G5(d) = max(0, 1 - x_cap/d))
}

_ROC_CACHE: Dict[Tuple[float, float, float], Tuple[float, float]] = {}
_SPEC_CACHE: Dict[Tuple[str, int, float, float], dict] = {}


def binormal_roc_capture_and_fpr(auc: float, g: float, phi_e: float) -> Tuple[float, float]:
    """Return (TPR C, routine FPR F_fp) at total routed share g in [0, 1]."""
    g_c = float(np.clip(g, 0.0, 1.0))
    if g_c <= 0.0:
        return 0.0, 0.0
    if g_c >= 1.0:
        return 1.0, 1.0
    auc_r = round(float(auc), 6)
    g_r = round(g_c, 6)
    pe_r = round(float(phi_e), 6)
    key = (auc_r, g_r, pe_r)
    if key in _ROC_CACHE:
        return _ROC_CACHE[key]
    if auc_r >= 1.0:
        c = min(1.0, g_c / phi_e)
        fp = max(0.0, (g_c - phi_e * c) / (1.0 - phi_e))
        _ROC_CACHE[key] = (c, fp)
        return c, fp
    if auc_r <= 0.5:
        _ROC_CACHE[key] = (g_c, g_c)
        return g_c, g_c
    d_roc = math.sqrt(2.0) * float(norm.ppf(min(0.999999, auc_r)))
    lo, hi = -11.0, 11.0
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        rate = phi_e * (1.0 - float(norm.cdf(mid - d_roc))) + (1.0 - phi_e) * (1.0 - float(norm.cdf(mid)))
        if rate > g_c:
            lo = mid
        else:
            hi = mid
    th = 0.5 * (lo + hi)
    c = float(np.clip(1.0 - float(norm.cdf(th - d_roc)), 0.0, 1.0))
    fp = float(np.clip(1.0 - float(norm.cdf(th)), 0.0, 1.0))
    _ROC_CACHE[key] = (c, fp)
    return c, fp


def binormal_roc_capture(auc: float, g: float, phi_e: float) -> float:
    """Return outside-frontier exception capture probability C(rho, g, phi_e)."""
    return binormal_roc_capture_and_fpr(auc, g, phi_e)[0]


def build_topology_spec(top: str, N: int, params: Dict[str, float] | None = None) -> dict:
    """Construct conserved 4:1 reporting and mentoring matrices for topology `top`."""
    P = CANONICAL_PARAMS if params is None else params
    r = int(round(P["r"]))
    if N % (r + 1) != 0:
        raise ValueError(f"Total specialists N={N} must be divisible by r+1={r+1}.")
    N_S = N // (r + 1)
    if top == "cent" and N_S % 3 != 0:
        raise ValueError(f"Centralized divisional topology requires 3 | N_S (got N_S={N_S}).")
    if top == "mat" and N_S < 3:
        raise ValueError(f"Small-world matrix topology requires N_S >= 3 (got N_S={N_S}).")
    if top == "pool" and N_S < 2:
        raise ValueError(f"Anonymous pooled topology requires N_S >= 2 (got N_S={N_S}).")

    key = (top, N, float(P["p_esc"]), float(P["p_mat"]))
    if key in _SPEC_CACHE and r == 4:
        return _SPEC_CACHE[key]

    N_J = r * N_S
    W_men = np.zeros((N_J, N_S), dtype=float)
    for i in range(N_J):
        W_men[i, i // r] = 1.0

    W_ver = np.zeros((N_J, N_S), dtype=float)
    W_esc = np.zeros((N_J, N_S), dtype=float)

    if top == "mod":
        W_ver[:, :] = W_men
    elif top == "mat":
        for i in range(N_J):
            k = i // r
            W_ver[i, k] += 0.50
            W_ver[i, (k - 1) % N_S] += 0.25
            W_ver[i, (k + 1) % N_S] += 0.25
            W_esc[i, (k - 2) % N_S] += P["p_mat"] / 2.0
            W_esc[i, (k + 2) % N_S] += P["p_mat"] / 2.0
    elif top == "pool":
        W_ver[:, :] = 1.0 / float(N_S)
    elif top == "cent":
        W_ver[:, :] = W_men
        for i in range(N_J):
            div = (i // r) // 3
            hub = 3 * div
            W_esc[i, hub] = P["p_esc"]
    else:
        raise ValueError(f"Unknown topology: {top}")

    w_cross = 1.0 - float((W_ver * W_men).sum()) / float(N_J)
    spec = {
        "top": top,
        "N": N,
        "N_S": N_S,
        "N_J": N_J,
        "W_men": W_men,
        "W_ver": W_ver,
        "W_esc": W_esc,
        "w_cross": w_cross,
    }
    if r == 4:
        _SPEC_CACHE[key] = spec
    return spec


def canonical_rhs_batch(
    S_mat: np.ndarray,
    spec: dict,
    d_nom: float | np.ndarray,
    params: Dict[str, float] | None = None,
    q_quota: float | np.ndarray = 0.0,
    v_bonus: float | np.ndarray = 0.0,
    g_route: float | np.ndarray = 1.0,
    auc: float = 0.85,
    guardrails: float | np.ndarray = 0.0,
    parallel: float | bool | np.ndarray = 0.0,
    zero_tau: bool = False,
    zero_delta: bool = False,
    concave_m1: bool = False,
    null_ctrl: bool = False,
    coach_relief_only: bool = False,
    external_replacement: bool = False,
    frozen_x_learn: float | None = None,
) -> Dict[str, np.ndarray]:
    """Evaluate exact vectorized deterministic RHS and observables over K states."""
    P = CANONICAL_PARAMS if params is None else params
    S_2d = np.atleast_2d(np.asarray(S_mat, dtype=float))
    K_batch, N_S = S_2d.shape
    phi_e = 1.0 - P["phi"]

    d_arr = np.broadcast_to(np.asarray(d_nom, dtype=float), (K_batch,))
    q_arr = np.broadcast_to(np.asarray(q_quota, dtype=float), (K_batch,))
    v_arr = np.broadcast_to(np.asarray(v_bonus, dtype=float), (K_batch,))
    g_arr = np.broadcast_to(np.asarray(g_route, dtype=float), (K_batch,))
    gr_arr = np.broadcast_to(np.asarray(guardrails, dtype=float), (K_batch,))

    x = np.clip(d_arr * (1.0 - q_arr) - v_arr, 0.0, 1.0)
    y = np.clip(g_arr * x, 0.0, 1.0)
    lam = P["lam0"] * (1.0 + P["alpha_ai"] * x * (1.0 - 0.35 * gr_arr))
    lam_req = P["lam0"] * (1.0 + P["alpha_ai"] * P["gamma_req"] * d_arr)
    B_av = np.maximum(0.15, P["B"] - P["b_exec"] - 0.34 * gr_arr)

    c_arr = np.empty(K_batch, dtype=float)
    fp_arr = np.empty(K_batch, dtype=float)
    for idx in range(K_batch):
        c_arr[idx], fp_arr[idx] = binormal_roc_capture_and_fpr(auc, float(g_arr[idx]), phi_e)

    tau_v = 0.0 if zero_tau else P["tau_v"]
    kappa0 = 0.0 if (zero_tau or null_ctrl) else P["kappa0"]
    sigma0 = 0.0 if null_ctrl else P["sigma0"]

    W_men = spec["W_men"]
    W_ver = spec["W_ver"]
    W_esc = np.zeros_like(spec["W_esc"]) if null_ctrl else spec["W_esc"]

    zeta = ((W_ver + W_esc) * (tau_v + kappa0 * (1.0 - W_men))).sum(axis=0)
    zeta_rew = (W_ver * (tau_v + kappa0 * (1.0 - W_men))).sum(axis=0)
    s_eff = np.ones_like(S_2d) if concave_m1 else (P["s_min"] + (1.0 - P["s_min"]) * S_2d)

    p_ens = np.broadcast_to(np.asarray(parallel, dtype=float), (K_batch,))
    par_mult = 1.0 + 0.65 * p_ens
    rew = np.zeros(K_batch, dtype=float) if zero_tau else (P["rho_rew"] * phi_e * P["p_out"] * x * (1.0 - c_arr))
    L_prim = ((par_mult * y * lam)[:, None] * zeta[None, :] + (lam * rew)[:, None] * zeta_rew[None, :]) / s_eff

    raw_ov = np.maximum(0.0, L_prim - B_av[:, None])
    if spec["top"] == "cent":
        n_div = N_S // 3
        ov_div = raw_ov.reshape(K_batch, n_div, 3).mean(axis=2)
        ov_share = np.repeat(ov_div, 3, axis=1)
    else:
        ov_share = raw_ov.mean(axis=1, keepdims=True)
    L = L_prim + P["psi_spill"] * ov_share

    u_tilde = L / B_av[:, None]
    u = np.minimum(1.0, u_tilde)
    chi = 1.0 / (1.0 + (0.36 * u_tilde) ** 4)

    local_num = (W_ver * W_men * tau_v).sum(axis=0)
    f_tau = np.where(zeta > 0.0, local_num / np.where(zeta > 0.0, zeta, 1.0), 0.0)

    u_coach = np.zeros_like(u) if coach_relief_only else u
    chi_coach = np.ones_like(chi) if coach_relief_only else chi
    m_eff = B_av[:, None] * (1.0 - u_coach) + sigma0 * B_av[:, None] * u_coach * chi_coach * f_tau[None, :]

    x_learn = (x * (1.0 - gr_arr)) if frozen_x_learn is None else np.full(K_batch, float(frozen_x_learn), dtype=float)
    sig_x = 1.0 / (1.0 + np.exp((x_learn - P["x0"]) / P["beta_a"]))
    A = sig_x[:, None] + (1.0 - sig_x)[:, None] * (P["A_min"] * (1.0 - u_coach) + P["A_cong"] * u_coach)
    b = P["J0"] + P["alpha_p"] * (1.0 - 0.65 * x_learn) - 0.004 * gr_arr

    drive = P["eta_m"] * m_eff * S_2d
    if concave_m1:
        hill = drive / (P["K_m"] + drive)
    else:
        hill = (drive ** 2) / (P["K_m"] ** 2 + drive ** 2)

    J = b[:, None] + A * P["mu0"] * hill
    if external_replacement:
        s0_ext = P["J0"] + P["alpha_p"] + P["mu0"] * ((P["eta_m"] * (P["B"] - P["b_exec"]) * 0.935) ** 2) / (
            P["K_m"] ** 2 + (P["eta_m"] * (P["B"] - P["b_exec"]) * 0.935) ** 2
        )
        J_in = np.full_like(J, s0_ext)
    elif spec["top"] == "cent":
        n_div = N_S // 3
        J_div = np.repeat(J.reshape(K_batch, n_div, 3).mean(axis=2), 3, axis=1)
        J_in = P["nu"] * J + (1.0 - P["nu"]) * J_div
    else:
        J_in = P["nu"] * J + (1.0 - P["nu"]) * J.mean(axis=1, keepdims=True)

    dS = np.zeros_like(S_2d) if zero_delta else P["delta"] * (J_in - S_2d)

    eff_rev = S_2d * chi
    q_prim = eff_rev @ W_ver.T
    q_prim = np.minimum(1.0, q_prim * (1.0 + 0.03 * p_ens[:, None]))
    esc_sum = eff_rev @ W_esc.T
    psi_two = q_prim + (1.0 - q_prim) * P["xi_esc"] * esc_sum
    psi_bar = psi_two.mean(axis=1)

    theta_ood = c_arr * psi_bar
    theta_in = fp_arr * psi_bar
    Q = P["h0"] + P["h1"] * theta_ood
    exp_def = (1.0 - phi_e) * P["p_in"] * (1.0 - P["catch_max"] * theta_in) + phi_e * P["p_out"] * (
        1.0 - P["catch_max"] * theta_ood
    )
    Y_gross = lam * (1.0 - P["c_def"] * x * exp_def)
    Y = Y_gross - P["pi_req"] * np.maximum(0.0, lam_req - lam)

    return {
        "dS": dS,
        "J": J,
        "u": u,
        "chi": chi,
        "m_eff": m_eff,
        "A": A,
        "Q": Q,
        "Y": Y,
        "Y_gross": Y_gross,
        "x": x,
        "y": y,
        "c_cap": c_arr,
        "psi_bar": psi_bar,
    }


def relax_steady_state(
    spec: dict,
    d_grid: np.ndarray,
    S_init: float | np.ndarray = 0.935,
    params: Dict[str, float] | None = None,
    max_iter: int = 3000,
    tol: float = 1e-10,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Relax each grid point to a strict fixed point with |dS|_inf < tol."""
    P = CANONICAL_PARAMS if params is None else params
    K = len(d_grid)
    N_S = spec["N_S"]
    if np.isscalar(S_init):
        S = np.full((K, N_S), float(S_init), dtype=float)
    else:
        arr = np.asarray(S_init, dtype=float)
        S = np.broadcast_to(arr, (K, N_S)).copy()
    step = min(0.95, 0.90 / max(1e-6, float(P["delta"])))
    for _ in range(max_iter):
        out = canonical_rhs_batch(S, spec, d_grid, params=params, **kwargs)
        if float(np.max(np.abs(out["dS"]))) < tol:
            break
        S = np.clip(S + step * out["dS"], 0.0, 1.0)
    out = canonical_rhs_batch(S, spec, d_grid, params=params, **kwargs)
    out["S"] = S
    out["max_residual"] = float(np.max(np.abs(out["dS"])))
    return out


def adiabatic_continuation(
    spec: dict,
    d_grid: np.ndarray,
    S0: float = 0.935,
    params: Dict[str, float] | None = None,
    tol: float = 1e-9,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Sequential 0 -> 1 -> 0 adiabatic continuation along `d_grid`."""
    N_S = spec["N_S"]
    K = len(d_grid)
    S_fwd = np.zeros((K, N_S), dtype=float)
    Q_fwd = np.zeros(K, dtype=float)
    Y_fwd = np.zeros(K, dtype=float)
    u_fwd = np.zeros((K, N_S), dtype=float)

    cur_S = np.full((1, N_S), float(S0), dtype=float)
    for idx, d_val in enumerate(d_grid):
        res = relax_steady_state(spec, np.array([d_val]), S_init=cur_S, params=params, max_iter=1200, tol=tol, **kwargs)
        cur_S = res["S"]
        S_fwd[idx] = cur_S[0]
        Q_fwd[idx] = res["Q"][0]
        Y_fwd[idx] = res["Y"][0]
        u_fwd[idx] = res["u"][0]

    S_bwd = np.zeros((K, N_S), dtype=float)
    Q_bwd = np.zeros(K, dtype=float)
    Y_bwd = np.zeros(K, dtype=float)
    for rev_idx in range(K - 1, -1, -1):
        res = relax_steady_state(spec, np.array([d_grid[rev_idx]]), S_init=cur_S, params=params, max_iter=1200, tol=tol, **kwargs)
        cur_S = res["S"]
        S_bwd[rev_idx] = cur_S[0]
        Q_bwd[rev_idx] = res["Q"][0]
        Y_bwd[rev_idx] = res["Y"][0]

    return {
        "d_grid": d_grid,
        "S_fwd": S_fwd,
        "Q_fwd": Q_fwd,
        "Y_fwd": Y_fwd,
        "u_fwd": u_fwd,
        "S_bwd": S_bwd,
        "Q_bwd": Q_bwd,
        "Y_bwd": Y_bwd,
        "hyst_Q": Q_fwd - Q_bwd,
    }


if __name__ == "__main__":
    P = CANONICAL_PARAMS
    assert P["J0"] + P["alpha_p"] + P["mu0"] <= 1.0
    d_grid = np.round(np.linspace(0.0, 1.0, 51), 4)

    mod15 = build_topology_spec("mod", 15)
    s_scan = np.linspace(0.005, 0.995, 1000)[:, None]
    for dv in [0.0, 0.5, 0.8, 1.0]:
        ds = canonical_rhs_batch(s_scan, mod15, dv, zero_tau=True)["dS"][:, 0]
        roots = [round(float(0.5 * (s_scan[j, 0] + s_scan[j + 1, 0])), 4) for j in range(len(ds) - 1) if ds[j] * ds[j + 1] <= 0]
        assert len(roots) == 1 and roots[0] > 0.5, f"u=0 root failure at d={dv}: {roots}"
    ds_d1 = canonical_rhs_batch(s_scan, mod15, 1.0)["dS"][:, 0]
    roots_d1 = [round(float(0.5 * (s_scan[j, 0] + s_scan[j + 1, 0])), 4) for j in range(len(ds_d1) - 1) if ds_d1[j] * ds_d1[j + 1] <= 0]
    assert len(roots_d1) == 1 and roots_d1[0] < 0.5, f"d=1 root failure: {roots_d1}"

    for N in [15, 30, 60, 120]:
        summary = {}
        for top in ["cent", "pool", "mat", "mod"]:
            sp = build_topology_spec(top, N)
            cont = adiabatic_continuation(sp, d_grid)
            jumps = np.diff(cont["Q_fwd"])
            idx = int(np.argmin(jumps))
            d_col = float(d_grid[idx + 1])
            s0 = cont["S_fwd"][0:1]
            y_sr = float(canonical_rhs_batch(s0, sp, min(1.0, d_col + 0.04))["Y"][0])
            y_lr75 = float(cont["Y_fwd"][37])
            fine_d = np.round(np.linspace(0.15, 0.60, 91), 4)
            fine_c = adiabatic_continuation(sp, fine_d)
            h_mask = fine_c["hyst_Q"] > 0.05
            h_width = round(float(fine_d[h_mask][-1] - fine_d[h_mask][0] + 0.005), 4) if np.any(h_mask) else 0.0
            summary[top] = (d_col, round(-float(jumps[idx]), 4), round(float(cont["hyst_Q"].max()), 4), h_width, round(float(cont["Q_fwd"][5]), 5), round(y_sr, 4), round(y_lr75, 4))
        print(f"N={N}: {summary}")
        assert summary["cent"][0] < summary["pool"][0] < summary["mod"][0]
        assert summary["mod"][0] - max(summary["cent"][0], summary["pool"][0], summary["mat"][0]) >= 0.08
        for top in ["cent", "pool", "mat", "mod"]:
            assert summary[top][1] >= 0.10, f"Jump < 0.10 for {top}: {summary[top]}"
            assert summary[top][2] >= 0.10, f"Hyst < 0.10 for {top}: {summary[top]}"
            assert summary[top][3] >= (0.05 if top == "mod" else 0.025), f"Width too narrow for {top}: {summary[top]}"

    q_nulls = [adiabatic_continuation(build_topology_spec(t, 30), d_grid, null_ctrl=True)["Q_fwd"] for t in ["mod", "mat", "pool", "cent"]]
    max_null_diff = max(float(np.max(np.abs(q_nulls[0] - qn))) for qn in q_nulls[1:])
    print("Max null trajectory diff (psi_spill=2.85 active):", max_null_diff)
    assert max_null_diff < 1e-12

    sp30 = build_topology_spec("mod", 30)
    intact_c = adiabatic_continuation(sp30, d_grid)
    print(f"Intact: Q(0.76)={intact_c['Q_fwd'][38]:.4f}, max_hyst={intact_c['hyst_Q'].max():.5f}")
    for name, kw in [
        ("zero_tau", dict(zero_tau=True)),
        ("zero_delta", dict(zero_delta=True)),
        ("concave_m1", dict(concave_m1=True)),
        ("coach_relief", dict(coach_relief_only=True)),
        ("ext_rep", dict(external_replacement=True)),
    ]:
        c = adiabatic_continuation(sp30, d_grid, **kw)
        print(f"Knockout {name}: Q(0.76)={c['Q_fwd'][38]:.4f}, max_hyst={c['hyst_Q'].max():.5f}")
        assert c["hyst_Q"].max() < 1e-4
        if name != "concave_m1":
            assert c["Q_fwd"][38] - intact_c["Q_fwd"][38] >= 0.10

    # Vectorized 25-grid governance sweep!
    for d_val in [0.70, 0.75, 0.80]:
        q_g5 = max(0.0, 1.0 - P["x_cap"] / d_val)
        for auc_val in [0.60, 0.75, 0.85, 0.95]:
            g_cand = np.linspace(0.24, 1.0, 25)
            r_g5 = relax_steady_state(sp30, np.full(25, d_val), S_init=0.935, q_quota=q_g5, g_route=g_cand, auc=auc_val)
            feas_idx = np.where(r_g5["Y"] >= 1.02)[0]
            best_i = int(feas_idx[np.argmax(r_g5["Q"][feas_idx])]) if len(feas_idx) else int(np.argmax(r_g5["Q"]))
            g_star, Q_g5, Y_g5 = float(g_cand[best_i]), float(r_g5["Q"][best_i]), float(r_g5["Y"][best_i])

            q_cand = np.linspace(0.0, 0.65, 25)
            r_qa = relax_steady_state(sp30, np.full(25, d_val), S_init=0.935, q_quota=q_cand, g_route=1.0, auc=auc_val)
            qa_feas = np.where(r_qa["Y"] >= Y_g5 - 0.02)[0]
            max_q_qa_feas = float(np.max(r_qa["Q"][qa_feas])) if len(qa_feas) else -1.0

            ra_cand = np.linspace(0.20, 1.0, 25)
            r_ra = relax_steady_state(sp30, np.full(25, d_val), S_init=0.935, q_quota=0.0, g_route=ra_cand, auc=auc_val)
            max_q_ra = float(np.max(r_ra["Q"]))

            r_rt = relax_steady_state(sp30, np.array([d_val]), S_init=0.935, q_quota=q_g5, g_route=g_star, auc=0.50)
            r_fl = relax_steady_state(sp30, np.array([d_val]), S_init=0.935, q_quota=q_g5, g_route=g_star, auc=auc_val, frozen_x_learn=d_val)
            rt_gap = Q_g5 - float(r_rt["Q"][0])
            fl_gap = Q_g5 - float(r_fl["Q"][0])
            print(
                f"d={d_val} AUC={auc_val}: G5=(g={g_star:.2f}, Q={Q_g5:.4f}, Y={Y_g5:.4f}) | "
                f"QA_feas={len(qa_feas)} maxQ_QA_feas={max_q_qa_feas:.4f} | RA_maxQ={max_q_ra:.4f} | "
                f"RT_gapQ={rt_gap:.4f} | FL_gapQ={fl_gap:.4f}"
            )
            if auc_val >= 0.75:
                assert len(qa_feas) == 0
                assert Q_g5 - max_q_ra >= 0.10
                assert rt_gap >= (0.03 if auc_val == 0.75 else (0.05 if auc_val == 0.85 else 0.08))
                assert fl_gap >= 0.10
    print("ALL CANONICAL SELF-CHECKS PASSED CLEANLY!")
