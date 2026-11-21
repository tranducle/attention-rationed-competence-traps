#!/usr/bin/env python3
"""SymPy symbolic proof and canonical implementation consistency verification.

Verifies Propositions 1, 2, 3 and Theorem 1 symbolically AND cross-checks exact numerical
agreement with canonical_org_model.py so the formal specification equations and canonical Python implementation remain in exact agreement.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import numpy as np
import sympy as sp

REPO_ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = REPO_ROOT / "src"
OUTPUT_DIR = Path(os.environ.get("MODEL_VERIFY_OUTPUT_DIR", str(REPO_ROOT / "results")))
sys.path.insert(0, str(SIM_DIR))
from canonical_org_model import (
    binormal_roc_capture,  # noqa: E402
    CANONICAL_PARAMS,
    adiabatic_continuation,
    build_topology_spec,
    canonical_rhs_batch,
    relax_steady_state,
)


def verify_all() -> dict:
    results: dict = {"verdict": "MATH_VERIFIED", "items": []}

    # 1. Proposition 1(i): Strict forward invariance on [0, 1] and Q in [h0, h0 + h1] <= 1
    J0, alpha_p, mu0, x, A, hill = sp.symbols("J0 alpha_p mu0 x A hill", nonnegative=True, real=True)
    J_upper = (J0 + alpha_p * (1 - sp.Rational(65, 100) * x) + A * mu0 * hill).subs({x: 0, A: 1, hill: 1})
    assert sp.simplify(J_upper - (J0 + alpha_p + mu0)) == 0
    P_sum = CANONICAL_PARAMS["J0"] + CANONICAL_PARAMS["alpha_p"] + CANONICAL_PARAMS["mu0"]
    assert 0.0 < P_sum <= 1.0, f"A0 violated: {P_sum}"
    assert 0.0 < CANONICAL_PARAMS["h0"] + CANONICAL_PARAMS["h1"] <= 1.0
    results["items"].append({
        "id": "Prop1_i_forward_invariance_and_Q_bounds",
        "status": "verified",
        "symbolic_upper_bound": str(J_upper),
        "canonical_sum_J0_alpha_p_mu0": round(P_sum, 6),
    })

    # 2. Proposition 1(ii): Exact quadratic saturation root d_sat(S0) and short-run boom derivative at d = 0+
    r, lam0, alpha_ai, tau_v, s_eff, B_av, d = sp.symbols(
        "r lam0 alpha_ai tau_v s_eff B_av d", positive=True, real=True
    )
    L_mod = r * lam0 * d * (1 + alpha_ai * d) * tau_v / s_eff
    roots_d = sp.solve(sp.Eq(L_mod, B_av), d)
    pos_roots = [rt for rt in roots_d if rt.subs({r: 4, lam0: 1, alpha_ai: 1, tau_v: 1, s_eff: 1, B_av: 1}) > 0]
    assert len(pos_roots) == 1
    d_sat_closed = (-1 + sp.sqrt(1 + 4 * alpha_ai * B_av * s_eff / (r * lam0 * tau_v))) / (2 * alpha_ai)
    assert sp.simplify(pos_roots[0] - d_sat_closed) == 0
    assert sp.simplify(L_mod.subs(d, d_sat_closed) - B_av) == 0

    c_def, phi_e, p_in, p_out, c_max, S0 = sp.symbols("c_def phi_e p_in p_out c_max S0", positive=True, real=True)
    u_tilde = L_mod / B_av
    chi_d = 1 / (1 + (sp.Rational(36, 100) * u_tilde) ** 4)
    q_eff_sr = S0 * chi_d
    p_bar = (1 - phi_e) * p_in + phi_e * p_out
    Y_sr = lam0 * (1 + alpha_ai * d) * (1 - c_def * d * p_bar * (1 - c_max * q_eff_sr))
    dY_dd_0 = sp.simplify(sp.diff(Y_sr, d).subs(d, 0))
    expected_dY0 = lam0 * (alpha_ai - c_def * p_bar * (1 - c_max * S0))
    assert sp.simplify(dY_dd_0 - expected_dY0) == 0
    results["items"].append({
        "id": "Prop1_ii_closed_form_d_sat_and_short_run_slope",
        "status": "verified",
        "d_sat_formula": str(d_sat_closed),
        "dY_dd_at_d0": str(dY_dd_0),
    })

    # 3. Proposition 1(iii)-(iv): Concave m=1 second derivative J''(S) <= 0, A4 root counts, and knockouts
    S, b, M_amp, eta_m, m_eff, K = sp.symbols("S b M_amp eta_m m_eff K", positive=True, real=True)
    J_concave = b + M_amp * (eta_m * m_eff * S) / (K + eta_m * m_eff * S)
    d2J_dS2 = sp.simplify(sp.diff(J_concave, S, 2))
    assert d2J_dS2.is_negative is True

    sp_mod = build_topology_spec("mod", 30)
    s_scan = np.linspace(0.002, 0.998, 2000)[:, None]
    for dv in [0.0, 0.25, 0.50, 0.75, 1.0]:
        ds_u0 = canonical_rhs_batch(s_scan, sp_mod[:, :1] if False else build_topology_spec("mod", 15), dv, zero_tau=True)["dS"][:, 0]
        r_u0 = [0.5 * (s_scan[j, 0] + s_scan[j + 1, 0]) for j in range(len(ds_u0) - 1) if ds_u0[j] * ds_u0[j + 1] <= 0]
        assert len(r_u0) == 1 and r_u0[0] > 0.5, f"A4 u=0 uniqueness failed at d={dv}: {r_u0}"

    mod15 = build_topology_spec("mod", 15)
    ds_0 = canonical_rhs_batch(s_scan, mod15, 0.0)["dS"][:, 0]
    r_0 = [0.5 * (s_scan[j, 0] + s_scan[j + 1, 0]) for j in range(len(ds_0) - 1) if ds_0[j] * ds_0[j + 1] <= 0]
    ds_1 = canonical_rhs_batch(s_scan, mod15, 1.0)["dS"][:, 0]
    r_1 = [0.5 * (s_scan[j, 0] + s_scan[j + 1, 0]) for j in range(len(ds_1) - 1) if ds_1[j] * ds_1[j + 1] <= 0]
    ds_mid = canonical_rhs_batch(s_scan, mod15, 0.44)["dS"][:, 0]
    r_mid = [0.5 * (s_scan[j, 0] + s_scan[j + 1, 0]) for j in range(len(ds_mid) - 1) if ds_mid[j] * ds_mid[j + 1] <= 0]
    assert len(r_0) == 1 and r_0[0] > 0.5
    assert len(r_1) == 1 and (CANONICAL_PARAMS["J0"] + 0.35 * CANONICAL_PARAMS["alpha_p"]) < r_1[0] < 0.5
    assert len(r_mid) == 3 and 0.0 < r_mid[0] < r_mid[1] < 0.5 < r_mid[2] < 1.0

    d_grid = np.round(np.linspace(0.0, 1.0, 51), 4)
    zt_cont = adiabatic_continuation(sp_mod, d_grid, zero_tau=True)
    m1_cont = adiabatic_continuation(sp_mod, d_grid, concave_m1=True)
    assert float(np.max(np.abs(zt_cont["hyst_Q"]))) < 1e-7
    assert float(np.max(np.abs(m1_cont["hyst_Q"]))) < 1e-7
    results["items"].append({
        "id": "Prop1_iii_iv_bistability_and_knockout_proofs",
        "status": "verified",
        "concave_second_derivative": str(d2J_dS2),
        "d_star_044_three_roots": [round(float(v), 5) for v in r_mid],
        "canonical_zero_tau_adiabatic_hysteresis": round(float(np.max(np.abs(zt_cont["hyst_Q"]))), 10),
    })

    # 4. Proposition 2 & Proposition 3: Pod-local + divisional hub load, two-pass gain, pooling friction, null parity
    NS = sp.Symbol("NS", integer=True, positive=True)
    p_esc, kappa0, sigma0, xi_esc = sp.symbols("p_esc kappa0 sigma0 xi_esc", positive=True, real=True)
    zeta_mod = 4 * tau_v
    zeta_cent_hub = 4 * (1 + 3 * p_esc) * tau_v + 8 * p_esc * kappa0
    diff_hub_mod = sp.simplify(zeta_cent_hub - zeta_mod)
    assert diff_hub_mod.is_positive is True and NS not in diff_hub_mod.free_symbols

    sp_cent30 = build_topology_spec("cent", 30)
    zeta_py_hub = float(((sp_cent30["W_ver"] + sp_cent30["W_esc"]) * (CANONICAL_PARAMS["tau_v"] + CANONICAL_PARAMS["kappa0"] * (1.0 - sp_cent30["W_men"]))).sum(axis=0)[0])
    zeta_sym_hub = float(zeta_cent_hub.subs({tau_v: CANONICAL_PARAMS["tau_v"], p_esc: CANONICAL_PARAMS["p_esc"], kappa0: CANONICAL_PARAMS["kappa0"]}))
    assert abs(zeta_py_hub - zeta_sym_hub) < 1e-12

    S_unit = sp.Symbol("S_unit", positive=True, real=True)
    two_pass_gain = sp.simplify((S_unit + p_esc * (1 - S_unit) * xi_esc * S_unit) - S_unit)
    assert sp.simplify(two_pass_gain - p_esc * S_unit * (1 - S_unit) * xi_esc) == 0

    NS_ge2 = sp.Symbol("k", integer=True, nonnegative=True) + 2
    diff_pool_mod = sp.simplify(4 * (tau_v + kappa0 * (1 - 1 / NS_ge2)) - 4 * tau_v)
    assert diff_pool_mod.is_positive is True

    q_nulls = [adiabatic_continuation(build_topology_spec(t, 30), d_grid, null_ctrl=True)["Q_fwd"] for t in ["mod", "mat", "pool", "cent"]]
    max_null_diff = max(float(np.max(np.abs(q_nulls[0] - qn))) for qn in q_nulls[1:])
    assert max_null_diff < 1e-12
    results["items"].append({
        "id": "Prop2_Prop3_scale_invariance_two_pass_pooling_and_null_parity",
        "status": "verified",
        "hub_minus_mod_weight": str(diff_hub_mod),
        "two_pass_uncongested_gain": str(two_pass_gain),
        "pool_minus_mod_weight_NS_ge_2": str(diff_pool_mod),
        "max_null_trajectory_difference": max_null_diff,
    })

    # 5. Theorem 1: Total-probability capture bound C <= g / phi_e and proportional throughput shortfall
    g_sym, phi_e_sym, tpr_sym, fpr_sym = sp.symbols("g phi_e tpr fpr", nonnegative=True, real=True)
    tpr_upper = sp.solve(sp.Eq(phi_e_sym * tpr_sym + (1 - phi_e_sym) * fpr_sym, g_sym), tpr_sym)[0]
    assert sp.simplify(g_sym / phi_e_sym - tpr_upper) == sp.simplify((1 - phi_e_sym) * fpr_sym / phi_e_sym)

    pi_req_s, gamma_req_s, x_s = sp.symbols("pi_req gamma_req x_s", positive=True, real=True)
    shortfall = sp.simplify(pi_req_s * (lam0 * (1 + alpha_ai * gamma_req_s * d) - lam0 * (1 + alpha_ai * x_s)))
    assert sp.simplify(shortfall - pi_req_s * lam0 * alpha_ai * (gamma_req_s * d - x_s)) == 0
    # Verify non-vacuous g_min in (0, 1) and Theorem 1(i) bound at d=0.75, bar_rho=0.95
    g_scan = np.linspace(0.01, 1.0, 200)
    r_hi = relax_steady_state(sp_mod, np.full(200, 0.75), S_init=0.935, q_quota=0.0, g_route=g_scan, auc=1.0)
    s_hi = r_hi["S"].mean(axis=1)
    g_min_75 = float(g_scan[np.where(s_hi < 0.50)[0][0]])
    assert 0.0 < g_min_75 < 1.0
    s_d0 = float(relax_steady_state(sp_mod, np.array([0.75]), S_init=0.935, q_quota=0.0, g_route=0.0, auc=0.95)["S"].mean())
    c_bar = binormal_roc_capture(0.95, g_min_75, 1.0 - CANONICAL_PARAMS["phi"])
    q_thm1_ub = CANONICAL_PARAMS["h0"] + CANONICAL_PARAMS["h1"] * max(0.5, c_bar * s_d0)
    r_ra95 = relax_steady_state(sp_mod, np.full(200, 0.75), S_init=0.935, q_quota=0.0, g_route=g_scan, auc=0.95)
    assert float(np.max(r_ra95["Q"])) <= q_thm1_ub + 1e-6
    results["items"].append({
        "id": "Thm1_roc_capture_and_sla_complementarity_bounds",
        "g_min_at_d075": round(g_min_75, 4),
        "thm1_i_upper_bound_at_auc095": round(q_thm1_ub, 4),
        "status": "verified",
        "g_over_phi_e_minus_C": str(sp.simplify(g_sym / phi_e_sym - tpr_upper)),
        "throughput_shortfall_formula": str(shortfall),
    })

    return results


def main() -> None:
    res = verify_all()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "sympy_proof_results.json"
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"SymPy proof verification: {res['verdict']} ({len(res['items'])} items verified, 0 mismatches with canonical_org_model.py)")


if __name__ == "__main__":
    main()
