#!/usr/bin/env python3
"""Targeted unit and docking verification tests for `sim_abm_org.py`."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
for _sub in ("src", "experiments", "scripts"):
    _p = str(_REPO_ROOT / _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import math
import time
import numpy as np

from canonical_org_model import CANONICAL_PARAMS, adiabatic_continuation, build_topology_spec
from sim_abm_org import (
    ASSIGN_ESCALATION_DEST,
    ASSIGN_ESCALATION_TRIGGER,
    ASSIGN_PRIMARY_DEST,
    ASSIGN_REWORK_DEST,
    STREAM_ASSIGN,
    _assign_escalation_pass,
    _assign_full_pass,
    _assignment_uniforms,
    _build_custom_topology_spec,
    _splitmix64,
    _validate_assignment_matrices,
    config_hash_of,
    simulate_adiabatic_sweep,
    simulate_batch,
    simulate_endogenous_d_controller,
    tune_arm_on_validation,
    tune_g5_blind_5x5,
)


def test_config_hash_ignores_seed():
    h1 = config_hash_of({"arm": "G5", "d": 0.75, "seed": 1000})
    h2 = config_hash_of({"arm": "G5", "d": 0.75, "seed": 1099})
    h3 = config_hash_of({"arm": "G5", "d": 0.80, "seed": 1000})
    assert h1 == h2
    assert h1 != h3


def test_assignment_counter_vectors_and_domains():
    raw = _splitmix64(np.array([0, 1], dtype=np.uint64))
    assert [int(value) for value in raw] == [0xE220A8397B1DCDAF, 0x910A2DEC89025CC1]
    token = np.array([0x0123456789ABCDEF], dtype=np.uint64)
    values = [
        _assignment_uniforms(token, 7, domain, 3, 8)
        for domain in (ASSIGN_PRIMARY_DEST, ASSIGN_ESCALATION_TRIGGER, ASSIGN_ESCALATION_DEST, ASSIGN_REWORK_DEST)
    ]
    assert all(np.all((item >= 0.0) & (item < 1.0)) for item in values)
    assert len({item.tobytes() for item in values}) == 4
    assert np.array_equal(values[0], _assignment_uniforms(token, 7, ASSIGN_PRIMARY_DEST, 3, 8))


def test_assignment_stream_consumption_and_exact_conservation():
    seeds = np.arange(900, 904)
    a = simulate_batch(seeds, "mat", 30, 0.55, T=24, assignment_stream_id=STREAM_ASSIGN)
    b = simulate_batch(seeds, "mat", 30, 0.55, T=24, assignment_stream_id=STREAM_ASSIGN)
    c = simulate_batch(seeds, "mat", 30, 0.55, T=24, assignment_stream_id=STREAM_ASSIGN + 1000)
    diagnostic_keys = (
        "assignment_primary_by_senior",
        "assignment_escalation_by_senior",
        "assignment_rework_by_senior",
    )
    for key in diagnostic_keys:
        assert np.array_equal(a[key], b[key])
    assert any(not np.array_equal(a[key], c[key]) for key in diagnostic_keys)
    assert np.array_equal(a["assignment_primary_tickets"], a["assignment_primary_destinations"])
    assert np.array_equal(a["assignment_escalation_triggers"], a["assignment_escalation_destinations"])
    assert np.array_equal(a["assignment_rework_tickets"], a["assignment_rework_destinations"])
    assert np.all(a["assignment_escalation_triggers"] <= a["assignment_primary_tickets"])
    assert np.all(a["assignment_conservation_max_abs"] == 0)


def test_assignment_probability_family_and_load_convergence():
    row_cases = []
    variants = [
        {},
        {"pass_conserved_hub": True},
        {"selective_hub": True},
        {"role_balanced_hub": True},
    ]
    for N in (15, 30, 60, 120):
        for top in ("mod", "mat", "pool", "cent"):
            applicable = variants if top == "cent" and N == 30 else [{}]
            for options in applicable:
                spec = _build_custom_topology_spec(top, N, CANONICAL_PARAMS, **options)
                _validate_assignment_matrices(spec["W_ver"], spec["W_esc"])
                for junior in range(spec["N_J"]):
                    row_cases.append((spec["W_ver"][junior], spec["W_esc"][junior], junior))

    positive_cells = 0
    positive_values = []
    for primary, escalation, _ in row_cases:
        positive_cells += int(np.count_nonzero(primary > 0.0))
        positive_values.extend(primary[primary > 0.0].tolist())
        esc_probability = float(escalation.sum())
        if esc_probability > 0.0:
            positive_cells += 1 + int(np.count_nonzero(escalation > 0.0))
            positive_values.append(esc_probability)
            positive_values.extend((escalation[escalation > 0.0] / esc_probability).tolist())

    sample_size = 200000
    assert sample_size * min(positive_values) >= 5000
    family_bound = math.sqrt(math.log(2.0 * positive_cells / 0.001) / (2.0 * sample_size))
    token = np.array([0xB6A7C8D9EAFB1021], dtype=np.uint64)
    for case_index, (primary, escalation, junior) in enumerate(row_cases):
        primary_u = _assignment_uniforms(token, case_index, ASSIGN_PRIMARY_DEST, junior, sample_size)[0]
        primary_dest = np.searchsorted(np.cumsum(primary), primary_u, side="right")
        primary_freq = np.bincount(primary_dest, minlength=len(primary)) / float(sample_size)
        assert np.all(primary_freq[primary == 0.0] == 0.0)
        for observed, expected in zip(primary_freq[primary > 0.0], primary[primary > 0.0]):
            assert abs(float(observed - expected)) <= family_bound
            assert abs(float(observed - expected)) / float(expected) <= 0.10

        esc_probability = float(escalation.sum())
        if esc_probability <= 0.0:
            continue
        trigger_u = _assignment_uniforms(token, case_index, ASSIGN_ESCALATION_TRIGGER, junior, sample_size)[0]
        trigger_mask = trigger_u < esc_probability
        trigger_freq = float(trigger_mask.mean())
        assert abs(trigger_freq - esc_probability) <= family_bound
        assert abs(trigger_freq - esc_probability) / esc_probability <= 0.10
        conditional = escalation / esc_probability
        destination_u = _assignment_uniforms(token, case_index, ASSIGN_ESCALATION_DEST, junior, sample_size)[0]
        destination = np.searchsorted(np.cumsum(conditional), destination_u, side="right")
        destination_freq = np.bincount(destination, minlength=len(conditional)) / float(sample_size)
        assert np.all(destination_freq[conditional == 0.0] == 0.0)
        for observed, expected in zip(destination_freq[conditional > 0.0], conditional[conditional > 0.0]):
            assert abs(float(observed - expected)) <= family_bound
            assert abs(float(observed - expected)) / float(expected) <= 0.10

    spec = _build_custom_topology_spec("mat", 30, CANONICAL_PARAMS)
    W_ver = spec["W_ver"]
    W_esc = spec["W_esc"]
    costs = CANONICAL_PARAMS["tau_v"] + CANONICAL_PARAMS["kappa0"] * (1.0 - spec["W_men"])
    counts = np.full((1, spec["N_J"]), sample_size, dtype=np.int64)
    primary_load, primary_n, _ = _assign_full_pass(
        counts, W_ver, costs, token, 11, ASSIGN_PRIMARY_DEST, 1.0
    )
    escalation_load, trigger_n, escalation_n, _ = _assign_escalation_pass(
        counts, W_esc, costs, token, 11, 1.0
    )
    expected_load = counts.astype(float) @ ((W_ver + W_esc) * costs)
    observed_load = primary_load + escalation_load
    relative_l1 = float(np.abs(observed_load - expected_load).sum() / expected_load.sum())
    assert relative_l1 <= 0.005
    assert np.array_equal(primary_n, counts.sum(axis=1))
    assert np.array_equal(trigger_n, escalation_n)


def test_ticket_id_placebo_bitwise_invariant():
    seeds = np.arange(900, 910)
    a = simulate_batch(seeds, "mod", 30, 0.75, T=80, permute_ticket_ids=False)
    b = simulate_batch(seeds, "mod", 30, 0.75, T=80, permute_ticket_ids=True)
    assert np.all(b["meta_perm_check"] > 0)
    assert float(np.max(np.abs(a["Q_ss"] - b["Q_ss"]))) < 1e-12
    assert float(np.max(np.abs(a["Y_ss"] - b["Y_ss"]))) < 1e-12


def test_isomorphic_agent_permutation_invariant():
    seeds = np.arange(900, 902)
    cases = (("mod", 30), ("mat", 30), ("pool", 60), ("cent", 60))
    for top, N in cases:
        senior_count = N // 5
        permutation = np.arange(senior_count).reshape(-1, 3)[:, [1, 2, 0]].reshape(-1)
        inverse = np.argsort(permutation)
        assert not np.array_equal(permutation, inverse)
        assert np.array_equal(np.sort(permutation), np.arange(senior_count))
        a = simulate_batch(seeds, top, N, 0.40, T=60, agent_perm=None, record_trajectory=True)
        b = simulate_batch(seeds, top, N, 0.40, T=60, agent_perm=permutation, record_trajectory=True)
        assert set(a) == set(b)
        for key in a:
            left = np.asarray(a[key])
            right = np.asarray(b[key])
            assert left.shape == right.shape, f"Failed shape permutation check for {top} N={N} {key}"
            if np.issubdtype(left.dtype, np.integer) or np.issubdtype(left.dtype, np.bool_):
                assert np.array_equal(left, right), f"Failed exact permutation check for {top} N={N} {key}"
                continue
            difference = np.abs(left - right)
            scale = np.maximum(1.0, np.maximum(np.abs(left), np.abs(right)))
            assert float(np.max(difference)) <= 1e-12, f"Failed absolute permutation check for {top} N={N} {key}"
            assert float(np.max(difference / scale)) <= 1e-10, f"Failed relative permutation check for {top} N={N} {key}"


def test_invalid_senior_permutations_are_rejected():
    invalid = (
        np.array([0, 1, 2, 3, 4]),
        np.array([0, 1, 2, 3, 4, 4]),
        np.array([0, 1, 2, 3, 4, 6]),
        np.array([-1, 1, 2, 3, 4, 5]),
    )
    for permutation in invalid:
        try:
            simulate_batch(np.array([900]), "mod", 30, 0.40, T=1, agent_perm=permutation)
        except ValueError:
            continue
        raise AssertionError(f"Invalid senior permutation was accepted: {permutation.tolist()}")


def test_positive_and_negative_controls():
    seeds = np.array([900, 901, 902])
    base0 = simulate_batch(seeds, "mod", 30, 0.0, T=160)
    pos = simulate_batch(seeds, "mod", 30, 0.0, T=160, m_max_zero=True)
    neg_tau = simulate_batch(seeds, "mod", 30, 0.75, T=160, zero_tau=True)
    neg_del = simulate_batch(seeds, "mod", 30, 0.75, T=160, zero_delta=True)
    assert float(base0["Q_ss"].mean() - pos["Q_ss"].mean()) >= 0.20
    assert float(neg_tau["Q_ss"].mean()) >= 0.74
    assert float(neg_del["S_ss"].mean()) >= 0.93


def test_abm_ode_docking_and_burst_null_pooling():
    d_grid = np.round(np.linspace(0.0, 1.0, 26), 4)
    seeds = np.arange(900, 910)
    t0 = time.time()
    abm = simulate_adiabatic_sweep(seeds, "mod", 30, d_grid, T_step=80)
    dt = time.time() - t0
    ode = adiabatic_continuation(build_topology_spec("mod", 30), d_grid)
    rmse_fwd = float(np.sqrt(np.mean((abm["Q_fwd"].mean(axis=0) - ode["Q_fwd"]) ** 2)))
    rmse_bwd = float(np.sqrt(np.mean((abm["Q_bwd"].mean(axis=0) - ode["Q_bwd"]) ** 2)))
    print(f"Docking N=30 (10 seeds, {dt:.2f}s): RMSE_fwd={rmse_fwd:.4f}, RMSE_bwd={rmse_bwd:.4f}, hyst={abm['max_hyst_Q'].mean():.4f}")
    assert rmse_fwd <= 0.04, f"Forward RMSE too high: {rmse_fwd}"
    assert rmse_bwd <= 0.04, f"Backward RMSE too high: {rmse_bwd}"
    assert float(abm["max_hyst_Q"].mean()) >= 0.10

    # Check high-burstiness null pooling advantage (pool beats mod when kappa0=sigma0=p_esc=0)
    d_null = np.round(np.linspace(0.16, 0.56, 21), 4)
    abm_mod_b = simulate_adiabatic_sweep(seeds, "mod", 30, d_null, T_step=60, null_ctrl=True, burstiness=0.55)
    abm_pool_b = simulate_adiabatic_sweep(seeds, "pool", 30, d_null, T_step=60, null_ctrl=True, burstiness=0.55)
    gap = float(abm_pool_b["d_collapse"].mean() - abm_mod_b["d_collapse"].mean())
    print(f"Null high-burst pooling gap (pool - mod): {gap:+.4f} (pool={abm_pool_b['d_collapse'].mean():.3f}, mod={abm_mod_b['d_collapse'].mean():.3f})")
    assert gap >= 0.02, f"Expected Erlang burst-pooling advantage >= 0.02, got {gap}"


def test_critic_extensions_and_endogenous_controller():
    seeds = np.array([900, 901, 902])
    # verify_relief_only and freeze_A run cleanly and produce finite outputs
    vr = simulate_batch(seeds, "mod", 30, 0.75, T=80, verify_relief_only=True)
    fa = simulate_batch(seeds, "mod", 30, 0.75, T=80, freeze_A=True)
    assert np.all(np.isfinite(vr["Q_ss"])) and np.all(np.isfinite(fa["Q_ss"]))

    # selective_hub and role_balanced_hub run cleanly on cent N=30
    sh = simulate_batch(seeds, "cent", 30, 0.40, T=80, selective_hub=True)
    rb = simulate_batch(seeds, "cent", 30, 0.40, T=80, role_balanced_hub=True)
    assert np.all(np.isfinite(sh["Q_ss"])) and np.all(np.isfinite(rb["Q_ss"]))

    # c_router deducts cleanly only for classifier-based arm, not for RT_random_thinning (AUC=0.50)
    g5_0 = simulate_batch(seeds, "mod", 30, 0.75, T=80, q_quota=0.25, g_route=0.52, auc=0.85, c_router=0.0)
    g5_c = simulate_batch(seeds, "mod", 30, 0.75, T=80, q_quota=0.25, g_route=0.52, auc=0.85, c_router=0.05)
    assert abs(float((g5_0["Y_ss"] - g5_c["Y_ss"]).mean()) - 0.05) < 1e-12
    rt_0 = simulate_batch(seeds, "mod", 30, 0.75, T=80, q_quota=0.25, g_route=0.52, auc=0.50, c_router=0.0)
    rt_c = simulate_batch(seeds, "mod", 30, 0.75, T=80, q_quota=0.25, g_route=0.52, auc=0.50, c_router=0.05)
    assert abs(float((rt_0["Y_ss"] - rt_c["Y_ss"]).mean())) < 1e-12

    # blind 5x5 G5 and RT_random_thinning validation tuning work
    bq, bg, bQ, bY = tune_g5_blind_5x5(seeds[:2], "mod", 30, 0.75, 0.85)
    assert 0.10 <= bq <= 0.45 and 0.30 <= bg <= 0.85
    rt_g, rt_Q, rt_Y = tune_arm_on_validation("RT_random_thinning", seeds[:2], "mod", 30, 0.75, 0.50)
    assert 0.24 <= rt_g <= 1.00

    # Endogenous delayed d(t) controller uses categorical assignment with exact conservation
    ec = simulate_endogenous_d_controller(seeds[:2], "mod", 30, T=60, delay_tau=8)
    ec_repeat = simulate_endogenous_d_controller(seeds[:2], "mod", 30, T=60, delay_tau=8)
    assert ec["Q_hist"].shape == (2, 60) and np.all(np.isfinite(ec["Q_hist"]))
    assert np.array_equal(ec["assignment_primary_tickets"], ec["assignment_primary_destinations"])
    assert np.array_equal(ec["assignment_escalation_triggers"], ec["assignment_escalation_destinations"])
    assert np.array_equal(ec["assignment_rework_tickets"], ec["assignment_rework_destinations"])
    assert np.all(ec["assignment_conservation_max_abs"] == 0)
    for key in (
        "assignment_primary_by_senior",
        "assignment_escalation_by_senior",
        "assignment_rework_by_senior",
    ):
        assert np.array_equal(ec[key], ec_repeat[key])

    ec_pool = simulate_endogenous_d_controller(seeds[:2], "pool", 30, T=20, delay_tau=8)
    ec_pool_alt = simulate_endogenous_d_controller(
        seeds[:2], "pool", 30, T=20, delay_tau=8, assignment_stream_id=STREAM_ASSIGN + 1000
    )
    assert any(
        not np.array_equal(ec_pool[key], ec_pool_alt[key])
        for key in ("assignment_primary_by_senior", "assignment_escalation_by_senior")
    )


if __name__ == "__main__":
    test_config_hash_ignores_seed()
    test_assignment_counter_vectors_and_domains()
    test_assignment_stream_consumption_and_exact_conservation()
    test_assignment_probability_family_and_load_convergence()
    test_ticket_id_placebo_bitwise_invariant()
    test_isomorphic_agent_permutation_invariant()
    test_invalid_senior_permutations_are_rejected()
    test_positive_and_negative_controls()
    test_abm_ode_docking_and_burst_null_pooling()
    test_critic_extensions_and_endogenous_controller()
    print("ALL SIM_ABM_ORG SELF-TESTS PASSED!")
