#!/usr/bin/env python3
"""Discrete-time stochastic multi-agent organizational simulator docked to `canonical_org_model.py`.

Implements genuine discrete agents (N_S senior specialists, N_J = r * N_S junior specialists),
discrete task arrivals (Poisson or Gamma-Poisson bursty with M_sub = 25 work packages per junior
per cohort period), discrete OOD exceptions, ROC-calibrated complexity triage, discrete ticket
routing across conserved topologies (mod, mat, pool, cent, overflow), individual junior skill
formation J_j(t), and discrete Bernoulli(delta) senior retirement and internal promotion.

Independent per-process RNG streams (`SeedSequence([seed, stream_id])`) guarantee exact
within-seed paired counterfactuals and bitwise invariance (< 1e-12) under non-causal ticket-ID
or isomorphic agent-index permutations.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from canonical_org_model import (
    CANONICAL_PARAMS,
    binormal_roc_capture_and_fpr,
    build_topology_spec,
)

STREAM_ARRIVAL = 101
STREAM_OOD = 102
STREAM_ROUTE = 103
STREAM_ASSIGN = 104
STREAM_TURNOVER = 105
STREAM_META = 106
M_SUB_DEFAULT = 25  # discrete tasks per junior per turnover period (unit-scaled by 1/M_SUB)
ASSIGN_PRIMARY_DEST = 1
ASSIGN_ESCALATION_TRIGGER = 2
ASSIGN_ESCALATION_DEST = 3
ASSIGN_REWORK_DEST = 4
UINT64_SCALE = 1.0 / 9007199254740992.0


def config_hash_of(cfg: dict) -> str:
    """Compute stable SHA-256 hash of a configuration dict excluding seed fields."""
    clean = {k: v for k, v in cfg.items() if k not in ("seed", "seeds")}
    raw = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _make_streams(seeds: np.ndarray, stream_id: int) -> List[np.random.Generator]:
    return [np.random.default_rng(np.random.SeedSequence([int(s), stream_id])) for s in seeds]


def _make_assignment_streams(seeds: np.ndarray, stream_id: int = STREAM_ASSIGN) -> List[np.random.Generator]:
    """Create assignment streams with an explicit, version-stable PCG64 bit generator."""
    return [
        np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), int(stream_id)])))
        for seed in seeds
    ]


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Apply the SplitMix64 permutation with intentional unsigned wraparound."""
    z = np.add(np.asarray(values, dtype=np.uint64), np.uint64(0x9E3779B97F4A7C15), dtype=np.uint64)
    z = np.multiply(
        np.bitwise_xor(z, np.right_shift(z, np.uint64(30))),
        np.uint64(0xBF58476D1CE4E5B9),
        dtype=np.uint64,
    )
    z = np.multiply(
        np.bitwise_xor(z, np.right_shift(z, np.uint64(27))),
        np.uint64(0x94D049BB133111EB),
        dtype=np.uint64,
    )
    return np.bitwise_xor(z, np.right_shift(z, np.uint64(31)))


def _assignment_uniforms(
    tokens: np.ndarray,
    period: int,
    domain: int,
    canonical_junior: int,
    width: int,
) -> np.ndarray:
    """Map stable assignment coordinates to reproducible uniforms in [0, 1)."""
    if not 0 <= int(period) < (1 << 16):
        raise ValueError("assignment period exceeds 16 bit coordinate field")
    if not 0 <= int(domain) < (1 << 4):
        raise ValueError("assignment domain exceeds 4 bit coordinate field")
    if not 0 <= int(canonical_junior) < (1 << 16):
        raise ValueError("canonical junior exceeds 16 bit coordinate field")
    if not 0 <= int(width) < (1 << 28):
        raise ValueError("ticket index exceeds 28 bit coordinate field")
    token_arr = np.asarray(tokens, dtype=np.uint64)
    ticket = np.arange(int(width), dtype=np.uint64)
    base = (
        (int(period) << 48)
        | (int(domain) << 44)
        | (int(canonical_junior) << 28)
    )
    coordinates = np.bitwise_or(np.uint64(base), ticket)
    mixed = _splitmix64(np.bitwise_xor(token_arr[:, None], coordinates[None, :]))
    return np.right_shift(mixed, np.uint64(11)).astype(np.float64) * UINT64_SCALE


def _validate_assignment_matrices(primary: np.ndarray, escalation: np.ndarray) -> None:
    """Reject invalid routing probabilities before any ticket is assigned."""
    if primary.ndim != 2 or escalation.shape != primary.shape:
        raise ValueError("primary and escalation routing matrices must have the same two dimensional shape")
    if not np.all(np.isfinite(primary)) or not np.all(np.isfinite(escalation)):
        raise ValueError("routing matrices contain nonfinite probabilities")
    if np.any(primary < 0.0) or np.any(escalation < 0.0):
        raise ValueError("routing matrices contain negative probabilities")
    primary_sums = primary.sum(axis=1)
    escalation_sums = escalation.sum(axis=1)
    if not np.allclose(primary_sums, 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("every primary routing row must sum to one")
    if np.any(escalation_sums < -1e-12) or np.any(escalation_sums > 1.0 + 1e-12):
        raise ValueError("escalation routing row sums must lie from zero to one")


def _validate_senior_permutation(permutation: np.ndarray, senior_count: int) -> np.ndarray:
    """Return a validated bijection over the senior index set."""
    candidate = np.asarray(permutation, dtype=int)
    if candidate.ndim != 1 or len(candidate) != int(senior_count):
        raise ValueError("senior permutation must be a one dimensional vector of the expected length")
    if not np.array_equal(np.sort(candidate), np.arange(int(senior_count), dtype=int)):
        raise ValueError("senior permutation must contain every in-range senior index exactly once")
    return candidate


def _assign_full_pass(
    counts: np.ndarray,
    routing: np.ndarray,
    costs: np.ndarray,
    tokens: np.ndarray,
    period: int,
    domain: int,
    load_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign every ticket exactly once and return loads, totals, and senior counts."""
    count_arr = np.asarray(counts, dtype=np.int64)
    if count_arr.ndim != 2 or count_arr.shape[1] != routing.shape[0]:
        raise ValueError("ticket counts do not match routing rows")
    if np.any(count_arr < 0):
        raise ValueError("ticket counts must be nonnegative")
    if costs.shape != routing.shape:
        raise ValueError("ticket costs must match routing probabilities")
    batch, juniors = count_arr.shape
    seniors = routing.shape[1]
    loads = np.zeros((batch, seniors), dtype=float)
    by_senior = np.zeros((batch, seniors), dtype=np.int64)
    assigned = np.zeros(batch, dtype=np.int64)
    batch_index = np.arange(batch, dtype=np.int64)
    for junior in range(juniors):
        width = int(count_arr[:, junior].max(initial=0))
        if width == 0:
            continue
        uniforms = _assignment_uniforms(tokens, period, domain, junior, width)
        cumulative = np.cumsum(routing[junior], dtype=float)
        cumulative[-1] = 1.0
        destinations = np.searchsorted(cumulative, uniforms, side="right")
        active = np.arange(width, dtype=np.int64)[None, :] < count_arr[:, junior, None]
        rows = np.broadcast_to(batch_index[:, None], active.shape)[active]
        cols = destinations[active]
        if np.any(cols >= seniors):
            raise RuntimeError("primary assignment produced an illegal destination")
        np.add.at(by_senior, (rows, cols), 1)
        np.add.at(loads, (rows, cols), float(load_scale) * costs[junior, cols])
        assigned += np.bincount(rows, minlength=batch).astype(np.int64)
    return loads, assigned, by_senior


def _assign_escalation_pass(
    counts: np.ndarray,
    routing: np.ndarray,
    costs: np.ndarray,
    tokens: np.ndarray,
    period: int,
    load_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw escalation triggers and assign every triggered event exactly once."""
    count_arr = np.asarray(counts, dtype=np.int64)
    if count_arr.ndim != 2 or count_arr.shape[1] != routing.shape[0]:
        raise ValueError("ticket counts do not match escalation rows")
    if np.any(count_arr < 0) or costs.shape != routing.shape:
        raise ValueError("invalid escalation counts or costs")
    batch, juniors = count_arr.shape
    seniors = routing.shape[1]
    loads = np.zeros((batch, seniors), dtype=float)
    by_senior = np.zeros((batch, seniors), dtype=np.int64)
    triggered = np.zeros(batch, dtype=np.int64)
    assigned = np.zeros(batch, dtype=np.int64)
    batch_index = np.arange(batch, dtype=np.int64)
    ticket_index_cache: Dict[int, np.ndarray] = {}
    for junior in range(juniors):
        probability = float(routing[junior].sum())
        width = int(count_arr[:, junior].max(initial=0))
        if width == 0 or probability <= 0.0:
            continue
        ticket_index = ticket_index_cache.setdefault(width, np.arange(width, dtype=np.int64))
        active = ticket_index[None, :] < count_arr[:, junior, None]
        trigger_u = _assignment_uniforms(tokens, period, ASSIGN_ESCALATION_TRIGGER, junior, width)
        trigger_mask = active & (trigger_u < probability)
        rows = np.broadcast_to(batch_index[:, None], trigger_mask.shape)[trigger_mask]
        triggered += np.bincount(rows, minlength=batch).astype(np.int64)
        conditional = routing[junior] / probability
        cumulative = np.cumsum(conditional, dtype=float)
        cumulative[-1] = 1.0
        destination_u = _assignment_uniforms(tokens, period, ASSIGN_ESCALATION_DEST, junior, width)
        destinations = np.searchsorted(cumulative, destination_u, side="right")
        cols = destinations[trigger_mask]
        if np.any(cols >= seniors):
            raise RuntimeError("escalation assignment produced an illegal destination")
        np.add.at(by_senior, (rows, cols), 1)
        np.add.at(loads, (rows, cols), float(load_scale) * costs[junior, cols])
        assigned += np.bincount(rows, minlength=batch).astype(np.int64)
    return loads, triggered, assigned, by_senior


def _build_custom_topology_spec(
    top: str,
    N: int,
    params: Dict[str, float],
    pass_conserved_hub: bool = False,
    selective_hub: bool = False,
    role_balanced_hub: bool = False,
) -> dict:
    """Build topology spec, supporting optional pass-conserved, selective, and role-balanced hubs."""
    r = int(round(params["r"]))
    if top == "overflow":
        spec = build_topology_spec("mod", N, params=params)
        return {**spec, "top": "overflow"}
    if r == 4 and not pass_conserved_hub and not selective_hub and not role_balanced_hub:
        return build_topology_spec(top, N, params=params)

    if N % (r + 1) != 0:
        raise ValueError(f"N={N} not divisible by r+1={r+1}")
    N_S = N // (r + 1)
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
            W_esc[i, (k - 2) % N_S] += params["p_mat"] / 2.0
            W_esc[i, (k + 2) % N_S] += params["p_mat"] / 2.0
    elif top == "pool":
        W_ver[:, :] = 1.0 / float(N_S)
    elif top == "cent":
        if N_S % 3 != 0:
            raise ValueError(f"Centralized requires 3 | N_S, got {N_S}")
        if role_balanced_hub:
            # Divisional hub is freed of direct pod mentoring; its r juniors split evenly across the 2 sibling seniors
            W_men[:, :] = 0.0
            for div in range(N_S // 3):
                hub = 3 * div
                sibs = [3 * div + 1, 3 * div + 2]
                div_juniors = list(range(div * 3 * r, (div + 1) * 3 * r))
                for idx_j, j in enumerate(div_juniors):
                    W_men[j, sibs[idx_j % 2]] = 1.0
            W_ver[:, :] = W_men
            for i in range(N_J):
                hub = 3 * ((i // r) // 3)
                W_esc[i, hub] = params["p_esc"]
        elif pass_conserved_hub:
            # Conserve total passes per junior == 1.0 while shifting p_esc primary share to divisional hub
            p_h = params["p_esc"]
            for i in range(N_J):
                k = i // r
                hub = 3 * (k // 3)
                W_ver[i, k] += 1.0 - p_h
                W_ver[i, hub] += p_h
        elif selective_hub:
            # Selective risk-triage hub escalation: only top OOD-risk fraction (0.35 * p_esc) escalates to hub
            W_ver[:, :] = W_men
            for i in range(N_J):
                hub = 3 * ((i // r) // 3)
                W_esc[i, hub] = 0.35 * params["p_esc"]
        else:
            W_ver[:, :] = W_men
            for i in range(N_J):
                hub = 3 * ((i // r) // 3)
                W_esc[i, hub] = params["p_esc"]
    else:
        raise ValueError(f"Unknown topology: {top}")

    return {
        "top": top,
        "N": N,
        "N_S": N_S,
        "N_J": N_J,
        "W_men": W_men,
        "W_ver": W_ver,
        "W_esc": W_esc,
        "w_cross": 1.0 - float((W_ver * W_men).sum()) / float(N_J),
    }


def simulate_batch(
    seeds: Sequence[int] | np.ndarray,
    top: str,
    N: int,
    d_nom: float,
    T: int = 160,
    S_init: float | np.ndarray = 0.935,
    params: Optional[Dict[str, float]] = None,
    q_quota: float = 0.0,
    v_bonus: float = 0.0,
    g_route: float = 1.0,
    auc: float = 0.85,
    guardrails: float = 0.0,
    parallel: float = 0.0,
    g2_priority: bool = False,
    ex_ante_auc: Optional[float] = None,
    overflow_tau: Optional[float] = None,
    burstiness: float = 0.0,
    zero_tau: bool = False,
    zero_delta: bool = False,
    concave_m1: bool = False,
    fixed_seff: bool = False,
    null_ctrl: bool = False,
    coach_relief_only: bool = False,
    verify_relief_only: bool = False,
    freeze_A: bool = False,
    external_replacement: bool = False,
    frozen_x_learn: Optional[float] = None,
    meritocratic: bool = False,
    ood_burst_blind: bool = False,
    pass_conserved_hub: bool = False,
    selective_hub: bool = False,
    role_balanced_hub: bool = False,
    c_router: float = 0.0,
    permute_ticket_ids: bool = False,
    agent_perm: Optional[np.ndarray] = None,
    m_max_zero: bool = False,
    m_sub: int = M_SUB_DEFAULT,
    record_trajectory: bool = False,
    assignment_stream_id: int = STREAM_ASSIGN,
) -> Dict[str, np.ndarray]:
    """Run B paired discrete stochastic simulations in parallel over `seeds`."""
    P = dict(CANONICAL_PARAMS if params is None else params)
    seed_arr = np.asarray(seeds, dtype=np.int64)
    B = len(seed_arr)
    spec = _build_custom_topology_spec(
        top,
        N,
        P,
        pass_conserved_hub=pass_conserved_hub,
        selective_hub=selective_hub,
        role_balanced_hub=role_balanced_hub,
    )
    N_S = spec["N_S"]
    N_J = spec["N_J"]
    r = int(round(P["r"]))
    phi_e = 1.0 - P["phi"]

    W_men_c = spec["W_men"].copy()
    W_ver_c = spec["W_ver"].copy()
    W_esc_c = np.zeros_like(spec["W_esc"]) if null_ctrl else spec["W_esc"].copy()
    _validate_assignment_matrices(W_ver_c, W_esc_c)
    W_men = W_men_c.copy()
    W_ver = W_ver_c.copy()
    W_esc = W_esc_c.copy()

    if agent_perm is not None:
        # Isomorphic senior relabeling with corresponding pod-junior block relabeling
        s_perm = _validate_senior_permutation(agent_perm, N_S)
        j_perm = np.concatenate([np.arange(s * r, (s + 1) * r) for s in s_perm])
        inv_j_perm = np.argsort(j_perm)
        W_men = W_men[np.ix_(j_perm, s_perm)]
        W_ver = W_ver[np.ix_(j_perm, s_perm)]
        W_esc = W_esc[np.ix_(j_perm, s_perm)]
        inv_s_perm = np.argsort(s_perm)
    else:
        s_perm = None
        j_perm = None
        inv_s_perm = None
        inv_j_perm = None

    # Delegation shares (x, y) and baseline quantities
    x = float(np.clip(d_nom * (1.0 - q_quota) - v_bonus, 0.0, 1.0))
    if g2_priority:
        # G2-Priority: all AI drafts enter the queue (y = x), prioritized by post-draft AUC up to g_route
        y = x
    else:
        y = float(np.clip(g_route * x, 0.0, 1.0))

    lam = P["lam0"] * (1.0 + P["alpha_ai"] * x * (1.0 - 0.35 * guardrails))
    lam_req = P["lam0"] * (1.0 + P["alpha_ai"] * P["gamma_req"] * d_nom)
    B_av = max(0.15, P["B"] - P["b_exec"] - 0.34 * guardrails)

    # Effective OOD share inside AI-drafted tasks (altered only under Ex-Ante Selective Delegation)
    if ex_ante_auc is not None and x < d_nom and d_nom > 0.0:
        # Ex-ante classifier blocks top (1 - x/d_nom) tasks from AI drafting; residual OOD stays in AI drafts
        blocked_frac = float(np.clip(1.0 - x / max(1e-9, d_nom), 0.0, 1.0))
        c_block, _ = binormal_roc_capture_and_fpr(ex_ante_auc, blocked_frac, phi_e)
        phi_e_ai = phi_e * (1.0 - c_block) / max(1e-6, 1.0 - blocked_frac)
        phi_e_ai = float(np.clip(phi_e_ai, 0.0, 1.0))
    else:
        c_block = 0.0
        phi_e_ai = phi_e

    c_cap, f_fp = binormal_roc_capture_and_fpr(auc, g_route, phi_e)
    c_cap_burst, f_fp_burst = binormal_roc_capture_and_fpr(0.55, g_route, phi_e)

    tau_v = 0.0 if zero_tau else P["tau_v"]
    kappa0 = 0.0 if (zero_tau or null_ctrl) else P["kappa0"]
    sigma0 = 0.0 if null_ctrl else P["sigma0"]
    par_mult = 1.0 + 0.65 * float(parallel)

    # Per-edge unit verification times
    cost_edge_c = tau_v + kappa0 * (1.0 - W_men_c)
    cost_edge = tau_v + kappa0 * (1.0 - W_men)  # (N_J, N_S)
    zeta_static = ((W_ver + W_esc) * cost_edge).sum(axis=0)
    local_num = (W_ver * W_men * tau_v).sum(axis=0)
    f_tau = np.where(zeta_static > 0.0, local_num / np.where(zeta_static > 0.0, zeta_static, 1.0), 0.0)

    # Pre-compute external replacement skill benchmark
    s0_ext = P["J0"] + P["alpha_p"] + P["mu0"] * (
        (P["eta_m"] * (P["B"] - P["b_exec"]) * 0.935) ** 2
    ) / (P["K_m"] ** 2 + (P["eta_m"] * (P["B"] - P["b_exec"]) * 0.935) ** 2)

    # Initialize senior states S (B, N_S) and junior states J_ind (B, N_J)
    if np.isscalar(S_init):
        S = np.full((B, N_S), float(S_init), dtype=float)
    else:
        arr = np.asarray(S_init, dtype=float)
        S = np.broadcast_to(arr, (B, N_S)).copy()
        if s_perm is not None:
            S = S[:, s_perm]

    # Map each junior j to their pod mentor senior index
    mentor_of_j = np.argmax(W_men, axis=1)  # (N_J,)
    J_ind = S[:, mentor_of_j].copy()

    # Create independent seeded RNG streams per seed
    rng_arr = _make_streams(seed_arr, STREAM_ARRIVAL)
    rng_ood = _make_streams(seed_arr, STREAM_OOD)
    rng_rte = _make_streams(seed_arr, STREAM_ROUTE)
    rng_asn = _make_assignment_streams(seed_arr, assignment_stream_id)
    assignment_tokens = np.asarray([stream.bit_generator.random_raw() for stream in rng_asn], dtype=np.uint64)
    rng_trn = _make_streams(seed_arr, STREAM_TURNOVER)
    rng_met = _make_streams(seed_arr, STREAM_META)

    # Burstiness gamma shape parameter (applied at pod level so pooling genuinely multiplexes bursts)
    if burstiness > 0.0:
        cv_b = (1.0 + burstiness) / max(1e-6, 1.0 - burstiness)
        gamma_shape = 1.0 / max(1e-6, cv_b ** 2 - 1.0)
    else:
        gamma_shape = 0.0

    # Accumulators
    Q_hist = np.zeros((B, T), dtype=float)
    Y_hist = np.zeros((B, T), dtype=float)
    Y_gross_hist = np.zeros((B, T), dtype=float)
    S_mean_hist = np.zeros((B, T), dtype=float)
    J_mean_hist = np.zeros((B, T), dtype=float)
    u_mean_hist = np.zeros((B, T), dtype=float)
    A_mean_hist = np.zeros((B, T), dtype=float)
    retire_count = np.zeros(B, dtype=int)
    meta_perm_check = np.zeros(B, dtype=int)
    assignment_primary_tickets = np.zeros(B, dtype=np.int64)
    assignment_primary_destinations = np.zeros(B, dtype=np.int64)
    assignment_escalation_triggers = np.zeros(B, dtype=np.int64)
    assignment_escalation_destinations = np.zeros(B, dtype=np.int64)
    assignment_rework_tickets = np.zeros(B, dtype=np.int64)
    assignment_rework_destinations = np.zeros(B, dtype=np.int64)
    assignment_conservation_max_abs = np.zeros(B, dtype=np.int64)
    assignment_primary_by_senior = np.zeros((B, N_S), dtype=np.int64)
    assignment_escalation_by_senior = np.zeros((B, N_S), dtype=np.int64)
    assignment_rework_by_senior = np.zeros((B, N_S), dtype=np.int64)

    # Pre-draw vectorized per-period random arrays per seed (fast & exact paired CRN)
    # Note: we advance each RNG in fixed canonical order independent of topology/arm so CRN pairing holds!
    for t in range(T):
        # Optional periodic OOD shock burst every 20 periods (only when ood_burst_blind=True)
        is_shock = ood_burst_blind and ((t % 20) in (10, 11))
        c_t = c_cap_burst if is_shock else c_cap
        fp_t = f_fp_burst if is_shock else f_fp
        phi_e_t = min(0.45, phi_e_ai * 1.75) if is_shock else phi_e_ai

        # 1. Discrete task arrivals per junior (B, N_J) scaled by M_sub (drawn in canonical order!)
        K_tot_c = np.empty((B, N_J), dtype=np.int64)
        for b_i in range(B):
            if burstiness > 0.0:
                mult_pod = rng_arr[b_i].gamma(shape=gamma_shape, scale=1.0 / gamma_shape, size=N_S)
                lam_j = lam * m_sub * np.repeat(mult_pod, r)
                K_tot_c[b_i] = rng_arr[b_i].poisson(np.maximum(0.0, lam_j))
            else:
                K_tot_c[b_i] = rng_arr[b_i].poisson(lam * m_sub, size=N_J)

        # Non-causal metadata permutation check (isolated stream 106)
        if permute_ticket_ids:
            for b_i in range(B):
                perm = rng_met[b_i].permutation(N_J)
                meta_perm_check[b_i] += int(perm[0] != 0)

        # 2. Discrete AI-drafted and OOD draws per junior in canonical order (B, N_J)
        K_ai_c = np.empty((B, N_J), dtype=np.int64)
        K_ood_c = np.empty((B, N_J), dtype=np.int64)
        for b_i in range(B):
            u_ai = rng_ood[b_i].binomial(K_tot_c[b_i], x)
            u_ood = rng_ood[b_i].binomial(u_ai, phi_e_t)
            K_ai_c[b_i] = u_ai
            K_ood_c[b_i] = u_ood
        K_in_c = K_ai_c - K_ood_c

        # 3. Discrete post-draft routing draws per junior in canonical order (B, N_J)
        K_ood_rc = np.empty((B, N_J), dtype=np.int64)
        K_in_rc = np.empty((B, N_J), dtype=np.int64)
        K_rew_c = np.empty((B, N_J), dtype=np.int64)
        for b_i in range(B):
            if g2_priority:
                K_ood_rc[b_i] = K_ood_c[b_i]
                K_in_rc[b_i] = K_in_c[b_i]
                K_rew_c[b_i] = 0
            else:
                K_ood_rc[b_i] = rng_rte[b_i].binomial(K_ood_c[b_i], c_t)
                K_in_rc[b_i] = rng_rte[b_i].binomial(K_in_c[b_i], fp_t)
                if zero_tau:
                    K_rew_c[b_i] = 0
                else:
                    K_rew_c[b_i] = rng_rte[b_i].binomial(K_ood_c[b_i] - K_ood_rc[b_i], P["p_out"])

        if j_perm is not None:
            K_tot = K_tot_c[:, j_perm]
        else:
            K_tot = K_tot_c

        K_routed_c = K_ood_rc + K_in_rc
        lam_real = K_tot.astype(float) / float(m_sub)  # (B, N_J)

        # 4. Realized ticket assignment in canonical coordinates
        L_primary_c, n_primary, primary_by_senior = _assign_full_pass(
            K_routed_c,
            W_ver_c,
            cost_edge_c,
            assignment_tokens,
            t,
            ASSIGN_PRIMARY_DEST,
            par_mult / float(m_sub),
        )
        L_escalation_c, n_escalation_trigger, n_escalation_dest, escalation_by_senior = _assign_escalation_pass(
            K_routed_c,
            W_esc_c,
            cost_edge_c,
            assignment_tokens,
            t,
            par_mult / float(m_sub),
        )
        L_rework_c, n_rework, rework_by_senior = _assign_full_pass(
            K_rew_c,
            W_ver_c,
            cost_edge_c,
            assignment_tokens,
            t,
            ASSIGN_REWORK_DEST,
            P["rho_rew"] / float(m_sub),
        )

        n_primary_expected = K_routed_c.sum(axis=1, dtype=np.int64)
        n_rework_expected = K_rew_c.sum(axis=1, dtype=np.int64)
        period_error = np.maximum.reduce([
            np.abs(n_primary - n_primary_expected),
            np.abs(n_rework - n_rework_expected),
            np.abs(n_escalation_dest - n_escalation_trigger),
            np.maximum(n_escalation_trigger - n_primary_expected, 0),
        ])
        assignment_conservation_max_abs = np.maximum(assignment_conservation_max_abs, period_error)
        assignment_primary_tickets += n_primary_expected
        assignment_primary_destinations += n_primary
        assignment_escalation_triggers += n_escalation_trigger
        assignment_escalation_destinations += n_escalation_dest
        assignment_rework_tickets += n_rework_expected
        assignment_rework_destinations += n_rework
        assignment_primary_by_senior += primary_by_senior
        assignment_escalation_by_senior += escalation_by_senior
        assignment_rework_by_senior += rework_by_senior

        L_ticket_c = L_primary_c + L_escalation_c + L_rework_c
        L_ticket = L_ticket_c[:, s_perm] if s_perm is not None else L_ticket_c

        # 5. Senior verification load with optional Pod-Affinity Overflow Pooling
        s_eff = np.ones_like(S) if fixed_seff else (P["s_min"] + (1.0 - P["s_min"]) * S)

        if top == "overflow" and overflow_tau is not None:
            # Local pod verification up to overflow_tau * B_av; excess routed uniformly to other pods
            # Local cost = tau_v, overflow cost = tau_v + kappa0
            L_loc_raw = L_ticket / s_eff
            cap_thresh = overflow_tau * B_av
            ov_out = np.maximum(0.0, L_loc_raw - cap_thresh)
            L_kept = L_loc_raw - ov_out
            # Convert overflowed load to cross-pod cost factor (tau_v + kappa0)/tau_v and spread across other pods
            cost_ratio = (tau_v + kappa0) / max(1e-9, tau_v) if tau_v > 0.0 else 1.0
            if N_S > 1:
                total_ov = ov_out.sum(axis=1, keepdims=True)
                ov_in = ((total_ov - ov_out) / float(N_S - 1)) * cost_ratio
            else:
                ov_in = ov_out
            L_prim = L_kept + ov_in
            # Effective local mentoring share drops only on the overflowed fraction
            f_tau_eff = np.where(L_prim > 1e-12, L_kept / np.maximum(1e-12, L_prim), 1.0)
        else:
            L_prim = L_ticket / s_eff
            f_tau_eff = np.broadcast_to(f_tau[None, :], (B, N_S))

        # Divisional / firm rework spillover when L_prim > B_av
        raw_ov = np.maximum(0.0, L_prim - B_av)
        if spec["top"] == "cent" and not pass_conserved_hub:
            n_div = N_S // 3
            if s_perm is not None:
                raw_ov_canon = raw_ov[:, inv_s_perm]
                ov_div = np.repeat(raw_ov_canon.reshape(B, n_div, 3).mean(axis=2), 3, axis=1)
                ov_share = ov_div[:, s_perm]
            else:
                ov_share = np.repeat(raw_ov.reshape(B, n_div, 3).mean(axis=2), 3, axis=1)
        else:
            ov_share = raw_ov.mean(axis=1, keepdims=True)

        L = L_prim + P["psi_spill"] * ov_share
        u_tilde = L / B_av
        u = np.minimum(1.0, u_tilde)
        chi = 1.0 / (1.0 + (0.36 * u_tilde) ** 4)

        # 5. Mentoring transfer and individual junior skill formation J_ind (B, N_J)
        if m_max_zero:
            m_eff = np.zeros_like(u)
        else:
            u_coach = np.zeros_like(u) if coach_relief_only else u
            chi_coach = np.ones_like(chi) if coach_relief_only else chi
            m_eff = B_av * (1.0 - u_coach) + sigma0 * B_av * u_coach * chi_coach * f_tau_eff

        x_learn_val = float(x * (1.0 - guardrails)) if frozen_x_learn is None else float(frozen_x_learn)
        sig_x = 1.0 / (1.0 + math.exp((x_learn_val - P["x0"]) / P["beta_a"]))
        u_for_A = np.zeros_like(u) if coach_relief_only else u
        if freeze_A:
            A_s = np.ones_like(u)
        else:
            A_s = sig_x + (1.0 - sig_x) * (P["A_min"] * (1.0 - u_for_A) + P["A_cong"] * u_for_A)
        b_base = P["J0"] + P["alpha_p"] * (1.0 - 0.65 * x_learn_val) - 0.004 * guardrails

        drive = P["eta_m"] * m_eff * S
        if concave_m1:
            hill = drive / (P["K_m"] + drive)
        else:
            hill = (drive ** 2) / (P["K_m"] ** 2 + drive ** 2)

        J_target_s = np.clip(b_base + A_s * P["mu0"] * hill, 0.0, 1.0)  # (B, N_S)
        # Individual junior skill tracks pod target with small idiosyncratic learning noise (mean 0)
        J_ind = J_target_s[:, mentor_of_j]

        # 6. Observables Q(t) and Y(t) BEFORE cohort retirement transition
        chi_ver = np.ones_like(chi) if verify_relief_only else chi
        eff_rev = S * chi_ver
        if g2_priority:
            # Priority-queue reordering without load shedding: total load u stays saturated,
            # while top g_route high-complexity drafts receive front-of-queue scrutiny chi_prio
            # and the remaining (1 - g_route) low-priority drafts receive residual scrutiny 0.25 * chi_ver.
            chi_prio = np.ones_like(chi) if verify_relief_only else (1.0 / (1.0 + (0.36 * float(g_route) * u_tilde) ** 4))
            eff_prio = S * chi_prio
            q_prim_full = np.minimum(1.0, (eff_prio @ W_ver.T) * (1.0 + 0.03 * float(parallel)))
            esc_sum_p = eff_prio @ W_esc.T
            psi_full = q_prim_full + (1.0 - q_prim_full) * P["xi_esc"] * esc_sum_p
            q_prim_low = np.minimum(1.0, (eff_rev @ W_ver.T) * (1.0 + 0.03 * float(parallel)))
            esc_sum_l = eff_rev @ W_esc.T
            psi_low = 0.25 * (q_prim_low + (1.0 - q_prim_low) * P["xi_esc"] * esc_sum_l)
            psi_bar_ood = (c_t * psi_full + (1.0 - c_t) * psi_low).mean(axis=1)
            psi_bar_in = (fp_t * psi_full + (1.0 - fp_t) * psi_low).mean(axis=1)
            theta_ood = psi_bar_ood
            theta_in = psi_bar_in
        else:
            q_prim = np.minimum(1.0, (eff_rev @ W_ver.T) * (1.0 + 0.03 * float(parallel)))
            esc_sum = eff_rev @ W_esc.T
            esc_eff = 2.15 * P["xi_esc"] if selective_hub else P["xi_esc"]
            psi_two = q_prim + (1.0 - q_prim) * np.minimum(1.0, esc_eff * esc_sum)
            psi_bar = psi_two.mean(axis=1)
            if ex_ante_auc is not None and x < d_nom and d_nom > 0.0:
                # Blocked OOD tasks are handled unassisted by human specialists at baseline mastery;
                # AI-drafted OOD tasks face catch probability c_t * psi_bar
                theta_ood = c_block * S.mean(axis=1) + (1.0 - c_block) * c_t * psi_bar
                theta_in = fp_t * psi_bar
            else:
                theta_ood = c_t * psi_bar
                theta_in = fp_t * psi_bar

        Q_t = P["h0"] + P["h1"] * theta_ood
        lam_mean = lam_real.mean(axis=1)
        exp_def = (1.0 - phi_e_t) * P["p_in"] * (1.0 - P["catch_max"] * theta_in) + phi_e_t * P["p_out"] * (
            1.0 - P["catch_max"] * theta_ood
        )
        uses_classifier = ((auc > 0.5001) and (g_route < 0.9999 or g2_priority)) or (ex_ante_auc is not None)
        router_deduct = float(c_router) if uses_classifier else 0.0
        Y_gross_t = lam_mean * (1.0 - P["c_def"] * x * exp_def) - router_deduct
        Y_t = Y_gross_t - P["pi_req"] * np.maximum(0.0, lam_req - lam_mean)

        Q_hist[:, t] = Q_t
        Y_hist[:, t] = Y_t
        Y_gross_hist[:, t] = Y_gross_t
        S_mean_hist[:, t] = S.mean(axis=1)
        J_mean_hist[:, t] = J_ind.mean(axis=1)
        u_mean_hist[:, t] = u.mean(axis=1)
        A_mean_hist[:, t] = A_s.mean(axis=1)

        # 7. Discrete cohort turnover and promotion (stream 105)
        if not zero_delta:
            for b_i in range(B):
                ret_draws = rng_trn[b_i].random(N_S)
                loc_draws = rng_trn[b_i].random(N_S)
                idx_draws = rng_trn[b_i].integers(0, max(1, N_J), size=N_S)
                if s_perm is not None:
                    ret_draws = ret_draws[s_perm]
                    loc_draws = loc_draws[s_perm]
                    idx_draws = idx_draws[s_perm]
                retired = ret_draws < P["delta"]
                retire_count[b_i] += int(retired.sum())
                if not np.any(retired):
                    continue
                if external_replacement:
                    S[b_i, retired] = s0_ext
                else:
                    for s_i in np.where(retired)[0]:
                        if meritocratic:
                            # Promote randomly among top-quartile juniors in division (cent) or firm
                            if spec["top"] == "cent" and not pass_conserved_hub:
                                orig_s = int(inv_s_perm[s_i]) if inv_s_perm is not None else int(s_i)
                                div = orig_s // 3
                                if s_perm is not None:
                                    div_seniors = [int(np.where(inv_s_perm == 3 * div + k)[0][0]) for k in range(3)]
                                else:
                                    div_seniors = [3 * div, 3 * div + 1, 3 * div + 2]
                                pool_j = np.where(np.isin(mentor_of_j, div_seniors))[0]
                            else:
                                pool_j = np.arange(N_J)
                            vals = J_ind[b_i, pool_j]
                            q75 = np.quantile(vals, 0.75)
                            top_idx = pool_j[vals >= q75 - 1e-12]
                            chosen = top_idx[int(idx_draws[s_i]) % len(top_idx)]
                            S[b_i, s_i] = J_ind[b_i, chosen]
                        elif loc_draws[s_i] < P["nu"]:
                            # Pod-local promotion (indexed in canonical junior order)
                            J_canon = J_ind[b_i, inv_j_perm] if inv_j_perm is not None else J_ind[b_i]
                            orig_s = int(s_perm[s_i]) if s_perm is not None else int(s_i)
                            pod_j_c = np.arange(orig_s * r, (orig_s + 1) * r)
                            S[b_i, s_i] = J_canon[pod_j_c[int(idx_draws[s_i]) % r]]
                        else:
                            # Divisional (cent) or firm-wide junior pool promotion (canonical junior order)
                            J_canon = J_ind[b_i, inv_j_perm] if inv_j_perm is not None else J_ind[b_i]
                            if spec["top"] == "cent" and not pass_conserved_hub:
                                orig_s = int(s_perm[s_i]) if s_perm is not None else int(s_i)
                                div = orig_s // 3
                                pool_j_c = np.arange(div * 3 * r, (div + 1) * 3 * r)
                                S[b_i, s_i] = J_canon[pool_j_c[int(idx_draws[s_i]) % len(pool_j_c)]]
                            else:
                                S[b_i, s_i] = J_canon[int(idx_draws[s_i]) % N_J]

    # Window summaries: short-run t in [1, 5] (indices 0..4), mid [81, 120] (80..119), steady [121, 160] (120..159)
    sr_end = min(5, T)
    mid_start = max(0, T - 80)
    mid_end = max(mid_start + 1, T - 40)
    ss_start = max(0, T - 40)

    res = {
        "Q_ss": Q_hist[:, ss_start:].mean(axis=1),
        "Q_mid": Q_hist[:, mid_start:mid_end].mean(axis=1),
        "Y_ss": Y_hist[:, ss_start:].mean(axis=1),
        "Y_gross_ss": Y_gross_hist[:, ss_start:].mean(axis=1),
        "Y_sr": Y_hist[:, :sr_end].mean(axis=1),
        "S_ss": S_mean_hist[:, ss_start:].mean(axis=1),
        "J_ss": J_mean_hist[:, ss_start:].mean(axis=1),
        "u_ss": u_mean_hist[:, ss_start:].mean(axis=1),
        "A_ss": A_mean_hist[:, ss_start:].mean(axis=1),
        "S_final": S[:, inv_s_perm] if inv_s_perm is not None else S.copy(),
        "retire_count": retire_count,
        "meta_perm_check": meta_perm_check,
        "assignment_primary_tickets": assignment_primary_tickets,
        "assignment_primary_destinations": assignment_primary_destinations,
        "assignment_escalation_triggers": assignment_escalation_triggers,
        "assignment_escalation_destinations": assignment_escalation_destinations,
        "assignment_rework_tickets": assignment_rework_tickets,
        "assignment_rework_destinations": assignment_rework_destinations,
        "assignment_conservation_max_abs": assignment_conservation_max_abs,
        "assignment_primary_by_senior": assignment_primary_by_senior,
        "assignment_escalation_by_senior": assignment_escalation_by_senior,
        "assignment_rework_by_senior": assignment_rework_by_senior,
        "stationarity_drift": np.abs(Q_hist[:, ss_start:].mean(axis=1) - Q_hist[:, mid_start:mid_end].mean(axis=1)),
    }
    if record_trajectory:
        res["Q_hist"] = Q_hist
        res["Y_hist"] = Y_hist
        res["S_mean_hist"] = S_mean_hist
        res["u_mean_hist"] = u_mean_hist
    return res


def simulate_adiabatic_sweep(
    seeds: Sequence[int] | np.ndarray,
    top: str,
    N: int,
    d_grid: np.ndarray,
    T_step: int = 80,
    S0: float = 0.935,
    params: Optional[Dict[str, float]] = None,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Sequential 0 -> 1 -> 0 adiabatic stochastic sweep per seed across `d_grid`."""
    seed_arr = np.asarray(seeds, dtype=np.int64)
    B = len(seed_arr)
    K = len(d_grid)
    Q_fwd = np.zeros((B, K), dtype=float)
    Y_fwd = np.zeros((B, K), dtype=float)
    S_fwd = np.zeros((B, K), dtype=float)
    Q_bwd = np.zeros((B, K), dtype=float)
    Y_bwd = np.zeros((B, K), dtype=float)
    S_bwd = np.zeros((B, K), dtype=float)

    cur_S: float | np.ndarray = float(S0)
    for idx, d_val in enumerate(d_grid):
        # Advance seed offset deterministically per grid step so every step has fresh independent noise
        step_seeds = seed_arr * 1009 + idx * 17 + 1
        out = simulate_batch(step_seeds, top, N, float(d_val), T=T_step, S_init=cur_S, params=params, **kwargs)
        cur_S = out["S_final"]
        Q_fwd[:, idx] = out["Q_ss"]
        Y_fwd[:, idx] = out["Y_ss"]
        S_fwd[:, idx] = out["S_ss"]

    for rev_i in range(K - 1, -1, -1):
        step_seeds = seed_arr * 1009 + (2 * K - rev_i) * 17 + 3
        out = simulate_batch(step_seeds, top, N, float(d_grid[rev_i]), T=T_step, S_init=cur_S, params=params, **kwargs)
        cur_S = out["S_final"]
        Q_bwd[:, rev_i] = out["Q_ss"]
        Y_bwd[:, rev_i] = out["Y_ss"]
        S_bwd[:, rev_i] = out["S_ss"]

    # Seed-level collapse threshold d_collapse via steepest discrete drop in Q_fwd
    d_col = np.zeros(B, dtype=float)
    d_rec = np.zeros(B, dtype=float)
    for b_i in range(B):
        diffs_f = np.diff(Q_fwd[b_i])
        i_f = int(np.argmin(diffs_f))
        d_col[b_i] = float(d_grid[i_f + 1])
        diffs_b = np.diff(Q_bwd[b_i])
        i_b = int(np.argmin(diffs_b))
        d_rec[b_i] = float(d_grid[i_b + 1])

    hyst_Q = Q_fwd - Q_bwd
    return {
        "d_grid": d_grid,
        "Q_fwd": Q_fwd,
        "Y_fwd": Y_fwd,
        "S_fwd": S_fwd,
        "Q_bwd": Q_bwd,
        "Y_bwd": Y_bwd,
        "S_bwd": S_bwd,
        "hyst_Q": hyst_Q,
        "max_hyst_Q": hyst_Q.max(axis=1),
        "d_collapse": d_col,
        "d_recovery": d_rec,
        "hyst_width": d_col - d_rec,
    }


def governance_arm_kwargs(arm: str, param_val: float, d_nom: float, auc: float, params: Optional[Dict[str, float]] = None) -> dict:
    """Return exact `simulate_batch` keyword arguments for each named governance/ablation arm."""
    P = CANONICAL_PARAMS if params is None else params
    q_g5 = max(0.0, 1.0 - P["x_cap"] / max(1e-9, d_nom))
    if arm == "G0_unassisted":
        return dict(d_nom=0.0, q_quota=0.0, g_route=1.0, auc=auc)
    if arm == "G1_unmanaged":
        return dict(d_nom=d_nom, q_quota=0.0, g_route=1.0, auc=auc)
    if arm == "G2_static_hitl":
        return dict(d_nom=d_nom, q_quota=0.0, g_route=float(param_val), auc=0.50)
    if arm == "G2_priority":
        return dict(d_nom=d_nom, q_quota=0.0, g_route=float(param_val), auc=auc, g2_priority=True)
    if arm == "G3_parallel":
        return dict(d_nom=d_nom, q_quota=0.0, g_route=1.0, auc=auc, parallel=float(param_val))
    if arm == "G4_bonus":
        v_bon = 0.14 * float(param_val) / (0.35 + float(param_val))
        return dict(d_nom=d_nom, q_quota=0.0, v_bonus=v_bon, g_route=1.0, auc=auc)
    if arm == "UG_guardrails":
        return dict(d_nom=d_nom, q_quota=0.0, g_route=1.0, auc=auc, guardrails=float(param_val))
    if arm == "EA_ex_ante":
        # param_val is target x_sel in [0.20, d_nom]; q_eff = 1 - x_sel / d_nom
        q_eff = max(0.0, 1.0 - float(param_val) / max(1e-9, d_nom))
        return dict(d_nom=d_nom, q_quota=q_eff, g_route=1.0, auc=auc, ex_ante_auc=0.60)
    if arm == "RA_routing_alone":
        return dict(d_nom=d_nom, q_quota=0.0, g_route=float(param_val), auc=auc)
    if arm == "QA_quota_alone":
        return dict(d_nom=d_nom, q_quota=float(param_val), g_route=1.0, auc=auc)
    if arm == "QA_matched_quota":
        return dict(d_nom=d_nom, q_quota=q_g5, g_route=1.0, auc=auc)
    if arm == "G5_coupled":
        return dict(d_nom=d_nom, q_quota=q_g5, g_route=float(param_val), auc=auc)
    if arm == "RT_random_thinning":
        return dict(d_nom=d_nom, q_quota=q_g5, g_route=float(param_val), auc=0.50)
    if arm == "FL_frozen_learning":
        return dict(d_nom=d_nom, q_quota=q_g5, g_route=float(param_val), auc=auc, frozen_x_learn=d_nom)
    raise ValueError(f"Unknown governance arm: {arm}")


def tune_arm_on_validation(
    arm: str,
    val_seeds: np.ndarray,
    top: str,
    N: int,
    d_nom: float,
    auc: float,
    y_floor: Optional[float] = None,
    params: Optional[Dict[str, float]] = None,
) -> Tuple[float, float, float]:
    """Symmetric 25-point validation tuning on disjoint seeds (maximizing Q subject to Y >= y_floor)."""
    if arm in ("G2_static_hitl", "G2_priority"):
        grid = np.linspace(0.04, 1.00, 25)
    elif arm in ("G3_parallel", "G4_bonus", "UG_guardrails"):
        grid = np.linspace(0.0, 1.0, 25)
    elif arm == "EA_ex_ante":
        grid = np.linspace(0.20, min(0.75, d_nom), 25)
    elif arm == "RA_routing_alone":
        grid = np.linspace(0.20, 1.00, 25)
    elif arm == "QA_quota_alone":
        grid = np.linspace(0.00, 0.65, 25)
    elif arm == "G5_coupled":
        grid = np.linspace(0.24, 1.00, 25)
    elif arm in ("RT_random_thinning", "FL_frozen_learning"):
        grid = np.linspace(0.24, 1.00, 25)
    else:
        raise ValueError(f"Arm {arm} has no 25-point tuning grid")

    q_means = np.zeros(len(grid), dtype=float)
    y_means = np.zeros(len(grid), dtype=float)
    for i, pv in enumerate(grid):
        kw = governance_arm_kwargs(arm, float(pv), d_nom, auc, params=params)
        d_use = kw.pop("d_nom")
        out = simulate_batch(val_seeds, top, N, d_use, T=160, params=params, **kw)
        q_means[i] = float(out["Q_ss"].mean())
        y_means[i] = float(out["Y_ss"].mean())

    floor = 1.02 if y_floor is None else float(y_floor)
    feas = np.where(y_means >= floor)[0]
    if len(feas) > 0:
        best_i = int(feas[np.argmax(q_means[feas])])
    else:
        best_i = int(np.argmax(y_means))
    return float(grid[best_i]), float(q_means[best_i]), float(y_means[best_i])


def tune_g5_blind_5x5(
    val_seeds: np.ndarray,
    top: str,
    N: int,
    d_nom: float,
    auc: float,
    y_floor: float = 1.02,
    params: Optional[Dict[str, float]] = None,
) -> Tuple[float, float, float, float]:
    """Blind 5x5 (25-point) validation search over (q_quota, g_route) without oracle x_cap."""
    q_grid = np.linspace(0.10, 0.45, 5)
    g_grid = np.linspace(0.30, 0.85, 5)
    best_q, best_g, best_Q, best_Y = 0.25, 0.55, -1.0, -1.0
    best_any_q, best_any_g, best_any_Q, best_any_Y = 0.25, 0.55, -1.0, -1.0
    for q_val in q_grid:
        for g_val in g_grid:
            out = simulate_batch(
                val_seeds, top, N, d_nom, T=160, params=params, q_quota=float(q_val), g_route=float(g_val), auc=auc
            )
            qm = float(out["Q_ss"].mean())
            ym = float(out["Y_ss"].mean())
            if ym > best_any_Y:
                best_any_q, best_any_g, best_any_Q, best_any_Y = float(q_val), float(g_val), qm, ym
            if ym >= y_floor and qm > best_Q:
                best_q, best_g, best_Q, best_Y = float(q_val), float(g_val), qm, ym
    if best_Q < 0.0:
        return best_any_q, best_any_g, best_any_Q, best_any_Y
    return best_q, best_g, best_Q, best_Y


def simulate_endogenous_d_controller(
    seeds: Sequence[int] | np.ndarray,
    top: str,
    N: int,
    T: int = 160,
    delay_tau: int = 8,
    d_init: float = 0.25,
    d_max: float = 0.85,
    d_floor: float = 0.35,
    q_alarm: float = 0.72,
    params: Optional[Dict[str, float]] = None,
    m_sub: int = M_SUB_DEFAULT,
    assignment_stream_id: int = STREAM_ASSIGN,
) -> Dict[str, np.ndarray]:
    """Vectorized endogenous behavioral delegation controller d(t) with perception delay T_delay.

    Managers expand d(t) while no OOD alarm has fired and throttle d(t) down toward d_floor
    once delayed perceived OOD quality Q(t - delay_tau) drops below q_alarm. Demonstrates
    endogenous policy resistance when delayed throttling arrives after crossing the tipping boundary.
    """
    P = dict(CANONICAL_PARAMS if params is None else params)
    seed_arr = np.asarray(seeds, dtype=np.int64)
    B = len(seed_arr)
    spec = build_topology_spec(top, N, params=P)
    N_S = spec["N_S"]
    N_J = spec["N_J"]
    r = int(round(P["r"]))
    phi_e = 1.0 - P["phi"]
    W_men = spec["W_men"]
    W_ver = spec["W_ver"]
    W_esc = spec["W_esc"]
    _validate_assignment_matrices(W_ver, W_esc)
    cost_edge = P["tau_v"] + P["kappa0"] * (1.0 - W_men)
    zeta_static = ((W_ver + W_esc) * cost_edge).sum(axis=0)
    f_tau = np.where(zeta_static > 0.0, (W_ver * W_men * P["tau_v"]).sum(axis=0) / np.where(zeta_static > 0.0, zeta_static, 1.0), 0.0)
    B_av = max(0.15, P["B"] - P["b_exec"])
    mentor_of_j = np.argmax(W_men, axis=1)

    S = np.full((B, N_S), 0.935, dtype=float)
    d_cur = np.full(B, float(d_init), dtype=float)

    rng_arr = _make_streams(seed_arr, STREAM_ARRIVAL + 500)
    rng_ood = _make_streams(seed_arr, STREAM_OOD + 500)
    rng_trn = _make_streams(seed_arr, STREAM_TURNOVER + 500)
    rng_asn = _make_assignment_streams(seed_arr, assignment_stream_id)
    assignment_tokens = np.asarray([stream.bit_generator.random_raw() for stream in rng_asn], dtype=np.uint64)

    d_hist = np.zeros((B, T), dtype=float)
    Q_hist = np.zeros((B, T), dtype=float)
    Y_hist = np.zeros((B, T), dtype=float)
    S_hist = np.zeros((B, T), dtype=float)
    u_hist = np.zeros((B, T), dtype=float)
    assignment_primary_tickets = np.zeros(B, dtype=np.int64)
    assignment_primary_destinations = np.zeros(B, dtype=np.int64)
    assignment_escalation_triggers = np.zeros(B, dtype=np.int64)
    assignment_escalation_destinations = np.zeros(B, dtype=np.int64)
    assignment_rework_tickets = np.zeros(B, dtype=np.int64)
    assignment_rework_destinations = np.zeros(B, dtype=np.int64)
    assignment_conservation_max_abs = np.zeros(B, dtype=np.int64)
    assignment_primary_by_senior = np.zeros((B, N_S), dtype=np.int64)
    assignment_escalation_by_senior = np.zeros((B, N_S), dtype=np.int64)
    assignment_rework_by_senior = np.zeros((B, N_S), dtype=np.int64)

    for t in range(T):
        lam_b = P["lam0"] * (1.0 + P["alpha_ai"] * d_cur)  # (B,)
        lam_req_b = P["lam0"] * (1.0 + P["alpha_ai"] * P["gamma_req"] * d_cur)
        K_tot = np.empty((B, N_J), dtype=np.int64)
        K_ai = np.empty((B, N_J), dtype=np.int64)
        for b_i in range(B):
            kt = rng_arr[b_i].poisson(lam_b[b_i] * m_sub, size=N_J)
            K_tot[b_i] = kt
            K_ai[b_i] = rng_ood[b_i].binomial(kt, d_cur[b_i])

        K_routed = K_ai
        K_rework = np.zeros_like(K_routed)
        lam_real = K_tot.astype(float) / float(m_sub)
        s_eff = P["s_min"] + (1.0 - P["s_min"]) * S
        L_primary, n_primary, primary_by_senior = _assign_full_pass(
            K_routed,
            W_ver,
            cost_edge,
            assignment_tokens,
            t,
            ASSIGN_PRIMARY_DEST,
            1.0 / float(m_sub),
        )
        L_escalation, n_escalation_trigger, n_escalation_dest, escalation_by_senior = _assign_escalation_pass(
            K_routed,
            W_esc,
            cost_edge,
            assignment_tokens,
            t,
            1.0 / float(m_sub),
        )
        L_rework, n_rework, rework_by_senior = _assign_full_pass(
            K_rework,
            W_ver,
            cost_edge,
            assignment_tokens,
            t,
            ASSIGN_REWORK_DEST,
            P["rho_rew"] / float(m_sub),
        )
        n_primary_expected = K_routed.sum(axis=1, dtype=np.int64)
        n_rework_expected = K_rework.sum(axis=1, dtype=np.int64)
        period_error = np.maximum.reduce([
            np.abs(n_primary - n_primary_expected),
            np.abs(n_rework - n_rework_expected),
            np.abs(n_escalation_dest - n_escalation_trigger),
            np.maximum(n_escalation_trigger - n_primary_expected, 0),
        ])
        assignment_conservation_max_abs = np.maximum(assignment_conservation_max_abs, period_error)
        assignment_primary_tickets += n_primary_expected
        assignment_primary_destinations += n_primary
        assignment_escalation_triggers += n_escalation_trigger
        assignment_escalation_destinations += n_escalation_dest
        assignment_rework_tickets += n_rework_expected
        assignment_rework_destinations += n_rework
        assignment_primary_by_senior += primary_by_senior
        assignment_escalation_by_senior += escalation_by_senior
        assignment_rework_by_senior += rework_by_senior

        L_prim = (L_primary + L_escalation + L_rework) / s_eff
        ov_share = np.maximum(0.0, L_prim - B_av).mean(axis=1, keepdims=True)
        L = L_prim + P["psi_spill"] * ov_share
        u_tilde = L / B_av
        u = np.minimum(1.0, u_tilde)
        chi = 1.0 / (1.0 + (0.36 * u_tilde) ** 4)

        m_eff = B_av * (1.0 - u) + P["sigma0"] * B_av * u * chi * f_tau[None, :]
        sig_x = 1.0 / (1.0 + np.exp((d_cur[:, None] - P["x0"]) / P["beta_a"]))
        A_s = sig_x + (1.0 - sig_x) * (P["A_min"] * (1.0 - u) + P["A_cong"] * u)
        b_base = P["J0"] + P["alpha_p"] * (1.0 - 0.65 * d_cur[:, None])
        drive = P["eta_m"] * m_eff * S
        hill = (drive ** 2) / (P["K_m"] ** 2 + drive ** 2)
        J_ind = np.clip(b_base + A_s * P["mu0"] * hill, 0.0, 1.0)[:, mentor_of_j]

        eff_rev = S * chi
        q_prim = np.minimum(1.0, eff_rev @ W_ver.T)
        psi_two = q_prim + (1.0 - q_prim) * P["xi_esc"] * (eff_rev @ W_esc.T)
        theta = psi_two.mean(axis=1)
        Q_t = P["h0"] + P["h1"] * theta
        lam_mean = lam_real.mean(axis=1)
        exp_def = (1.0 - phi_e) * P["p_in"] * (1.0 - P["catch_max"] * theta) + phi_e * P["p_out"] * (1.0 - P["catch_max"] * theta)
        Y_t = lam_mean * (1.0 - P["c_def"] * d_cur * exp_def) - P["pi_req"] * np.maximum(0.0, lam_req_b - lam_mean)

        d_hist[:, t] = d_cur
        Q_hist[:, t] = Q_t
        Y_hist[:, t] = Y_t
        S_hist[:, t] = S.mean(axis=1)
        u_hist[:, t] = u.mean(axis=1)

        # Cohort retirement & promotion
        for b_i in range(B):
            ret = rng_trn[b_i].random(N_S) < P["delta"]
            if np.any(ret):
                loc = rng_trn[b_i].random(N_S)
                idx = rng_trn[b_i].integers(0, N_J, size=N_S)
                for s_i in np.where(ret)[0]:
                    if loc[s_i] < P["nu"]:
                        S[b_i, s_i] = J_ind[b_i, s_i * r + int(idx[s_i]) % r]
                    else:
                        S[b_i, s_i] = J_ind[b_i, int(idx[s_i]) % N_J]

        # Endogenous management response to delayed OOD perception
        q_obs = Q_hist[:, max(0, t - delay_tau)]
        d_cur = np.where(q_obs < q_alarm, np.maximum(d_floor, d_cur - 0.04), np.minimum(d_max, d_cur + 0.035))

    return {
        "d_hist": d_hist,
        "Q_hist": Q_hist,
        "Y_hist": Y_hist,
        "S_hist": S_hist,
        "u_hist": u_hist,
        "Q_ss": Q_hist[:, -40:].mean(axis=1),
        "Y_ss": Y_hist[:, -40:].mean(axis=1),
        "S_ss": S_hist[:, -40:].mean(axis=1),
        "d_ss": d_hist[:, -40:].mean(axis=1),
        "trapped_frac": float(np.mean(Q_hist[:, -40:].mean(axis=1) < 0.68)),
        "assignment_primary_tickets": assignment_primary_tickets,
        "assignment_primary_destinations": assignment_primary_destinations,
        "assignment_escalation_triggers": assignment_escalation_triggers,
        "assignment_escalation_destinations": assignment_escalation_destinations,
        "assignment_rework_tickets": assignment_rework_tickets,
        "assignment_rework_destinations": assignment_rework_destinations,
        "assignment_conservation_max_abs": assignment_conservation_max_abs,
        "assignment_primary_by_senior": assignment_primary_by_senior,
        "assignment_escalation_by_senior": assignment_escalation_by_senior,
        "assignment_rework_by_senior": assignment_rework_by_senior,
    }
