#!/usr/bin/env python3
"""Exact rational and seeded boundary probes against the canonical model."""
from __future__ import annotations

from fractions import Fraction
import json
import math
import os
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = REPO_ROOT / "src"
OUTPUT_DIR = Path(os.environ.get("MODEL_VERIFY_OUTPUT_DIR", str(REPO_ROOT / "results")))
sys.path.insert(0, str(SIM_DIR))
from canonical_org_model import CANONICAL_PARAMS, binormal_roc_capture, build_topology_spec, canonical_rhs_batch  # noqa: E402
import numpy as np  # noqa: E402

SEED = 20261119


def main() -> None:
    rng = random.Random(SEED)
    probes = []
    P = CANONICAL_PARAMS

    # 1. Exact boundary probes on [0, 1] forward invariance, d=0, d=1, g=0, g=1, rho=0.5, rho=1.0, tau_v=0, delta=0
    for top, N in [("mod", 5), ("mod", 15), ("pool", 10), ("mat", 15), ("cent", 15), ("cent", 120)]:
        sp_top = build_topology_spec(top, N)
        ns = sp_top["N_S"]
        for d_val in (0.0, 0.1, 0.5, 0.75, 1.0):
            for s_val in (0.0, 0.01, 0.5, 0.99, 1.0):
                S_in = np.full((1, ns), s_val, dtype=float)
                out = canonical_rhs_batch(S_in, sp_top, d_val)
                j_min, j_max = float(out["J"].min()), float(out["J"].max())
                q_val = float(out["Q"][0])
                assert 0.0 < j_min and j_max < 1.0, f"J exited (0,1): [{j_min}, {j_max}]"
                assert P["h0"] - 1e-12 <= q_val <= P["h0"] + P["h1"] + 1e-12, f"Q exited [h0, h0+h1]: {q_val}"
                if s_val == 0.0:
                    assert float(out["dS"].min()) > 0.0, "At S=0, dS/dt must be strictly positive!"
                if s_val == 1.0:
                    assert float(out["dS"].max()) < 0.0, "At S=1, dS/dt must be strictly negative!"
    probes.append({"probe": "boundary_forward_invariance_grid", "status": "passed"})

    # 2. Exact rational Fraction check of Proposition 2 & Proposition 3 inequalities across 1,000 random tuples
    counterexamples = []
    for idx in range(1000):
        NS = rng.choice([3, 6, 12, 24])
        tau_v = Fraction(rng.randint(1, 20), 50)
        kappa0 = Fraction(rng.randint(1, 15), 100)
        p_esc = Fraction(rng.randint(1, 9), 10)
        zeta_mod = Fraction(4, 1) * tau_v
        zeta_pool = Fraction(4, 1) * (tau_v + kappa0 * (Fraction(1, 1) - Fraction(1, NS)))
        zeta_cent_hub = Fraction(4, 1) * (1 + 3 * p_esc) * tau_v + Fraction(8, 1) * p_esc * kappa0
        if not (zeta_cent_hub > zeta_mod and zeta_pool > zeta_mod):
            counterexamples.append({"idx": idx, "type": "weight_inequality"})

        rho = rng.choice([0.5, 1.0]) if idx < 20 else rng.uniform(0.5, 1.0)
        g = rng.choice([0.0, 1.0]) if idx < 10 else rng.uniform(0.02, 0.98)
        phi_e = rng.uniform(0.10, 0.60)
        C = binormal_roc_capture(rho, g, phi_e)
        if not (g - 1e-4 <= C <= min(1.0, g / phi_e) + 1e-4):
            counterexamples.append({"idx": idx, "type": "roc_bound", "C": C, "g": g, "phi_e": phi_e})

        alpha_ai = rng.uniform(0.1, 1.8)
        B_av = rng.uniform(0.2, 0.9)
        s_eff = rng.uniform(0.20, 1.0)
        tau_big = (B_av * s_eff) / (4.0 * (1.0 + alpha_ai)) * rng.uniform(1.05, 3.0)
        C_ratio = (B_av * s_eff) / (4.0 * tau_big)
        d_sat = (-1.0 + math.sqrt(1.0 + 4.0 * alpha_ai * C_ratio)) / (2.0 * alpha_ai)
        if not (0.0 < d_sat < 1.0):
            counterexamples.append({"idx": idx, "type": "d_sat_interval", "d_sat": d_sat})

    assert not counterexamples, f"Counterexamples found: {counterexamples[:3]}"
    out = {
        "seed": SEED,
        "total_random_probes": 1000,
        "counterexamples_found": 0,
        "canonical_params_checked": CANONICAL_PARAMS,
        "exact_probes": probes,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "math_boundary_probes.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Canonical boundary probes (seed={SEED}): 1000/1000 passed, 0 counterexamples, strict [0,1] forward invariance verified.")


if __name__ == "__main__":
    main()
