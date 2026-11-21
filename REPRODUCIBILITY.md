# Reproducibility Guide

## Environment

Tested on Python 3.10+ (reference freeze recorded in `results/environment_freeze.json`). Pinned dependencies are listed in `requirements.txt`:

- `numpy==2.4.6`
- `scipy==1.17.1`
- `pandas==2.3.3`
- `sympy==1.14.0`
- `statsmodels==0.14.6`
- `pytest==9.0.2`

Install into a fresh virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Fast End-to-End Verification (Recommended First Check)

Run all analytical proofs, boundary checks, `pytest` unit/regression suites, diagnostics, sensitivity verification, 27-contrast precision reconciliation, figure data regeneration, and SHA-256 verification without mutating tracked files:

```bash
bash scripts/run_quick_verify.sh
```

Expected terminal line on completion: `ALL_REPRODUCIBILITY_CHECKS_PASSED`.

## Individual Verification Commands

- Symbolic proofs (Propositions 1 to 3 and Theorem 1): `python3 scripts/verify_proofs_sympy.py`
- Rational and numerical boundary invariance probes: `python3 scripts/probe_math_boundaries.py`
- Unit and docking test suite: `pytest -q -p no:cacheprovider tests/`
- Model diagnostics and structural sensitivity tables:
  - `python3 experiments/run_model_diagnostics.py`
  - `python3 scripts/verify_sensitivity_tables.py`
- 27-contrast precision family and Appendix table check:
  - `python3 experiments/generate_confirmatory_effect_summary.py results/artifacts results/confirmatory_effect_summary.json results/confirmatory_effect_mc_audit.json`
  - `python3 scripts/verify_precision_table.py`
- Figure series regeneration: `python3 experiments/generate_figure_data.py results results/figure_data`

## Full Simulation Campaign Regeneration

Seed discipline separates disjoint validation tuning seeds (`900..929`, `n = 30`) from locked confirmatory evaluation seeds (`1000..1099`, `n = 100`). To recompute the simulation campaign cleanly into a separate output root:

```bash
export RELEASE_OUTPUT_ROOT=/path/to/recompute_dir
python3 experiments/run_official_experiments.py
python3 experiments/run_official_breadth_supplement.py
python3 experiments/run_confirmatory_extensions.py
```

Running the full campaign regenerates the agent-based Monte Carlo sweeps and Latin hypercube samples; the quick verification script cross-checks all derived summaries, symbolic invariants, and deterministic generators within a few seconds.
