# Attention-Rationed Competence Traps

Code, verification suite, and audited simulation outputs for our System Dynamics Review manuscript on capability erosion, verification overload, communication topology, and governance in AI-assisted expert workflows.

## What Is Included

- `src/canonical_org_model.py`: Deterministic continuous-state ODE and organizational topology matrices (`mod`, `mat`, `pool`, `cent`, and overflow variants).
- `src/sim_abm_org.py`: Discrete stochastic multi-agent organizational simulator (junior learning `J`, senior skill `S`, task congestion, ROC-based triage, and discrete staff turnover).
- `src/effect_resolver_core.py`: Seed-paired bootstrap and Bonferroni interval calculator for the 27-contrast simulation precision family.
- `experiments/`: Full simulation campaign scripts, structural diagnostics, figure CSV generators, and external classroom reanalysis script.
- `scripts/`: Symbolic SymPy proof checks, boundedness probes, sensitivity checks, 27-contrast precision table verification, and SHA-256 payload manifest tools.
- `configs/precision_table_spec.json`: Exact 27-contrast specification corresponding to Appendix Table C2 in the manuscript.
- `tests/`: Unit, regression, and invariance tests covering the discrete simulator and precision reconciliation.
- `results/`: Audited simulation CSV/JSON tables, figure panel series (`results/figure_data/`), detailed run artifacts (`results/artifacts/`), and derived classroom regression summaries.

## Quick Start

Install dependencies in a clean Python 3.10+ environment and run the full fast verification pipeline:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_quick_verify.sh
```

`bash scripts/run_quick_verify.sh` checks symbolic proofs (SymPy), exact rational boundary probes, `pytest` unit tests, model diagnostics, sensitivity table summaries, byte-exact regeneration of the 27 confirmatory Bonferroni contrasts and figure panel CSVs, and SHA-256 integrity across the full repository payload.

## Repository Map

- `DATA.md`: Inventory of committed simulation datasets, hashes in `DATA_MANIFEST.json`, and provenance instructions for third-party classroom microdata (Bastani et al., 2025, PNAS; not redistributed here).
- `REPRODUCIBILITY.md`: Pinned dependency versions, seed discipline (`900..929` validation tuning, `1000..1099` confirmatory evaluation), and instructions to regenerate all results from scratch.
- `LICENSE`: MIT License (anonymized for double-blind peer review).
