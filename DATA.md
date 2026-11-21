# Data Provenance and Artifact Inventory

## External Empirical Dataset (Not Redistributed)

Secondary econometric robustness checks re-analyze public classroom experiment microdata from Bastani et al. (2025), *PNAS* (DOI: `10.1073/pnas.2422633122`). Raw participant files (`final_data.csv` and `gpt_answers_full.csv`) remain with the original publication archive and are not mirrored in this repository.

To re-run the secondary audit locally after downloading the PNAS replication archive:

```bash
export BASTANI_DATA_DIR=/path/to/main_regressions
python3 experiments/run_bastani_audit.py
```

The repository commits only the derived, aggregated outputs produced by that script:
- `results/classroom_prior_reanalysis.json`
- `results/classroom_prior_regressions.csv`

## Generated Simulation Outputs and Tables

All files under `results/` are deterministic outputs generated from the ODE solver, multi-agent simulator, Latin hypercube sampling diagnostics, and bootstrap precision tools in `src/` and `experiments/`. Every committed CSV and JSON file is indexed with its SHA-256 digest, license, and generating provenance in `DATA_MANIFEST.json` at the repository root.

Key groups inside `results/`:
- `results/confirmatory_effect_summary.json` and `results/confirmatory_effect_mc_audit.json`: The 27 Bonferroni-corrected paired contrasts (`n = 100` confirmatory seeds, 10,000 primary / 100,000 reference resamples).
- `results/model_diagnostics.json`, `results/structural_lhs_500_draws.csv`, `results/topology_capture_sensitivity.csv`, and `results/governance_defect_scrutiny_sensitivity.csv`: Structural sensitivity, partial-capture ROC grids, and alternative scrutiny-decay families.
- `results/figure_data/`: Exact figure panel CSV series (`fig2a_hysteresis.csv` through `fig5c_xcap_dose.csv`) and `figure_data_manifest.json` backing Figures 2 through 5 in the manuscript.
- `results/artifacts/`: Detailed per-experiment JSON logs underlying all reported summary metrics.
- `results/run_register.csv` and `results/supplemental_register.csv`: Seed-level and condition-level execution logs across primary and breadth campaigns.
- `results/sha256_manifest.json` and `results/scientific_artifact_hashes.json`: Repository-wide cryptographic integrity manifests verified by `python3 scripts/payload_manifest.py verify`.
