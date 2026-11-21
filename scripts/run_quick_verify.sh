#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERIFY_TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/arc_verify.XXXXXX")"
trap 'rm -rf "$VERIFY_TMP_ROOT"' EXIT

MODEL_VERIFY_OUTPUT_DIR="$VERIFY_TMP_ROOT" python3 scripts/verify_proofs_sympy.py
MODEL_VERIFY_OUTPUT_DIR="$VERIFY_TMP_ROOT" python3 scripts/probe_math_boundaries.py

pytest -q -p no:cacheprovider tests/

MODEL_VERIFY_OUTPUT_DIR="$VERIFY_TMP_ROOT" python3 experiments/run_model_diagnostics.py > /dev/null
MODEL_VERIFY_OUTPUT_DIR="$VERIFY_TMP_ROOT" python3 scripts/verify_sensitivity_tables.py

python3 experiments/generate_confirmatory_effect_summary.py   results/artifacts   "$VERIFY_TMP_ROOT/confirmatory_effect_summary.json"   "$VERIFY_TMP_ROOT/confirmatory_effect_mc_audit.json"
cmp "$VERIFY_TMP_ROOT/confirmatory_effect_summary.json" results/confirmatory_effect_summary.json
cmp "$VERIFY_TMP_ROOT/confirmatory_effect_mc_audit.json" results/confirmatory_effect_mc_audit.json

python3 scripts/verify_precision_table.py   "$VERIFY_TMP_ROOT/confirmatory_effect_summary.json"   configs/precision_table_spec.json

python3 experiments/generate_figure_data.py results "$VERIFY_TMP_ROOT/figure_data"
for expected in results/figure_data/*.csv; do
  cmp "$VERIFY_TMP_ROOT/figure_data/$(basename "$expected")" "$expected"
done
cmp "$VERIFY_TMP_ROOT/figure_data/figure_data_manifest.json" results/figure_data/figure_data_manifest.json

python3 scripts/payload_manifest.py verify
echo "ALL_REPRODUCIBILITY_CHECKS_PASSED"
