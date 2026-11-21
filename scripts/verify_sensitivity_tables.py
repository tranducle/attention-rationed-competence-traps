"""Verify row-level sensitivity artifacts against their published summaries."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_RESULTS = ROOT / "results"
OUTPUT_DIR = os.environ.get("MODEL_VERIFY_OUTPUT_DIR")
GENERATED_RESULTS = Path(OUTPUT_DIR).resolve() if OUTPUT_DIR else ARCHIVE_RESULTS


def read_rows(name: str, generated: bool = False) -> list[dict[str, str]]:
    path = (GENERATED_RESULTS if generated else ARCHIVE_RESULTS) / name
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_columns(rows: list[dict[str, str]], required: set[str], label: str) -> None:
    observed = set(rows[0]) if rows else set()
    missing = required - observed
    if missing:
        raise SystemExit(f"{label}: missing columns {sorted(missing)}")


def fraction(rows: list[dict[str, str]], column: str, predicate) -> float:
    return round(sum(predicate(float(row[column])) for row in rows) / len(rows), 4)


def compare(label: str, observed: dict[str, float], expected: dict[str, float]) -> None:
    if observed != expected:
        raise SystemExit(f"{label}: summary mismatch, observed={observed}, expected={expected}")


def require_generated_matches_archive(names: list[str]) -> dict[str, str]:
    if GENERATED_RESULTS == ARCHIVE_RESULTS:
        return {name: "in_place" for name in names}
    matches = {}
    for name in names:
        generated_path = GENERATED_RESULTS / name
        archive_path = ARCHIVE_RESULTS / name
        if not generated_path.exists() or not archive_path.exists():
            raise SystemExit(f"generated archive comparison: missing {name}")
        generated_digest = hashlib.sha256(generated_path.read_bytes()).hexdigest()
        archive_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        if generated_digest != archive_digest:
            raise SystemExit(
                f"generated archive comparison: {name} differs, generated={generated_digest}, archive={archive_digest}"
            )
        matches[name] = generated_digest
    return matches


def main() -> None:
    global_rows = read_rows("global_lhs_500_draws.csv")
    structural_rows = read_rows("structural_lhs_500_draws.csv", generated=True)
    topology_capture_rows = read_rows("topology_capture_sensitivity.csv", generated=True)
    defect_scrutiny_rows = read_rows("governance_defect_scrutiny_sensitivity.csv", generated=True)
    phase_rows = read_rows("topology_kappa_sigma_contour.csv")

    if len(global_rows) != 500:
        raise SystemExit(f"global sensitivity: expected 500 rows, found {len(global_rows)}")
    if len(structural_rows) != 500:
        raise SystemExit(f"structural sensitivity: expected 500 rows, found {len(structural_rows)}")
    if len(topology_capture_rows) != 48:
        raise SystemExit(f"topology capture sensitivity: expected 48 rows, found {len(topology_capture_rows)}")
    if len(defect_scrutiny_rows) != 12:
        raise SystemExit(f"governance defect and scrutiny sensitivity: expected 12 rows, found {len(defect_scrutiny_rows)}")
    if len(phase_rows) != 121:
        raise SystemExit(f"topology friction map: expected 121 rows, found {len(phase_rows)}")

    require_columns(
        global_rows,
        {
            "c1_hyst_Q",
            "c2_mod_minus_max_rival",
            "c3_g5_minus_best_single",
            "c3_g5_minus_rt",
        },
        "global sensitivity",
    )
    require_columns(
        structural_rows,
        {
            "draw_id",
            "s_min",
            "psi_spill",
            "x0",
            "beta_a",
            "A_cong",
            "rho_rew",
            "c_def",
            "mod_max_hyst_q",
            "mod_minus_pool_d_col",
            "g5_minus_best_ra_q",
        },
        "structural sensitivity",
    )
    require_columns(
        topology_capture_rows,
        {
            "scrutiny_family",
            "capture_g",
            "p_out",
            "mod_max_hyst_q",
            "mod_d_col",
            "pool_d_col",
            "mod_minus_pool_d_col",
        },
        "topology capture sensitivity",
    )
    require_columns(
        defect_scrutiny_rows,
        {
            "scrutiny_family",
            "p_out",
            "g5_minus_best_ra_q",
            "g5_y",
            "g5_minus_best_ra_y",
        },
        "governance defect and scrutiny sensitivity",
    )
    require_columns(
        phase_rows,
        {"kappa0", "sigma0", "d_col_mod", "d_col_pool", "gap_mod_minus_pool"},
        "topology friction map",
    )

    global_summary_path = ARCHIVE_RESULTS / "artifacts" / "global_lhs_500.json"
    global_expected = json.loads(global_summary_path.read_text(encoding="utf-8"))["survival_fractions"]
    global_observed = {
        "C1_hyst_ge_005": fraction(global_rows, "c1_hyst_Q", lambda value: value >= 0.05),
        "C1_hyst_ge_010": fraction(global_rows, "c1_hyst_Q", lambda value: value >= 0.10),
        "C2_mod_beats_all_gt_0": fraction(global_rows, "c2_mod_minus_max_rival", lambda value: value > 0.0),
        "C2_mod_gap_ge_008": fraction(global_rows, "c2_mod_minus_max_rival", lambda value: value >= 0.08),
        "C3_g5_beats_singles_gt_0": fraction(global_rows, "c3_g5_minus_best_single", lambda value: value > 0.0),
        "C3_g5_gain_ge_010": fraction(global_rows, "c3_g5_minus_best_single", lambda value: value >= 0.10),
        "C3_g5_beats_rt_gt_0": fraction(global_rows, "c3_g5_minus_rt", lambda value: value > 0.0),
    }
    compare("global sensitivity", global_observed, global_expected)

    diagnostics = json.loads((GENERATED_RESULTS / "model_diagnostics.json").read_text(encoding="utf-8"))
    structural_expected = diagnostics["structural_sensitivity"]["structural_7param_lhs_500"]
    structural_observed = {
        "n_draws": len(structural_rows),
        "C1_surv_ge_010": fraction(structural_rows, "mod_max_hyst_q", lambda value: value >= 0.10),
        "C1_surv_ge_005": fraction(structural_rows, "mod_max_hyst_q", lambda value: value >= 0.05),
        "C2_mod_gt_pool": fraction(structural_rows, "mod_minus_pool_d_col", lambda value: value > 0.0),
        "C2_surv_ge_008": fraction(structural_rows, "mod_minus_pool_d_col", lambda value: value >= 0.08),
        "C3_gt_RA": fraction(structural_rows, "g5_minus_best_ra_q", lambda value: value > 0.0),
        "C3_surv_ge_010": fraction(structural_rows, "g5_minus_best_ra_q", lambda value: value >= 0.10),
    }
    structural_target = {key: structural_expected[key] for key in structural_observed}
    compare("structural sensitivity", structural_observed, structural_target)

    topology_summary = diagnostics["topology_capture_sensitivity"]
    expected_families = {"quartic", "overload_exponential"}
    expected_p_out = {0.20, 0.40, 0.60, 0.80, 0.94, 1.00}
    expected_capture = {0.35, 0.60, 0.85, 1.00}
    observed_topology_families = {row["scrutiny_family"] for row in topology_capture_rows}
    if observed_topology_families != expected_families:
        raise SystemExit(f"topology capture sensitivity: families {sorted(observed_topology_families)}")
    observed_capture = {round(float(row["capture_g"]), 2) for row in topology_capture_rows}
    if observed_capture != expected_capture:
        raise SystemExit(f"topology capture sensitivity: capture grid mismatch {sorted(observed_capture)}")
    for family in sorted(expected_families):
        discriminating_c1 = False
        discriminating_c2 = False
        for capture_g in sorted(expected_capture):
            rows = [
                row for row in topology_capture_rows
                if row["scrutiny_family"] == family and round(float(row["capture_g"]), 2) == capture_g
            ]
            observed_p_out = {round(float(row["p_out"]), 2) for row in rows}
            if observed_p_out != expected_p_out:
                raise SystemExit(f"{family} capture {capture_g:.2f}: p_out grid mismatch {sorted(observed_p_out)}")
            c1 = [float(row["mod_max_hyst_q"]) for row in rows]
            c2 = [float(row["mod_minus_pool_d_col"]) for row in rows]
            c1_span = max(c1) - min(c1)
            c2_span = max(c2) - min(c2)
            observed = {
                "capture_g": capture_g,
                "n_p_out_values": len(rows),
                "p_out_range": [min(observed_p_out), max(observed_p_out)],
                "C1_all_ge_010": all(value >= 0.10 for value in c1),
                "C1_min_max_hyst_Q": round(min(c1), 4),
                "C1_max_max_hyst_Q": round(max(c1), 4),
                "C1_span_over_p_out": round(c1_span, 4),
                "C1_discriminated_by_p_out": c1_span > 1e-8,
                "C2_all_mod_gt_pool": all(value > 0.0 for value in c2),
                "C2_min_gap": round(min(c2), 4),
                "C2_max_gap": round(max(c2), 4),
                "C2_span_over_p_out": round(c2_span, 4),
                "C2_discriminated_by_p_out": c2_span > 1e-8,
            }
            compare(
                f"topology capture sensitivity {family} {capture_g:.2f}",
                observed,
                topology_summary["by_family_and_capture"][family][f"capture_g_{capture_g:.2f}"],
            )
            if capture_g < 1.0:
                discriminating_c1 = discriminating_c1 or c1_span > 1e-8
                discriminating_c2 = discriminating_c2 or c2_span > 1e-8
        if not discriminating_c1 or not discriminating_c2:
            raise SystemExit(f"{family}: partial-capture grid does not discriminate both C1 and C2")

    defect_summary = diagnostics["governance_defect_scrutiny_sensitivity"]
    observed_families = {row["scrutiny_family"] for row in defect_scrutiny_rows}
    if observed_families != expected_families:
        raise SystemExit(f"governance defect and scrutiny sensitivity: families {sorted(observed_families)}")
    for family in sorted(expected_families):
        rows = [row for row in defect_scrutiny_rows if row["scrutiny_family"] == family]
        observed_p_out = {round(float(row["p_out"]), 2) for row in rows}
        if observed_p_out != expected_p_out:
            raise SystemExit(f"{family}: p_out grid mismatch {sorted(observed_p_out)}")
        c3 = [float(row["g5_minus_best_ra_q"]) for row in rows]
        g5_y = [float(row["g5_y"]) for row in rows]
        y_gap = [float(row["g5_minus_best_ra_y"]) for row in rows]
        observed = {
            "n_p_out_values": len(rows),
            "p_out_range": [min(observed_p_out), max(observed_p_out)],
            "C3_all_g5_gt_RA": all(value > 0.0 for value in c3),
            "C3_min_gap": round(min(c3), 4),
            "G5_min_Y": round(min(g5_y), 4),
            "G5_min_Y_minus_RA": round(min(y_gap), 4),
        }
        compare(f"governance defect and scrutiny sensitivity {family}", observed, defect_summary["by_family"][family])

    kappa_values = {row["kappa0"] for row in phase_rows}
    sigma_values = {row["sigma0"] for row in phase_rows}
    if len(kappa_values) != 11 or len(sigma_values) != 11:
        raise SystemExit("topology friction map: expected an 11 by 11 grid")

    generated_matches_archive = require_generated_matches_archive([
        "model_diagnostics.json",
        "structural_lhs_500_draws.csv",
        "topology_capture_sensitivity.csv",
        "governance_defect_scrutiny_sensitivity.csv",
    ])

    print(json.dumps({
        "global_rows": len(global_rows),
        "structural_rows": len(structural_rows),
        "topology_capture_rows": len(topology_capture_rows),
        "defect_scrutiny_rows": len(defect_scrutiny_rows),
        "phase_map_rows": len(phase_rows),
        "global_summary_match": True,
        "structural_summary_match": True,
        "topology_capture_summary_match": True,
        "defect_scrutiny_summary_match": True,
        "generated_matches_archive": generated_matches_archive,
        "phase_map_shape": "11 by 11",
    }, sort_keys=True))
    print("SENSITIVITY_TABLES_VERIFIED")


if __name__ == "__main__":
    main()
