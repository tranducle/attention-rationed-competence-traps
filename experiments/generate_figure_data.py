#!/usr/bin/env python3
"""Regenerate all manuscript figure CSV files from audited release results."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
for _sub in ("src", "experiments", "scripts"):
    _p = str(_REPO_ROOT / _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from canonical_org_model import adiabatic_continuation, build_topology_spec


FIGURE_FILES = (
    "fig2a_hysteresis.csv",
    "fig2b_bbw.csv",
    "fig2c_endogenous.csv",
    "fig3a_poisson_scaling.csv",
    "fig3b_bursty_scaling.csv",
    "fig3c_reversal.csv",
    "fig4a_k4_forest.csv",
    "fig4b_k9_forest.csv",
    "fig4c_k12_forest.csv",
    "fig5a_pareto.csv",
    "fig5b_auc_dose.csv",
    "fig5c_xcap_dose.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rounded(value: Any, digits: int) -> str:
    number = round(float(value), digits)
    if number == 0:
        number = 0.0
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite figure value: {value}")
    return str(number)


def require_count(name: str, rows: list[Any], expected: int) -> None:
    if len(rows) != expected:
        raise ValueError(f"{name} has {len(rows)} rows, expected {expected}")


def generate_fig2(results: Path, output: Path) -> None:
    source_fields, source_rows = read_csv(results / "modular_N30_hysteresis_curves.csv")
    require_count("e3 hysteresis source", source_rows, 51)
    expected_fields = [
        "d",
        "ode_Q_fwd",
        "ode_Q_bwd",
        "abm_Q_fwd_mean",
        "abm_Q_fwd_lo",
        "abm_Q_fwd_hi",
        "abm_Q_bwd_mean",
        "abm_Q_bwd_lo",
        "abm_Q_bwd_hi",
    ]
    if source_fields != expected_fields:
        raise ValueError("Unexpected E3 hysteresis schema")
    d_grid = np.asarray([float(row["d"]) for row in source_rows], dtype=float)
    zero_tau = adiabatic_continuation(
        build_topology_spec("mod", 30),
        d_grid,
        zero_tau=True,
    )["Q_fwd"]
    fig2a_rows = []
    for row, q_value in zip(source_rows, zero_tau, strict=True):
        fig2a_rows.append({**row, "ode_Q_zero_tau": rounded(q_value, 8)})
    write_csv(
        output / "fig2a_hysteresis.csv",
        expected_fields + ["ode_Q_zero_tau"],
        fig2a_rows,
    )

    bbw_fields, bbw_rows = read_csv(results / "abm_bbw_trajectory_100seeds.csv")
    if len(bbw_rows) < 60:
        raise ValueError("E3 trajectory has fewer than 60 periods")
    write_csv(output / "fig2b_bbw.csv", bbw_fields, bbw_rows[:60])

    _, controller_rows = read_csv(results / "endogenous_d_controller.csv")
    delays = (0, 4, 8, 12)
    by_delay_time: dict[tuple[int, int], str] = {}
    for row in controller_rows:
        key = (int(row["delay_tau"]), int(row["t"]))
        by_delay_time[key] = row["Q_mean"]
    times = sorted({time for delay, time in by_delay_time if delay == 0})
    require_count("endogenous controller periods", times, 160)
    fig2c_rows = []
    for time in times:
        row: dict[str, Any] = {"t": time}
        for delay in delays:
            key = (delay, time)
            if key not in by_delay_time:
                raise ValueError(f"Missing endogenous controller cell: {key}")
            row[f"Q_d{delay}"] = by_delay_time[key]
        fig2c_rows.append(row)
    write_csv(
        output / "fig2c_endogenous.csv",
        ["t", "Q_d0", "Q_d4", "Q_d8", "Q_d12"],
        fig2c_rows,
    )


def generate_fig3(results: Path, output: Path) -> None:
    _, source_rows = read_csv(results / "topology_scaling_100seeds.csv")
    lookup = {
        (row["regime"], int(row["N"]), row["top"]): row["d_collapse_mean"]
        for row in source_rows
    }
    scales = (15, 30, 60, 120)
    topologies = ("cent", "mat", "mod", "overflow", "pool")
    for regime, filename in (
        ("poisson", "fig3a_poisson_scaling.csv"),
        ("bursty_0326", "fig3b_bursty_scaling.csv"),
    ):
        rows = []
        for scale in scales:
            row: dict[str, Any] = {"N": scale}
            for topology in topologies:
                key = (regime, scale, topology)
                if key not in lookup:
                    raise ValueError(f"Missing E8 topology cell: {key}")
                row[topology] = lookup[key]
            rows.append(row)
        write_csv(output / filename, ["N", *topologies], rows)

    e8 = load_json(results / "artifacts" / "abm_multi_topology.json")
    null_cells = e8["null_burst_pooling"]
    reversal_rows = []
    for index, scale in enumerate((30, 60, 120), start=1):
        poisson_gap = float(lookup[("poisson", scale, "mod")]) - float(
            lookup[("poisson", scale, "pool")]
        )
        bursty_gap = float(lookup[("bursty_0326", scale, "mod")]) - float(
            lookup[("bursty_0326", scale, "pool")]
        )
        null_summary = null_cells[f"N{scale}"]["pool_minus_mod_null_burst055"]
        if not isinstance(null_summary, dict) or "diff_mean" not in null_summary:
            raise ValueError(f"Unexpected null reversal schema at N={scale}")
        null_gap = -float(null_summary["diff_mean"])
        reversal_rows.append(
            {
                "N": scale,
                "x_idx": index,
                "canon_poisson": str(poisson_gap),
                "canon_bursty": str(bursty_gap),
                "null_burst055": str(null_gap),
            }
        )
    write_csv(
        output / "fig3c_reversal.csv",
        ["N", "x_idx", "canon_poisson", "canon_bursty", "null_burst055"],
        reversal_rows,
    )


def generate_fig4(results: Path, output: Path) -> None:
    _, k4_rows = read_csv(results / "mechanism_ablation_summary.csv")
    k4 = {row["arm"]: row for row in k4_rows}
    k4_order = (
        ("placebo_ticket_id", "Ticket-ID placebo"),
        ("concave_m1_fixed_seff", "Joint $m{=}1$ and fixed speed"),
        ("freeze_A_only", "Freeze absorption $A$"),
        ("verify_relief_only", "Verif. relief only"),
        ("coach_relief_only", "Coaching relief only"),
        ("external_replacement", "External replacement"),
        ("zero_delta", "Zero turnover ($\\delta=0$)"),
        ("zero_tau_v", "Zero verif. cost ($\\tau_v=0$)"),
        ("zero_tau_and_zero_delta", "Zero $\\tau_v$ + Zero $\\delta$"),
    )
    fig4a_rows = []
    for index, (arm, label) in enumerate(k4_order, start=1):
        if arm not in k4:
            raise ValueError(f"Missing K4 arm: {arm}")
        source = k4[arm]
        fig4a_rows.append(
            {
                "y": index,
                "arm": arm,
                "label": label,
                "delta": rounded(source["delta_Q"], 5),
                "lo": rounded(source["ci95_lo"], 5),
                "hi": rounded(source["ci95_hi"], 5),
            }
        )
    write_csv(
        output / "fig4a_k4_forest.csv",
        ["y", "arm", "label", "delta", "lo", "hi"],
        fig4a_rows,
    )

    _, k9_rows = read_csv(results / "topology_channel_ablations.csv")
    k9 = {row["channel"]: row for row in k9_rows}
    k9_order = (
        (
            "Null parity (kappa0=sigma0=0)",
            "Both channels zero (kappa0=sigma0=0)",
            "Both channels zero",
        ),
        (
            "R6b: sigma0 = 0 (mod - pool)",
            "R6b: sigma0 = 0 (mod - pool)",
            "Zero coaching ($\\sigma_0=0$)",
        ),
        (
            "R6a: kappa0 = 0 (mod - pool)",
            "R6a: kappa0 = 0 (mod - pool)",
            "Zero context ($\\kappa_0=0$)",
        ),
        ("Canon (mod - pool)", "Canon (mod - pool)", "Canon: Mod vs Pool"),
        (
            "Selective hub (mod - cent_sel)",
            "Selective hub (mod - cent_sel)",
            "Selective hub (35\\%)",
        ),
        (
            "Role-balanced hub (mod - cent_rb)",
            "Role-balanced hub (mod - cent_rb)",
            "Role-balanced hub",
        ),
        (
            "Pass-conserved hub (mod - cent_pc)",
            "Pass-conserved hub (mod - cent_pc)",
            "Pass-conserved hub",
        ),
        ("Canon (mod - cent)", "Canon (mod - cent)", "Canon: Mod vs Cent"),
    )
    fig4b_rows = []
    for index, (source_name, channel, label) in enumerate(k9_order, start=1):
        if source_name not in k9:
            raise ValueError(f"Missing K9 channel: {source_name}")
        source = k9[source_name]
        fig4b_rows.append(
            {
                "y": index,
                "channel": channel,
                "label": label,
                "delta": rounded(source["mean"], 4),
                "lo": rounded(source["ci95_lo"], 4),
                "hi": rounded(source["ci95_hi"], 4),
            }
        )
    write_csv(
        output / "fig4b_k9_forest.csv",
        ["y", "channel", "label", "delta", "lo", "hi"],
        fig4b_rows,
    )

    _, k12_rows = read_csv(results / "governance_ablations_100seeds.csv")
    k12 = {row["contrast"]: row for row in k12_rows}
    k12_order = (
        ("G5_minus_RT_tuned_AUC050", "G5 vs Random Thin (tun)"),
        ("G5_minus_RT_fixed_g_placebo", "G5 vs Random Thin (fix)"),
        ("G5_minus_FL_tuned_frozen", "G5 vs Frozen Learn (tun)"),
        ("synergy_2x2_G5_minus_RA_minus_QA", "2x2 Quota x Route Synergy"),
        ("G5_minus_RA_routing_alone", "G5 vs Routing Alone"),
        ("G5_minus_FL_fixed_g_frozen", "G5 vs Frozen Learn (fix)"),
        ("G5_minus_QA_matched_quota", "G5 vs Matched Quota"),
    )
    fig4c_rows = []
    for index, (contrast, label) in enumerate(k12_order, start=1):
        if contrast not in k12:
            raise ValueError(f"Missing K12 contrast: {contrast}")
        source = k12[contrast]
        fig4c_rows.append(
            {
                "y": index,
                "contrast": contrast,
                "label": label,
                "delta": rounded(source["diff_mean"], 5),
                "lo": rounded(source["ci95_lo"], 5),
                "hi": rounded(source["ci95_hi"], 5),
            }
        )
    write_csv(
        output / "fig4c_k12_forest.csv",
        ["y", "contrast", "label", "delta", "lo", "hi"],
        fig4c_rows,
    )


def generate_fig5(results: Path, output: Path) -> None:
    e11 = load_json(results / "artifacts" / "governance_tournament.json")
    arms = e11["confirmatory_N30_d075_arms"]
    arm_order = (
        "G0_unassisted",
        "G1_unmanaged",
        "G2_static_hitl",
        "G2_priority",
        "G3_parallel",
        "G4_bonus",
        "UG_guardrails",
        "EA_ex_ante",
        "RA_routing_alone",
        "QA_quota_alone",
        "RT_random_thinning",
        "G5_blind_5x5",
        "G5_coupled",
        "QA_matched_quota",
        "G5_blind_5x5_sla",
    )
    plot_groups = {
        "G0_unassisted": 4,
        "G1_unmanaged": 1,
        "G2_static_hitl": 1,
        "G2_priority": 1,
        "G3_parallel": 1,
        "G4_bonus": 1,
        "UG_guardrails": 1,
        "EA_ex_ante": 2,
        "RA_routing_alone": 2,
        "QA_quota_alone": 2,
        "RT_random_thinning": 3,
        "G5_blind_5x5": 0,
        "G5_coupled": 5,
        "QA_matched_quota": 1,
        "G5_blind_5x5_sla": 6,
    }
    metric_fields = (
        "Q_mean",
        "Q_sd",
        "Y_mean",
        "Y_sd",
        "Y_gross_mean",
        "S_mean",
        "u_mean",
    )
    rows = []
    for arm in arm_order:
        if arm not in arms:
            raise ValueError(f"Missing E11 governance arm: {arm}")
        source = arms[arm]
        row: dict[str, Any] = {"arm": arm}
        for field in metric_fields:
            if field not in source:
                raise ValueError(f"Missing E11 field: {arm}.{field}")
            row[field] = rounded(source[field], 6)
        row["plot_group"] = plot_groups[arm]
        rows.append(row)
    write_csv(
        output / "fig5a_pareto.csv",
        ["arm", *metric_fields, "plot_group"],
        rows,
    )

    shutil.copyfile(
        results / "governance_auc_dose_100seeds.csv",
        output / "fig5b_auc_dose.csv",
    )
    shutil.copyfile(
        results / "governance_xcap_dose_100seeds.csv",
        output / "fig5c_xcap_dose.csv",
    )


def build_manifest(results: Path, output: Path) -> dict[str, Any]:
    source_paths = (
        results / "modular_N30_hysteresis_curves.csv",
        results / "abm_bbw_trajectory_100seeds.csv",
        results / "endogenous_d_controller.csv",
        results / "mechanism_ablation_summary.csv",
        results / "topology_scaling_100seeds.csv",
        results / "topology_channel_ablations.csv",
        results / "governance_auc_dose_100seeds.csv",
        results / "governance_xcap_dose_100seeds.csv",
        results / "governance_ablations_100seeds.csv",
        results / "artifacts" / "abm_multi_topology.json",
        results / "artifacts" / "governance_tournament.json",
        Path(__file__).resolve(),
    )
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing figure sources: {missing}")
    return {
        "schema": "attention-rationed-traps-figure-data-manifest/v1",
        "sources": {
            str(path.relative_to(results)) if path.is_relative_to(results) else path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in source_paths
        },
        "outputs": {
            filename: {
                "bytes": (output / filename).stat().st_size,
                "sha256": sha256_file(output / filename),
            }
            for filename in FIGURE_FILES
        },
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python3 generate_figure_data.py RESULTS_ROOT OUTPUT_DIR", file=sys.stderr)
        return 2
    results = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    if not results.is_dir():
        raise NotADirectoryError(results)
    output.mkdir(parents=True, exist_ok=True)
    generate_fig2(results, output)
    generate_fig3(results, output)
    generate_fig4(results, output)
    generate_fig5(results, output)
    manifest = build_manifest(results, output)
    manifest_path = output / "figure_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "outputs": len(FIGURE_FILES), "manifest": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
