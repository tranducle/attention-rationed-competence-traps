"""Generate the fixed 27 contrast confirmatory family from paired official artifacts."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
for _sub in ("src", "experiments", "scripts"):
    _p = str(_REPO_ROOT / _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import json
import math
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

from effect_resolver_core import Samples, resolve


EXPECTED_SEEDS = tuple(range(1000, 1100))
BASE_CONFIDENCE = 0.99
FAMILY_ALPHA = 0.01
COMPARISONS = 27
PRIMARY_RESAMPLES = 10000
REFERENCE_RESAMPLES = 100000
BOOTSTRAP_SEED = 20261119
ENDPOINT_TOLERANCE = 0.005

ARTIFACT_FILES = {
    "e3": "abm_docking_hysteresis.json",
    "e4": "mechanism_loop_ablations.json",
    "e8": "abm_multi_topology.json",
    "e11": "governance_tournament.json",
}

CONTRAST_SPECS = (
    {"id": "C1_K3_hyst_vs_tau0", "source": "e3", "method": "per_seed_max_hyst_Q", "baseline": "per_seed_tau0_hyst_Q", "metric": "max_hyst_Q", "metric_tex": r"$\max Q_{\mathrm{hyst}}$", "label_tex": r"Mech.: Modular hyst. vs $\tau_v = 0$ ($N=30$)", "threshold": 0.10},
    {"id": "C1_K4_zero_tau_v", "source": "e4", "method": "N30_arms.zero_tau_v.per_seed_Q", "baseline": "N30_arms.full_model.per_seed_Q", "metric": "ood_accuracy_Q", "metric_tex": r"$Q$", "label_tex": r"Mech.: Zero $\tau_v$ vs unmanaged ($d=0.75$)", "threshold": 0.10},
    {"id": "C1_K4_zero_delta", "source": "e4", "method": "N30_arms.zero_delta.per_seed_Q", "baseline": "N30_arms.full_model.per_seed_Q", "metric": "ood_accuracy_Q", "metric_tex": r"$Q$", "label_tex": r"Mech.: Zero $\delta$ vs unmanaged ($d=0.75$)", "threshold": 0.10},
    {"id": "C1_K4_coach_relief_only", "source": "e4", "method": "N30_arms.coach_relief_only.per_seed_Q", "baseline": "N30_arms.full_model.per_seed_Q", "metric": "ood_accuracy_Q", "metric_tex": r"$Q$", "label_tex": r"Mech.: Joint coach + absorp. relief ($d=0.75$)", "threshold": 0.10},
    {"id": "C1_K4_verify_relief_only", "source": "e4", "method": "N30_arms.verify_relief_only.per_seed_Q", "baseline": "N30_arms.full_model.per_seed_Q", "metric": "ood_accuracy_Q", "metric_tex": r"$Q$", "label_tex": r"Mech.: Verify-fatigue relief alone ($d=0.75$)", "threshold": 0.10},
    {"id": "C1_K4_external_replacement", "source": "e4", "method": "N30_arms.external_replacement.per_seed_Q", "baseline": "N30_arms.full_model.per_seed_Q", "metric": "ood_accuracy_Q", "metric_tex": r"$Q$", "label_tex": r"Mech.: External senior replacement ($d=0.75$)", "threshold": 0.10},
    {"id": "C2_K8_poisson_N15_mod_vs_maxrival", "source": "e8", "method": "scales.poisson.N15.per_seed_mod", "baseline_max": ("scales.poisson.N15.per_seed_cent", "scales.poisson.N15.per_seed_pool", "scales.poisson.N15.per_seed_mat"), "metric": "d_collapse", "metric_tex": r"$d_{\mathrm{col}}$", "label_tex": r"Topo.: Poisson $N=15$, $\mathrm{mod} - \max\text{ rival}$", "threshold": 0.08},
    {"id": "C2_K8_poisson_N30_mod_vs_maxrival", "source": "e8", "method": "scales.poisson.N30.per_seed_mod", "baseline_max": ("scales.poisson.N30.per_seed_cent", "scales.poisson.N30.per_seed_pool", "scales.poisson.N30.per_seed_mat"), "metric": "d_collapse", "metric_tex": r"$d_{\mathrm{col}}$", "label_tex": r"Topo.: Poisson $N=30$, $\mathrm{mod} - \max\text{ rival}$", "threshold": 0.08},
    {"id": "C2_K8_poisson_N60_mod_vs_maxrival", "source": "e8", "method": "scales.poisson.N60.per_seed_mod", "baseline_max": ("scales.poisson.N60.per_seed_cent", "scales.poisson.N60.per_seed_pool", "scales.poisson.N60.per_seed_mat"), "metric": "d_collapse", "metric_tex": r"$d_{\mathrm{col}}$", "label_tex": r"Topo.: Poisson $N=60$, $\mathrm{mod} - \max\text{ rival}$", "threshold": 0.08},
    {"id": "C2_K8_poisson_N120_mod_vs_maxrival", "source": "e8", "method": "scales.poisson.N120.per_seed_mod", "baseline_max": ("scales.poisson.N120.per_seed_cent", "scales.poisson.N120.per_seed_pool", "scales.poisson.N120.per_seed_mat"), "metric": "d_collapse", "metric_tex": r"$d_{\mathrm{col}}$", "label_tex": r"Topo.: Poisson $N=120$, $\mathrm{mod} - \max\text{ rival}$", "threshold": 0.08},
    {"id": "C2_K8_poisson_N30_mod_vs_mat", "source": "e8", "method": "scales.poisson.N30.per_seed_mod", "baseline": "scales.poisson.N30.per_seed_mat", "metric": "d_collapse", "metric_tex": r"$d_{\mathrm{col}}$", "label_tex": r"Topo.: Poisson $N=30$, $\mathrm{mod} - \mathrm{mat}$", "threshold": 0.08},
    {"id": "C2_K8_poisson_N30_mod_vs_pool", "source": "e8", "method": "scales.poisson.N30.per_seed_mod", "baseline": "scales.poisson.N30.per_seed_pool", "metric": "d_collapse", "metric_tex": r"$d_{\mathrm{col}}$", "label_tex": r"Topo.: Poisson $N=30$, $\mathrm{mod} - \mathrm{pool}$", "threshold": 0.08},
    {"id": "C2_K8_poisson_N30_mod_vs_cent", "source": "e8", "method": "scales.poisson.N30.per_seed_mod", "baseline": "scales.poisson.N30.per_seed_cent", "metric": "d_collapse", "metric_tex": r"$d_{\mathrm{col}}$", "label_tex": r"Topo.: Poisson $N=30$, $\mathrm{mod} - \mathrm{cent}$", "threshold": 0.08},
    {"id": "C2_K8_bursty_N120_mod_vs_maxrival", "source": "e8", "method": "scales.bursty_0326.N120.per_seed_mod", "baseline_max": ("scales.bursty_0326.N120.per_seed_cent", "scales.bursty_0326.N120.per_seed_pool", "scales.bursty_0326.N120.per_seed_mat"), "metric": "d_collapse", "metric_tex": r"$d_{\mathrm{col}}$", "label_tex": r"Topo.: Bursty $N=120$, $\mathrm{mod} - \max\text{ rival}$", "threshold": 0.08},
    {"id": "C3_G5_vs_G1_unmanaged", "source": "e11", "method": "confirmatory_N30_d075_arms.G5_coupled.per_seed_Q", "baseline": "confirmatory_N30_d075_arms.G1_unmanaged.per_seed_Q", "metric": "ood_accuracy_Q", "metric_tex": r"$Q$", "label_tex": r"Gov.: $G_5 - G_1$ (unmanaged)", "threshold": 0.10},
    {"id": "C3_G5_vs_G2_static_hitl", "source": "e11", "method": "confirmatory_N30_d075_arms.G5_coupled.per_seed_Q", "baseline": "confirmatory_N30_d075_arms.G2_static_hitl.per_seed_Q", "metric": "ood_accuracy_Q", "metric_tex": r"$Q$", "label_tex": r"Gov.: $G_5 - G_2$ (static HITL)", "threshold": 0.10},
    {"id": "C3_G5_vs_G2_priority", "source": "e11", "method": "confirmatory_N30_d075_arms.G5_coupled.per_seed_Q", "baseline": "confirmatory_N30_d075_arms.G2_priority.per_seed_Q", "metric": "ood_accuracy_Q", "metric_tex": r"$Q$", "label_tex": r"Gov.: $G_5 - G_{2,\mathrm{prio}}$ (priority HITL)", "threshold": 0.10},
    {"id": "C3_G5_vs_RA_routing_alone", "source": "e11", "method": "confirmatory_N30_d075_arms.G5_coupled.per_seed_Q", "baseline": "confirmatory_N30_d075_arms.RA_routing_alone.per_seed_Q", "metric": "ood_accuracy_Q", "metric_tex": r"$Q$", "label_tex": r"Gov.: $G_5 - \mathrm{RA}$ (routing alone)", "threshold": 0.10},
    {"id": "C3_G5_vs_QA_matched_quota", "source": "e11", "method": "confirmatory_N30_d075_arms.G5_coupled.per_seed_Q", "baseline": "confirmatory_N30_d075_arms.QA_matched_quota.per_seed_Q", "metric": "ood_accuracy_Q", "metric_tex": r"$Q$", "label_tex": r"Gov.: $G_5 - \mathrm{QA}_{\mathrm{match}}$ (matched quota)", "threshold": 0.10},
    {"id": "C3_G5_vs_FL_fixed_g_frozen", "source": "e11", "method": "confirmatory_N30_d075_arms.G5_coupled.per_seed_Q", "baseline": "confirmatory_N30_d075_arms.FL_fixed_g_frozen.per_seed_Q", "metric": "ood_accuracy_Q", "metric_tex": r"$Q$", "label_tex": r"Gov.: $G_5 - \mathrm{FL}_{\mathrm{fixed}}$ (frozen learn.)", "threshold": 0.10},
    {"id": "C3_G5_vs_FL_frozen_learning", "source": "e11", "method": "confirmatory_N30_d075_arms.G5_coupled.per_seed_Q", "baseline": "confirmatory_N30_d075_arms.FL_frozen_learning.per_seed_Q", "metric": "ood_accuracy_Q", "metric_tex": r"$Q$", "label_tex": r"Gov.: $G_5 - \mathrm{FL}_{\mathrm{tuned}}$ (frozen learn.)", "threshold": 0.10},
    {"id": "C3_G5_vs_RT_fixed_g_placebo", "source": "e11", "method": "confirmatory_N30_d075_arms.G5_coupled.per_seed_Q", "baseline": "confirmatory_N30_d075_arms.RT_fixed_g_placebo.per_seed_Q", "metric": "ood_accuracy_Q", "metric_tex": r"$Q$", "label_tex": r"Gov.: $G_5 - \mathrm{RT}_{\mathrm{fixed}}$ (random thin.)", "threshold": 0.05},
    {"id": "C3_G5_vs_RT_random_thinning", "source": "e11", "method": "confirmatory_N30_d075_arms.G5_coupled.per_seed_Q", "baseline": "confirmatory_N30_d075_arms.RT_random_thinning.per_seed_Q", "metric": "ood_accuracy_Q", "metric_tex": r"$Q$", "label_tex": r"Gov.: $G_5 - \mathrm{RT}_{\mathrm{tuned}}$ (random thin.)", "threshold": 0.05},
    {"id": "C3_G5_Y_vs_G0_unassisted", "source": "e11", "method": "confirmatory_N30_d075_arms.G5_coupled.per_seed_Y", "baseline": "confirmatory_N30_d075_arms.G0_unassisted.per_seed_Y", "metric": "net_output_Y", "metric_tex": r"$Y$", "label_tex": r"Gov.: $G_5 - G_0$ (unassisted ref.)", "threshold": 0.02},
    {"id": "C3_G5_Y_vs_QA_quota_alone", "source": "e11", "method": "confirmatory_N30_d075_arms.G5_coupled.per_seed_Y", "baseline": "confirmatory_N30_d075_arms.QA_quota_alone.per_seed_Y", "metric": "net_output_Y", "metric_tex": r"$Y$", "label_tex": r"Gov.: $G_5 - \mathrm{QA}_{\mathrm{free}}$ (free quota)", "threshold": 0.02},
    {"id": "C3_G5_Y_vs_EA_ex_ante", "source": "e11", "method": "confirmatory_N30_d075_arms.G5_coupled.per_seed_Y", "baseline": "confirmatory_N30_d075_arms.EA_ex_ante.per_seed_Y", "metric": "net_output_Y", "metric_tex": r"$Y$", "label_tex": r"Gov.: $G_5 - \mathrm{EA}_{\mathrm{ex\text{-}ante}}$ (ex-ante cap)", "threshold": 0.02},
    {"id": "C3_G5_Y_vs_RT_random_thinning", "source": "e11", "method": "confirmatory_N30_d075_arms.G5_coupled.per_seed_Y", "baseline": "confirmatory_N30_d075_arms.RT_random_thinning.per_seed_Y", "metric": "net_output_Y", "metric_tex": r"$Y$", "label_tex": r"Gov.: $G_5 - \mathrm{RT}_{\mathrm{tuned}}$ (random thin.)", "threshold": 0.02},
)


def _object_pairs(pairs: Iterable[Tuple[str, Any]]) -> dict:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_json(path: Path) -> dict:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON number: {value}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_object_pairs,
        parse_constant=reject_constant,
    )


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _path_value(document: dict, dotted: str) -> Any:
    value: Any = document
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"missing required artifact field: {dotted}")
        value = value[part]
    return value


def _vector(document: dict, dotted: str) -> Tuple[float, ...]:
    raw = _path_value(document, dotted)
    if not isinstance(raw, list) or len(raw) != len(EXPECTED_SEEDS):
        raise ValueError(f"{dotted} must contain exactly {len(EXPECTED_SEEDS)} paired values")
    values = tuple(float(item) for item in raw)
    if any(not math.isfinite(item) for item in values):
        raise ValueError(f"{dotted} contains a nonfinite value")
    return values


def _load_artifacts(artifact_dir: Path, require_seed_ids: bool) -> Dict[str, dict]:
    documents = {
        name: _read_json(artifact_dir / filename)
        for name, filename in ARTIFACT_FILES.items()
    }
    for name, document in documents.items():
        seed_ids = document.get("seed_ids")
        if seed_ids is None and not require_seed_ids:
            continue
        if seed_ids != list(EXPECTED_SEEDS):
            raise ValueError(f"{name} does not bind the ordered confirmatory seed identifiers")
    return documents


def _paired_values(documents: Dict[str, dict], spec: dict) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    document = documents[str(spec["source"])]
    method = _vector(document, str(spec["method"]))
    if "baseline" in spec:
        baseline = _vector(document, str(spec["baseline"]))
    else:
        rivals = [_vector(document, path) for path in spec["baseline_max"]]
        baseline = tuple(max(items) for items in zip(*rivals))
    return method, baseline


def generate_family_payload(
    artifact_dir: Path,
    resamples: int = PRIMARY_RESAMPLES,
    require_seed_ids: bool = True,
) -> dict:
    if len(CONTRAST_SPECS) != COMPARISONS:
        raise ValueError("the confirmatory family must contain exactly 27 contrasts")
    identifiers = [str(item["id"]) for item in CONTRAST_SPECS]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("confirmatory contrast identifiers must be unique")
    documents = _load_artifacts(Path(artifact_dir), require_seed_ids)
    contrasts: Dict[str, dict] = {}
    for spec in CONTRAST_SPECS:
        method, baseline = _paired_values(documents, spec)
        report = resolve(
            Samples(method, baseline, "paired"),
            float(spec["threshold"]),
            metric=str(spec["metric"]),
            confidence=BASE_CONFIDENCE,
            comparisons=COMPARISONS,
            resamples=int(resamples),
            seed=BOOTSTRAP_SEED,
        )
        contrasts[str(spec["id"])] = asdict(report)
    return {
        "meta": {
            "confidence": BASE_CONFIDENCE,
            "alpha_family": FAMILY_ALPHA,
            "bonferroni_comparisons": COMPARISONS,
            "bootstrap_resamples": int(resamples),
            "seed": BOOTSTRAP_SEED,
        },
        "contrasts": contrasts,
    }


def generate_precision_audit_payload(primary: dict, reference: dict) -> dict:
    rows: Dict[str, dict] = {}
    all_adequate = True
    for identifier in (str(spec["id"]) for spec in CONTRAST_SPECS):
        first = primary["contrasts"][identifier]
        second = reference["contrasts"][identifier]
        first_interval = first["interval_decision"]
        second_interval = second["interval_decision"]
        if first_interval is None or second_interval is None:
            shifts = None
            adequate = False
        else:
            shifts = [
                abs(float(first_interval[0]) - float(second_interval[0])),
                abs(float(first_interval[1]) - float(second_interval[1])),
            ]
            adequate = first["status"] == second["status"] and max(shifts) <= ENDPOINT_TOLERANCE
        all_adequate = all_adequate and adequate
        rows[identifier] = {
            "primary_status": first["status"],
            "reference_status": second["status"],
            "primary_interval_decision": first_interval,
            "reference_interval_decision": second_interval,
            "endpoint_abs_shift": shifts,
            "adequate": adequate,
        }
    return {
        "meta": {
            "primary_resamples": PRIMARY_RESAMPLES,
            "reference_resamples": REFERENCE_RESAMPLES,
            "endpoint_tolerance": ENDPOINT_TOLERANCE,
            "seed": BOOTSTRAP_SEED,
        },
        "all_adequate": all_adequate,
        "contrasts": rows,
    }


def generate_outputs(
    artifact_dir: Path,
    primary_path: Path,
    audit_path: Path,
    require_seed_ids: bool = True,
) -> Tuple[dict, dict]:
    primary = generate_family_payload(artifact_dir, PRIMARY_RESAMPLES, require_seed_ids)
    repeat = generate_family_payload(artifact_dir, PRIMARY_RESAMPLES, require_seed_ids)
    if json.dumps(primary, indent=2) != json.dumps(repeat, indent=2):
        raise RuntimeError("confirmatory family generation is not deterministic")
    reference = generate_family_payload(artifact_dir, REFERENCE_RESAMPLES, require_seed_ids)
    audit = generate_precision_audit_payload(primary, reference)
    _atomic_json(Path(primary_path), primary)
    _atomic_json(Path(audit_path), audit)
    return primary, audit


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) not in (2, 3):
        raise SystemExit(
            "usage: generate_confirmatory_effect_summary.py artifact_dir primary_output optional_audit_output"
        )
    artifact_dir = Path(args[0])
    primary_path = Path(args[1])
    audit_path = Path(args[2]) if len(args) == 3 else primary_path.with_name("confirmatory_effect_mc_audit.json")
    primary, audit = generate_outputs(artifact_dir, primary_path, audit_path, require_seed_ids=True)
    print(json.dumps({
        "contrasts": len(primary["contrasts"]),
        "all_adequate": audit["all_adequate"],
        "primary": str(primary_path),
        "audit": str(audit_path),
    }, sort_keys=True))
    return 0 if audit["all_adequate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

