"""Verify the 27-contrast precision table specification against confirmatory_effect_summary.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from generate_confirmatory_effect_summary import CONTRAST_SPECS  # noqa: E402


def _fmt_signed(value: float) -> str:
    return f"{float(value):+.4f}"


def table_rows(summary: dict[str, Any], spec_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contrasts = summary["contrasts"]
    expected_rows = []
    for index, (contrast_spec, table_spec) in enumerate(zip(CONTRAST_SPECS, spec_rows), start=1):
        identifier = str(contrast_spec["id"])
        row = contrasts[identifier]
        low, high = row["interval_decision"]
        expected_rows.append(
            {
                "index": index,
                "id": identifier,
                "effect_str": _fmt_signed(row["effect"]),
                "interval_str": f"[{_fmt_signed(low)}, {_fmt_signed(high)}]",
                "reported_contrast": table_spec["reported_contrast"],
                "metric": table_spec["metric"],
            }
        )
    return expected_rows


def verify(summary_path: Path, spec_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    observed = json.loads(spec_path.read_text(encoding="utf-8"))
    if len(observed) != len(CONTRAST_SPECS):
        raise AssertionError(f"Expected {len(CONTRAST_SPECS)} table rows, found {len(observed)}")
    if [row["id"] for row in observed] != [str(spec["id"]) for spec in CONTRAST_SPECS]:
        raise AssertionError("Contrast order in precision_table_spec.json does not match CONTRAST_SPECS")
    expected = table_rows(summary, observed)
    for exp_row, obs_row in zip(expected, observed):
        if (
            int(obs_row["index"]) != exp_row["index"]
            or obs_row["effect_str"] != exp_row["effect_str"]
            or obs_row["interval_str"] != exp_row["interval_str"]
        ):
            raise AssertionError(
                f"Precision table mismatch at contrast {exp_row['index']} ({exp_row['id']}): "
                f"expected effect={exp_row['effect_str']} interval={exp_row['interval_str']}, "
                f"observed effect={obs_row['effect_str']} interval={obs_row['interval_str']}"
            )
    return {
        "rows_verified": len(observed),
        "family_alpha": summary["meta"]["alpha_family"],
        "confidence": summary["meta"]["confidence"],
        "status": "PRECISION_TABLE_VERIFIED",
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    summary_path = Path(args[0]) if len(args) >= 1 else REPO_ROOT / "results" / "confirmatory_effect_summary.json"
    spec_path = Path(args[1]) if len(args) >= 2 else REPO_ROOT / "configs" / "precision_table_spec.json"
    result = verify(summary_path, spec_path)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
