"""Regression checks for Appendix 27-contrast precision table reconciliation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from verify_precision_table import verify  # noqa: E402

SUMMARY = REPO_ROOT / "results" / "confirmatory_effect_summary.json"
SPEC = REPO_ROOT / "configs" / "precision_table_spec.json"


def test_precision_table_matches_summary():
    result = verify(SUMMARY, SPEC)
    assert result["rows_verified"] == 27
    assert result["status"] == "PRECISION_TABLE_VERIFIED"


def test_precision_table_rejects_mutated_effect(tmp_path: Path):
    rows = json.loads(SPEC.read_text(encoding="utf-8"))
    rows[0]["effect_str"] = "+0.9999"
    mutated = tmp_path / "mutated_spec.json"
    mutated.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with pytest.raises(AssertionError, match="Precision table mismatch"):
        verify(SUMMARY, mutated)


def test_precision_table_rejects_reordered_rows(tmp_path: Path):
    rows = json.loads(SPEC.read_text(encoding="utf-8"))
    rows[0], rows[1] = rows[1], rows[0]
    reordered = tmp_path / "reordered_spec.json"
    reordered.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with pytest.raises(AssertionError, match="Contrast order"):
        verify(SUMMARY, reordered)
