"""Regression checks for the fixed 27 contrast family generator."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from generate_confirmatory_effect_summary import (  # noqa: E402
    COMPARISONS,
    CONTRAST_SPECS,
    PRIMARY_RESAMPLES,
    generate_family_payload,
)

ARTIFACT_DIR = REPO_ROOT / "results" / "artifacts"
EXPECTED = REPO_ROOT / "results" / "confirmatory_effect_summary.json"


def test_registered_family_reproduces_existing_bytes():
    payload = generate_family_payload(ARTIFACT_DIR, PRIMARY_RESAMPLES, require_seed_ids=True)
    observed = json.dumps(payload, indent=2).encode("utf-8")
    assert observed == EXPECTED.read_bytes()
    assert len(payload["contrasts"]) == COMPARISONS == len(CONTRAST_SPECS)


def test_contrast_ids_match_precision_spec():
    spec_rows = json.loads((REPO_ROOT / "configs" / "precision_table_spec.json").read_text(encoding="utf-8"))
    assert [row["id"] for row in spec_rows] == [str(spec["id"]) for spec in CONTRAST_SPECS]
