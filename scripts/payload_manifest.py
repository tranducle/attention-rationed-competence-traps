"""Build or verify the reproducibility package payload manifest."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "results" / "sha256_manifest.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def inventory() -> dict[str, dict[str, int | str]]:
    records: dict[str, dict[str, int | str]] = {}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == MANIFEST:
            continue
        relative = path.relative_to(ROOT)
        if relative.as_posix() == "DATA_MANIFEST.json" or ".git" in relative.parts or ".pytest_cache" in relative.parts or "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        records[relative.as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
    return records


def build() -> None:
    records = inventory()
    MANIFEST.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"payload_files": len(records), "manifest": str(MANIFEST)}, sort_keys=True))


def verify() -> None:
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    observed = inventory()
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    mismatches = sorted(
        path for path in set(expected) & set(observed) if expected[path] != observed[path]
    )
    result = {
        "expected_files": len(expected),
        "observed_files": len(observed),
        "missing": missing,
        "extra": extra,
        "mismatches": mismatches,
    }
    print(json.dumps(result, sort_keys=True))
    if missing or extra or mismatches:
        raise SystemExit("PAYLOAD_MANIFEST_MISMATCH")
    print("PAYLOAD_MANIFEST_VERIFIED")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"build", "verify"}:
        raise SystemExit("usage: payload_manifest.py build or verify")
    if sys.argv[1] == "build":
        build()
    else:
        verify()


if __name__ == "__main__":
    main()
