"""Paths and helpers shared by the whole suite.

FIXTURES_DIR points to tests/fixtures/, which are real excerpts of EUR-Lex
XHTML (see tests/fixtures/PROVENANCE.md), not synthetic documents.

The "corpus" marker (registered in pyproject.toml) groups the tests that need
data/raw/, which is in .gitignore under D-002 and therefore does NOT exist in a
clean checkout or in CI. They are skipped automatically when it is missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
# data/manifest.json IS versioned (unlike data/raw/), so the manifest contract
# tests also run in a clean checkout.
MANIFEST_PATH = Path(__file__).parent.parent / "data" / "manifest.json"


def fixture_path(name: str) -> str:
    return str(FIXTURES_DIR / f"{name}.xhtml")


def pytest_collection_modifyitems(config, items):
    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        return
    skip = pytest.mark.skip(reason="data/raw/ is missing in this checkout (D-002: not versioned)")
    for item in items:
        if "corpus" in item.keywords:
            item.add_marker(skip)
