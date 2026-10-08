"""Paths and helpers shared by the whole suite.

FIXTURES_DIR points to tests/fixtures/, which are real excerpts of EUR-Lex
XHTML (see tests/fixtures/PROVENANCE.md), not synthetic documents.

The "corpus" marker (registered in pyproject.toml) groups the tests that need
data/raw/, which is in .gitignore under D-002 and therefore does NOT exist in a
clean checkout or in CI. They are skipped automatically when it is missing.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from ingest import hierarchy as h

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
# data/manifest.json IS versioned (unlike data/raw/), so the manifest contract
# tests also run in a clean checkout.
MANIFEST_PATH = Path(__file__).parent.parent / "data" / "manifest.json"


def fixture_path(name: str) -> str:
    return str(FIXTURES_DIR / f"{name}.xhtml")


# ---------------------------------------------------------------------------
# reconstruction invariant (M-006): the same principle as scripts/measure_nodes.py,
# written again here on purpose, so that the tests do not depend on a script
# ---------------------------------------------------------------------------


def reinsert(node: h.Node) -> str:
    if node.kind == "apartado":
        return f"{node.marker}. {node.own_text}"
    if node.kind == "punto":
        return f"({node.marker}) {node.own_text}"
    # the article, a subparagraph and an unnumbered paragraph print no marker
    return node.own_text


def reconstruct(article: h.Article) -> str:
    """'Article <path> <title>' + every node's marker and own_text, in node order."""
    head = f"Article {article.path} {article.title}"
    return h._normalize(" ".join([head] + [reinsert(n) for n in article.nodes]))


def same_words(article: h.Article) -> bool:
    """No word of Article.text lost, none duplicated, order aside. Article.text comes
    from an independent code path (string cleaning), so agreement means that the
    attribution of text to nodes lost nothing and repeated nothing."""
    return Counter(reconstruct(article).split()) == Counter(article.text.split())


def pytest_collection_modifyitems(config, items):
    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        return
    skip = pytest.mark.skip(reason="data/raw/ is missing in this checkout (D-002: not versioned)")
    for item in items:
        if "corpus" in item.keywords:
            item.add_marker(skip)
