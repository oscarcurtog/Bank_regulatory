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


# ---------------------------------------------------------------------------
# chunk layer invariants (T2): a second implementation, written from the design
# and not from ingest/chunks.py, so that the chunker is checked by an oracle and
# not by itself
# ---------------------------------------------------------------------------

SUBDIVISION_KINDS = ("subparagraph", "unnumbered_paragraph")


def expected_context(act: str, article: h.Article, root_path: str) -> str:
    """Heading, then the own text of every ancestor that governs root_path. An
    ancestor whose way down goes through an explicit sub_<k>/unp_<k> is skipped:
    its own text is the implicit first subparagraph, a sibling of the chunk."""
    by_path = {n.path: n for n in article.nodes}
    labels = root_path.split(".")
    lines = []
    for i in range(1, len(labels)):
        ancestor = by_path[".".join(labels[:i])]
        if by_path[".".join(labels[: i + 1])].kind in SUBDIVISION_KINDS:
            continue
        if ancestor.own_text:
            lines.append(reinsert(ancestor))
    heading = f"{act} Article {article.path}" + (f": {article.title}" if article.title else "")
    return heading + ("\n\n" + "\n".join(lines) if lines else "")


def assert_chunk_invariants(article, act, chunks, max_tokens, count) -> None:
    """Every invariant of the chunk layer, for one article and its chunks."""
    nodes = {n.path: n for n in article.nodes}
    order = {n.path: i for i, n in enumerate(article.nodes)}
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_id must be unique"
    for c in chunks:
        assert c.chunk_id == f"{act}:{c.root_path}" + (":own" if c.coverage == "own" else "")
        # traceability: every source path is one real node of this article
        assert c.root_path in nodes and all(p in nodes for p in c.source_paths)
        assert [order[p] for p in c.source_paths] == sorted(order[p] for p in c.source_paths)
        # coverage: what a chunk owns is exactly its root, or its whole subtree
        subtree = [
            p
            for p in nodes
            if (p == c.root_path or p.startswith(c.root_path + ".")) and nodes[p].own_text
        ]
        assert c.source_paths == ([c.root_path] if c.coverage == "own" else subtree)
        # body is nothing but its sources; context is the governing structure
        assert c.body == "\n".join(reinsert(nodes[p]) for p in c.source_paths)
        assert c.context == expected_context(act, article, c.root_path)
        assert c.text == c.context + "\n\n" + c.body
        # tokens recounted with the real tokenizer, budget never silent
        assert c.token_count == count(c.text)
        assert (c.token_count > max_tokens) == (c.budget_exception is not None)
    # lossless ownership: every non-empty own text is in exactly one body
    owned = Counter(p for c in chunks for p in c.source_paths)
    sources = {p for p, n in nodes.items() if n.own_text}
    assert set(owned) == sources, "source partition"
    assert all(v == 1 for v in owned.values()), "a source owned twice"
