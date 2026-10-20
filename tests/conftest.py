"""Paths and helpers shared by the whole suite.

FIXTURES_DIR points to tests/fixtures/, which are real excerpts of EUR-Lex
XHTML (see tests/fixtures/PROVENANCE.md), not synthetic documents.

The "corpus" marker (registered in pyproject.toml) groups the tests that need
data/raw/, which is in .gitignore under D-002 and therefore does NOT exist in a
clean checkout or in CI. They are skipped automatically when it is missing.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
from collections import Counter
from pathlib import Path

import pytest

from ingest import evaluation as ev
from ingest import hierarchy as h

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
# data/manifest.json IS versioned (unlike data/raw/), so the manifest contract
# tests also run in a clean checkout.
MANIFEST_PATH = Path(__file__).parent.parent / "data" / "manifest.json"
# The private TEST area (D-054) is never versioned either. The tests marked
# "requires_private_holdout_environment" check the real installation in it, and are
# skipped where it does not exist; every other holdout test makes a synthetic one.
PRIVATE_HOLDOUT_DIR = Path(__file__).parent.parent / "data" / "benchmark" / "test"


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


# The "evaluator" tests need ir_measures (D-020), which is in requirements-dev.lock.txt
# and deliberately not in the Docker images: its backend publishes no Linux ARM64
# wheel. They are skipped ONLY where the environment declares it, as the dev image
# does. Anywhere else a missing ir_measures fails the tests that need it.
EVALUATOR_ENV = "FUNDAMENTO_EVALUATOR"
# The "embedding_model" tests need torch, transformers and the weights of the embedding
# model (requirements-embed.lock.txt and one download). CI and the dev image install
# neither, and say so with this variable. Anywhere else a missing model fails them.
EMBEDDING_MODEL_ENV = "FUNDAMENTO_EMBEDDING_MODEL"
# The "postgres" tests need a PostgreSQL server with pgvector (compose.yaml locally, a
# service container in CI), reached through the standard libpq variables. The dev image
# has none and says so with this variable. Anywhere else an unreachable server fails them.
POSTGRES_ENV = "FUNDAMENTO_POSTGRES"


def pytest_collection_modifyitems(config, items):
    if os.environ.get(EVALUATOR_ENV) == "unsupported":
        skip_evaluator = pytest.mark.skip(
            reason=f"{EVALUATOR_ENV}=unsupported: this environment does not ship ir_measures "
            f"by design (D-020); these tests run in the venv and in CI"
        )
        for item in items:
            if "evaluator" in item.keywords:
                item.add_marker(skip_evaluator)
    if os.environ.get(EMBEDDING_MODEL_ENV) == "unsupported":
        skip_model = pytest.mark.skip(
            reason=f"{EMBEDDING_MODEL_ENV}=unsupported: this environment installs neither torch "
            f"nor the embedding model by design; these tests run where the 'embed' extra is"
        )
        for item in items:
            if "embedding_model" in item.keywords:
                item.add_marker(skip_model)
    if os.environ.get(POSTGRES_ENV) == "unsupported":
        skip_postgres = pytest.mark.skip(
            reason=f"{POSTGRES_ENV}=unsupported: this environment runs no PostgreSQL by design; "
            f"these tests run against compose.yaml's server and in CI"
        )
        for item in items:
            if "postgres" in item.keywords:
                item.add_marker(skip_postgres)
    if not (PRIVATE_HOLDOUT_DIR / "provenance").is_dir():
        skip_private = pytest.mark.skip(
            reason="data/benchmark/test/ is missing in this checkout (D-054: the private "
            "TEST area is not versioned)"
        )
        for item in items:
            if "requires_private_holdout_environment" in item.keywords:
                item.add_marker(skip_private)
    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        return
    skip = pytest.mark.skip(reason="data/raw/ is missing in this checkout (D-002: not versioned)")
    for item in items:
        if "corpus" in item.keywords:
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# CI evidence. The job log of a public repository needs a signed-in reader, but
# check-run annotations can be read by anyone through the public API. In GitHub
# Actions the summary of the run is also printed as workflow commands, which
# GitHub turns into annotations: the counts, the skip reasons and the backend
# that computed the metrics, with no credential needed to check them.
# ---------------------------------------------------------------------------


def _command_text(text: str, title: bool = False) -> str:
    """Escaped for a workflow command: a newline would end it early."""
    text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return text.replace(":", "%3A").replace(",", "%2C") if title else text


def evaluator_backend() -> str:
    try:
        found = ", ".join(
            f"{d} {importlib.metadata.version(d)}" for d in ("ir-measures", "pytrec-eval-terrier")
        )
    except importlib.metadata.PackageNotFoundError as e:
        found = f"not installed ({e})"
    return f"{found}; {platform.system()} {platform.machine()}, Python {platform.python_version()}"


def ci_annotations(stats: dict[str, list], backend: str) -> list[str]:
    labels = {  # pytest's keys in terminalreporter.stats, and how they are printed
        "passed": "passed",
        "skipped": "skipped",
        "failed": "failed",
        "error": "errors",
        "warnings": "warnings",
    }
    counts = ", ".join(f"{len(stats.get(k, []))} {label}" for k, label in labels.items())
    lines = [f"::notice title=pytest::{counts}"]
    reasons = Counter(
        str(r.longrepr[2]).removeprefix("Skipped: ") for r in stats.get("skipped", [])
    )
    for reason, n in sorted(reasons.items()):
        title = _command_text(f"skipped ({n})", title=True)
        lines.append(f"::notice title={title}::{_command_text(reason)}")
    lines.append(f"::notice title=evaluator backend::{_command_text(backend)}")
    for w in stats.get("warnings", []):
        lines.append(f"::warning title=pytest warning::{_command_text(str(w.message))}")
    return lines


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    for line in ci_annotations(terminalreporter.stats, evaluator_backend()):
        terminalreporter.write_line(line)


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


# ---------------------------------------------------------------------------
# retrieval benchmark (D-019, D-020): an excerpt of the real 500-token layer for
# fixture N002, so that its known values also run where data/raw/ is missing.
# tests/integration/test_evaluation_fixtures.py checks it against the real layer.
# ---------------------------------------------------------------------------

# MiCA chunks of the official 500-token layer that fixture N002 touches, plus the
# two that close its known ranking. Checked against the real layer below.
N002_EXCERPT = (
    ("MiCA:40", ("40.1", "40.2", "40.3"), 167),
    ("MiCA:50", ("50.1", "50.2", "50.3"), 170),
    ("MiCA:3.1.6", ("3.1.6",), 73),
    ("MiCA:3.1.7", ("3.1.7",), 61),
    ("MiCA:3.1.5", ("3.1.5",), 58),
    (
        "MiCA:39",
        ("39.1", "39.2", "39.2.a", "39.2.b", "39.2.c", "39.2.d", "39.2.e", "39.2.sub_2", "39.3"),
        442,
    ),
    ("MiCA:3.1.15", ("3.1.15",), 78),
)
N002_RANKING = ["MiCA:40", "MiCA:50", "MiCA:3.1.6", "MiCA:3.1.7", "MiCA:3.1.5", "MiCA:39"]


def n002_excerpt_layer() -> ev.ChunkLayer:
    refs = [
        ev.ChunkRef(cid, tuple(("MiCA", p) for p in paths), tokens)
        for cid, paths, tokens in N002_EXCERPT
    ]
    return ev.layer_from_refs(refs, max_tokens=500, parser_version="1.4.0", sources={})
