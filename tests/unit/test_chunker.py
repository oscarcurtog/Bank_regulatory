"""Unit tests for the chunking algorithm, on small synthetic articles.

A word counter stands in for the tokenizer, so that every budget below can be
checked by hand: the expected texts are written out literally, which also pins
the rendering template without going through the module's own render functions.
The real tokenizer is used in tests/integration/test_chunker_fixtures.py.
"""

from __future__ import annotations

import json

from ingest import chunks as ch
from ingest import hierarchy as h


def words(text: str) -> int:
    return len(text.split())


def _article(path: str, title: str, nodes: list[tuple[str, str, str, str]]) -> h.Article:
    built = [
        h.Node(path=p, level=p.count(".") + 1, marker=m, kind=k, own_text=t)  # type: ignore[arg-type]
        for p, k, m, t in nodes
    ]
    text = " ".join(n.own_text for n in built if n.own_text) or "text"
    return h.Article(
        article_id=f"art_{path}",
        path=path,
        title=title,
        text=text,
        max_depth=max(n.level for n in built),
        nodes=built,
    )


# Article 7: two numbered paragraphs; paragraph 1 has an implicit first
# subparagraph (its own text and points a, b) and an explicit second one.
ART7 = _article(
    "7",
    "Test article",
    [
        ("7", "articulo", "7", ""),
        ("7.1", "apartado", "1", "Member States shall ensure that:"),
        ("7.1.a", "punto", "a", "first obligation;"),
        ("7.1.b", "punto", "b", "second obligation."),
        (
            "7.1.sub_2",
            "subparagraph",
            "2",
            "For the purposes of the first subparagraph, providers shall:",
        ),
        ("7.1.sub_2.a", "punto", "a", "keep records;"),
        ("7.1.sub_2.b", "punto", "b", "report yearly."),
        ("7.2", "apartado", "2", "Method A"),
    ],
)
HEAD7 = "T Article 7: Test article"
P1_SUBTREE = (
    "1. Member States shall ensure that:\n"
    "(a) first obligation;\n"
    "(b) second obligation.\n"
    "For the purposes of the first subparagraph, providers shall:\n"
    "(a) keep records;\n"
    "(b) report yearly."
)

# Article 8: no numbered paragraphs; its own text is its implicit first
# unnumbered paragraph, then points, then a second unnumbered paragraph.
ART8 = _article(
    "8",
    "",
    [
        ("8", "articulo", "8", "The Authority shall:"),
        ("8.a", "punto", "a", "collect data;"),
        ("8.b", "punto", "b", "publish it."),
        ("8.unp_2", "unnumbered_paragraph", "2", "It shall report every year to the Commission."),
    ],
)


def _ids(chunks: list[ch.Chunk]) -> list[str]:
    return [c.chunk_id for c in chunks]


def _by_id(chunks: list[ch.Chunk]) -> dict[str, ch.Chunk]:
    return {c.chunk_id: c for c in chunks}


# ---------------------------------------------------------------------------
# the largest unit that fits
# ---------------------------------------------------------------------------


def test_an_article_that_fits_is_one_chunk_with_its_whole_subtree():
    chunks = ch.chunk_article(ART7, "T", 1000, words)
    assert _ids(chunks) == ["T:7"]
    c = chunks[0]
    assert c.coverage == "subtree"
    assert c.source_paths == [
        "7.1",
        "7.1.a",
        "7.1.b",
        "7.1.sub_2",
        "7.1.sub_2.a",
        "7.1.sub_2.b",
        "7.2",
    ]
    assert c.context == HEAD7
    assert c.body == P1_SUBTREE + "\n2. Method A"
    assert c.text == HEAD7 + "\n\n" + P1_SUBTREE + "\n2. Method A"
    assert c.token_count == words(c.text)
    assert c.budget_exception is None


def test_an_article_that_does_not_fit_goes_down_to_its_paragraphs():
    """The article node has no own text, so there is no ':own' chunk for it."""
    p1_text = HEAD7 + "\n\n" + P1_SUBTREE
    chunks = ch.chunk_article(ART7, "T", words(p1_text), words)
    assert _ids(chunks) == ["T:7.1", "T:7.2"]
    assert _by_id(chunks)["T:7.1"].text == p1_text
    assert _by_id(chunks)["T:7.2"].text == HEAD7 + "\n\n2. Method A"


def test_a_split_parent_keeps_its_own_text_in_an_own_chunk():
    """Ownership: the parent's own text is the body of 'T:7.1:own' and of no other
    chunk; its children repeat it as context, on purpose."""
    chunks = ch.chunk_article(ART7, "T", 18, words)
    by = _by_id(chunks)
    assert _ids(chunks) == [
        "T:7.1:own",
        "T:7.1.a",
        "T:7.1.b",
        "T:7.1.sub_2:own",
        "T:7.1.sub_2.a",
        "T:7.1.sub_2.b",
        "T:7.2",
    ]
    own = by["T:7.1:own"]
    assert own.coverage == "own"
    assert own.source_paths == ["7.1"]
    assert own.body == "1. Member States shall ensure that:"
    assert by["T:7.1.a"].context == HEAD7 + "\n\n1. Member States shall ensure that:"


def test_implicit_first_subparagraph_is_not_context_for_a_later_one():
    """7.1's own text IS its first subparagraph. For 7.1.sub_2 and anything under
    it, that text is a sibling: it is never inherited. The lead-in of sub_2 is."""
    by = _by_id(ch.chunk_article(ART7, "T", 18, words))
    assert by["T:7.1.sub_2:own"].context == HEAD7
    assert by["T:7.1.sub_2.a"].context == (
        HEAD7 + "\n\nFor the purposes of the first subparagraph, providers shall:"
    )
    assert "Member States shall ensure" not in by["T:7.1.sub_2.a"].text


def test_implicit_first_unnumbered_paragraph_is_not_context_for_a_later_one():
    by = _by_id(ch.chunk_article(ART8, "T", 11, words))
    assert list(by) == ["T:8:own", "T:8.a", "T:8.b", "T:8.unp_2"]
    assert by["T:8.a"].context == "T Article 8\n\nThe Authority shall:"
    assert by["T:8.unp_2"].context == "T Article 8"


def test_a_small_structural_chunk_is_valid():
    """No minimum size and no sibling packing in v1."""
    tiny = _by_id(ch.chunk_article(ART7, "T", 18, words))["T:7.2"]
    assert tiny.body == "2. Method A"
    assert tiny.token_count == 8


# ---------------------------------------------------------------------------
# budget exceptions: never silent
# ---------------------------------------------------------------------------


def test_a_leaf_over_budget_is_kept_whole_and_flagged():
    chunks = ch.chunk_article(ART7, "T", 6, words)
    leaf = _by_id(chunks)["T:7.1.a"]
    assert leaf.coverage == "subtree"
    assert leaf.token_count > 6
    assert leaf.budget_exception == "leaf_over_budget"


def test_an_own_text_over_budget_is_flagged():
    own = _by_id(ch.chunk_article(ART7, "T", 6, words))["T:7.1:own"]
    assert own.token_count > 6
    assert own.budget_exception == "own_over_budget"


def test_no_chunk_is_over_budget_without_an_exception():
    for max_tokens in (1, 6, 8, 11, 18, 20, 1000):
        for c in ch.chunk_article(ART7, "T", max_tokens, words):
            assert (c.token_count > max_tokens) == (c.budget_exception is not None)


# ---------------------------------------------------------------------------
# ownership, determinism, configuration
# ---------------------------------------------------------------------------


def test_every_own_text_is_owned_exactly_once_at_any_budget():
    sources = [n.path for n in ART7.nodes if n.own_text]
    for max_tokens in (1, 6, 11, 18, 20, 1000):
        owned = [p for c in ch.chunk_article(ART7, "T", max_tokens, words) for p in c.source_paths]
        assert sorted(owned) == sorted(sources)


def test_chunking_is_deterministic():
    first = [c.model_dump(mode="json") for c in ch.chunk_article(ART7, "T", 18, words)]
    second = [c.model_dump(mode="json") for c in ch.chunk_article(ART7, "T", 18, words)]
    assert json.dumps(first) == json.dumps(second)


def test_max_tokens_is_configuration_not_version():
    """Two budgets, two different results, the same chunker version."""
    version = ch.CHUNKER_VERSION
    assert _ids(ch.chunk_article(ART7, "T", 1000, words)) != _ids(
        ch.chunk_article(ART7, "T", 18, words)
    )
    assert ch.CHUNKER_VERSION == version


def test_the_heading_has_no_colon_without_a_title():
    assert ch.article_heading("T", ART8) == "T Article 8"
    assert ch.article_heading("T", ART7) == HEAD7


def test_governing_ancestors_skip_the_sibling_first_subparagraph():
    assert [n.path for n in ch.governing_ancestors(ART7, "7.1.a")] == ["7.1"]
    assert [n.path for n in ch.governing_ancestors(ART7, "7.1.sub_2")] == []
    assert [n.path for n in ch.governing_ancestors(ART7, "7.1.sub_2.b")] == ["7.1.sub_2"]
    assert [n.path for n in ch.governing_ancestors(ART8, "8.unp_2")] == []
    assert [n.path for n in ch.governing_ancestors(ART8, "8.b")] == ["8"]
