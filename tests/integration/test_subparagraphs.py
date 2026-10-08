"""Integration tests for the subparagraph structure (parser 1.4.0), on real
EUR-Lex excerpts (tests/fixtures/, see PROVENANCE.md).

The rule they pin is the one measured in T2 phase 1 on the five raw documents:
every text block at paragraph or article level opens a subparagraph, points
belong to the last block before them, and subparagraph 1 stays implicit. The
expectations below are the structure that measurement approved, including its
accepted limitations: the "Method X" headings of PSD2 Art. 9(1) become
subparagraphs of their own, and nothing is split inside a point.
"""

from __future__ import annotations

import json

import pytest

from ingest import hierarchy as h
from tests.conftest import fixture_path, same_words

NEW_KINDS = ("subparagraph", "unnumbered_paragraph")

PSD2_ART9_PATHS = [
    "9",
    "9.1",
    "9.1.sub_2",
    "9.1.sub_3",
    "9.1.sub_4",
    "9.1.sub_5",
    "9.1.sub_5.a",
    "9.1.sub_5.b",
    "9.1.sub_5.c",
    "9.1.sub_5.d",
    "9.1.sub_5.e",
    "9.1.sub_6",
    "9.1.sub_7",
    "9.1.sub_7.a",
    "9.1.sub_7.a.i",
    "9.1.sub_7.a.ii",
    "9.1.sub_7.a.iii",
    "9.1.sub_7.a.iv",
    "9.1.sub_7.b",
    "9.1.sub_7.b.i",
    "9.1.sub_7.b.ii",
    "9.1.sub_7.b.iii",
    "9.1.sub_7.b.iv",
    "9.1.sub_7.b.v",
    "9.2",
    "9.2.a",
    "9.2.b",
    "9.3",
]

PSD2_ART9 = [("psd2_art9", "consolidado"), ("psd2_art9_original", "doue")]


def _parse(name: str, family: str) -> h.Article:
    return h.parse(fixture_path(name), family)[0]


def _by_path(article: h.Article) -> dict[str, h.Node]:
    return {n.path: n for n in article.nodes}


def _children(article: h.Article, path: str) -> list[str]:
    depth = path.count(".") + 1
    return [
        n.path
        for n in article.nodes
        if n.path.startswith(path + ".") and n.path.count(".") == depth
    ]


# ---------------------------------------------------------------------------
# PSD2 Art. 9: the case that motivated the change, in both markup families
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,family", PSD2_ART9)
def test_psd2_art9_structure(name, family):
    """Method B and Method C are two lists inside paragraph 1: each lives in its
    own subparagraph, so no path repeats. Node order is document order."""
    article = _parse(name, family)
    assert [n.path for n in article.nodes] == PSD2_ART9_PATHS
    assert article.max_depth == 5


@pytest.mark.parametrize("name,family", PSD2_ART9)
def test_psd2_art9_methods_are_subparagraphs(name, family):
    """The three "Method" headings are text blocks of their own in the markup, so
    the structural rule makes each one a subparagraph (an accepted false positive
    of T2 phase 1: no rule reads the word "Method")."""
    by = _by_path(_parse(name, family))
    for path, heading in (("9.1.sub_2", "A"), ("9.1.sub_4", "B"), ("9.1.sub_6", "C")):
        assert by[path].own_text == f"Method {heading}"
    for k in range(2, 8):
        node = by[f"9.1.sub_{k}"]
        assert node.kind == "subparagraph"
        assert node.marker == str(k)
        assert node.level == 3
    assert by["9.1.sub_3"].own_text.startswith(
        "The payment institution's own funds shall amount to at least 10 %"
    )
    # paragraph 1 keeps only its own first subparagraph, the lead-in
    assert by["9.1"].own_text.endswith("in accordance with national legislation:")


@pytest.mark.parametrize("name,family", PSD2_ART9)
def test_psd2_art9_points_and_subpoints_are_separated(name, family):
    """Up to 1.3.0 the two copies of 9.1.a shared one fused text (Method B (a)
    followed by Method C (a)). Now each point owns only its text."""
    article = _parse(name, family)
    by = _by_path(article)
    assert by["9.1.sub_5.a"].own_text == "4,0 % of the slice of PV up to EUR 5 million; plus"
    assert _children(article, "9.1.sub_5") == [f"9.1.sub_5.{m}" for m in "abcde"]
    assert _children(article, "9.1.sub_5.a") == []
    assert by["9.1.sub_7.a"].own_text.startswith(
        "The relevant indicator is the sum of the following:"
    )
    assert _children(article, "9.1.sub_7") == ["9.1.sub_7.a", "9.1.sub_7.b"]
    assert _children(article, "9.1.sub_7.a") == [
        f"9.1.sub_7.a.{m}" for m in ("i", "ii", "iii", "iv")
    ]
    assert _children(article, "9.1.sub_7.b") == [
        f"9.1.sub_7.b.{m}" for m in ("i", "ii", "iii", "iv", "v")
    ]
    # Known, accepted limitation: inside a point nothing is split, so the closing
    # text of Method C (a), after its sub-points, stays in the point's own_text.
    assert "Each element shall be included in the sum" in by["9.1.sub_7.a"].own_text


def test_psd2_art9_original_and_consolidated_agree():
    """The Official Journal text writes each Method as a table with an empty first
    cell; the consolidated text writes it as blocks. With unmarked items walked as
    transparent containers, both families give exactly the same nodes."""
    consolidated = _parse("psd2_art9", "consolidado")
    original = _parse("psd2_art9_original", "doue")

    def nodes(a):
        return [(n.path, n.level, n.marker, n.kind, n.own_text) for n in a.nodes]

    assert nodes(original) == nodes(consolidated)


# ---------------------------------------------------------------------------
# DORA Art. 36: a second list in a later subparagraph, without a lettering clash
# ---------------------------------------------------------------------------

DORA_ART36_PATHS = [
    "36",
    "36.1",
    "36.1.a",
    "36.1.b",
    "36.1.sub_2",
    "36.1.sub_2.i",
    "36.1.sub_2.ii",
    "36.1.sub_2.iii",
    "36.1.sub_2.iv",
    "36.2",
    "36.2.sub_2",
    "36.2.sub_2.a",
    "36.2.sub_2.b",
    "36.2.sub_2.c",
    "36.2.sub_2.d",
    "36.2.sub_2.e",
    "36.3",
    "36.3.a",
    "36.3.b",
    "36.3.sub_2",
]


def test_dora_art36_structure():
    article = _parse("dora_art36", "doue")
    assert [n.path for n in article.nodes] == DORA_ART36_PATHS
    assert article.max_depth == 4


def test_dora_art36_first_subparagraph_keeps_its_paths():
    """The points of the first subparagraph keep their 1.3.0 paths and text: the
    implicit subparagraph 1 is what keeps the common case unchanged."""
    by = _by_path(_parse("dora_art36", "doue"))
    assert by["36.1.a"].own_text == "in Article 35(1), point (a); and"
    assert by["36.1.b"].own_text.startswith("in Article 35(1), point (b), in accordance with")
    assert by["36.1"].own_text.startswith("When oversight objectives cannot be attained")


def test_dora_art36_second_subparagraph_owns_its_points():
    """ "The powers referred to in the first subparagraph..." is the second
    subparagraph of 36(1) (the text says so itself), and its points (i) to (iv)
    hang from it instead of from the paragraph."""
    article = _parse("dora_art36", "doue")
    by = _by_path(article)
    sub = by["36.1.sub_2"]
    assert sub.kind == "subparagraph"
    assert sub.marker == "2"
    assert sub.own_text.startswith("The powers referred to in the first subparagraph")
    assert _children(article, "36.1.sub_2") == [f"36.1.sub_2.{m}" for m in ("i", "ii", "iii", "iv")]
    assert "36.1.i" not in by


# ---------------------------------------------------------------------------
# DORA Art. 15: unnumbered paragraphs of an article without numbered paragraphs
# ---------------------------------------------------------------------------


def test_dora_art15_unnumbered_paragraphs():
    """An article without numbered paragraphs is divided into unnumbered
    paragraphs. The first one (the lead-in and its points (a) to (g)) stays
    implicit; the three that follow the list become unp_2..unp_4."""
    article = _parse("dora_art15", "doue")
    by = _by_path(article)
    assert [n.path for n in article.nodes] == (
        ["15"] + [f"15.{m}" for m in "abcdefg"] + ["15.unp_2", "15.unp_3", "15.unp_4"]
    )
    for k in (2, 3, 4):
        node = by[f"15.unp_{k}"]
        assert node.kind == "unnumbered_paragraph"
        assert node.marker == str(k)
        assert node.level == 2
    assert by["15"].own_text.startswith("The ESAs shall, through the Joint Committee")
    assert by["15.unp_2"].own_text.startswith(
        "When developing those draft regulatory technical standards"
    )


# ---------------------------------------------------------------------------
# regressions: articles without more than one block per paragraph do not change
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,family",
    [
        ("dora_art45", "doue"),
        ("dora_art60", "doue"),
        ("psd2_art69", "consolidado"),
        ("psd2_art111", "consolidado"),
    ],
)
def test_articles_without_subparagraphs_are_unchanged(name, family):
    """None of these has a second block in a paragraph or article, so the rule adds
    nothing (their exact 1.3.0 paths are pinned in test_parse_end_to_end.py and
    test_corpus.py). DORA Art. 60 and PSD2 Art. 111 are amending articles with
    quoted paragraphs inside points: inside a point nothing is split."""
    article = _parse(name, family)
    assert not [n.path for n in article.nodes if n.kind in NEW_KINDS]


# ---------------------------------------------------------------------------
# invariants
# ---------------------------------------------------------------------------

ALL_FIXTURES = [
    ("dora_art45", "doue"),
    ("dora_art60", "doue"),
    ("dora_art15", "doue"),
    ("dora_art36", "doue"),
    ("psd2_art9_original", "doue"),
    ("psd2_art69", "consolidado"),
    ("psd2_art111", "consolidado"),
    ("psd2_art9", "consolidado"),
]


@pytest.mark.parametrize("name,family", ALL_FIXTURES)
def test_paths_are_unique_and_parents_come_first(name, family):
    """Every path appears once, and every node comes after its parent: node order
    is document order, a pre-order of the tree."""
    article = _parse(name, family)
    paths = [n.path for n in article.nodes]
    assert len(paths) == len(set(paths))
    for i, path in enumerate(paths[1:], start=1):
        assert path.rsplit(".", 1)[0] in paths[:i], path


@pytest.mark.parametrize("name,family", ALL_FIXTURES)
def test_parse_is_deterministic(name, family):
    """Same input, same output, byte for byte once serialized."""
    first = [a.model_dump(mode="json") for a in h.parse(fixture_path(name), family)]
    second = [a.model_dump(mode="json") for a in h.parse(fixture_path(name), family)]
    assert json.dumps(first, ensure_ascii=False) == json.dumps(second, ensure_ascii=False)


@pytest.mark.parametrize("name,family", PSD2_ART9 + [("dora_art15", "doue")])
def test_round_trip_with_subparagraphs(name, family):
    """An article with the new kinds survives model -> JSON -> model unchanged,
    validators included."""
    article = _parse(name, family)
    as_json = json.loads(json.dumps(article.model_dump(mode="json"), ensure_ascii=False))
    assert h.Article.model_validate(as_json) == article


@pytest.mark.parametrize(
    "name,family",
    [f for f in ALL_FIXTURES if f[0] != "dora_art60"],
)
def test_no_source_text_lost_or_duplicated(name, family):
    """The words of the nodes, markers reinserted, are exactly the words of
    Article.text: nothing lost, nothing repeated, order aside. PSD2 Art. 9 passes
    too, although its order is not exact (the known limitation inside a point).
    DORA Art. 60 is left out because it writes its quoted markers as "'(a)"
    (marker format, M-006), which changes the words, not the text."""
    assert same_words(_parse(name, family))
