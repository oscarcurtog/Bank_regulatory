"""Integration tests: from a real XHTML file to an Article, going through the
whole path (ET.fromstring, _preclean, _nodes_doue/_nodes_consolidado,
_clean_text, _normalize). They use the happy-path fixtures, which are the
smallest real article of each family with three real levels
(article > paragraph > point) and an exact reconstruction invariant.
"""

from __future__ import annotations

import re

import pytest

from ingest import hierarchy as h
from tests.conftest import fixture_path, reconstruct

# ---------------------------------------------------------------------------
# happy path: one per family
# ---------------------------------------------------------------------------


def test_parse_doue_happy_path():
    """DORA Art. 45, doue family: <div id="NNN.NNN"> as a real paragraph,
    points as one-row <table>s."""
    articles = h.parse(fixture_path("dora_art45"), "doue")
    assert len(articles) == 1
    a = articles[0]
    assert a.article_id == "art_45"
    assert a.path == "45"
    assert "Information-sharing" in a.title
    assert len(a.nodes) == 7
    assert a.max_depth == 3

    paths = [n.path for n in a.nodes]
    assert paths == ["45", "45.1", "45.1.a", "45.1.b", "45.1.c", "45.2", "45.3"]

    by_path = {n.path: n for n in a.nodes}
    assert by_path["45"].kind == "articulo"
    assert by_path["45"].level == 1
    assert by_path["45.1"].kind == "apartado"
    assert by_path["45.1"].marker == "1"
    assert by_path["45.1"].level == 2
    assert by_path["45.1.a"].kind == "punto"
    assert by_path["45.1.a"].marker == "a"
    assert by_path["45.1.a"].level == 3


def test_parse_consolidado_happy_path():
    """PSD2 Art. 69, consolidado family: the paragraph number in
    span.no-parag, a sibling of the list, not its parent."""
    articles = h.parse(fixture_path("psd2_art69"), "consolidado")
    assert len(articles) == 1
    a = articles[0]
    assert a.article_id == "art_69"
    assert a.path == "69"
    assert len(a.nodes) == 5
    assert a.max_depth == 3

    paths = [n.path for n in a.nodes]
    assert paths == ["69", "69.1", "69.1.a", "69.1.b", "69.2"]

    by_path = {n.path: n for n in a.nodes}
    assert by_path["69.1"].kind == "apartado"
    assert by_path["69.1.a"].kind == "punto"
    assert by_path["69.1.a"].marker == "a"


# ---------------------------------------------------------------------------
# invariants: they hold for ANY node of ANY article, not for one case
# ---------------------------------------------------------------------------

FIXTURE_FAMILIES = [
    ("dora_art45", "doue"),
    ("dora_art60", "doue"),
    ("dora_art15", "doue"),
    ("dora_art36", "doue"),
    ("psd2_art9_original", "doue"),
    ("mica_art3", "consolidado"),
    ("psd2_art69", "consolidado"),
    ("psd2_art111", "consolidado"),
    ("psd2_art9", "consolidado"),
]


def _all_articles():
    out = []
    for name, family in FIXTURE_FAMILIES:
        out.extend(h.parse(fixture_path(name), family))
    return out


def test_level_matches_path_segments():
    """Invariant (the 1.2.0 bug): level == number of path segments, for every
    node of every article, not only for the case that was reviewed by hand at
    the time."""
    for article in _all_articles():
        for node in article.nodes:
            assert node.level == node.path.count(".") + 1, (article.article_id, node.path)


_LTREE_LABEL = re.compile(r"^[0-9A-Za-z]+$|^(sub|unp)_[0-9]+$")


def test_paths_are_ltree_safe():
    """Invariant (D-004): every path segment is a valid ltree label: a marker,
    alphanumeric only, or a subparagraph label sub_<k> / unp_<k> (1.4.0), and
    nothing else with an underscore. DORA Art. 60 has a "'c" marker that would
    have broken this without the _MARKER_KEEP whitelist."""
    for article in _all_articles():
        for node in article.nodes:
            for segment in node.path.split("."):
                assert _LTREE_LABEL.match(segment), (article.article_id, node.path, segment)


def test_max_depth_matches_deepest_node():
    """Invariant (found in manual review, DORA Art. 35): max_depth must match the
    level of the deepest node that REALLY exists, not a counter that also counts
    dashes or bullets without a citable marker."""
    for article in _all_articles():
        deepest = max((n.level for n in article.nodes), default=1)
        assert article.max_depth == deepest, article.article_id


@pytest.mark.parametrize(
    "name,family",
    [
        ("dora_art45", "doue"),
        ("dora_art15", "doue"),
        ("dora_art36", "doue"),
        ("mica_art3", "consolidado"),
        ("psd2_art69", "consolidado"),
        ("psd2_art111", "consolidado"),
    ],
)
def test_article_reconstructs_from_nodes(name, family):
    """Reconstruction invariant (M-006): reinserting each marker in front of
    own_text, in node order, reproduces Article.text exactly. Two independent
    code paths (string cleaning vs. attribution over the tree) that agree are
    evidence that neither of them loses or duplicates text.

    DORA Art. 15 and Art. 36 are here since 1.4.0: up to 1.3.0 their text after
    a list was attributed to the parent and came out before the list. The two
    fixtures left out are explained: DORA Art. 60 writes its quoted markers as
    "'(a)" (marker format, M-006), and PSD2 Art. 9 keeps one reordering inside a
    point (see tests/integration/test_subparagraphs.py)."""
    article = h.parse(fixture_path(name), family)[0]
    assert reconstruct(article) == article.text
