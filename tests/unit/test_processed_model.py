"""Unit tests for the processed layer contract (EF-3, contract 2): the
ProcessedDocument, Article and Node models in isolation, without touching disk.

The tests whose names start with "test_accepts_" are the important ones to
read: they pin what the schema must NOT reject, with the real number that backs
it. One constraint too many is as bad as one too few, because it turns a
correct parse into a hard failure.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ingest.hierarchy import Article, Node
from ingest.processed import ProcessedDocument

VALID_NODE = {
    "path": "45.1.a",
    "level": 3,
    "marker": "a",
    "kind": "punto",
    "own_text": "aims to enhance the digital operational resilience of financial entities;",
}

VALID_ARTICLE = {
    "article_id": "art_45",
    "path": "45",
    "title": "Information-sharing arrangements",
    "text": "Article 45 Information-sharing arrangements 1. Financial entities may exchange.",
    "max_depth": 3,
    "nodes": [
        {"path": "45", "level": 1, "marker": "45", "kind": "articulo", "own_text": ""},
        VALID_NODE,
    ],
}

VALID_DOC = {
    "source_celex": "32022R2554",
    "source_role": "original",
    "source_sha256": "cf50d8f023d23b2a56f535aaee0760eba7f1925ef08a9858c5c31d43d414f46b",
    "parser_version": "1.3.0",
    "family": "doue",
    "articles": [VALID_ARTICLE],
}


def _node(**overrides):
    return {**VALID_NODE, **overrides}


def _article(**overrides):
    return {**VALID_ARTICLE, **overrides}


def _doc(**overrides):
    return {**VALID_DOC, **overrides}


# ---------------------------------------------------------------------------
# what the schema must ACCEPT, and the number that backs it
# ---------------------------------------------------------------------------


def test_accepts_empty_own_text():
    """267 of the 3,379 real nodes have an empty own_text: the article is
    almost always a pure container (M-006). A min_length=1 here would reject 8%
    of the corpus."""
    assert Node.model_validate(_node(own_text="")).own_text == ""


def test_accepts_empty_title():
    """_article_title returns "" when an article has no title element. It does
    not happen in any of the 331 today, but it is a legitimate branch of the
    producer: requiring a title would turn a correct parse into a failure."""
    assert Article.model_validate(_article(title="")).title == ""


@pytest.mark.parametrize(
    "level,kind",
    [(2, "punto"), (3, "apartado"), (4, "apartado")],
)
def test_accepts_kind_and_level_that_do_not_correspond(level, kind):
    """kind and level are DIFFERENT axes. PSD2 Art. 52 is a numbered list of
    points at level 2; DORA Art. 60 has quoted paragraphs at levels 3 and 4. A
    "paragraph if and only if level 2" rule would reject 248 real nodes."""
    node = Node.model_validate(
        _node(level=level, kind=kind, path="52.1" if level == 2 else "60.1.a.3")
    )
    assert node.kind == kind


def test_accepts_duplicate_paths_within_an_article():
    """PSD2 Art. 9 has 9.1.a and 9.1.b twice each, because Method B and Method
    C restart the lettering (D-004, open point). Uniqueness CANNOT live in the
    schema: it would make it impossible to load the corpus."""
    dup = _article(
        article_id="art_9",
        path="9",
        nodes=[
            {"path": "9", "level": 1, "marker": "9", "kind": "articulo", "own_text": ""},
            {"path": "9.1.a", "level": 3, "marker": "a", "kind": "punto", "own_text": "Method B"},
            {"path": "9.1.a", "level": 3, "marker": "a", "kind": "punto", "own_text": "Method C"},
        ],
    )
    assert len(Article.model_validate(dup).nodes) == 3


def test_accepts_level_that_disagrees_with_the_path():
    """level == path segments is a DERIVED relation and its place is
    tests/integration/test_parse_end_to_end.py, not the schema. Here it passes
    on purpose, so that it is written down where that check lives."""
    assert Node.model_validate(_node(path="45.1.a", level=99)).level == 99


# ---------------------------------------------------------------------------
# Node: what it does reject
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["seccion", "Articulo", "", "chunk"])
def test_invalid_node_kind_is_rejected(kind):
    with pytest.raises(ValidationError) as excinfo:
        Node.model_validate(_node(kind=kind))
    assert "kind" in str(excinfo.value)


def test_empty_marker_is_rejected():
    """0 of the 3,379 nodes have an empty marker, and the parser already guards
    with `if marker:` before creating the node: a node without a marker is not
    citable."""
    with pytest.raises(ValidationError) as excinfo:
        Node.model_validate(_node(marker=""))
    assert "marker" in str(excinfo.value)


@pytest.mark.parametrize("path", ["45..1", "45.'c", "45 1", ".45", "45.", ""])
def test_path_that_is_not_a_valid_ltree_label_is_rejected(path):
    """D-004: every path segment has to be a valid label. The "'c" marker of
    DORA Art. 60 is the real case that motivated the whitelist."""
    with pytest.raises(ValidationError) as excinfo:
        Node.model_validate(_node(path=path))
    assert "path" in str(excinfo.value)


@pytest.mark.parametrize("level", [0, -1])
def test_non_positive_level_is_rejected(level):
    with pytest.raises(ValidationError) as excinfo:
        Node.model_validate(_node(level=level))
    assert "level" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Article
# ---------------------------------------------------------------------------


def test_empty_text_is_rejected():
    """Empty text is not an article: it is a failed parse. The real minimum
    observed is 71 characters."""
    with pytest.raises(ValidationError) as excinfo:
        Article.model_validate(_article(text=""))
    assert "text" in str(excinfo.value)


@pytest.mark.parametrize("article_id", ["45", "article_45", "ART_45", ""])
def test_article_id_without_the_art_prefix_is_rejected(article_id):
    with pytest.raises(ValidationError) as excinfo:
        Article.model_validate(_article(article_id=article_id))
    assert "article_id" in str(excinfo.value)


def test_max_depth_below_one_is_rejected():
    with pytest.raises(ValidationError) as excinfo:
        Article.model_validate(_article(max_depth=0))
    assert "max_depth" in str(excinfo.value)


def test_a_bad_node_invalidates_the_whole_article():
    """Nested structure: the article is not half-built if one of its nodes
    breaks the contract."""
    with pytest.raises(ValidationError) as excinfo:
        Article.model_validate(_article(nodes=[VALID_NODE, _node(kind="invented")]))
    assert "kind" in str(excinfo.value)


# ---------------------------------------------------------------------------
# ProcessedDocument
# ---------------------------------------------------------------------------


def test_valid_document_is_accepted():
    doc = ProcessedDocument.model_validate(VALID_DOC)
    assert doc.family == "doue"
    assert len(doc.articles) == 1
    assert doc.articles[0].nodes[0].kind == "articulo"


@pytest.mark.parametrize("family", ["consolidada", "DOUE", "", "xhtml"])
def test_invalid_family_is_rejected(family):
    """hierarchy.parse already raises ValueError for an unknown family: the
    schema's closed set is backed by the code as well."""
    with pytest.raises(ValidationError) as excinfo:
        ProcessedDocument.model_validate(_doc(family=family))
    assert "family" in str(excinfo.value)


@pytest.mark.parametrize("version", ["1.3", "v1.3.0", "1.3.0-dev", "", "one"])
def test_malformed_parser_version_is_rejected(version):
    with pytest.raises(ValidationError) as excinfo:
        ProcessedDocument.model_validate(_doc(parser_version=version))
    assert "parser_version" in str(excinfo.value)


def test_document_without_articles_is_rejected():
    with pytest.raises(ValidationError) as excinfo:
        ProcessedDocument.model_validate(_doc(articles=[]))
    assert "articles" in str(excinfo.value)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("source_celex", "celex:32022R2554"),
        ("source_sha256", "cf50d8f0"),
        ("source_role", "consolidate"),
    ],
)
def test_provenance_fields_reuse_the_manifest_constraints(field, bad_value):
    """The three provenance fields are a literal quote of the manifest and use
    THE SAME constraints as FetchResult, so that the two contracts cannot
    disagree."""
    with pytest.raises(ValidationError) as excinfo:
        ProcessedDocument.model_validate(_doc(**{field: bad_value}))
    assert field in str(excinfo.value)


# ---------------------------------------------------------------------------
# round-trip
# ---------------------------------------------------------------------------


def test_round_trip_preserves_shape_and_field_order():
    """model -> JSON -> model, and the JSON comes out with the same fields in
    the same order as the file already on disk. Order matters because it is
    what keeps data/processed/*.json identical byte for byte."""
    doc = ProcessedDocument.model_validate(VALID_DOC)
    as_json = doc.model_dump(mode="json")
    assert as_json == VALID_DOC
    assert list(as_json) == list(VALID_DOC)
    assert list(as_json["articles"][0]) == list(VALID_ARTICLE)
    assert list(as_json["articles"][0]["nodes"][0]) == list(VALID_NODE)
    assert ProcessedDocument.model_validate(as_json) == doc
