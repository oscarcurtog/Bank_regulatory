"""Explicit cross-references and the one-hop expansion (ingest/references.py)."""

from __future__ import annotations

import pytest

from ingest import evaluation as ev
from ingest import hierarchy as h
from ingest import references as xr
from ingest import retrieval as rt

ACTS = {
    "Regulation (EU) 2022/2554": "DORA",
    "Regulation (EU) 2023/1114": "MiCA",
    "Directive (EU) 2015/2366": "PSD2",
}


def refs(text: str, act: str = "MiCA", path: str = "46.2") -> list[tuple[str, str, str]]:
    found, _ = xr.extract(act, path, text, ACTS)
    return [(r.form, r.act, r.path) for r in found]


def skipped(text: str) -> list[str]:
    _, skip = xr.extract("MiCA", "46.2", text, ACTS)
    return [s.reason for s in skip]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("set out in Article 46", [("article", "MiCA", "46")]),
        ("using the methodology in Article 36(11)", [("article", "MiCA", "36.11")]),
        ("as defined in Article 3(1), point (7)", [("article", "MiCA", "3.1.7")]),
        ("in Article 35(1) and (3)", [("article", "MiCA", "35.1"), ("article", "MiCA", "35.3")]),
        ("under point (a) of Article 59(1)", [("points_of_article", "MiCA", "59.1.a")]),
        (
            "points (a) to (c) of Article 2(1)",
            [("points_of_article", "MiCA", f"2.1.{x}") for x in "abc"],
        ),
        (
            "Articles 16 to 18",
            [("articles", "MiCA", "16"), ("articles", "MiCA", "17"), ("articles", "MiCA", "18")],
        ),
        (
            "set up pursuant to Articles 11 and 12 of Regulation (EU) 2022/2554",
            [("articles", "DORA", "11"), ("articles", "DORA", "12")],
        ),
        (
            "within the meaning of Article 4, point (16), of Directive (EU) 2015/2366",
            [("article", "PSD2", "4.16")],
        ),
        ("referred to in paragraph 3", [("paragraphs", "MiCA", "46.3")]),
        (
            "paragraphs 1 and 4",
            [("paragraphs", "MiCA", "46.1"), ("paragraphs", "MiCA", "46.4")],
        ),
        ("point (b) of paragraph 1", [("points_of_paragraph", "MiCA", "46.1.b")]),
        (
            "paragraph 2 of Article 47 of this Regulation",
            [("paragraphs_of_article", "MiCA", "47.2")],
        ),
        ("the first subparagraph, Title II, Chapter 3 and Annex I", []),
    ],
)
def test_each_reference_form_names_its_path(text, expected):
    assert refs(text) == expected


def test_references_come_out_in_text_order_and_once():
    text = "in accordance with point (a) of Article 59(1), Article 60 and paragraph 3"
    assert refs(text) == [
        ("points_of_article", "MiCA", "59.1.a"),
        ("article", "MiCA", "60"),
        ("paragraphs", "MiCA", "46.3"),
    ]


def test_other_acts_and_anaphora_are_not_followed():
    assert refs("in accordance with Article 16 of Regulation (EU) No 1093/2010") == []
    assert skipped("in accordance with Article 16 of Regulation (EU) No 1093/2010") == [
        "outside_the_corpus"
    ]
    assert skipped("referred to in Article 5 of that Directive") == ["anaphora"]
    assert skipped("Article 114 of the Treaty on the Functioning of the European Union") == [
        "outside_the_corpus"
    ]
    assert refs("Article 4 of Directive 2013/36/EU") == []


def _layer() -> tuple[ev.ChunkLayer, dict[tuple[str, str], str], list[str]]:
    """MiCA 46 in three chunks and 47 in one; PSD2 4 in one; MiCA 46.2 cites the others."""
    texts = {
        ("MiCA", "46"): "",
        ("MiCA", "46.1"): "Issuers shall draw up a recovery plan.",
        ("MiCA", "46.2"): "The plan referred to in paragraph 1 and Article 47 includes the "
        "conditions of Article 4, point (9), of Directive (EU) 2015/2366 and Article 99.",
        ("MiCA", "46.3"): "The authority may require changes to Article 46(7).",
        ("MiCA", "47"): "Issuers shall draw up a redemption plan.",
        ("PSD2", "4.9"): "'payment service provider' means ...",
    }
    chunks = [
        ("MiCA:46.1", [("MiCA", "46.1")]),
        ("MiCA:46.2", [("MiCA", "46.2")]),
        ("MiCA:46.3", [("MiCA", "46.3")]),
        ("MiCA:47", [("MiCA", "47")]),
        ("PSD2:4.9", [("PSD2", "4.9")]),
    ]
    layer = ev.layer_from_refs(
        [ev.ChunkRef(c, tuple(n), 50) for c, n in chunks],
        max_tokens=500,
        parser_version=h.PARSER_VERSION,
        sources={},
    )
    return layer, texts, [c for c, _ in chunks]


def test_the_graph_resolves_to_the_owning_chunks_and_to_the_nearest_ancestor():
    layer, texts, order = _layer()
    graph = xr.build_graph(layer, texts, order, ACTS)
    edges = graph.edges["MiCA:46.2"]
    assert [(e.reference.path, e.resolved_path, e.targets) for e in edges] == [
        ("46.1", "46.1", ("MiCA:46.1",)),
        ("47", "47", ("MiCA:47",)),
        ("4.9", "4.9", ("PSD2:4.9",)),
    ]
    # Article 99 does not exist: unresolved, never guessed
    assert [r.path for r in graph.unresolved] == ["99"]
    # 46(7) does not exist: its nearest ancestor, Article 46, owns three chunks
    [edge] = graph.edges["MiCA:46.3"]
    assert edge.resolved_path == "46"
    assert edge.targets == ("MiCA:46.1", "MiCA:46.2", "MiCA:46.3")


def test_expansion_places_each_anchors_targets_after_it_once_and_one_hop_only():
    layer, texts, order = _layer()
    graph = xr.build_graph(layer, texts, order, ACTS)
    base = rt.Ranked(
        "q", ("MiCA:46.2", "MiCA:46.3", "MiCA:47", "PSD2:4.9", "MiCA:46.1"), (5, 4, 3, 2, 1)
    )
    expanded, trace = xr.expand(base, graph, k=2)
    assert expanded.chunk_ids == ("MiCA:46.2", "MiCA:46.1", "MiCA:47", "PSD2:4.9", "MiCA:46.3")
    assert [(p.position, p.chunk_id, p.anchor, p.target_node) for p in trace] == [
        (2, "MiCA:46.1", "MiCA:46.2", "MiCA:46.1"),
        (3, "MiCA:47", "MiCA:46.2", "MiCA:47"),
        (4, "PSD2:4.9", "MiCA:46.2", "PSD2:4.9"),
    ]
    assert list(expanded.scores) == sorted(expanded.scores, reverse=True)
    # with one anchor that cites nothing, the base ranking is unchanged
    plain = rt.Ranked("q", ("MiCA:47", "MiCA:46.2"), (2, 1))
    assert xr.expand(plain, graph, k=1)[0].chunk_ids == ("MiCA:47", "MiCA:46.2")
    with pytest.raises(rt.RetrievalError):
        xr.expand(base, graph, k=0)
