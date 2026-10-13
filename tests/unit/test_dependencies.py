"""Anchor-conditioned dependency retrieval (ingest/dependencies.py): the glossary, the
relations an anchor yields, the bridges and the candidate list."""

from __future__ import annotations

import numpy as np

from ingest import dependencies as dp
from ingest import evaluation as ev
from ingest import hierarchy as h
from ingest import references as xr
from ingest import retrieval as rt

ACTS = {"Directive (EU) 2015/2366": "PSD2"}
TEXTS = {
    ("DORA", "2.1"): "This Regulation applies to payment institutions and account information "
    "service providers.",
    ("DORA", "3.36"): "'payment institution' means a payment institution as defined in Article 4, "
    "point (4), of Directive (EU) 2015/2366;",
    ("DORA", "3.37"): "'account information service provider' means an account information "
    "service provider as referred to in Article 33(1) of Directive (EU) 2015/2366;",
    ("DORA", "3.38"): "'ICT risk' or 'information risk' means any risk;",
    ("DORA", "5"): "The management body shall define the policy referred to in Article 6.",
    ("DORA", "6"): "Financial entities shall have an ICT risk management framework.",
    ("PSD2", "4.4"): "'payment institution' means a legal person authorised in accordance with "
    "Article 11;",
    ("PSD2", "33"): "Account information service providers shall be exempted.",
}


def _layer() -> ev.ChunkLayer:
    refs = [ev.ChunkRef(f"{a}:{p}", ((a, p),), 50) for a, p in TEXTS]
    return ev.layer_from_refs(refs, max_tokens=500, parser_version=h.PARSER_VERSION, sources={})


def _index() -> dp.DependencyIndex:
    layer = _layer()
    graph = xr.build_graph(layer, TEXTS, list(layer.chunks), ACTS)
    return dp.build_index(layer, TEXTS, graph, dp.glossary(TEXTS))


def test_the_glossary_reads_every_quoted_term_that_opens_a_definition():
    terms = {(d.act, d.path, d.term) for d in dp.glossary(TEXTS)}
    assert terms == {
        ("DORA", "3.36", "payment institution"),
        ("DORA", "3.37", "account information service provider"),
        ("DORA", "3.38", "ICT risk"),
        ("DORA", "3.38", "information risk"),
        ("PSD2", "4.4", "payment institution"),
    }


def test_terms_match_as_whole_words_in_any_case_and_in_the_plural():
    finder = dp.TermFinder(dp.glossary(TEXTS))
    found = [d.term for d in finder.find("DORA", "Payment institutions and an ICT risk register")]
    assert found == ["payment institution", "ICT risk"]
    # another act's definitions do not apply, and a term inside a longer word is not a use
    assert finder.find("MiCA", "payment institutions") == []
    assert finder.find("DORA", "prepayment institutions") == []


def test_an_anchor_yields_its_references_its_definitions_and_their_references():
    index = _index()
    kinds = {(k, d, node, t) for k, d, node, t in index.relations["DORA:2.1"]}
    assert ("definition", "payment institution", "DORA:3.36", "DORA:3.36") in kinds
    assert ("definition", "account information service provider", "DORA:3.37", "DORA:3.37") in kinds
    # definitions by reference lead, one step, into the other act
    assert ("definition_reference", "payment institution", "PSD2:4.4", "PSD2:4.4") in kinds
    assert (
        "definition_reference",
        "account information service provider",
        "PSD2:33",
        "PSD2:33",
    ) in kinds
    assert ("reference", "Article 6", "DORA:6", "DORA:6") in set(index.relations["DORA:5"])
    # a definition never points at itself
    assert all(t != "DORA:3.38" for *_, t in index.relations["DORA:3.38"])


def test_bridges_are_the_terms_closest_to_the_query():
    base = rt.Ranked("q", ("DORA:2.1", "DORA:5"), (2.0, 1.0))
    _, terms = dp.anchor_relations(base, _index(), anchors=2)
    assert [t for t, _ in terms] == ["account information service provider", "payment institution"]
    vectors = {
        dp.bridge_text("account information service provider"): np.array([0.0, 1.0]),
        dp.bridge_text("payment institution"): np.array([1.0, 0.0]),
    }
    assert dp.choose_bridges(np.array([1.0, 0.0]), terms, vectors, limit=1) == [
        "What is payment institution?"
    ]


def test_the_candidate_list_keeps_the_dense_head_and_orders_the_pool_by_probe_score():
    base = rt.Ranked("q", ("X:1", "X:2", "X:3", "X:4", "X:5"), (5.0, 4.0, 3.0, 2.0, 1.0))
    relations = [dp.Relation("X:1", 1, "reference", "Article 5", "A:5", "X:5")]
    hits = {"What is a?": ["X:4", "X:1"]}
    scores = {"X:5": 0.2, "X:4": 0.9}
    qc = dp.candidate_list(base, relations, hits, scores.__getitem__, keep=2)
    assert qc.ranked.chunk_ids == ("X:1", "X:2", "X:4", "X:5", "X:3")
    assert qc.pool_size == 4
    assert [r.kind for r in qc.provenance["X:5"]] == ["reference"]
    assert [r.kind for r in qc.provenance["X:4"]] == ["bridge"]


def test_the_structure_matrix_spreads_each_chunk_evenly_over_its_distinct_targets():
    index = _index()
    ids = list(_layer().chunks)
    w = dp.structure_matrix(index, ids)
    rows = w.sum(axis=1)
    assert all(abs(r - 1) < 1e-12 or r == 0 for r in rows)
    assert w[ids.index("DORA:5"), ids.index("DORA:6")] == 1.0
    # DORA 6 uses the defined term "ICT risk": one step to its definition
    assert w[ids.index("DORA:6"), ids.index("DORA:3.38")] == 1.0
    assert w[ids.index("PSD2:33")].sum() == 0  # no relation: a dangling node


def test_personalized_pagerank_is_the_fixed_point_of_its_equation():
    # 0 -> 1 -> 2, node 2 dangling; all reset mass on node 0
    w = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    s = np.array([1.0, 0.0, 0.0])
    p = dp.personalized_pagerank(s, w, follow=0.5)
    assert abs(p.sum() - 1) < 1e-12
    expected = (1 - 0.5) * s + 0.5 * (w.T @ p + p[2] * s)
    assert np.abs(p - expected).max() < 1e-12
    assert p[0] > p[1] > p[2] > 0


def test_a_ppr_ranking_puts_reached_chunks_first_and_keeps_the_base_order_for_the_rest():
    index = _index()
    ids = list(_layer().chunks)
    w = dp.structure_matrix(index, ids)
    base = rt.Ranked(
        "q", tuple(["DORA:5", *[c for c in ids if c != "DORA:5"]]), tuple([1.0] * len(ids))
    )
    scores = dict.fromkeys(ids, 0.5)
    ranked = dp.ppr_ranking(base, scores, w, ids, seeds=1, follow=0.5)
    # DORA 5 cites Article 6, which uses "ICT risk": two steps of mass, in that order
    reached = ("DORA:5", "DORA:6", "DORA:3.38")
    assert ranked.chunk_ids[:3] == reached
    assert list(ranked.chunk_ids[3:]) == [c for c in base.chunk_ids if c not in reached]


def test_the_paired_bootstrap_resamples_needs_and_counts_queries():
    from ingest import benchmark as bm
    from ingest import run_report as rr

    queries = [
        bm.Query.model_validate(
            {
                "query_id": q,
                "need_id": q[:4],
                "lang": "en",
                "style": "natural",
                "origin": "human",
                "text": "x",
            }
        )
        for q in ("N001-en-1", "N001-en-2", "N002-en-1")
    ]
    out = rr.paired_bootstrap({"N001-en-1": 0.5, "N001-en-2": 0.0, "N002-en-1": -0.5}, queries)
    assert out["mean"] == 0.0 and (out["better"], out["equal"], out["worse"]) == (1, 1, 1)
    assert out["ci95_low"] <= out["mean"] <= out["ci95_high"]
