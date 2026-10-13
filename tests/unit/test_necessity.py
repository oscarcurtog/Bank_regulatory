"""Anchor-conditioned necessity scoring and evidence sets (ingest/necessity.py): the citation
contexts, the relation labels, the population built from R1's provenance, the three inputs,
the evidence sets and the discrimination measures."""

from __future__ import annotations

import math

import pytest

from ingest import benchmark as bm
from ingest import dependencies as dp
from ingest import evaluation as ev
from ingest import hierarchy as h
from ingest import necessity as nc
from ingest import references as xr
from ingest import retrieval as rt

ACTS = {"Directive (EU) 2015/2366": "PSD2"}
TEXTS = {
    ("DORA", "2"): "This Regulation applies to payment institutions.",
    ("DORA", "3.36"): "'payment institution' means a payment institution as defined in Article 4, "
    "point (4), of Directive (EU) 2015/2366;",
    ("DORA", "3.38"): "'ICT risk' means any risk;",
    ("DORA", "6"): "Financial entities shall manage risk.",
    ("DORA", "6.1"): "Financial entities shall have an ICT risk management framework.",
    ("DORA", "6.2"): "The framework referred to in paragraph 1 shall be reviewed yearly.",
    ("DORA", "7"): "Payment institutions shall, subject to Article 6, keep records. By way of "
    "derogation from Article 2, microenterprises may keep fewer records.",
    ("PSD2", "4.4"): "'payment institution' means a legal person authorised in accordance with "
    "Article 11;",
    ("PSD2", "11"): "Member States shall require undertakings to obtain authorisation.",
}
CHUNKS = tuple(f"{a}:{p}" for a, p in TEXTS)


def _structure() -> nc.Structure:
    refs = [ev.ChunkRef(f"{a}:{p}", ((a, p),), 50) for a, p in TEXTS]
    layer = ev.layer_from_refs(refs, max_tokens=500, parser_version=h.PARSER_VERSION, sources={})
    graph = xr.build_graph(layer, TEXTS, list(layer.chunks), ACTS)
    definitions = dp.glossary(TEXTS)
    index = dp.build_index(layer, TEXTS, graph, definitions)
    article = {f"{a}:{p}": f"{a}:{p.split('.')[0]}" for a, p in TEXTS}
    return nc.Structure(TEXTS, layer, graph, index, tuple(definitions), article)


def _e5() -> rt.Ranked:
    order = (
        *("DORA:7", "DORA:6.2", "DORA:3.38", "DORA:2", "DORA:3.36", "DORA:6", "DORA:6.1"),
        *("PSD2:4.4", "PSD2:11"),
    )
    return rt.Ranked("q", order, tuple(float(len(order) - i) for i in range(len(order))))


def _record(structure: nc.Structure) -> dict[str, object]:
    """R1's provenance for the query, as scripts/dependency_retrieval.py records it."""
    base = _e5()
    relations, _ = dp.anchor_relations(base, structure.index, anchors=10)
    hits = {"What is payment institution?": ["PSD2:4.4", "DORA:3.36"]}
    qc = dp.candidate_list(base, relations, hits, lambda c: 0.0, keep=3)
    return {
        "query_id": "q",
        "bridges": qc.bridges,
        "candidates": {c: [r.__dict__ for r in rels] for c, rels in qc.provenance.items()},
    }


def _population() -> dict[tuple[str, str], nc.Triple]:
    structure = _structure()
    triples = nc.population(_e5(), _record(structure), structure, anchors=2, outside=3)
    return {(t.anchor, t.candidate): t for t in triples}


def test_the_citation_context_is_the_clause_that_holds_the_mention():
    text = "First clause; second clause with Article 6 here. Third sentence."
    start = text.index("Article 6")
    lo, clause = nc.segment_at(text, start, start + len("Article 6"))
    assert clause == "second clause with Article 6 here."
    assert text[lo:].startswith("second clause")


def test_a_long_clause_is_cut_around_the_mention_at_spaces_and_says_so():
    text = "alpha " * 100 + "Article 6" + " omega" * 100
    start = text.index("Article 6")
    _, clause = nc.segment_at(text, start, start + len("Article 6"), limit=100)
    assert clause.startswith("...") and clause.endswith("...") and "Article 6" in clause
    assert len(clause) <= 100 + 6
    assert all(w in ("alpha", "omega", "Article", "6") for w in clause.strip(".").split())


def test_exception_and_condition_cues_are_read_only_inside_the_clause():
    for text, expected in [
        ("Financial entities shall, subject to Article 6, keep records.", "condition"),
        ("By way of derogation from Article 2, they may keep fewer.", "exception"),
        ("Notwithstanding Article 2, they may keep fewer.", "exception"),
        ("The records referred to in Article 6 shall be kept.", None),
        ("Subject to the rules. Article 6 applies.", None),  # the cue is in another clause
    ]:
        start = text.index("Article")
        lo, _ = nc.segment_at(text, start, start + 9)
        assert nc.cue(text, start, lo) == expected, text


def test_the_population_links_each_anchor_to_its_candidates_with_typed_relations():
    pop = _population()
    # anchors are E5's first 2; candidates outside E5's first 3; one triple per pair
    assert set(pop) == {
        ("DORA:7", "DORA:2"),
        ("DORA:7", "DORA:3.36"),
        ("DORA:7", "DORA:6"),
        ("DORA:7", "DORA:6.1"),
        ("DORA:7", "PSD2:4.4"),
        ("DORA:6.2", "DORA:6.1"),
    }
    condition = pop[("DORA:7", "DORA:6")]
    assert condition.kinds == ("reference",)
    assert condition.relation == "condition: the anchor cites the candidate as a condition"
    assert condition.context == '"Payment institutions shall, subject to Article 6, keep records."'
    assert pop[("DORA:7", "DORA:2")].relation.startswith("exception:")
    hierarchy = pop[("DORA:6.2", "DORA:6.1")]
    assert hierarchy.relation.endswith("(hierarchy: same article)")
    definition = pop[("DORA:7", "DORA:3.36")]
    assert definition.kinds == ("definition", "bridge")
    assert "the candidate defines 'payment institution'" in definition.relation
    # a definition by reference into another act, also reached by the bridge
    cross = pop[("DORA:7", "PSD2:4.4")]
    assert cross.kinds == ("definition_reference", "bridge")
    assert "definition by reference" in cross.relation and "(cross-act)" in cross.relation
    assert "Directive (EU) 2015/2366" in cross.context
    assert [t.anchor_rank for t in pop.values()] == [1, 1, 1, 1, 1, 2]


def test_the_population_refuses_a_relation_the_structure_does_not_hold():
    structure = _structure()
    record = _record(structure)
    candidates = record["candidates"]
    assert isinstance(candidates, dict)
    candidates["DORA:6"] = [
        {
            "anchor": "DORA:7",
            "anchor_rank": 1,
            "kind": "reference",
            "detail": "Article 99",
            "target_node": "DORA:99",
            "candidate": "DORA:6",
        }
    ]
    with pytest.raises(nc.NecessityError):
        nc.population(_e5(), record, structure, anchors=2, outside=3)


def test_the_three_inputs_add_the_anchor_then_the_relation_and_name_no_label():
    triple = nc.Triple("q", "A:1", 1, "B:2", ("reference",), "condition: x", '"subject to"')
    text = {"A:1": "Anchor text", "B:2": "Candidate text"}
    a1, a2, a3 = (nc.model_input(v, "Which rule applies?", triple, text) for v in nc.VARIANTS)
    assert a1[1] == "QUESTION: Which rule applies?"
    assert a2[1] == "QUESTION: Which rule applies?\nANCHOR: Anchor text"
    assert a3[1] == a2[1] + '\nRELATION: condition: x\nCITATION CONTEXT: "subject to"'
    assert a1[2] == a2[2] == a3[2] == "CANDIDATE: Candidate text"
    assert a2[0] == a3[0] != a1[0]
    # the words the inputs add around the texts name no label nor what is sought
    added = " ".join(nc.INSTRUCTIONS.values()) + " QUESTION ANCHOR RELATION CITATION CONTEXT"
    for banned in ("qrel", "grade", "gold", "relevant", "answer"):
        assert banned not in (added + " CANDIDATE").lower()


def test_p_yes_is_the_logistic_of_the_logit_difference_without_overflow():
    assert nc.p_yes(0.0) == 0.5
    assert abs(nc.p_yes(2.0) - 1 / (1 + math.exp(-2.0))) < 1e-15
    assert nc.p_yes(-1000.0) == 0.0 and nc.p_yes(1000.0) == 1.0


def test_an_evidence_set_keeps_the_head_adds_the_supports_then_follows_e5():
    e5 = rt.Ranked("q", ("X:1", "X:2", "X:3", "X:4", "Y:1", "Y:2", "Y:3"), (7, 6, 5, 4, 3, 2, 1))
    probability = {
        ("X:1", "Y:1"): 0.72,
        ("X:2", "Y:2"): 0.95,
        ("X:1", "Y:2"): 0.60,
        ("X:3", "Y:3"): 0.99,  # its anchor is not kept with a head of 2
        ("X:2", "Y:3"): 0.50,
    }
    ranked, chosen = nc.evidence_set(e5, probability, head=2, supports=2, threshold=0.70)
    assert ranked.chunk_ids == ("X:1", "X:2", "Y:2", "Y:1", "X:3", "X:4", "Y:3")
    assert [(s.candidate, s.anchor) for s in chosen] == [("Y:2", "X:2"), ("Y:1", "X:1")]
    # at most k supports, and nothing below the threshold
    ranked, chosen = nc.evidence_set(e5, probability, head=2, supports=1, threshold=0.96)
    assert chosen == [] and ranked.chunk_ids == e5.chunk_ids
    ranked, chosen = nc.evidence_set(e5, probability, head=2, supports=1, threshold=0.5)
    assert [s.candidate for s in chosen] == ["Y:2"]


def test_average_precision_and_auroc_on_known_lists():
    assert nc.average_precision([True, False, True]) == pytest.approx((1 + 2 / 3) / 2)
    assert nc.auroc([0.9, 0.8, 0.7, 0.6], [True, False, True, False]) == 0.75
    assert nc.auroc([0.5, 0.5], [True, False]) == 0.5
    with pytest.raises(nc.NecessityError):
        nc.average_precision([False, False])


def test_discrimination_scores_a_candidate_by_its_best_anchor():
    queries = [
        bm.Query.model_validate(
            {
                "query_id": "N001-en-1",
                "need_id": "N001",
                "lang": "en",
                "style": "natural",
                "origin": "human",
                "text": "x",
            }
        )
    ]
    q = "N001-en-1"
    triples = {
        q: [
            nc.Triple(q, "A:1", 1, "C:1", ("definition",), "", ""),
            nc.Triple(q, "A:1", 1, "C:2", ("definition",), "", ""),
            nc.Triple(q, "A:2", 2, "C:2", ("reference",), "", ""),
            nc.Triple(q, "A:2", 2, "C:3", ("reference",), "", ""),
        ]
    }
    score = {(q, "A:1", "C:1"): 2.0, (q, "A:1", "C:2"): -1.0, (q, "A:2", "C:2"): 3.0}
    score |= {(q, "A:2", "C:3"): 0.0}
    grades = {q: {"C:2": 2, "C:3": 1}}
    out = nc.discrimination(
        triples,
        score,
        grades,
        missing={q: frozenset({"C:2", "C:9"})},
        dependents={q: frozenset({"C:2"})},
        queries=queries,
        k=1,
    )
    # by best score: C:2 (3.0), C:1 (2.0), C:3 (0.0): the only relevant one is first
    assert out["average_precision"] == 1.0
    # anchor A:1 ranks C:1 first (wrong), A:2 ranks C:2 first (right)
    assert out["precision_at_1"] == 0.5 and out["anchor_groups_with_a_relevant_candidate"] == 2
    assert out["support_recall@1"] == 0.5 and out["support_recall_ceiling"] == 0.5
    assert out["dependent_support_recall@1"] == 1.0
    assert out["probability_by_grade"]["2"]["candidates"] == 1
    assert out["probability_by_grade"]["0 or unjudged"]["candidates"] == 1


def test_a_query_without_candidates_still_counts_in_the_support_recalls():
    def query(qid: str) -> bm.Query:
        return bm.Query.model_validate(
            {
                "query_id": qid,
                "need_id": qid[:4],
                "lang": "en",
                "style": "natural",
                "origin": "human",
                "text": "x",
            }
        )

    queries = [query("N001-en-1"), query("N002-en-1")]
    triples = {
        "N001-en-1": [
            nc.Triple("N001-en-1", "A:1", 1, "C:1", ("definition",), "", ""),
            nc.Triple("N001-en-1", "A:1", 1, "C:2", ("reference",), "", ""),
        ]
    }
    out = nc.discrimination(
        triples,
        {("N001-en-1", "A:1", "C:1"): 1.0, ("N001-en-1", "A:1", "C:2"): -1.0},
        {"N001-en-1": {"C:1": 2}, "N002-en-1": {"C:9": 2}},
        missing={"N001-en-1": frozenset({"C:1"}), "N002-en-1": frozenset({"C:9"})},
        dependents={"N001-en-1": frozenset({"C:1"}), "N002-en-1": frozenset({"C:9"})},
        queries=queries,
        k=4,
    )
    # N002 has evidence to recover and no candidate: a miss, not an absence
    assert out["support_recall@4"] == 0.5 and out["support_recall_ceiling"] == 0.5
    assert out["dependent_support_recall@4"] == 0.5
    assert out["average_precision"] == 1.0 and out["queries_measured"] == 1
