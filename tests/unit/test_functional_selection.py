"""C's expanded population and legal-function set selection (ingest/functional_selection.py):
the hop rules, the two-hop limit, inherited-context visibility, the window prompt, the parsing of
the model's answer and the fixed merge."""

from __future__ import annotations

from typing import Any

from ingest import dependencies as dp
from ingest import evaluation as ev
from ingest import functional_selection as fs
from ingest import hierarchy as h
from ingest import necessity as nc
from ingest import references as xr
from ingest import retrieval as rt

TEXTS = {
    ("DORA", "3.1"): "'ICT risk' means any risk;",
    ("DORA", "5"): "Financial entities shall comply with Article 8.",
    ("DORA", "8"): "Entities shall keep an ICT risk register in accordance with Article 9.",
    ("DORA", "9"): "The register shall follow Article 10.",
    ("DORA", "10"): "Records shall be kept.",
    ("DORA", "11"): "An unrelated provision.",
}


def _setup() -> tuple[nc.Structure, rt.Ranked, dict[str, Any]]:
    refs = [ev.ChunkRef(f"{a}:{p}", ((a, p),), 50) for a, p in TEXTS]
    layer = ev.layer_from_refs(refs, max_tokens=500, parser_version=h.PARSER_VERSION, sources={})
    graph = xr.build_graph(layer, TEXTS, list(layer.chunks), {})
    definitions = dp.glossary(TEXTS)
    index = dp.build_index(layer, TEXTS, graph, definitions)
    article = {f"{a}:{p}": f"{a}:{p.split('.')[0]}" for a, p in TEXTS}
    structure = nc.Structure(TEXTS, layer, graph, index, tuple(definitions), article)
    order = ("DORA:5", "DORA:11", "DORA:3.1", "DORA:8", "DORA:9", "DORA:10")
    e5 = rt.Ranked("q", order, tuple(float(6 - i) for i in range(6)))
    relations, _ = dp.anchor_relations(e5, index, anchors=1)
    qc = dp.candidate_list(e5, relations, {}, lambda c: 0.0, keep=2)
    record = {
        "query_id": "q",
        "candidates": {c: [r.__dict__ for r in rels] for c, rels in qc.provenance.items()},
    }
    return structure, e5, record


def _population(question: str = "Do entities face ICT risk rules?") -> dict[str, fs.Member]:
    structure, e5, record = _setup()
    members = fs.expanded_population(
        e5,
        e5,
        record,
        structure,
        question,
        dp.TermFinder(structure.definitions),
        {},
        head=2,
        anchors_n=1,
    )
    return {m.chunk: m for m in members}


def test_hops_follow_the_anchor_its_references_and_the_terms_and_stop_at_two():
    pop = _population()
    assert {c: m.hop for c, m in pop.items()} == {
        "DORA:5": 0,  # E5's first 2 chunks
        "DORA:11": 0,
        "DORA:8": 1,  # the anchor's reference
        "DORA:3.1": 1,  # a term the question uses ("ICT risk"), also used by DORA 8
        "DORA:9": 2,  # DORA 8's reference: the second hop
    }
    assert "DORA:10" not in pop  # a third hop is never taken
    assert pop["DORA:3.1"].routes[0].kind == "question_term"
    assert {r.kind for r in pop["DORA:3.1"].routes} == {"question_term", "definition"}
    assert pop["DORA:9"].routes[0].source == "DORA:8"
    assert "that provision" in pop["DORA:8"].routes[0].relation  # not "the anchor"


def test_members_follow_the_candidate_generators_order():
    structure, e5, record = _setup()
    members = fs.expanded_population(
        e5,
        e5,
        record,
        structure,
        "x",
        dp.TermFinder(structure.definitions),
        {},
        head=2,
        anchors_n=1,
    )
    assert [m.position for m in members] == sorted(m.position for m in members)


def test_visibility_reads_the_inherited_context_within_an_article():
    contexts = {"X:1": "", "X:1.a": "X Article 1\n\nThe lead-in:", "Y:1.a": "The lead-in:"}
    owned = {"X:1": [("X", "1")], "X:1.a": [("X", "1.a")], "Y:1.a": [("Y", "1.a")]}
    node_text = {("X", "1"): "The lead-in:", ("X", "1.a"): "point a", ("Y", "1.a"): "other"}
    article = {"X:1": "X:art_1", "X:1.a": "X:art_1", "Y:1.a": "Y:art_1"}
    assert fs.visibility(contexts, owned, node_text, article) == {"X:1": ["X:1.a"]}


def test_the_window_prompt_asks_the_counterfactual_question_and_shortens_clauses():
    route = fs.Route("DORA:5", "reference", 1, "reference: that provision cites it", "w " * 200)
    member = fs.Member("DORA:8", 1, 3, (route,), ("DORA:8.a",))
    messages = fs.window_prompt("Which rule applies?", [member], {"DORA:8": "DORA Article 8"})
    user = messages[1]["content"]
    assert user.startswith("QUESTION: Which rule applies?")
    assert "if this candidate were removed" in user
    assert (
        "[1] DORA Article 8\nFound from DORA 5 (reference: that provision cites it; hop 1)" in user
    )
    clause = user.split("clause: ")[1].split("\n")[0]
    assert clause.endswith("...") and len(clause) <= fs.CLAUSE_CHARS + 3
    assert "heading context of: DORA 8.a" in user
    for banned in ("qrel", "grade", "gold"):
        assert banned not in (messages[0]["content"] + user).lower()


def test_judgments_are_read_line_by_line_in_the_required_form():
    out = (
        "[1] governing rule | yes\n[2] Merely related | No\n[3] condition | necessary: yes\n"
        "[2] exception | yes\n[9] condition | yes\n[4] a new function | yes\nnoise"
    )
    parsed = fs.parse_judgments(out, n=4)
    assert parsed[1] == fs.Judgment("governing rule", True)
    assert parsed[2] == fs.Judgment("merely related", False)  # the first line wins
    assert parsed[3] == fs.Judgment("condition", True)
    assert parsed[4] == fs.Judgment("unparsed", True)
    assert 9 not in parsed


def test_the_merge_covers_functions_in_turn_and_stops_at_the_limit():
    members = [fs.Member(f"C:{i}", 0, i, (), ()) for i in range(1, 8)]
    judged = {
        "C:1": fs.Judgment("governing rule", True),
        "C:2": fs.Judgment("governing rule", True),
        "C:3": fs.Judgment("condition", True),
        "C:4": fs.Judgment("legal category or definition", True),
        "C:5": fs.Judgment("merely related", True),
        "C:6": fs.Judgment("exception", False),
        "C:7": fs.Judgment("unparsed", True),
    }
    assert fs.merge(judged, members) == ["C:1", "C:4", "C:3", "C:2"]
    assert fs.merge(judged, members, limit=2) == ["C:1", "C:4"]
    e5 = rt.Ranked("q", ("C:5", "C:1", "C:2"), (3.0, 2.0, 1.0))
    assert fs.evidence_ranking(e5, ["C:2"]).chunk_ids == ("C:2", "C:5", "C:1")
