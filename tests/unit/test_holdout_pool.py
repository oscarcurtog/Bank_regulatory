"""The exposure pool of the design of 2026-10-22 and exposure v2 of D-066
(config/holdout/evidence-discovery.md), as functions: what a search received in full and what
its search pages named, each search's exposure v2 and the pool of three searches, the audit
search's new nodes, the batches of judge B's pool variant, its answer's contract, the candidates
of judge D and the diagnostic, D-061's review metrics, and the controller's outer deadline.
Synthetic data only: no corpus, no model, no TEST text."""

from __future__ import annotations

from typing import Any

import pytest

from ingest import benchmark as bm
from ingest import holdout_admission as ha
from ingest import holdout_controller as hct
from ingest import holdout_judging as hj
from ingest import holdout_packets as hp
from ingest import ir_metrics

TEXTS: dict[bm.NodeKey, str] = {
    ("MiCA", "3.1"): "a definition",
    ("MiCA", "3.2"): "another definition",
    ("MiCA", "60.1"): "a duty",
    ("MiCA", "60.2"): "its exception",
    ("DORA", "19.1"): "a report",
    ("PSD2", "3.1"): "an excluded article",
    ("PSD2", "4.1"): "a definition",
    ("MiCA", "60"): "",  # a structural parent: its text is all in its children
}
EXCLUDED = frozenset({"3"})


def entry(op: str, *nodes: str) -> dict[str, Any]:
    return {"n": 1, "op": op, "args": {}, "page": 1, "pages": 1, "chars": 1, "nodes": list(nodes)}


def file(*nodes: tuple[str, int], answerability: str = "complete") -> dict[str, Any]:
    return {
        "candidates": [
            {
                "node": n,
                "scope": [],
                "grade": g,
                "ambiguous": False,
                "scope_basis": [n] if g == 0 else [],
                "rationale": f"label of {n}",
            }
            for n, g in nodes
        ],
        "answerability": answerability,
        "answerability_reason": "why",
        "gap_note": None,
    }


# ---------------------------------------------------------------------------
# exposure and the pool
# ---------------------------------------------------------------------------


def test_only_article_and_nodes_pages_are_a_full_reading():
    log = [
        entry("titles"),
        entry("search", "MiCA:3.1", "MiCA:60.1"),  # snippets only
        entry("article", "MiCA:60.1", "MiCA:60.2"),
        entry("nodes", "DORA:19.1"),
        entry("article", "MiCA:60.2"),
    ]
    assert hj.exposure(log) == ["DORA:19.1", "MiCA:60.1", "MiCA:60.2"]
    assert hj.exposure([entry("search", "MiCA:3.1")]) == []  # a snippet alone enters nothing


def test_an_exposed_unlisted_node_enters_the_pool_and_a_listed_one_enters_unread():
    files = {
        "search_1": file(("MiCA:60.1", 3)),
        "search_2": file(("MiCA:60.1", 2)),
        "search_3": file(("DORA:19.1", 1)),  # listed, never read in full: in the pool too
    }
    exposures = {
        "search_1": ["MiCA:60.1", "MiCA:60.2"],
        "search_2": ["MiCA:3.1", "PSD2:3.1", "MiCA:99.9", "MiCA:60"],  # excluded; none; empty
        "search_3": [],
    }
    found = hj.exposure_pool(files, exposures, {}, TEXTS, EXCLUDED)
    assert found.listed == ["MiCA:60.1", "DORA:19.1"]
    assert found.unlisted == ["MiCA:3.1", "MiCA:60.2"]
    assert found.pool == ["DORA:19.1", "MiCA:3.1", "MiCA:60.1", "MiCA:60.2"]
    assert "DORA:19.1" not in found.exposed_by  # listed without being read in full
    assert found.first["MiCA:60.1"]["grade"] == 3  # search_1's, before search_2's
    # the same inputs, the same pool, in any run
    assert hj.exposure_pool(files, exposures, {}, TEXTS, EXCLUDED) == found


def test_the_audit_nodes_are_split_by_the_normal_pool():
    audit = file(("MiCA:60.1", 1), ("PSD2:4.1", 2))
    nodes = hj.audit_nodes(
        audit,
        ["MiCA:3.2", "MiCA:60.2", "PSD2:3.1"],
        [],
        ["MiCA:60.1", "MiCA:60.2"],
        TEXTS,
        EXCLUDED,
    )
    assert list(nodes.listed) == ["PSD2:4.1"]  # listed, outside the pool: its label first
    assert nodes.exposed == ["MiCA:3.2"]  # received in full, outside the pool, not listed
    assert list(nodes.on_pool) == ["MiCA:60.1"]  # a label on a pool node: provenance only


# ---------------------------------------------------------------------------
# exposure v2, rule A (D-066): snippets followed up and the ancestor closure, by the corpus's
# structure alone; no article is expanded
# ---------------------------------------------------------------------------

# what exposure v2 reads of a corpus: an article whose own node has a lead-in and one whose
# text is all in its paragraphs, a paragraph whose text is all in its points, a point with
# subpoints, articles 1 and 10 of one act, article 19 in two acts, and an excluded PSD2 article
V2: dict[bm.NodeKey, str] = {
    ("DORA", "1"): "the first article",
    ("DORA", "1.1"): "its paragraph",
    ("DORA", "10"): "the tenth article's lead-in",
    ("DORA", "10.1"): "its paragraph",
    ("DORA", "19"): "an article's lead-in",
    ("DORA", "19.1"): "a report",
    ("DORA", "19.2"): "",
    ("DORA", "19.2.a"): "a first point, with subpoints:",
    ("DORA", "19.2.a.i"): "a subpoint",
    ("DORA", "19.2.b"): "a second point",
    ("DORA", "20"): "",
    ("DORA", "20.1"): "a template",
    ("DORA", "20.2"): "its review",
    ("MiCA", "19"): "another act's article 19",
    ("MiCA", "19.1"): "its paragraph",
    ("PSD2", "3"): "an excluded article's lead-in",
    ("PSD2", "3.1"): "an excluded paragraph",
    ("PSD2", "4"): "a definitions article's lead-in",
    ("PSD2", "4.1"): "a definition",
    ("PSD2", "4.2"): "another definition",
}


def expected(listed=(), full=(), named=()) -> set[str]:
    """Rule A computed here, not by hj.reach: the nodes listed, the nodes received in full or
    named that can be judged, and the ancestors that can be judged of all of them."""

    def ok(node: str) -> bool:
        act, _, path = node.partition(":")
        text = V2.get((act, path))
        return bool(text and text.strip()) and not (act == "PSD2" and path.split(".")[0] == "3")

    found = set(listed) | {n for n in [*full, *named] if ok(n)}
    for node in list(found):
        act, _, path = node.partition(":")
        parts = path.split(".")
        found |= {
            f"{act}:{'.'.join(parts[:i])}"
            for i in range(1, len(parts))
            if ok(f"{act}:{'.'.join(parts[:i])}")
        }
    return found


def test_v2_a_a_node_a_search_page_named_is_followed_up_alone():
    log = [entry("search", "DORA:20.1", "PSD2:3.1"), entry("titles"), entry("nodes", "DORA:19.1")]
    assert hj.snippets(log) == ["DORA:20.1", "PSD2:3.1"]  # a search page's ids, and only those
    assert hj.exposure(log) == ["DORA:19.1"]  # a snippet is still no full reading
    # followed up: the node named, never its article's other nodes nor an excluded node
    assert hj.reach([], [], hj.snippets(log), V2, EXCLUDED) == {"DORA:20.1": ["snippet_follow_up"]}


def test_v2_an_unknown_or_repeated_snippet_id_enters_once_or_never():
    log = [
        entry("search", "DORA:20.1", "DORA:99.9"),  # a node the corpus does not hold
        entry("search", "DORA:20.1"),  # the same node on another page
    ]
    assert hj.snippets(log) == ["DORA:20.1", "DORA:99.9"]
    found = hj.reach([], [], hj.snippets(log), V2, EXCLUDED)
    assert found == {"DORA:20.1": ["snippet_follow_up"]}  # once; the unknown id, never
    assert found == {
        n: hj.reach([], [], [n], V2, EXCLUDED)[n] for n in expected(named=["DORA:20.1"])
    }
    # the same snippet seen by two searches: one pool node, reached by both
    files = {s: file(("DORA:20.2", 2)) for s in ("search_1", "search_2", "search_3")}
    named = {"search_1": ["DORA:20.1"], "search_3": ["DORA:20.1", "DORA:20.1"]}
    pool = hj.exposure_pool(files, {}, named, V2, EXCLUDED)
    assert pool.pool == ["DORA:20.1", "DORA:20.2"]
    assert pool.reached["DORA:20.1"] == {
        "search_1": ["snippet_follow_up"],
        "search_3": ["snippet_follow_up"],
    }


def test_v2_b_c_an_ancestor_with_its_own_text_is_a_candidate_one_without_is_not():
    found = hj.reach([], ["DORA:19.2.b"], [], V2, EXCLUDED)
    assert found == {"DORA:19": ["ancestor_closure"], "DORA:19.2.b": ["full_text"]}
    assert "DORA:19.2" not in found  # its text is all in its points: nothing to judge
    assert "DORA:20" not in hj.reach([], [], ["DORA:20.1"], V2, EXCLUDED)


def test_v2_every_ancestor_level_with_its_own_text_enters():
    found = hj.reach([], [], ["DORA:19.2.a.i"], V2, EXCLUDED)
    assert found == {
        "DORA:19": ["ancestor_closure"],
        "DORA:19.2.a": ["ancestor_closure"],
        "DORA:19.2.a.i": ["snippet_follow_up"],
    }
    assert set(found) == expected(named=["DORA:19.2.a.i"])
    assert hj.ancestors("DORA:19.2.a.i") == ["DORA:19", "DORA:19.2", "DORA:19.2.a"]


def test_v2_ancestors_are_whole_path_segments_of_the_same_act():
    # DORA:10.1's article is 10, never 1
    assert hj.reach([], ["DORA:10.1"], [], V2, EXCLUDED) == {
        "DORA:10": ["ancestor_closure"],
        "DORA:10.1": ["full_text"],
    }
    assert hj.ancestors("DORA:10.1") == ["DORA:10"]
    # article 19 of MiCA brings MiCA's article 19, never DORA's
    found = hj.reach([], [], ["MiCA:19.1"], V2, EXCLUDED)
    assert found == {"MiCA:19": ["ancestor_closure"], "MiCA:19.1": ["snippet_follow_up"]}
    assert set(found) == expected(named=["MiCA:19.1"])


def test_v2_d_an_excluded_ancestor_never_enters():
    # a node of an excluded PSD2 article brings neither its lead-in nor its article
    assert hj.reach([], ["PSD2:3.1"], ["PSD2:3.1"], V2, EXCLUDED) == {}
    assert hj.reach(["PSD2:3.1"], [], [], V2, EXCLUDED) == {"PSD2:3.1": ["listed"]}
    # the same structure outside the excluded articles: the lead-in enters
    found = hj.reach([], [], ["PSD2:4.1"], V2, EXCLUDED)
    assert found == {"PSD2:4": ["ancestor_closure"], "PSD2:4.1": ["snippet_follow_up"]}


def test_v2_e_f_no_article_is_expanded_by_any_path():
    # a listed paragraph brings its lead-in, never its siblings
    assert hj.reach(["DORA:19.1"], [], [], V2, EXCLUDED) == {
        "DORA:19": ["ancestor_closure"],
        "DORA:19.1": ["listed"],
    }
    # an article's own node, listed, received in full or named, is one node like any other
    for found in (
        hj.reach(["DORA:19"], [], [], V2, EXCLUDED),
        hj.reach([], ["DORA:19"], [], V2, EXCLUDED),
        hj.reach([], [], ["DORA:19"], V2, EXCLUDED),
    ):
        assert set(found) == {"DORA:19"}
    # received in full or named: never the article's other nodes
    found = hj.reach([], ["DORA:20.1"], ["DORA:19.1"], V2, EXCLUDED)
    assert set(found) == expected(full=["DORA:20.1"], named=["DORA:19.1"])
    assert not {"DORA:20.2", "DORA:19.2.a", "DORA:19.2.b"} & set(found)


def test_v2_g_a_node_reached_by_several_paths_and_searches_enters_once():
    files = {
        "search_1": file(("DORA:19.1", 0)),
        "search_2": file(("DORA:20.1", 3)),
        "search_3": file(("DORA:20.1", 3)),
    }
    exposures = {"search_2": ["DORA:19.1", "DORA:19.2.a.i"]}
    named = {"search_3": ["DORA:19.1", "DORA:19.2.a.i"]}
    found = hj.exposure_pool(files, exposures, named, V2, EXCLUDED)
    assert set(found.pool) == expected(
        ["DORA:19.1", "DORA:20.1"], exposures["search_2"], named["search_3"]
    )
    assert found.pool == ["DORA:19", "DORA:19.1", "DORA:19.2.a", "DORA:19.2.a.i", "DORA:20.1"]
    assert found.reached["DORA:19.1"] == {
        "search_1": ["listed"],
        "search_2": ["full_text"],
        "search_3": ["snippet_follow_up"],
    }
    assert found.unlisted == ["DORA:19", "DORA:19.2.a", "DORA:19.2.a.i"]
    record = found.record()
    assert record["exposure_policy"] == "exposure.v2"
    assert record["paths"]["DORA:19.1"] == ["listed", "full_text", "snippet_follow_up"]
    assert record["reached_by"]["DORA:19.1"] == ["search_1", "search_2", "search_3"]
    assert record["by_path"] == {
        "listed": 2,
        "full_text": 2,
        "snippet_follow_up": 2,
        "ancestor_closure": 2,
    }
    assert record["beyond_full_text"] == 2  # DORA:19 and DORA:19.2.a


def test_v2_the_pool_and_its_record_do_not_depend_on_any_order():
    files = {
        "search_1": file(("DORA:19.1", 2), ("DORA:20.1", 1)),
        "search_2": file(("DORA:20.1", 2), ("DORA:19.1", 1)),
        "search_3": file(("PSD2:4.1", 1)),
    }
    exposures = {"search_1": ["DORA:19.2.a.i", "DORA:10.1"], "search_2": ["MiCA:19.1"]}
    named = {"search_3": ["PSD2:4.2", "DORA:1.1"]}
    one = hj.exposure_pool(files, exposures, named, V2, EXCLUDED)
    shuffled = hj.exposure_pool(
        files,
        {k: list(reversed(v)) for k, v in reversed(exposures.items())},
        {k: list(reversed(v)) for k, v in named.items()},
        V2,
        EXCLUDED,
    )
    assert shuffled.pool == one.pool == sorted(one.pool)
    assert shuffled.reached == one.reached and shuffled.record()["paths"] == one.record()["paths"]
    for listed, full, nm in ((["DORA:19.1"], ["DORA:10.1"], ["MiCA:19.1"]),):
        a = hj.reach(listed, full, nm, V2, EXCLUDED)
        b = hj.reach(list(reversed(listed)), list(reversed(full)), list(reversed(nm)), V2, EXCLUDED)
        assert list(a) == list(b) == sorted(a)


def test_v2_h_i_the_audit_search_is_measured_by_the_same_rules_against_the_closed_pool():
    files = {stage: file(("DORA:20.1", 2)) for stage in ("search_1", "search_2", "search_3")}
    pool = hj.exposure_pool(files, {"search_2": ["DORA:19.2.b"]}, {}, V2, EXCLUDED).pool
    assert pool == ["DORA:19", "DORA:19.2.b", "DORA:20.1"]  # DORA:19 by the ancestor closure
    audit = file(("DORA:19.1", 1), ("DORA:20.1", 2))
    nodes = hj.audit_nodes(audit, ["DORA:19.2.a.i"], ["PSD2:4.1", "DORA:19"], pool, V2, EXCLUDED)
    assert list(nodes.listed) == ["DORA:19.1"] and list(nodes.on_pool) == ["DORA:20.1"]
    # its own exposure v2 outside the closed normal pool: DORA:19 is in the pool already
    assert nodes.exposed == ["DORA:19.2.a", "DORA:19.2.a.i", "PSD2:4", "PSD2:4.1"]
    assert nodes.sources == {
        "DORA:19.1": ["listed"],
        "DORA:19.2.a": ["ancestor_closure"],
        "DORA:19.2.a.i": ["full_text"],
        "PSD2:4": ["ancestor_closure"],
        "PSD2:4.1": ["snippet_follow_up"],
    }


def test_v2_the_audit_reach_is_exactly_a_normal_search_reach():
    """Parity: the audit search's exposure v2 is reach() itself, so that against an empty pool
    every node it reaches is new, with the paths a normal search would get, and against the
    pool of three searches that did exactly what it did, nothing is new."""
    listed, full, named = ["DORA:19.1", "PSD2:4.1"], ["DORA:10.1", "PSD2:3.1"], ["MiCA:19.1"]
    audit = file(*[(n, 1) for n in listed])
    alone = hj.audit_nodes(audit, full, named, [], V2, EXCLUDED)
    assert alone.sources == hj.reach(listed, full, named, V2, EXCLUDED)
    assert set(alone.sources) == expected(listed, full, named)
    same = {s: file(*[(n, 1) for n in listed]) for s in ("search_1", "search_2", "search_3")}
    pool = hj.exposure_pool(
        same, {s: full for s in same}, {s: named for s in same}, V2, EXCLUDED
    ).pool
    again = hj.audit_nodes(audit, full, named, pool, V2, EXCLUDED)
    assert again.listed == {} and again.exposed == [] and again.sources == {}


# ---------------------------------------------------------------------------
# the batches of judge B's pool variant
# ---------------------------------------------------------------------------


def renderer(sizes: dict[str, int]):
    """A packet whose token count is a fixed part plus each node's size."""
    return lambda batch: " ".join(["x"] * 10 + [n for n in batch for _ in range(sizes.get(n, 1))])


def words(text: str) -> int:
    return len(text.split())


def test_a_batch_holds_at_most_34_nodes_a_ceiling_not_a_target():
    nodes = [f"MiCA:{i}.1" for i in range(1, 71)]
    batches = hp.pool_batches(nodes, renderer({}), words, max_tokens=10**6)
    assert [len(b) for b in batches] == [34, 34, 2]
    assert hp.pool_batches(nodes[:5], renderer({}), words, max_tokens=10**6) == [nodes[:5]]
    assert hp.POOL_BATCH_MAX_NODES == 34 and hp.POOL_BATCH_MAX_TOKENS == 9773


def test_the_token_preflight_closes_a_batch_earlier_and_never_splits_a_node():
    nodes = [f"MiCA:{i}.1" for i in range(1, 11)]
    sizes = {n: 30 for n in nodes}
    batches = hp.pool_batches(nodes, renderer(sizes), words, max_tokens=10 + 30 * 3)
    assert [len(b) for b in batches] == [3, 3, 3, 1]
    assert [n for b in batches for n in b] == nodes  # every node once, in its order
    for batch in batches:
        assert words(renderer(sizes)(batch)) <= 10 + 30 * 3
    # a node that alone passes the ceiling is no packet the protocol allows
    with pytest.raises(hp.PacketError, match="one node alone"):
        hp.pool_batches(["MiCA:1.1"], renderer({"MiCA:1.1": 500}), words, max_tokens=100)


def test_the_batches_are_deterministic():
    nodes = [f"DORA:{i}.1" for i in range(1, 50)]
    sizes = {n: i % 7 + 1 for i, n in enumerate(nodes)}
    one = hp.pool_batches(nodes, renderer(sizes), words, max_tokens=60)
    assert one == hp.pool_batches(list(nodes), renderer(dict(sizes)), words, max_tokens=60)


# ---------------------------------------------------------------------------
# judge B's pool variant: its answer and its routing
# ---------------------------------------------------------------------------

IDS = {"C-" + "1" * 12: "MiCA:3.1", "C-" + "2" * 12: "MiCA:60.2"}


def grades(*labels: tuple[str, Any, bool]) -> dict[str, Any]:
    return {
        "need_id": "R-x",
        "grades": [
            {
                "item_id": i,
                "grade": g,
                "ambiguous": a,
                "scope_basis": [IDS[i]] if g == 0 else [],
                "reason": "quotes the node",
            }
            for i, g, a in labels
        ],
    }


ONE, TWO = "C-" + "1" * 12, "C-" + "2" * 12


def test_the_pool_variant_answers_every_node_by_its_id_and_no_answerability():
    good = grades((ONE, 0, False), (TWO, 2, False))
    assert ha.pool_problems(good, IDS) == []  # the exact set, each explicit: accepted
    assert hj.pool_judgments(good, IDS)["MiCA:60.2"]["grade"] == 2
    missing = grades((ONE, 0, False))
    assert any("missing" in p for p in ha.pool_problems(missing, IDS))
    unknown = grades((ONE, 0, False), ("C-" + "0" * 12, 1, False))
    assert any("unknown" in p for p in ha.pool_problems(unknown, IDS))
    named = grades(("C-" + "1" * 12, 0, False), ("C-" + "2" * 12, 2, False))
    named["grades"][1]["target"] = "MiCA:61.2"
    assert any("identity" in p for p in ha.pool_problems(named, IDS))
    no_basis = grades(("C-" + "1" * 12, 0, False), ("C-" + "2" * 12, 2, False))
    no_basis["grades"][0]["scope_basis"] = []
    assert ha.pool_problems(no_basis, IDS)  # a grade 0 names its scope, as judge B's


def test_the_exact_coverage_contract_refuses_every_answer_short_of_one_explicit_label_each():
    """The person's decision of 2026-10-22: an absent judgment is never a negative one."""
    null = grades((ONE, None, False), (TWO, 2, False))
    assert any("grade not 0 to 3" in p for p in ha.pool_problems(null, IDS))
    twice = grades((ONE, 0, False), (TWO, 2, False), (TWO, 2, False))
    assert any("named twice" in p for p in ha.pool_problems(twice, IDS))
    silent = grades((ONE, 0, False), (TWO, 2, False))
    del silent["grades"][1]["ambiguous"]
    assert any("ambiguous not given" in p for p in ha.pool_problems(silent, IDS))
    replaced = grades((ONE, 0, False), (TWO, 2, False))
    replaced["grades"][1]["item_id"] = "C-" + "3" * 12  # another id in place of the shown one
    found = ha.pool_problems(replaced, IDS)
    assert any("unknown" in p for p in found) and any("missing" in p for p in found)
    assert ha.pool_problems({"need_id": "R-x", "grades": []}, IDS)  # nothing answered


@pytest.mark.parametrize(
    ("label", "route"),
    [
        ({"grade": 0, "ambiguous": False}, "final_negative"),  # an explicit grade 0
        ({"grade": 1, "ambiguous": False}, "second_judgment"),
        ({"grade": 3, "ambiguous": False}, "second_judgment"),
        ({"grade": 0, "ambiguous": True}, "second_judgment"),
    ],
)
def test_an_explicit_label_routes_its_node(label, route):
    assert hj.pool_route(label) == route


def test_an_unjudged_label_is_never_read_as_a_grade_0():
    with pytest.raises(ValueError, match="never a grade 0"):
        hj.pool_route({"grade": None, "ambiguous": False})
    with pytest.raises(ValueError):
        hj.pool_route({})


def test_the_two_blind_judgments_agree_or_judge_c_decides():
    one = {"grade": 1, "ambiguous": False, "rationale": "P1"}
    two = {"grade": 0, "ambiguous": False, "scope_basis": ["MiCA:60.2"], "rationale": "P2"}
    a, b = {"MiCA:60.2": one}, {"MiCA:60.2": two}
    with pytest.raises(ValueError, match="disputes"):
        hj.assemble_audit("N101", ["MiCA:60.2"], a, b, {}, {("MiCA", "60.2"): "x"})
    decided = hj.assemble_audit(
        "N101", ["MiCA:60.2"], a, b, {"MiCA:60.2": dict(two)}, {("MiCA", "60.2"): "x"}
    )
    assert decided.adjudications[0].c is not None and decided.qrels[0].grade == 0
    agreed = hj.assemble_audit(
        "N101", ["MiCA:60.2"], a, {"MiCA:60.2": dict(one)}, {}, {("MiCA", "60.2"): "x"}
    )
    assert agreed.adjudications[0].c is None and agreed.qrels[0].grade == 1


# ---------------------------------------------------------------------------
# judge D's and the diagnostic's candidates
# ---------------------------------------------------------------------------


def test_judge_d_and_the_diagnostic_receive_the_exact_union():
    listed = ["DORA:19.1", "MiCA:60.1"]
    pool_labels: dict[str, list[dict[str, Any]]] = {
        "MiCA:3.1": [{"grade": 0, "ambiguous": False}],  # an explicit negative: out
        # a positive first judgment, then judge C leaves it unjudged: in, by the first
        "MiCA:3.2": [{"grade": 2}, {"grade": 0}, {"grade": None, "ambiguous": False}],
        "MiCA:60.2": [{"grade": 1, "ambiguous": False}, {"grade": 0, "ambiguous": False}],
        "PSD2:4.1": [{"grade": 0, "ambiguous": True}, {"grade": 0, "ambiguous": True}],
    }
    final = {"DORA:19.1": 3, "MiCA:60.1": 0, "MiCA:60.2": 0, "PSD2:4.1": 0}
    assert hj.diagnostic_set(listed, pool_labels, final) == [
        "DORA:19.1",
        "MiCA:3.2",
        "MiCA:60.1",
        "MiCA:60.2",
        "PSD2:4.1",
    ]
    # a final grade >= 1 alone brings a pool node in, and a listed node needs nothing else
    assert "MiCA:3.1" in hj.diagnostic_set([], {"MiCA:3.1": []}, {"MiCA:3.1": 1})
    assert hj.diagnostic_set(["MiCA:60.1", "MiCA:60.1"], {}, {}) == ["MiCA:60.1"]


# ---------------------------------------------------------------------------
# D-061's review metrics, defined operationally (D-066): measurements, never a gate
# ---------------------------------------------------------------------------


def test_the_review_metrics_count_exactly_what_they_define():
    needs = [
        # search_3 alone reaches a grade-2 node: the third search's increment
        {
            "reached_by": {"A": ["search_1", "search_2", "search_3"], "B": ["search_3"]},
            "grades": {"A": 3, "B": 2},
        },
        # search_3 reaches a grade-2 node search_1 reached already: no increment
        {
            "reached_by": {"C": ["search_1", "search_3"], "D": ["search_2"]},
            "grades": {"C": 2, "D": 1},
        },
        # no node of grade >= 2: in the third search's denominator, not in recall's
        {"reached_by": {"E": ["search_1"]}, "grades": {"E": 1}},
    ]
    found = hj.review_metrics(needs)
    third = found["third_search_incremental_g2"]
    assert (third["numerator"], third["denominator"]) == (1, 3)
    assert third["rate"] == pytest.approx(1 / 3)
    assert third["review_flag"] is True and third["review_flag_threshold"] == 0.07
    recall = found["per_search_recall_g2"]
    # need 1: search_1 1/2, search_2 1/2, search_3 2/2; need 2: 1/1, 0/1, 1/1
    assert recall["denominator"] == 6 and recall["needs_without_grade_2"] == 1
    assert recall["mean"] == pytest.approx(4 / 6) and recall["median"] == pytest.approx(0.75)
    assert recall["min"] == 0 and recall["below_reference"] == 3 and recall["reference"] == 0.90
    none = hj.review_metrics([])
    assert none["third_search_incremental_g2"]["rate"] is None
    assert none["third_search_incremental_g2"]["review_flag"] is False
    assert none["per_search_recall_g2"]["mean"] is None


def test_the_audit_review_counts_new_nodes_by_final_grade_and_first_path():
    entries = [
        {
            "additional_final_grades": {"A": 0, "B": 1, "C": 2, "F": 1},
            "sources": {
                "A": ["snippet_follow_up"],
                "B": ["ancestor_closure"],
                "C": ["listed", "full_text"],
                "F": ["listed_article_closure"],  # a path no category names: unknown
            },
        },
        # judge C left D unjudged (never a 0); E has no recorded path
        {"additional_final_grades": {"D": None, "E": 0}, "sources": {"D": ["full_text"]}},
    ]
    found = hj.audit_review(entries)
    totals = {
        k: found[k] for k in ("audit_new", "grade_0", "grade_1", "grade_2_or_more", "unjudged")
    }
    assert totals == {
        "audit_new": 6,
        "grade_0": 2,
        "grade_1": 2,
        "grade_2_or_more": 1,
        "unjudged": 1,
    }
    by = found["by_category"]
    assert set(by) == {
        "listed",
        "other_full_text_reach",
        "snippet_follow_up",
        "ancestor_closure",
        "unknown",
    }
    assert by["snippet_follow_up"]["grade_0"] == 1
    assert by["ancestor_closure"]["grade_1"] == 1
    assert by["listed"]["grade_2_or_more"] == 1
    assert by["other_full_text_reach"]["unjudged"] == 1
    # never folded into another category, and diagnosable: which path, or none recorded
    assert (by["unknown"]["audit_new"], by["unknown"]["grade_0"], by["unknown"]["grade_1"]) == (
        2,
        1,
        1,
    )
    assert found["unknown_paths"] == {"<none>": 1, "listed_article_closure": 1}
    assert hj.audit_review([])["audit_new"] == 0 and hj.audit_review([])["unknown_paths"] == {}


# ---------------------------------------------------------------------------
# an explicit negative out of the qrels changes no metric of the S4 plan
# ---------------------------------------------------------------------------


@pytest.mark.evaluator
def test_leaving_an_explicit_negative_out_of_the_qrels_changes_no_metric_of_the_plan():
    """The chunk an explicit negative falls in, judged 0 or unjudged: Recall, MRR and nDCG
    (the S4 plan's metrics) are the same. Judged@10, a diagnostic of the evaluation code outside
    the plan, is the one that tells them apart, and the pool's record keeps the negative."""
    rankings = {"q1": ["c-negative", "c-relevant", "c-other"], "q2": ["c-other", "c-negative"]}
    with_zero = {
        "q1": {"c-relevant": 3, "c-negative": 0},
        "q2": {"c-relevant": 2, "c-negative": 0},
    }
    without = {"q1": {"c-relevant": 3}, "q2": {"c-relevant": 2}}
    one = ir_metrics.standard_scores(with_zero, rankings)
    two = ir_metrics.standard_scores(without, rankings)
    plan = ["Recall@5", "Recall@10", "Recall@20", "MRR@10", "nDCG@10"]
    for query in rankings:
        assert {m: one[query][m] for m in plan} == {m: two[query][m] for m in plan}
    assert one["q1"]["Judged@10"] != two["q1"]["Judged@10"]


# ---------------------------------------------------------------------------
# the searches' template and the controller's outer deadline
# ---------------------------------------------------------------------------


def test_the_search_template_suggests_no_number_of_candidates():
    text = hp.template("judge_a.v3").template
    assert "Usually 6 to 16" not in text and "Do not list nodes that nobody would consider." in text
    assert "Usually 6 to 16 candidates." in hp.template("judge_a.v2").template  # history


def test_the_controllers_outer_deadline_is_72_hours_and_a_stop():
    assert hct.INVOCATION_SECONDS == 72 * 3600
    assert hct.supervise.__defaults__ is not None
    assert hct.INVOCATION_SECONDS in hct.supervise.__defaults__
    assert hct.STALL_SECONDS == 5400  # unchanged: no new event for 90 minutes still stops
