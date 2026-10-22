"""The exposure pool of the design of 2026-10-22 (config/holdout/evidence-discovery.md), as
functions: what a search received in full, the pool of three searches, the batches of judge B's
pool variant, its answer's contract, the candidates of judge D and the diagnostic, and the
controller's outer deadline. Synthetic data only: no corpus, no model, no TEST text."""

from __future__ import annotations

from typing import Any

import pytest

from ingest import benchmark as bm
from ingest import holdout_admission as ha
from ingest import holdout_controller as hct
from ingest import holdout_judging as hj
from ingest import holdout_packets as hp

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
    found = hj.exposure_pool(files, exposures, TEXTS, EXCLUDED)
    assert found.listed == ["MiCA:60.1", "DORA:19.1"]
    assert found.unlisted == ["MiCA:3.1", "MiCA:60.2"]
    assert found.pool == ["DORA:19.1", "MiCA:3.1", "MiCA:60.1", "MiCA:60.2"]
    assert "DORA:19.1" not in found.exposed_by  # listed without being read in full
    assert found.first["MiCA:60.1"]["grade"] == 3  # search_1's, before search_2's
    # the same inputs, the same pool, in any run
    assert hj.exposure_pool(files, exposures, TEXTS, EXCLUDED) == found


def test_the_audit_nodes_are_split_by_the_normal_pool():
    audit = file(("MiCA:60.1", 1), ("PSD2:4.1", 2))
    nodes = hj.audit_nodes(
        audit, ["MiCA:3.2", "MiCA:60.2", "PSD2:3.1"], ["MiCA:60.1", "MiCA:60.2"], TEXTS, EXCLUDED
    )
    assert list(nodes.listed) == ["PSD2:4.1"]  # listed, outside the pool: its label first
    assert nodes.exposed == ["MiCA:3.2"]  # received in full, outside the pool, not listed
    assert list(nodes.on_pool) == ["MiCA:60.1"]  # a label on a pool node: provenance only


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


def test_the_pool_variant_answers_every_node_by_its_id_and_no_answerability():
    good = grades(("C-" + "1" * 12, 0, False), ("C-" + "2" * 12, 2, False))
    assert ha.pool_problems(good, IDS) == []
    assert hj.pool_judgments(good, IDS)["MiCA:60.2"]["grade"] == 2
    missing = grades(("C-" + "1" * 12, 0, False))
    assert any("missing" in p for p in ha.pool_problems(missing, IDS))
    unknown = grades(("C-" + "1" * 12, 0, False), ("C-" + "0" * 12, 1, False))
    assert any("unknown" in p for p in ha.pool_problems(unknown, IDS))
    named = grades(("C-" + "1" * 12, 0, False), ("C-" + "2" * 12, 2, False))
    named["grades"][1]["target"] = "MiCA:61.2"
    assert any("identity" in p for p in ha.pool_problems(named, IDS))
    no_basis = grades(("C-" + "1" * 12, 0, False), ("C-" + "2" * 12, 2, False))
    no_basis["grades"][0]["scope_basis"] = []
    assert ha.pool_problems(no_basis, IDS)  # a grade 0 names its scope, as judge B's


@pytest.mark.parametrize(
    ("label", "second"),
    [
        ({"grade": 0, "ambiguous": False}, False),  # final 0
        ({"grade": None, "ambiguous": False}, False),  # final: left unjudged
        ({"grade": 1, "ambiguous": False}, True),
        ({"grade": 3, "ambiguous": False}, True),
        ({"grade": 0, "ambiguous": True}, True),
    ],
)
def test_a_positive_or_ambiguous_label_needs_a_second_judgment(label, second):
    assert hj.needs_second_judgment(label) is second


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
        "MiCA:3.1": [{"grade": 0, "ambiguous": False}],  # nobody listed, judged 0: out
        "MiCA:3.2": [{"grade": None, "ambiguous": False}],  # left unjudged: out
        "MiCA:60.2": [{"grade": 1, "ambiguous": False}, {"grade": 0, "ambiguous": False}],
        "PSD2:4.1": [{"grade": 0, "ambiguous": True}, {"grade": 0, "ambiguous": True}],
    }
    final = {"DORA:19.1": 3, "MiCA:60.1": 0, "MiCA:60.2": 0, "PSD2:4.1": 0}
    assert hj.diagnostic_set(listed, pool_labels, final) == [
        "DORA:19.1",
        "MiCA:60.1",
        "MiCA:60.2",
        "PSD2:4.1",
    ]
    # a final grade >= 1 alone brings a pool node in, and a listed node needs nothing else
    assert "MiCA:3.1" in hj.diagnostic_set([], {"MiCA:3.1": []}, {"MiCA:3.1": 1})
    assert hj.diagnostic_set(["MiCA:60.1", "MiCA:60.1"], {}, {}) == ["MiCA:60.1"]


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
