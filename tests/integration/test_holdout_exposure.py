"""The exposure pool of the design of 2026-10-22 and exposure v2, rule A, of D-066
(config/holdout/evidence-discovery.md), end to end: every node of a normal search's exposure
v2 is judged, listed or not (received in full, named on a search page and followed up, or an
ancestor with its own text), and no article is expanded; judge B's pool variant, the second
blind judgment and judge C; the batches; the audit search's own exposure v2 and its judgment
discrepancies; judge D's and the diagnostic's candidates; replay from the logs. On the real grid
and corpus with synthetic replies (tests/holdout_fakes.py); no model runs, and no TEST text."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from ingest import benchmark as bm
from ingest import holdout as ho
from ingest import holdout_admission as ha
from ingest import holdout_judging as hj
from ingest import holdout_packets as hp
from ingest import holdout_pipeline as hpl
from tests.holdout_fakes import (
    EXTRA_NODE,
    Scenario,
    _candidates_in,
    load_base,
    make,
    reopen,
    rule_a,
)

pytestmark = pytest.mark.corpus


@pytest.fixture(scope="module")
def base():
    return load_base()


@pytest.fixture(scope="module")
def order(base):
    return ho.cell_order(base["grid"])


@pytest.fixture(scope="module")
def first(order, base):
    omission = set(ho.omission_cells(base["grid"]))
    return next(c for c in order if c not in omission)


@pytest.fixture(scope="module")
def audited(order, base):
    return next(c for c in order if c in set(ho.omission_cells(base["grid"])))


def article_nodes(base: dict[str, Any], act: str, *numbers: str) -> list[str]:
    """Nodes with their own text of the given articles, in document order: none of them is
    evidence of the synthetic replies."""
    corpus = base["corpus"]
    found = [
        f"{a}:{p}"
        for (a, p) in base["texts"]
        if a == act and p.split(".")[0] in numbers and "." in p
    ]
    ordered: list[str] = corpus.sort(found)
    return ordered


def block_of(packet: str, node: str) -> str:
    """One candidate as a judge reads it: from its CANDIDATE line to the next blank line."""
    found = re.search(
        rf"^CANDIDATE C-[0-9a-f]{{12}}\n\[{re.escape(node)}\].*?(?=\n\n|\Z)", packet, re.M | re.S
    )
    assert found is not None, node
    return found.group(0).rstrip("\n")  # the last block ends the packet


def value_of(pipeline: hpl.Pipeline, pid: str, stage: str) -> Any:
    row = pipeline.store.stage(pid, stage)
    assert row is not None, stage
    return row["value"]


def built(base, tmp_path, cells, scenario: Scenario | None = None):
    pipeline, executor = make(base, tmp_path / "test", scenario)
    for cell in cells:
        pipeline.build_cell(cell)
    return pipeline, executor


def stage_names(pipeline: hpl.Pipeline, pid: str) -> set[str]:
    return {s["stage"] for s in pipeline.store.stages() if s["proposal_id"] == pid}


def ready_of(pipeline: hpl.Pipeline, cell: str) -> hpl.ReadyNeed:
    return next(r for r in pipeline.ready.values() if r.cell_id == cell)


def graded(ready: hpl.ReadyNeed) -> dict[str, int]:
    return {f"{q.act}:{q.path}": q.grade for q in ready.qrels}


def packets_of(executor, role: str) -> list[str]:
    return [p for r, _, _, p in executor.sent if r == role]


def search_log(pipeline: hpl.Pipeline, pid: str, stage: str) -> list[dict[str, Any]]:
    run = next(
        r
        for r in pipeline.store.runs()
        if r["proposal_id"] == pid and r["stage"] == stage and r["status"] == "admitted"
    )
    kept = json.loads(run["response"])["kept"]
    return list(json.loads(pipeline.store.text(kept, "kept"))["run"]["log"])


# ---------------------------------------------------------------------------
# what enters the pool
# ---------------------------------------------------------------------------


def test_an_exposed_unlisted_node_enters_the_pool_and_is_judged_blind(base, first, tmp_path):
    exposed = article_nodes(base, "DORA", "28")[:2]
    scenario = Scenario(expose={(first, "searcher_2"): exposed})
    pipeline, executor = built(base, tmp_path, [first], scenario)
    pid = f"{first}-a1"
    pool = value_of(pipeline, pid, "exposure_pool")
    # rule A, computed here: the exposed nodes and their ancestors with their own text (the
    # first is its point's lead-in), and no other node of their article
    assert rule_a(base, pool["listed_by"], exposed) - set(pool["listed_by"]) == set(exposed)
    assert all(pool["exposed_by"][n] == ["search_2"] for n in exposed)
    assert all(pool["paths"][n][0] == "full_text" for n in exposed)
    assert pool["unlisted"] == len(exposed) and pool["second_judgment"] == []
    # judged once, blind, explicitly 0: an explicit negative, in the pool's record, no qrel
    assert set(pool["explicit_negative"]) == set(exposed)
    for n in exposed:
        assert pool["pool_adjudication"][n] == {
            "first": {"grade": 0, "ambiguous": False},
            "second": None,
            "judge_c": None,
            "outcome": "explicit_negative",
            "final": {"grade": 0, "ambiguous": False},
        }
    stages = stage_names(pipeline, pid)
    assert "pool_b1:1" in stages and not any(s.startswith("pool_b2") for s in stages)
    assert not set(exposed) & set(graded(ready_of(pipeline, first)))
    (packet,) = packets_of(executor, "pool_judge_1")
    assert {node for _, node in _candidates_in(packet)} == set(exposed)
    # nothing of the searches reaches the pool judge: no label, no rationale; and it is asked
    # for no answerability
    assert "A quotes the node" not in packet
    assert "answerability" not in packet.split("OUTPUT", 1)[1].split("THE NEED", 1)[0]


def test_a_node_a_search_page_named_is_followed_up_and_judged_on_its_whole_text(
    base, first, tmp_path
):
    """Exposure v2 (D-066), test A: a node only a `search` page named, which no search listed or
    opened, enters the pool and is judged blind, explicitly; its judge reads the node's whole
    own text from the frozen corpus, never the snippet the search saw."""
    node = max(
        article_nodes(base, "DORA", "28"), key=lambda n: len(base["texts"][hp.Corpus.key(n)])
    )
    text = base["texts"][hp.Corpus.key(node)]
    middle = text[len(text) // 2 : len(text) // 2 + 60]
    # a snippet holds at most 90 characters before the match and 110 after: never this text
    assert len(text) > 90 + len(middle) + 110
    scenario = Scenario(snippet={(first, "searcher_1"): re.escape(middle)})
    pipeline, executor = built(base, tmp_path, [first], scenario)
    pid = f"{first}-a1"
    log = search_log(pipeline, pid, "search_1")
    named = {n for e in log if e["op"] == "search" for n in e["nodes"]}
    assert node in named and hj.snippets(log) == sorted(named)
    pool = value_of(pipeline, pid, "exposure_pool")
    assert node not in pool["listed_by"] and node not in pool["exposed_by"]  # never opened
    assert pool["reached_by"][node] == ["search_1"]
    assert pool["paths"][node] == ["snippet_follow_up"]
    assert pool["by_path"]["snippet_follow_up"] >= 1 and pool["beyond_full_text"] >= 1
    assert pool["pool_adjudication"][node]["first"] == {"grade": 0, "ambiguous": False}
    (packet,) = [
        p for p in packets_of(executor, "pool_judge_1") if node in dict(_candidates_in(p)).values()
    ]
    assert f"text: {text}" in block_of(packet, node)  # the whole own text, from the corpus
    # nothing tells the judge how the node reached the pool, nor which search reached it
    assert not any(p in packet for p in [*hj.EXPOSURE_PATHS, "search_1", "exposure"])


def test_an_ancestor_with_its_own_text_becomes_a_candidate_judged_explicitly(base, first, tmp_path):
    """Exposure v2 (D-066), test B: the lead-in of a point a search received in full is no longer
    only the point's context: it enters the pool and is graded on its own."""
    point = next(
        n
        for n in article_nodes(base, "DORA", "28")
        if n.count(".") >= 2 and base["texts"].get(hp.Corpus.key(n.rsplit(".", 1)[0]), "").strip()
    )
    lead_in = point.rsplit(".", 1)[0]
    scenario = Scenario(
        expose={(first, "searcher_2"): [point]},
        pool_grade={(first, "pool_judge_1"): {lead_in: 1}, (first, "pool_judge_2"): {lead_in: 1}},
    )
    pipeline, executor = built(base, tmp_path, [first], scenario)
    pool = value_of(pipeline, f"{first}-a1", "exposure_pool")
    assert pool["paths"][lead_in] == ["ancestor_closure"]
    assert pool["reached_by"][lead_in] == ["search_2"] and lead_in not in pool["exposed_by"]
    assert pool["pool_adjudication"][lead_in]["outcome"] == "agreed"
    assert graded(ready_of(pipeline, first))[lead_in] == 1  # a qrel by its own grade
    (packet,) = packets_of(executor, "pool_judge_1")
    assert {lead_in, point} <= {n for _, n in _candidates_in(packet)}
    assert f"context ({lead_in})" not in block_of(packet, point)  # a candidate beside it


def test_a_listed_node_enters_the_pool_even_if_never_read_in_full(base, first, tmp_path):
    pipeline, _ = built(base, tmp_path, [first])
    pool = value_of(pipeline, f"{first}-a1", "exposure_pool")
    never_read = set(pool["listed_by"]) - set(pool["exposed_by"])
    assert never_read and all("listed" in pool["paths"][n] for n in never_read)
    # rule A: the listed nodes and their ancestors with their own text, and nothing else
    expected = rule_a(base, pool["listed_by"])
    assert set(pool["reached_by"]) == expected and pool["pool"] == len(expected)
    assert pool["unlisted"] == len(expected - set(pool["listed_by"]))


def test_the_pool_replays_identically_from_the_logs(base, first, tmp_path, monkeypatch):
    exposed = article_nodes(base, "DORA", "28")[:3]
    scenario = Scenario(
        expose={(first, "searcher_1"): exposed[:2], (first, "searcher_3"): exposed[1:]},
        pool_grade={(first, "pool_judge_1"): {exposed[0]: 1}},
    )
    pipeline, _ = built(base, tmp_path, [first], scenario)
    pid = f"{first}-a1"
    before = pipeline.store.verify()
    pool = value_of(pipeline, pid, "exposure_pool")
    records = {n: r.record() for n, r in pipeline.ready.items()}
    # the replayed state holds the judgments as given: an explicit 0 is the integer 0
    first_grades = {g["grade"] for g in value_of(pipeline, pid, "pool_b1:1")["grades"]}
    assert first_grades == {0, 1} and None not in first_grades
    assert pool["pool_adjudication"][exposed[1]]["outcome"] == "explicit_negative"
    pipeline.store.close()
    again, _ = reopen(base, tmp_path / "test", scenario)
    with again.store.no_writes():
        again._reset()
        again.build_cell(first)
    assert again.store.verify() == before
    assert value_of(again, pid, "exposure_pool") == pool
    assert {n: r.record() for n, r in again.ready.items()} == records
    assert again._exposure_problems() == []
    # an exposure that is not its broker log's is refused at every checkpoint and the freeze
    monkeypatch.setattr(hj, "exposure", lambda log: [])
    assert any("not its broker log's" in p for p in again._exposure_problems())


def test_snippets_that_are_not_the_broker_logs_are_refused_at_the_freeze(
    base, first, tmp_path, monkeypatch
):
    """The pool is rebuilt from the recorded logs: a value whose snippets are not its own log's,
    by any change, is refused at every checkpoint and at the freeze."""
    scenario = Scenario(snippet={(first, "searcher_2"): "crypto-asset service provider"})
    pipeline, _ = built(base, tmp_path, [first], scenario)
    assert pipeline._exposure_problems() == []
    real = hj.snippets
    for changed in (lambda log: [], lambda log: [*real(log), "DORA:1"]):
        monkeypatch.setattr(hj, "snippets", changed)
        found = pipeline._exposure_problems()
        assert found and all("its snippets are not its broker log's" in p for p in found)
        assert any("snippets are not" in p for p in pipeline.freeze_problems())


# ---------------------------------------------------------------------------
# the first blind judgment, the second, and judge C
# ---------------------------------------------------------------------------


def adjudication_of(ready: hpl.ReadyNeed, node: str) -> bm.Adjudication:
    return next(a for a in ready.adjudications if a.target == node)


def test_a_positive_first_judgment_gets_a_second_and_judge_c_decides_a_difference(
    base, first, tmp_path
):
    node = article_nodes(base, "DORA", "28")[0]
    scenario = Scenario(
        expose={(first, "searcher_2"): [node]},
        pool_grade={(first, "pool_judge_1"): {node: 1}},
    )
    pipeline, executor = built(base, tmp_path, [first], scenario)
    pid = f"{first}-a1"
    assert {"pool_b1:1", "pool_b2:1", "judge_c"} <= stage_names(pipeline, pid)
    (second,) = packets_of(executor, "pool_judge_2")
    (first_packet,) = packets_of(executor, "pool_judge_1")
    # the same immutable id in both, and the second judge sees nothing of the first: the node
    # as the first judge saw it, and no label of anyone
    ids = {n: i for i, n in _candidates_in(first_packet)}
    assert _candidates_in(second) == [(ids[node], node)]
    assert block_of(second, node) == block_of(first_packet, node)
    assert "A quotes the node" not in second
    ready = ready_of(pipeline, first)
    record = adjudication_of(ready, node)
    assert (record.a.grade, record.b.grade) == (1, 0)
    assert record.c is not None and record.c.grade == 0 and graded(ready)[node] == 0


def test_an_ambiguous_first_judgment_gets_a_second(base, first, tmp_path):
    node = article_nodes(base, "DORA", "28")[0]
    scenario = Scenario(
        expose={(first, "searcher_2"): [node]},
        pool_grade={(first, "pool_judge_1"): {node: {"grade": 0, "ambiguous": True}}},
    )
    pipeline, _ = built(base, tmp_path, [first], scenario)
    pid = f"{first}-a1"
    assert "pool_b2:1" in stage_names(pipeline, pid)
    record = adjudication_of(ready_of(pipeline, first), node)
    assert record.a.ambiguous and not record.b.ambiguous and record.c is not None


def test_two_agreeing_blind_judgments_are_final_without_judge_c(base, first, tmp_path):
    node = article_nodes(base, "DORA", "28")[0]
    scenario = Scenario(
        expose={(first, "searcher_3"): [node]},
        pool_grade={(first, "pool_judge_1"): {node: 1}, (first, "pool_judge_2"): {node: 1}},
    )
    pipeline, executor = built(base, tmp_path, [first], scenario)
    record = adjudication_of(ready_of(pipeline, first), node)
    assert (record.a.grade, record.b.grade, record.c) == (1, 1, None)
    assert graded(ready_of(pipeline, first))[node] == 1
    assert not packets_of(executor, "judge_c")


def test_the_orchestrator_never_gives_a_pool_node_its_label(base, first, tmp_path):
    """Every unlisted node's final label is one a judge gave it: the first blind judgment when
    it is an explicit grade 0, the agreed one, or judge C's; never one the orchestrator implies
    or repairs."""
    zero = article_nodes(base, "DORA", "28")[0]
    two = EXTRA_NODE  # grade 2 to every judge of this cell, judge D too
    scenario = Scenario(
        expose={(first, "searcher_1"): [zero, two]},
        pool_grade={(first, "pool_judge_1"): {zero: 0, two: 2}, (first, "pool_judge_2"): {two: 2}},
        extra_grade={first: 2},
    )
    pipeline, executor = built(base, tmp_path, [first], scenario)
    pool = value_of(pipeline, f"{first}-a1", "exposure_pool")
    assert rule_a(base, pool["listed_by"], [zero, two]) - set(pool["listed_by"]) == {zero, two}
    assert pool["explicit_negative"] == [zero]
    assert pool["pool_adjudication"][two]["outcome"] == "agreed"
    assert pool["pool_adjudication"][two]["final"] == {"grade": 2, "ambiguous": False}
    ready = ready_of(pipeline, first)
    assert graded(ready)[two] == 2 and zero not in graded(ready)
    # only pool nodes were judged by the pool variant, and each once in its first judgment
    judged = [
        node
        for _, _, _, packet in executor.sent
        if "CANDIDATE" in packet and packet in packets_of(executor, "pool_judge_1")
        for _, node in _candidates_in(packet)
    ]
    assert sorted(set(judged)) == sorted({zero, two})


# ---------------------------------------------------------------------------
# the exact-coverage contract: an absent judgment is never a negative one
# ---------------------------------------------------------------------------


def until_stop(pipeline: hpl.Pipeline, order: list[str], cell: str) -> None:
    with pytest.raises(hpl.TechnicalStop):
        for c in order:
            pipeline.build_cell(c)
            if c == cell:
                break


def failures_at(pipeline: hpl.Pipeline, stage: str) -> list[dict[str, Any]]:
    runs = [r["run_id"] for r in pipeline.store.runs() if r["stage"] == stage]
    return pipeline.store.failure_payloads(runs)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "unknown", "wrong"])
def test_a_first_pool_judgment_short_of_the_exact_set_is_technical(
    base, first, order, tmp_path, fault
):
    nodes = article_nodes(base, "DORA", "28")[:2]
    scenario = Scenario(
        expose={(first, "searcher_2"): nodes}, identity={(first, "pool_judge_1"): [fault]}
    )
    pipeline, _ = make(base, tmp_path / "test", scenario)
    until_stop(pipeline, order, first)
    (failure,) = failures_at(pipeline, "pool_b1:1")
    assert (failure["reason"], failure["check"], failure["violations"]) == (
        "malformed_session",
        "judge_answer_complete",
        [],
    )
    # nothing was finalized from the failed answer: no pool record, no negative inferred
    assert pipeline.store.stage(f"{first}-a1", "exposure_pool") is None


def test_an_ungraded_node_is_technical_three_times_and_never_a_zero(base, first, order, tmp_path):
    """A null grade, every execution: the role's contract fails three times and the stage is
    unresolved for a person; never a grade 0, never semantic_inadmissible."""
    node = article_nodes(base, "DORA", "28")[0]
    scenario = Scenario(
        expose={(first, "searcher_2"): [node]},
        pool_grade={(first, "pool_judge_1"): {node: None}},
    )
    pipeline, _ = make(base, tmp_path / "test", scenario)
    states = []
    for n in range(3):
        pipeline._reset()
        pipeline.resuming = n > 0
        until_stop(pipeline, order, first)
        states.append(pipeline.store.status())
    assert states == ["pending_technical", "pending_technical", "pending_technical_unresolved"]
    failures = failures_at(pipeline, "pool_b1:1")
    assert len(failures) == 3
    assert all(f["check"] == "judge_answer_complete" and f["violations"] == [] for f in failures)
    assert all(a["outcome"] == "accepted" for a in pipeline.store.attempts())  # no rejection
    assert pipeline.store.stage(f"{first}-a1", "exposure_pool") is None


def test_a_second_pool_judgment_missing_a_routed_node_is_technical(base, first, order, tmp_path):
    node = article_nodes(base, "DORA", "28")[0]
    scenario = Scenario(
        expose={(first, "searcher_2"): [node]},
        pool_grade={(first, "pool_judge_1"): {node: 1}},
        identity={(first, "pool_judge_2"): ["missing"]},
    )
    pipeline, _ = make(base, tmp_path / "test", scenario)
    until_stop(pipeline, order, first)
    (failure,) = failures_at(pipeline, "pool_b2:1")
    assert (failure["reason"], failure["check"]) == ("malformed_session", "judge_answer_complete")
    assert pipeline.store.stage(f"{first}-a1", "pool_b1:1") is not None


def test_an_audit_pool_judgment_missing_a_node_is_technical(base, audited, order, tmp_path):
    """The audit can never turn a missing answer into a grade 0."""
    node = article_nodes(base, "DORA", "28")[0]
    scenario = Scenario(
        expose={(audited, "audit_searcher"): [node]},
        identity={(audited, "pool_judge_1"): ["missing"]},
    )
    pipeline, _ = make(base, tmp_path / "test", scenario)
    until_stop(pipeline, order, audited)
    (failure,) = failures_at(pipeline, "audit_b1:1")
    assert (failure["reason"], failure["check"]) == ("malformed_session", "judge_answer_complete")
    assert pipeline.store.stage(f"{audited}-a1", "omission_audit") is None
    assert not pipeline.omission_log  # no audit was decided from the failed answer


def test_the_qrels_leave_an_explicit_negative_out_and_the_provenance_keeps_it(
    base, first, tmp_path
):
    node = article_nodes(base, "DORA", "28")[0]
    scenario = Scenario(expose={(first, "searcher_3"): [node]})
    pipeline, _ = built(base, tmp_path, [first], scenario)
    record = ready_of(pipeline, first).record()
    assert node not in {f"{q['act']}:{q['path']}" for q in record["qrels"]}
    assert node not in {a["target"] for a in record["adjudications"]}
    discovery = record["provenance"]["evidence_discovery"]
    assert discovery["explicit_negative"] == [node]
    assert discovery["pool_adjudication"][node]["first"] == {"grade": 0, "ambiguous": False}
    report = pipeline.report()["evidence_discovery"]
    assert report["explicit_negative"] == 1
    assert report["pool_by_path_full_text"] == discovery["by_path"]["full_text"] >= 1
    assert report["pool_beyond_full_text"] == discovery["beyond_full_text"] == 0


def test_a_broken_pool_identity_is_a_technical_failure(base, first, order, tmp_path):
    node = article_nodes(base, "DORA", "28")[0]
    scenario = Scenario(
        expose={(first, "searcher_2"): [node]}, identity={(first, "pool_judge_1"): ["unknown"]}
    )
    pipeline, _ = make(base, tmp_path / "test", scenario)
    with pytest.raises(hpl.TechnicalStop):
        for cell in order:
            pipeline.build_cell(cell)
            if cell == first:
                break
    summary = pipeline.store.summary()
    assert summary["technical_failures"] == {"malformed_session": 1}
    assert summary["waiting"][0]["stage"] == "pool_b1:1"
    pipeline.resuming = True
    pipeline._reset()
    pipeline.resuming = True
    for cell in order:
        pipeline.build_cell(cell)
        if cell == first:
            break
    assert any(r.cell_id == first for r in pipeline.ready.values())


# ---------------------------------------------------------------------------
# the batches
# ---------------------------------------------------------------------------


def test_a_pool_larger_than_a_batch_runs_in_deterministic_batches(base, first, tmp_path):
    nodes = article_nodes(base, "DORA", "28", "29", "30")[:60]
    assert len(nodes) == 60
    scenario = Scenario(
        expose={
            (first, "searcher_1"): nodes[:20],
            (first, "searcher_2"): nodes[20:40],
            (first, "searcher_3"): nodes[40:],
        }
    )
    pipeline, executor = built(base, tmp_path, [first], scenario)
    pid = f"{first}-a1"
    batches = [p for p in packets_of(executor, "pool_judge_1")]
    assert len(batches) >= 2
    shown = [[node for _, node in _candidates_in(p)] for p in batches]
    assert all(len(b) <= hp.POOL_BATCH_MAX_NODES for b in shown)
    assert all(hp.packet_tokens(p) <= hp.POOL_BATCH_MAX_TOKENS for p in batches)
    # every node once, never split, in document order across the batches
    pool = value_of(pipeline, pid, "exposure_pool")
    unlisted = rule_a(base, pool["listed_by"], nodes) - set(pool["listed_by"])
    assert [n for b in shown for n in b] == base["corpus"].sort(list(unlisted))
    stages = sorted(s for s in stage_names(pipeline, pid) if s.startswith("pool_b1:"))
    assert stages == [f"pool_b1:{i}" for i in range(1, len(batches) + 1)]
    pool = value_of(pipeline, pid, "exposure_pool")
    assert pool["batches"] == {"pool_b1": len(batches), "pool_b2": 0}


# ---------------------------------------------------------------------------
# the audit search's exposure, and its judgment discrepancies
# ---------------------------------------------------------------------------


def test_the_audit_search_exposure_catches_an_unlisted_node_read_in_full(base, audited, tmp_path):
    node = article_nodes(base, "DORA", "28")[0]
    scenario = Scenario(expose={(audited, "audit_searcher"): [node]})
    pipeline, _ = built(base, tmp_path, [audited], scenario)
    pid = f"{audited}-a1"
    assert "audit_b1:1" in stage_names(pipeline, pid)
    logged = next(o for o in pipeline.omission_log if o["proposal_id"] == pid)
    assert logged["additional_nodes"] == [node]
    assert logged["additional_final_grades"] == {node: 0}  # a diagnostic, never a qrel
    assert node not in graded(ready_of(pipeline, audited))


def test_a_new_audit_node_read_in_full_and_adjudicated_1_or_more_stops(base, audited, tmp_path):
    node = article_nodes(base, "DORA", "28")[0]
    scenario = Scenario(
        expose={(audited, "audit_searcher"): [node]},
        pool_grade={(audited, "pool_judge_1"): {node: 1}, (audited, "pool_judge_2"): {node: 1}},
    )
    pipeline, _ = make(base, tmp_path / "test", scenario)
    with pytest.raises(hpl.ConstructionStop, match="outside the exposure pool") as stop:
        pipeline.build()
    assert (stop.value.reason, stop.value.stage) == ("omission_audit", "omission_audit")
    audit = value_of(pipeline, f"{audited}-a1", "omission_audit")
    assert audit["final_grades"] == {node: 1}  # the two blind judgments agree: no judge C


def test_an_audit_label_on_a_pooled_node_is_provenance_only(base, audited, tmp_path):
    """The audit search grades 1 a node the normal pool judged 0: an audit judgment
    discrepancy, counted, never an omission, a label, a judge C or a stop."""
    scenario = Scenario(
        expose={(audited, "searcher_1"): [EXTRA_NODE]},
        pool_grade={(audited, "pool_judge_1"): {EXTRA_NODE: 0}},
        extra_grade={audited: 1},  # the audit search lists EXTRA_NODE, grade 1
    )
    pipeline, _ = built(base, tmp_path, [audited], scenario)
    pid = f"{audited}-a1"
    pool = value_of(pipeline, pid, "exposure_pool")
    assert pool["explicit_negative"] == [EXTRA_NODE]
    audit = value_of(pipeline, pid, "omission_audit")
    # the audit's listing expands no article: nothing of it is new to the normal pool
    assert audit["new_listed"] == [] and audit["new_exposed"] == [] and audit["sources"] == {}
    assert audit["judgment_discrepancies"] == [{"node": EXTRA_NODE, "direction": "0_to_1_or_more"}]
    assert "audit_judge_c" not in stage_names(pipeline, pid)
    assert EXTRA_NODE not in graded(ready_of(pipeline, audited))
    report = pipeline.report()["omission_audit"]
    assert report["judgment_discrepancies"] == {"0_to_1_or_more": 1}
    assert pipeline.progress_problems([audited]) == []


# ---------------------------------------------------------------------------
# judge D's and the diagnostic's candidates
# ---------------------------------------------------------------------------


def contested_scenario(base, cell: str, **more: Any) -> tuple[Scenario, str, str]:
    negative, contested = article_nodes(base, "DORA", "28")[0], article_nodes(base, "DORA", "30")[0]
    scenario = Scenario(
        expose={(cell, "searcher_2"): [negative, contested, EXTRA_NODE]},
        pool_grade={(cell, "pool_judge_1"): {contested: 1}},
        extra_grade={cell: 2},  # EXTRA_NODE is grade 2 to every judge: final 2
        **more,
    )
    return scenario, negative, contested


def test_the_diagnostic_receives_exactly_the_listed_and_the_contested_or_relevant_nodes(
    base, first, tmp_path
):
    scenario, negative, contested = contested_scenario(base, first)
    pipeline, executor = built(base, tmp_path, [first], scenario)
    ready = ready_of(pipeline, first)
    pool = value_of(pipeline, f"{first}-a1", "exposure_pool")
    expected = {*pool["listed_by"], contested, EXTRA_NODE}
    prompt = executor.packets[("qwen_diagnostic", first, 1)]
    assert {node for _, node in _candidates_in(prompt)} == expected
    # a pool node with no positive signal reaches neither (its text may show as a context)
    assert negative not in {node for _, node in _candidates_in(prompt)}
    assert graded(ready)[EXTRA_NODE] == 2 and graded(ready)[contested] == 0


def test_judge_d_receives_the_same_candidates(base, first, tmp_path):
    scenario, negative, contested = contested_scenario(base, first, qwen_overlength={first})
    pipeline, executor = built(base, tmp_path, [first], scenario)
    pool = value_of(pipeline, f"{first}-a1", "exposure_pool")
    expected = {*pool["listed_by"], contested, EXTRA_NODE}
    packet = executor.packets[("judge_d", first, 1)]
    assert {node for _, node in _candidates_in(packet)} == expected
    d = value_of(pipeline, f"{first}-a1", "judge_d")
    assert len(d["items"]) == len(expected) + 1  # every candidate and the answerability
    assert negative not in {node for _, node in _candidates_in(packet)}
    assert pipeline.progress_problems([first]) == []


# ---------------------------------------------------------------------------
# the searches' listing
# ---------------------------------------------------------------------------


def test_a_search_may_list_any_number_of_candidates(base):
    nodes = article_nodes(base, "DORA", "28", "29")[:24]
    found: dict[str, Any] = {
        "need_id": "N101",
        "candidates": [
            {
                "node": n,
                "scope": [],
                "grade": 3 if i == 0 else 1,
                "ambiguous": False,
                "scope_basis": [],
                "rationale": "quotes the node",
            }
            for i, n in enumerate(nodes)
        ],
        "answerability": "complete",
        "answerability_reason": "the node answers it",
        "gap_note": None,
        "drivers": {},
        "problem": None,
    }
    assert ha.search_problems(found, base["texts"], base["excluded"]) == []
