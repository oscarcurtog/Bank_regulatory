"""Evidence discovery for every TEST need (config/holdout/evidence-discovery.md): three blind,
independent searches, their listed candidates joined and the exposure pool built from what they
received in full; the omission audit's search in its 12 cells; each search's declared problem
with its transition; the audit bound to the attempt that becomes READY; the diagnostic's
preflight; and the failures nothing else classifies. On the real grid and corpus with synthetic
replies (tests/holdout_fakes.py); no model runs."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from ingest import benchmark as bm
from ingest import holdout as ho
from ingest import holdout_judging as hj
from ingest import holdout_packets as hp
from ingest import holdout_pipeline as hpl
from ingest import holdout_roles as hr
from tests.holdout_fakes import (
    B_EXTRA_NODE,
    EXTRA_NODE,
    Crash,
    Scenario,
    load_base,
    make,
    reopen,
)

pytestmark = pytest.mark.corpus
NORMAL = ("searcher_1", "searcher_2", "searcher_3")


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


def built(base, tmp_path, cells, scenario: Scenario | None = None):
    pipeline, executor = make(base, tmp_path / "test", scenario)
    for cell in cells:
        pipeline.build_cell(cell)
    return pipeline, executor


def ready_of(pipeline: hpl.Pipeline, cell: str) -> hpl.ReadyNeed:
    return next(r for r in pipeline.ready.values() if r.cell_id == cell)


def attempts_of(pipeline: hpl.Pipeline, cell: str) -> list[tuple[int, str | None]]:
    return [(a["number"], a["outcome"]) for a in pipeline.store.attempts() if a["cell_id"] == cell]


def without_ref(packet: str) -> str:
    return re.sub(r"R-[0-9a-f]{12}", "R-*", packet)


# ---------------------------------------------------------------------------
# the union and the pool, as functions
# ---------------------------------------------------------------------------


def file(*nodes: tuple[str, int], answerability: str = "complete") -> dict[str, Any]:
    return {
        "candidates": [
            {
                "node": n,
                "scope": [f"scope-of-{n}"],
                "grade": g,
                "ambiguous": False,
                "scope_basis": [n] if g == 0 else [],
                "rationale": f"label of {n}",
            }
            for n, g in nodes
        ],
        "answerability": answerability,
        "answerability_reason": "why",
        "gap_note": None if answerability == "complete" else "lacks",
    }


def test_the_union_keeps_each_node_once_with_its_first_judgment_and_pools_what_was_read(base):
    excluded = next(
        f"{a}:{p}"
        for (a, p) in base["texts"]
        if a == "PSD2" and p.split(".")[0] in base["excluded"]
    )
    files = {
        "search_1": file(("MiCA:149.2", 3), ("DORA:64", 0)),
        "search_2": file(("MiCA:149.2", 1), ("MiCA:143.2", 2), answerability="partial"),
        "search_3": file(("MiCA:143.2", 1), ("MiCA:143.3", 0)),
    }
    exposures = {
        "search_1": ["MiCA:149.2", "MiCA:149.1"],
        "search_2": ["MiCA:149.1", excluded, "MiCA:999.9"],  # excluded, and no such node
        "search_3": [],
    }
    found = hj.exposure_pool(files, exposures, base["texts"], base["excluded"])
    assert found.listed == ["MiCA:149.2", "DORA:64", "MiCA:143.2", "MiCA:143.3"]
    # the first search that listed a node gives its first judgment, in the order 1, 2, 3
    assert found.first["MiCA:149.2"]["grade"] == 3
    assert found.first["MiCA:143.2"]["grade"] == 2
    assert found.scope["MiCA:143.2"] == ["scope-of-MiCA:143.2"]
    # the answerability's first judgment is search_1's; the others' are provenance only
    assert found.first[bm.ANSWERABILITY_TARGET]["answerability"] == "complete"
    assert found.listed_by == {
        "MiCA:149.2": ["search_1", "search_2"],
        "DORA:64": ["search_1"],
        "MiCA:143.2": ["search_2", "search_3"],
        "MiCA:143.3": ["search_3"],
    }
    # what a search received in full enters the pool, listed or not, if it can be judged
    assert found.exposed_by == {"MiCA:149.2": ["search_1"], "MiCA:149.1": ["search_1", "search_2"]}
    assert found.unlisted == ["MiCA:149.1"]
    assert found.pool == sorted([*found.listed, "MiCA:149.1"])
    record = found.record()
    assert (
        record["listed_union"],
        record["listed_by_one"],
        record["pool"],
        record["unlisted"],
    ) == (
        4,
        2,
        5,
        1,
    )
    assert record["answerabilities_agree"] is False
    # the same inputs, the same pool: nothing in it depends on anything else
    assert hj.exposure_pool(files, exposures, base["texts"], base["excluded"]) == found


# ---------------------------------------------------------------------------
# three blind searches for every need
# ---------------------------------------------------------------------------


def test_every_need_gets_three_blind_searches_and_one_adjudication(base, first, tmp_path):
    pipeline, executor = built(base, tmp_path, [first], Scenario(b_extra={first: 2}))
    roles = [role for role, cell, _ in executor.calls if cell == first]
    assert [roles.count(r) for r in NORMAL] == [1, 1, 1]
    assert "audit_searcher" not in roles
    ready = ready_of(pipeline, first)
    graded = {f"{q.act}:{q.path}": q.grade for q in ready.qrels}
    assert graded[B_EXTRA_NODE] == 2  # listed by search_2 only, adjudicated with the rest
    discovery = ready.provenance["evidence_discovery"]
    assert discovery["listed_by"][B_EXTRA_NODE] == ["search_2"]
    assert discovery["listed_by_one"] == 1
    report = pipeline.report()["evidence_discovery"]
    assert report["listed_by_one_grade_1_or_more"] == 1
    assert set(ready.provenance["search_consumption"]) == {"search_1", "search_2", "search_3"}


def test_the_second_and_third_searches_are_as_blind_as_the_first(base, first, tmp_path):
    """The same frozen packet from the locked need alone, a reference of its own, and no input
    but the lock: nothing of one search reaches another."""
    pipeline, executor = built(base, tmp_path, [first], Scenario(b_extra={first: 2}))
    packets = [executor.packets[(role, first, 1)] for role in NORMAL]
    assert len(set(packets)) == 3 and len({without_ref(p) for p in packets}) == 1
    pid = f"{first}-a1"
    lock = pipeline.store.stage(pid, "lock")
    need = bm.Need.model_validate(lock["value"]["need"])
    cell = pipeline.world.cell(first)
    assert packets[2] == hp.judge_a_broker_packet(need, cell, ref=pipeline._ref("searcher_3", pid))
    inputs = {r["stage"]: json.loads(r["inputs"]) for r in pipeline.store.runs()}
    assert set(inputs["search_1"]) == set(inputs["search_2"]) == set(inputs["search_3"]) == {"lock"}
    for role in (*NORMAL, "audit_searcher"):
        assert hr.ROLE_CONTRACTS[role].template == "judge_a.v3"
        assert "the other searches' candidates" in " ".join(hr.ROLE_CONTRACTS[role].forbidden)


def test_a_need_planned_none_gets_the_three_searches(base, order, tmp_path):
    cell = next(c for c in order if _planned(base, c) == "none")
    pipeline, executor = built(base, tmp_path, [cell])
    roles = [role for role, c, _ in executor.calls if c == cell]
    assert set(NORMAL) <= set(roles)


def _planned(base, cell_id: str) -> str:
    return str(next(c.answerability for c in base["grid"].cells if c.cell_id == cell_id))


@pytest.mark.parametrize(
    ("role", "stage"),
    [("searcher_1", "search_1"), ("searcher_2", "search_2"), ("searcher_3", "search_3")],
)
def test_a_normal_search_declaring_a_problem_makes_the_need_judge_problem(
    base, first, tmp_path, role, stage
):
    """P1 #2: before any of its candidates is read; never a KeyError, never a retry."""
    scenario = Scenario(problem={(first, role): "the need cannot be judged"})
    pipeline, _ = built(base, tmp_path, [first], scenario)
    attempts = attempts_of(pipeline, first)
    assert attempts[0] == (1, "judge_problem")
    assert pipeline.store.summary()["technical_failures"] == {}
    rejected = [a for a in pipeline.store.attempts() if a["number"] == 1][0]
    assert rejected["stage"] == stage
    # the problem was admitted as the search's answer, and never consumed as candidates
    assert pipeline.store.stage(f"{first}-a1", "judge_b") is None


# ---------------------------------------------------------------------------
# the omission audit's search
# ---------------------------------------------------------------------------


def test_the_audit_search_runs_in_its_cells_blind_to_the_others(base, audited, tmp_path):
    pipeline, executor = built(base, tmp_path, [audited], Scenario(extra_grade={audited: 0}))
    audit = executor.packets[("audit_searcher", audited, 1)]
    one = executor.packets[("searcher_1", audited, 1)]
    assert without_ref(audit) == without_ref(one)
    inputs = {r["stage"]: json.loads(r["inputs"]) for r in pipeline.store.runs()}
    assert set(inputs["audit_search"]) == {"lock"}
    # the normal adjudication never reads the audit: the same in an audited cell as elsewhere
    judged = json.loads(next(r["inputs"] for r in pipeline.store.runs() if r["stage"] == "judge_b"))
    assert set(judged) == {"lock", "search_1", "search_2", "search_3"}


def test_a_node_new_to_the_audit_adjudicated_1_or_more_stops(base, audited, tmp_path):
    pipeline, _ = make(base, tmp_path / "test", Scenario(extra_grade={audited: 1}))
    with pytest.raises(hpl.ConstructionStop, match="outside the exposure pool") as stop:
        pipeline.build()
    assert stop.value.reason == "omission_audit"
    summary = pipeline.store.summary()
    assert (summary["status"], summary["stop"]["reason"]) == ("stopped", "omission_audit")


def test_a_node_new_to_the_audit_adjudicated_0_is_a_diagnostic(base, audited, tmp_path):
    pipeline, _ = built(base, tmp_path, [audited], Scenario(extra_grade={audited: 0}))
    ready = ready_of(pipeline, audited)
    graded = {f"{q.act}:{q.path}": q.grade for q in ready.qrels}
    assert EXTRA_NODE not in graded  # a diagnostic of the audit, no qrel of the need
    logged = [o for o in pipeline.omission_log if o["cell_id"] == audited]
    assert logged[0]["additional_final_grades"] == {EXTRA_NODE: 0}
    report = pipeline.report()["omission_audit"]
    assert (report["cells"], report["grade_0_additions"], report["attempts_audited"]) == (1, 1, 1)
    assert pipeline.progress_problems([audited]) == []


def test_a_node_a_normal_search_also_found_is_no_omission(base, audited, tmp_path):
    """The audit search lists, grade 3, the node only search_2 found: it is in the pool
    already, so nothing is new to the audit and nothing stops."""
    pipeline, _ = built(base, tmp_path, [audited], Scenario(b_extra={audited: 3}))
    ready = ready_of(pipeline, audited)
    assert ready.provenance["evidence_discovery"]["listed_by"][B_EXTRA_NODE] == ["search_2"]
    logged = [o for o in pipeline.omission_log if o["cell_id"] == audited]
    assert logged[0]["additional_nodes"] == []
    assert logged[0]["judgment_discrepancies"] == []  # the audit agrees with the final grade


def test_the_audit_search_declaring_a_problem_stops_the_construction(base, audited, tmp_path):
    scenario = Scenario(problem={(audited, "audit_searcher"): "cannot be judged"})
    pipeline, _ = make(base, tmp_path / "test", scenario)
    with pytest.raises(hpl.ConstructionStop) as stop:
        pipeline.build()
    assert stop.value.reason == "omission_audit_incomplete"
    summary = pipeline.store.summary()
    assert summary["stop"]["reason"] == "omission_audit_incomplete"
    assert summary["technical_failures"] == {}
    # no rejection: the audit's cells get no rejection the others do not get
    assert all(
        a["outcome"] in (None, "accepted")
        for a in pipeline.store.attempts()
        if a["cell_id"] == audited
    )
    # a stopped construction stays stopped
    pipeline.store.close()
    again, _ = reopen(base, tmp_path / "test", scenario)
    with pytest.raises(hpl.ConstructionStop, match="a person decides"):
        again.resume()


def test_the_audit_that_counts_is_the_ready_attempts(base, audited, tmp_path):
    """P1 #3: an earlier attempt of the cell, audited and then rejected, is provenance; the
    checkpoint and the freeze read the audit of the attempt that became READY."""
    # the diagnostic unavailable, so that judge D grades every item, and doubts attempt 1's
    scenario = Scenario(
        extra_grade={audited: 0}, qwen_overlength={audited}, d_uncertain={(audited, 1)}
    )
    pipeline, _ = built(base, tmp_path, [audited], scenario)
    assert attempts_of(pipeline, audited) == [(1, "adjudication_uncertain"), (2, "accepted")]
    logged = [o for o in pipeline.omission_log if o["cell_id"] == audited]
    assert [o["proposal_id"] for o in logged] == [f"{audited}-a1", f"{audited}-a2"]
    assert pipeline.progress_problems([audited]) == []
    report = pipeline.report()["omission_audit"]
    assert (report["cells"], report["attempts_audited"]) == (1, 2)
    # and the READY attempt's own audit is what is checked
    pipeline.omission_log[:] = [o for o in logged if o["proposal_id"].endswith("a1")]
    assert any(
        "omission audit of its READY attempt" in p for p in pipeline.progress_problems([audited])
    )


# ---------------------------------------------------------------------------
# the diagnostic's preflight, and what nothing else classifies
# ---------------------------------------------------------------------------


def test_a_prompt_over_the_limit_makes_the_diagnostic_unavailable_once(base, first, tmp_path):
    """P1 #6: one execution, never three identical ones; judge D grades every item."""
    pipeline, executor = built(base, tmp_path, [first], Scenario(qwen_overlength={first}))
    assert [r for r, c, _ in executor.calls if c == first].count("qwen_diagnostic") == 1
    assert pipeline.store.summary()["technical_failures"] == {}
    value = pipeline.store.stage(f"{first}-a1", "diagnostic")["value"]
    assert hj.diagnostic_unavailable(value) and value["defects"] == ["input_exceeds_context"]
    d = pipeline.store.stage(f"{first}-a1", "judge_d")["value"]
    candidates = [a.target for a in ready_of(pipeline, first).adjudications]
    ids = pipeline._item_ids(f"{first}-a1", "judge_d", candidates)
    assert {i["item_id"] for i in d["items"]} == set(ids.values())
    assert pipeline.report()["diagnostic"]["unavailable_defects"] == {"input_exceeds_context": 1}


def test_a_failure_nothing_classifies_is_recorded_as_a_stop(base, first, tmp_path, monkeypatch):
    def broken(*args: Any, **kwargs: Any) -> None:
        raise KeyError("a field nobody expected")

    monkeypatch.setattr(hpl.Pipeline, "_evidence_gate", broken)
    pipeline, _ = make(base, tmp_path / "test")
    with pytest.raises(hpl.ConstructionStop) as stop:
        pipeline.build()
    assert stop.value.reason == "unclassified_failure"
    summary = pipeline.store.summary()
    assert (summary["status"], summary["stop"]["reason"]) == ("stopped", "unclassified_failure")
    assert "KeyError" in summary["stop"]["detail"]
    assert "a field nobody expected" not in summary["stop"]["detail"]  # never the text


def test_an_audit_that_cannot_read_its_run_fails_closed_and_is_retried(
    base, first, tmp_path, monkeypatch
):
    real = hr.audit_role_run
    calls = {"n": 0}

    def once(run: hr.RoleRun, *args: Any, **kwargs: Any) -> hr.RoleAudit:
        calls["n"] += 1
        if run.role == "searcher_2" and calls.setdefault("broke", 0) == 0:
            calls["broke"] = 1
            raise ValueError("an unreadable session")
        return real(run, *args, **kwargs)

    monkeypatch.setattr(hr, "audit_role_run", once)
    pipeline, _ = make(base, tmp_path / "test")
    with pytest.raises(hpl.TechnicalStop):
        pipeline.build()
    summary = pipeline.store.summary()
    assert summary["technical_failures"] == {"isolation": 1}
    assert summary["waiting"][0]["stage"] == "search_2"
    pipeline.store.close()
    again, _ = reopen(base, tmp_path / "test")
    again.resuming = True
    again.build_cell(first)
    assert any(r.cell_id == first for r in again.ready.values())


def test_three_searches_replay_deterministically_with_every_write_refused(
    base, first, audited, tmp_path
):
    scenario = Scenario(b_extra={first: 2}, extra_grade={audited: 0})
    pipeline, _ = make(base, tmp_path / "test", scenario)
    for cell in ho.cell_order(base["grid"]):
        pipeline.build_cell(cell)
        if cell == audited:
            break
    before = pipeline.store.summary()
    pipeline.store.close()
    again, _ = reopen(base, tmp_path / "test", scenario)
    with again.store.no_writes():
        for cell in ho.cell_order(base["grid"]):
            again.build_cell(cell)
            if cell == audited:
                break
    assert again.store.summary() == before
    assert {n: r.record() for n, r in again.ready.items()} == {
        n: r.record() for n, r in pipeline.ready.items()
    }


def test_a_failure_of_the_state_medium_is_never_recorded_over(base, tmp_path, monkeypatch):
    """The state's own medium failing: nothing is written over it, not even a stop; the next
    open finds the state as the last whole transaction left it."""
    pipeline, _ = make(base, tmp_path / "test")
    real = pipeline.store.record

    def record(kind: str, *args: Any, **kwargs: Any) -> bool:
        if kind == "run_started" and pipeline.store.summary()["runs"]:
            raise __import__("sqlite3").OperationalError("disk I/O error")
        return real(kind, *args, **kwargs)

    monkeypatch.setattr(pipeline.store, "record", record)
    with pytest.raises(__import__("sqlite3").OperationalError):
        pipeline.build()
    monkeypatch.undo()
    summary = pipeline.store.verify()
    assert summary["status"] == "building" and summary["stop"] is None


def test_a_recovery_that_fails_keeps_its_own_reason(base, first, tmp_path, monkeypatch):
    """The process died after the call; reading the spool back raises a classified failure:
    its reason is recorded, never relabelled infrastructure."""
    scenario = Scenario(die_after=(first, "judge_b"))
    pipeline, _ = make(base, tmp_path / "test", scenario)
    with pytest.raises(Crash):
        pipeline.build()
    pipeline.store.close()
    again, executor = reopen(base, tmp_path / "test")

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise hpl.TechnicalFailure("transport", "the judge exited 1")

    monkeypatch.setattr(executor, "recover_headless", refuse)
    with pytest.raises(hpl.TechnicalStop):
        again.resume()
    assert again.store.summary()["technical_failures"] == {"transport": 1}


def test_an_answer_that_cannot_be_written_as_text_is_a_failed_execution(base, first, tmp_path):
    """A lone surrogate arrives through a JSON escape: refused at admission, never a write
    that fails in the state."""
    answer: dict[str, Any] = {
        "need_id": "x",
        "grades": [],
        "answerability": "complete",
        "answerability_reason": "\ud800",
        "missing": None,
    }
    scenario = Scenario(override={(first, "judge_b"): answer})
    pipeline, _ = make(base, tmp_path / "test", scenario)
    with pytest.raises(hpl.TechnicalStop):
        pipeline.build()
    summary = pipeline.store.verify()
    assert summary["technical_failures"] == {"malformed_session": 1}


def test_three_formulations_of_the_same_text_are_a_failed_execution():
    with pytest.raises(hpl.TechnicalFailure) as failure:
        hpl.parse_queries("en formal: Same?\nen natural: same?\nes natural: Otra?", "N101")
    assert failure.value.check == "query_formulations"
    with pytest.raises(hpl.TechnicalFailure):
        hpl.parse_queries("en formal: Same?\nen natural:   SAME?\nes natural: Otra?", "N101")
    assert len(hpl.parse_queries("en formal: A?\nen natural: B?\nes natural: C?", "N101")) == 3


def test_a_query_template_that_cannot_render_is_the_packets_failure(monkeypatch):
    """A statement that cites a provision is the statement's; a broken template is not."""
    found = hpl.statement_problems("Does Article 5 apply?", [])
    assert found["statement_cites_no_provision"]

    def broken(*args: Any, **kwargs: Any) -> str:
        raise bm.BenchmarkError("template query_variants.v1 must contain the marker once")

    monkeypatch.setattr(bm, "render_query_prompt", broken)
    with pytest.raises(bm.BenchmarkError):
        hpl.statement_problems("A firm asks a question.", [])


def test_a_state_artifact_that_cannot_be_written_is_the_mediums(base, tmp_path):
    pipeline, _ = make(base, tmp_path / "test")
    kept = pipeline.store.state / "artifacts"
    kept.mkdir(exist_ok=True)
    kept.chmod(0o500)
    try:
        with pytest.raises(__import__("ingest.holdout_state", fromlist=["x"]).MediumError):
            pipeline.build()
    finally:
        kept.chmod(0o755)
    summary = pipeline.store.verify()
    assert summary["stop"] is None  # nothing was written over the medium
