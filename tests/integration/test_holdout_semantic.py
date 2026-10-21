"""The semantic admissibility of an answer (the person's rule of 2026-10-21): a well-formed answer
that violates a frozen semantic invariant tying its labels to the need's evidence or
answerability is one failed technical execution; the same invariant violated in all three
executions of search A, search B or the metadata adjudicator rejects the attempt
(semantic_inadmissible). A failure of the role's contract keeps its technical exhaustion, and so
does search C. On the real grid and corpus with synthetic replies; no model runs."""

from __future__ import annotations

import json
from typing import Any

import pytest

from ingest import benchmark as bm
from ingest import holdout as ho
from ingest import holdout_admission as ha
from ingest import holdout_judging as hj
from ingest import holdout_pipeline as hpl
from tests.holdout_fakes import COMPLETE, Scenario, load_base, make, reopen

pytestmark = pytest.mark.corpus

# metadata answers of the right form, each violating one frozen invariant on COMPLETE's grades
NEGATION = {"type": "direct", "difficulty": "medium", "drivers": {"negation": ["MiCA:149.3"]}}
UNSUPPORTED = {
    "type": "direct",
    "difficulty": "medium",
    "drivers": {"cross_reference": ["MiCA:149.1"]},  # a grade 1: no relevant support
}
TYPE = {"type": "partial_out_of_scope", "difficulty": "medium", "drivers": {}}
PROSE = "I would rather explain the metadata in prose."


@pytest.fixture(scope="module")
def base():
    return load_base()


@pytest.fixture(scope="module")
def order(base):
    return ho.cell_order(base["grid"])


def answerability(base, cell_id: str) -> str:
    return str(next(c.answerability for c in base["grid"].cells if c.cell_id == cell_id))


@pytest.fixture(scope="module")
def complete(base, order):
    """The first cell of the order the fakes answer complete, outside the omission audit."""
    omission = set(ho.omission_cells(base["grid"]))
    return next(c for c in order if c not in omission and answerability(base, c) == "complete")


@pytest.fixture(scope="module")
def audited(base, order):
    omission = set(ho.omission_cells(base["grid"]))
    return next(c for c in order if c in omission and answerability(base, c) == "complete")


def run_to(pipeline: hpl.Pipeline, order: list[str], target: str, first: bool) -> None:
    """START (first) or one RESUME, up to and with the target cell: the cells before it are
    built or replayed, the target's waiting stage gets one more technical execution."""
    pipeline._reset()
    pipeline.resuming = not first
    for cell in order:
        pipeline.build_cell(cell)
        if cell == target:
            return


def executions(pipeline, order, target, times: int) -> list[str]:
    """START, then RESUME, times in all, each technical stop recorded by its kind."""
    found = []
    for n in range(times):
        try:
            run_to(pipeline, order, target, first=n == 0)
            found.append("done")
            return found
        except hpl.TechnicalStop:
            found.append(pipeline.store.status())
    return found


def attempts_of(pipeline, cell: str) -> list[tuple[int, str | None]]:
    return [(a["number"], a["outcome"]) for a in pipeline.store.attempts() if a["cell_id"] == cell]


def failures(pipeline: hpl.Pipeline, stage: str) -> list[dict[str, Any]]:
    runs = [
        r["run_id"]
        for r in pipeline.store.runs()
        if r["stage"] == stage and r["status"] == "failed"
    ]
    return pipeline.store.failure_payloads(runs)


# ---------------------------------------------------------------------------
# the rule, as a function
# ---------------------------------------------------------------------------


def payload(check="metadata_answer", violations=("negation_without_grade_3",), ok=True):
    return {"check": check, "violations": list(violations), "audit": {"ok": ok}}


def test_three_executions_violating_one_invariant_are_semantic_exhaustion():
    found = hpl.semantic_exhaustion([payload(), payload(), payload()])
    assert found == ("metadata_answer", ["negation_without_grade_3"])
    shared = hpl.semantic_exhaustion(
        [
            payload(violations=["negation_without_grade_3", "type_against_answerability"]),
            payload(violations=["negation_without_grade_3"]),
            payload(violations=["negation_without_grade_3", "driver_without_relevant_support"]),
        ]
    )
    assert shared == ("metadata_answer", ["negation_without_grade_3"])


@pytest.mark.parametrize(
    "failed",
    [
        [payload(), payload()],  # two executions only
        [payload(), payload(), payload(violations=["type_against_answerability"])],  # no shared
        [payload(), payload(), payload(check="judge_answer_complete")],  # another check
        [payload(), payload(), payload(check="reply_is_json", violations=())],  # a contract one
        [payload(), payload(), payload(ok=False)],  # an execution that failed its audit
        [payload(check="query_formulations")] * 3,  # a check that is never semantic
        [payload(violations=())] * 3,  # no invariant at all
    ],
)
def test_anything_else_is_no_semantic_exhaustion(failed):
    assert hpl.semantic_exhaustion(failed) is None


def test_the_metadata_verdict_refuses_exactly_what_the_frozen_rule_refuses():
    """The split into contract and invariants changes nothing of what is admitted."""
    qrels = [
        bm.Qrel(
            need_id="N101",
            act=n.split(":")[0],  # type: ignore[arg-type]
            path=n.split(":")[1],
            grade=g,
            rationale="r",
            text_sha256="0" * 64,
        )
        for n, g in COMPLETE
    ]
    cases = [
        NEGATION,
        UNSUPPORTED,
        TYPE,
        {"type": "direct", "difficulty": "medium", "drivers": {}},
        {"type": "nonsense", "difficulty": "medium", "drivers": {}},
        {"type": "direct", "difficulty": "trivial", "drivers": {}},
        {"type": "direct", "difficulty": "easy", "drivers": {"cross_act": ["MiCA:149.2"]}},
        {"type": "direct", "difficulty": "easy", "drivers": {"negation": []}},
        {"type": "direct", "difficulty": "easy", "drivers": {"negation": ["MiCA:1.1"]}},
        {"type": "direct", "difficulty": "easy", "drivers": {"negation": ["gap_note"]}},
        {"type": "direct", "difficulty": "easy", "drivers": {"numeric": ["gap_note"]}},
        {"type": "direct", "difficulty": "easy", "drivers": {"negation": ["MiCA:149.2"]}},
        {"type": "direct", "difficulty": "easy", "drivers": {"lexical_twin": ["DORA:64"]}},
    ]
    for answer in cases:
        for kind in ("complete", "partial"):
            verdict = ha.metadata_verdict(answer, qrels, kind)
            assert verdict.admitted == (hj.metadata_problems(answer, qrels, kind) == []), answer
    assert ha.metadata_verdict(NEGATION, qrels, "complete").violations == [
        "negation_without_grade_3"
    ]
    assert ha.metadata_verdict(UNSUPPORTED, qrels, "complete").violations == [
        "driver_without_relevant_support"
    ]
    assert ha.metadata_verdict(TYPE, qrels, "complete").violations == ["type_against_answerability"]
    numeric = {"type": "direct", "difficulty": "easy", "drivers": {"numeric": ["gap_note"]}}
    assert ha.metadata_verdict(numeric, qrels, "complete").violations == [
        "gap_driver_on_complete_need"
    ]
    # the role's contract is never an invariant: not shown, not of the rubric, no node named
    for contract in cases[4:10]:
        assert ha.metadata_verdict(contract, qrels, "complete").violations == []


# ---------------------------------------------------------------------------
# the transition, on the stand-in construction
# ---------------------------------------------------------------------------


def scenario(cell: str, role: str, replies: list[Any]) -> Scenario:
    return Scenario(per_execution={(cell, role): list(replies)})


def test_one_and_two_failures_of_the_same_invariant_are_technical_retries(
    base, order, complete, tmp_path
):
    pipeline, _ = make(
        base, tmp_path / "test", scenario(complete, "metadata_adjudicator", [NEGATION] * 2)
    )
    assert executions(pipeline, order, complete, 2) == ["pending_technical", "pending_technical"]
    assert attempts_of(pipeline, complete)[-1][1] == "accepted"  # not rejected yet
    found = failures(pipeline, "metadata")
    assert [f["violations"] for f in found] == [["negation_without_grade_3"]] * 2
    assert [f["state"] for f in found] == ["pending_technical"] * 2


@pytest.mark.parametrize(
    "answer", [NEGATION, UNSUPPORTED, TYPE], ids=["negation", "driver", "type"]
)
def test_three_failures_of_the_same_metadata_invariant_reject_the_attempt(
    base, order, complete, tmp_path, answer
):
    pipeline, _ = make(
        base, tmp_path / "test", scenario(complete, "metadata_adjudicator", [answer] * 3)
    )
    assert executions(pipeline, order, complete, 3) == [
        "pending_technical",
        "pending_technical",
        "done",
    ]
    assert attempts_of(pipeline, complete) == [(1, "semantic_inadmissible"), (2, "accepted")]
    invariant = ha.metadata_verdict(answer, _complete_qrels(), "complete").violations[0]
    rejected = next(a for a in pipeline.store.attempts() if a["proposal_id"] == f"{complete}-a1")
    assert rejected["note"] == f"metadata_answer: {invariant} in 3 executions"
    assert rejected["stage"] == "metadata"
    found = failures(pipeline, "metadata")
    assert [f["violations"] for f in found] == [[invariant]] * 3
    assert [f["state"] for f in found] == ["pending_technical", "pending_technical", "judging"]
    summary = pipeline.store.verify()
    assert summary["status"] == "building" and summary["technical_failures"] == {
        "malformed_session": 3
    }
    assert any(r.cell_id == complete for r in pipeline.ready.values())
    assert pipeline.report()["rejections"]["semantic_inadmissible"] == {rejected["note"]: 1}


def _complete_qrels() -> list[bm.Qrel]:
    return [
        bm.Qrel(
            need_id="N101",
            act=n.split(":")[0],  # type: ignore[arg-type]
            path=n.split(":")[1],
            grade=g,
            rationale="r",
            text_sha256="0" * 64,
        )
        for n, g in COMPLETE
    ]


def test_three_different_invariants_are_no_rejection(base, order, complete, tmp_path):
    replies = [NEGATION, UNSUPPORTED, TYPE]
    pipeline, _ = make(base, tmp_path / "test", scenario(complete, "metadata_adjudicator", replies))
    assert executions(pipeline, order, complete, 3)[-1] == "pending_technical_unresolved"
    assert attempts_of(pipeline, complete)[-1] == (1, "accepted")


@pytest.mark.parametrize(
    ("role", "replies", "stage"),
    [
        pytest.param("metadata_adjudicator", [PROSE] * 3, "metadata", id="malformed_json"),
        pytest.param(
            "metadata_adjudicator", [NEGATION, PROSE, NEGATION], "metadata", id="one_contract"
        ),
        pytest.param(
            "query_writer",
            ["en formal: Same?\nen natural: same?\nes natural: Otra?"] * 3,
            "query_formulation",
            id="duplicate_queries",
        ),
        pytest.param(
            "gatekeeper",
            [{"proposal_id": "x", "verdict": "paraphrase", "closest": "N999", "reason": "x"}] * 3,
            "overlap_gate",
            id="unseen_need",
        ),
    ],
)
def test_a_contract_failure_three_times_keeps_its_technical_exhaustion(
    base, order, complete, tmp_path, role, replies, stage
):
    pipeline, _ = make(base, tmp_path / "test", scenario(complete, role, replies))
    assert executions(pipeline, order, complete, 3)[-1] == "pending_technical_unresolved"
    summary = pipeline.store.verify()
    assert summary["waiting"][0]["stage"] == stage
    assert all(
        a["outcome"] in (None, "accepted")
        for a in pipeline.store.attempts()
        if a["cell_id"] == complete
    )


def search_without_grade_3() -> dict[str, Any]:
    """A search's final answer of the right form whose complete answerability its own grades
    do not support: no grade 3."""
    return {
        "candidates": [
            {
                "node": n,
                "scope": [],
                "grade": min(g, 2),
                "ambiguous": False,
                "scope_basis": [n] if g == 0 else [],
                "rationale": "quotes the node",
            }
            for n, g in COMPLETE
        ],
        "answerability": "complete",
        "answerability_reason": "A",
        "gap_note": None,
        "drivers": {},
        "problem": None,
    }


@pytest.mark.parametrize(
    ("role", "stage"), [("judge_a", "judge_a"), ("second_searcher", "second_search")]
)
def test_a_search_a_or_b_without_support_three_times_rejects_the_attempt(
    base, order, complete, tmp_path, role, stage
):
    replies = [search_without_grade_3() for _ in range(3)]
    pipeline, _ = make(base, tmp_path / "test", scenario(complete, role, replies))
    assert executions(pipeline, order, complete, 3)[-1] == "done"
    assert attempts_of(pipeline, complete) == [(1, "semantic_inadmissible"), (2, "accepted")]
    rejected = next(
        a for a in pipeline.store.attempts() if a["number"] == 1 and a["cell_id"] == complete
    )
    assert rejected["note"] == "judge_answer_complete: complete_without_grade_3 in 3 executions"
    assert rejected["stage"] == stage


def test_search_c_without_support_three_times_keeps_its_technical_exhaustion(
    base, order, audited, tmp_path
):
    """The audit's cells get no rejection the others do not get."""
    replies = [search_without_grade_3() for _ in range(3)]
    pipeline, _ = make(base, tmp_path / "test", scenario(audited, "omission_searcher", replies))
    assert executions(pipeline, order, audited, 3)[-1] == "pending_technical_unresolved"
    assert pipeline.store.verify()["waiting"][0]["stage"] == "omission_search"


def test_three_attempts_rejected_so_leave_the_cell_unfilled(base, order, complete, tmp_path):
    pipeline, _ = make(
        base, tmp_path / "test", scenario(complete, "metadata_adjudicator", [NEGATION] * 9)
    )
    assert executions(pipeline, order, complete, 9)[-1] == "done"
    assert attempts_of(pipeline, complete) == [(n, "semantic_inadmissible") for n in (1, 2, 3)]
    assert next(c for c in pipeline.store.cells() if c["cell_id"] == complete)["state"] == (
        "unfilled_after_attempt_limit"
    )


def test_the_rejection_replays_to_the_same_durable_result_and_nothing_is_repaired(
    base, order, complete, tmp_path
):
    pipeline, _ = make(
        base, tmp_path / "test", scenario(complete, "metadata_adjudicator", [NEGATION] * 3)
    )
    executions(pipeline, order, complete, 3)
    before = pipeline.store.verify()
    records = {n: r.record() for n, r in pipeline.ready.items()}
    pipeline.store.close()
    again, _ = reopen(base, tmp_path / "test")  # no special reply: nothing may run
    with again.store.no_writes():
        again._reset()
        for cell in order:
            again.build_cell(cell)
            if cell == complete:
                break
    assert again.store.verify() == before
    assert {n: r.record() for n, r in again.ready.items()} == records
    # the rejected attempt has no admitted metadata, and the invalid label is nowhere admitted
    assert again.store.stage(f"{complete}-a1", "metadata") is None
    rows = [again.store.stage(s["proposal_id"], s["stage"]) for s in again.store.stages()]
    admitted = json.dumps([row["value"] for row in rows if row is not None])
    assert '"negation"' not in admitted
    ready = next(r for r in again.ready.values() if r.cell_id == complete)
    assert ready.proposal_id == f"{complete}-a2" and "negation" not in ready.metadata.drivers
