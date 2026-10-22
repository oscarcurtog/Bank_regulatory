"""The identity of what a judge grades is the machine's (the person's decision of 2026-10-22):
judges B, C and D, and the diagnostic, receive each candidate under an opaque id the
orchestrator gave it and answer by that exact id; the orchestrator alone maps it to the legal
node. A missing, unknown, repeated or other id, or an identity written beside the id that names
another node, is a technical failure of the role's contract, never repaired, never mapped back by
position, and never a semantic rejection. On the real grid and corpus with synthetic replies;
no model runs, and no TEST text."""

from __future__ import annotations

import re
from typing import Any

import pytest

from ingest import holdout as ho
from ingest import holdout_admission as ha
from ingest import holdout_judging as hj
from ingest import holdout_pipeline as hpl
from tests.holdout_fakes import COMPLETE, Scenario, load_base, make, misbind, reopen

pytestmark = pytest.mark.corpus

DECIDED = "MiCA:149.3"  # judge B grades it one step off: judge C decides it
SECOND = "MiCA:149.4"


@pytest.fixture(scope="module")
def base():
    return load_base()


@pytest.fixture(scope="module")
def order(base):
    return ho.cell_order(base["grid"])


@pytest.fixture(scope="module")
def cell(base, order):
    """The first cell of the order the fakes answer complete, outside the omission audit."""
    omission = set(ho.omission_cells(base["grid"]))
    return next(
        c.cell_id
        for c in sorted(base["grid"].cells, key=lambda c: order.index(c.cell_id))
        if c.cell_id not in omission and c.answerability == "complete"
    )


def run_to(pipeline: hpl.Pipeline, order: list[str], target: str, first: bool) -> None:
    pipeline._reset()
    pipeline.resuming = not first
    for c in order:
        pipeline.build_cell(c)
        if c == target:
            return


def executions(pipeline, order, target, times: int) -> list[str]:
    found = []
    for n in range(times):
        try:
            run_to(pipeline, order, target, first=n == 0)
            found.append("done")
            return found
        except hpl.TechnicalStop:
            found.append(pipeline.store.status())
    return found


def final_grades(pipeline: hpl.Pipeline, cell: str) -> dict[str, int]:
    ready = next(r for r in pipeline.ready.values() if r.cell_id == cell)
    return {f"{q.act}:{q.path}": q.grade for q in ready.qrels}


def judge_c_failures(pipeline: hpl.Pipeline) -> list[dict[str, Any]]:
    runs = [
        r["run_id"]
        for r in pipeline.store.runs()
        if r["stage"] == "judge_c" and r["status"] == "failed"
    ]
    return pipeline.store.failure_payloads(runs)


# ---------------------------------------------------------------------------
# the ids
# ---------------------------------------------------------------------------


def test_an_id_names_nothing_and_is_owned_by_one_stage_of_one_attempt():
    salt = b"salt"
    one = ho.item_id(salt, "G01-a1:judge_c", "MiCA:149.3")
    assert re.fullmatch(r"C-[0-9a-f]{12}", one) and "149" not in one
    assert one == ho.item_id(salt, "G01-a1:judge_c", "MiCA:149.3")  # deterministic: replay
    assert one != ho.item_id(salt, "G01-a1:judge_b", "MiCA:149.3")  # another stage
    assert one != ho.item_id(salt, "G01-a2:judge_c", "MiCA:149.3")  # another attempt
    assert one != ho.item_id(b"other", "G01-a1:judge_c", "MiCA:149.3")  # another salt


# ---------------------------------------------------------------------------
# judge C, one item and several
# ---------------------------------------------------------------------------


def test_a_one_item_judge_c_answering_by_its_id_decides_its_own_node(base, order, cell, tmp_path):
    pipeline, executor = make(base, tmp_path / "test", Scenario(b_shift={cell: [DECIDED]}))
    assert executions(pipeline, order, cell, 1) == ["done"]
    packet = executor.packets[("judge_c", cell, 1)]
    assert len(re.findall(r"^ITEM C-[0-9a-f]{12}", packet, re.M)) == 1
    assert final_grades(pipeline, cell)[DECIDED] == dict(COMPLETE)[DECIDED]  # C's, on its node


@pytest.mark.parametrize(
    "fault", ["wrong", "missing", "unknown", "duplicate", "conflicting_target"]
)
def test_a_judge_c_answer_with_a_broken_identity_is_a_technical_failure(
    base, order, cell, tmp_path, fault
):
    """Never repaired, never mapped back by position: the failed execution has no consumer,
    and the next one, in a new context, answers by the right id."""
    scenario = Scenario(b_shift={cell: [DECIDED]}, identity={(cell, "judge_c"): [fault]})
    pipeline, _ = make(base, tmp_path / "test", scenario)
    assert executions(pipeline, order, cell, 2) == ["pending_technical", "done"]
    (failure,) = judge_c_failures(pipeline)
    assert (failure["reason"], failure["check"], failure["violations"]) == (
        "malformed_session",
        "judge_answer_complete",
        [],
    )
    assert final_grades(pipeline, cell)[DECIDED] == dict(COMPLETE)[DECIDED]


def test_judge_c_naming_another_node_of_the_same_act_three_times_is_technical_exhaustion(
    base, order, cell, tmp_path
):
    """The class of the fourth run's stop, rebuilt with no TEST text: the right id, and beside
    it a node of the same act with other article and paragraph numbers. Never a semantic
    rejection: the role did not follow its contract; nothing shows the need inadmissible."""
    faults = ["conflicting_target"] * 3
    scenario = Scenario(b_shift={cell: [DECIDED]}, identity={(cell, "judge_c"): faults})
    pipeline, _ = make(base, tmp_path / "test", scenario)
    assert executions(pipeline, order, cell, 3)[-1] == "pending_technical_unresolved"
    failures = judge_c_failures(pipeline)
    assert len(failures) == 3 and all(f["violations"] == [] for f in failures)
    (attempt,) = [a for a in pipeline.store.attempts() if a["cell_id"] == cell]
    assert attempt["outcome"] == "accepted"  # still the gate's: never semantic_inadmissible
    pid = attempt["proposal_id"]
    assert pipeline.store.stage(pid, "judge_b") is not None
    assert pipeline.store.stage(pid, "judge_c") is None  # nothing admitted, nothing repaired


def test_a_multi_item_judge_c_cannot_swap_its_decisions_between_candidates(
    base, order, cell, tmp_path
):
    """Its entries in another order bind by their ids: each decision stays on its node."""
    shifted = Scenario(b_shift={cell: [DECIDED, SECOND]})
    reference, _ = make(base, tmp_path / "ordered" / "test", shifted)
    assert executions(reference, order, cell, 1) == ["done"]
    reversed_ = Scenario(
        b_shift={cell: [DECIDED, SECOND]}, identity={(cell, "judge_c"): ["reversed"]}
    )
    pipeline, executor = make(base, tmp_path / "reversed" / "test", reversed_)
    assert executions(pipeline, order, cell, 1) == ["done"]
    packet = executor.packets[("judge_c", cell, 1)]
    assert len(re.findall(r"^ITEM C-[0-9a-f]{12}", packet, re.M)) == 2  # the two nodes
    assert final_grades(pipeline, cell) == final_grades(reference, cell)


def test_a_reason_that_names_another_provision_leaves_the_item_its_own(base, order, cell, tmp_path):
    scenario = Scenario(
        b_shift={cell: [DECIDED]}, c_reason={cell: "See also MiCA:150.1 and DORA:28.4."}
    )
    pipeline, _ = make(base, tmp_path / "test", scenario)
    assert executions(pipeline, order, cell, 1) == ["done"]
    grades = final_grades(pipeline, cell)
    assert grades[DECIDED] == dict(COMPLETE)[DECIDED]
    assert "MiCA:150.1" not in grades and "DORA:28.4" not in grades


# ---------------------------------------------------------------------------
# judges B and D, and the diagnostic, by the same contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fault", ["missing", "unknown", "duplicate", "wrong"])
def test_a_judge_b_answer_with_a_broken_identity_is_a_technical_failure(
    base, order, cell, tmp_path, fault
):
    scenario = Scenario(identity={(cell, "judge_b"): [fault]})
    pipeline, _ = make(base, tmp_path / "test", scenario)
    assert executions(pipeline, order, cell, 2) == ["pending_technical", "done"]
    assert pipeline.store.summary()["technical_failures"] == {"malformed_session": 1}


def test_a_judge_b_answer_in_another_order_binds_by_its_ids(base, order, cell, tmp_path):
    reference, _ = make(base, tmp_path / "ordered" / "test")
    executions(reference, order, cell, 1)
    pipeline, _ = make(
        base, tmp_path / "reversed" / "test", Scenario(identity={(cell, "judge_b"): ["reversed"]})
    )
    assert executions(pipeline, order, cell, 1) == ["done"]
    assert final_grades(pipeline, cell) == final_grades(reference, cell)


def test_a_judge_d_answer_with_an_unknown_id_is_a_technical_failure(base, order, cell, tmp_path):
    scenario = Scenario(qwen_overlength={cell}, identity={(cell, "judge_d"): ["unknown"]})
    pipeline, _ = make(base, tmp_path / "test", scenario)
    assert executions(pipeline, order, cell, 2) == ["pending_technical", "done"]
    waiting = [r for r in pipeline.store.runs() if r["status"] == "failed"]
    assert [r["stage"] for r in waiting] == ["judge_d"]


def test_an_identity_failure_replays_to_the_same_durable_result(base, order, cell, tmp_path):
    scenario = Scenario(b_shift={cell: [DECIDED]}, identity={(cell, "judge_c"): ["unknown"]})
    pipeline, _ = make(base, tmp_path / "test", scenario)
    executions(pipeline, order, cell, 2)
    before = pipeline.store.verify()
    records = {n: r.record() for n, r in pipeline.ready.items()}
    pipeline.store.close()
    again, _ = reopen(base, tmp_path / "test", Scenario(b_shift={cell: [DECIDED]}))
    with again.store.no_writes():
        again._reset()
        for c in order:
            again.build_cell(c)
            if c == cell:
                break
    assert again.store.verify() == before
    assert {n: r.record() for n, r in again.ready.items()} == records


# ---------------------------------------------------------------------------
# the rule, on the answers themselves
# ---------------------------------------------------------------------------


def test_misbinding_never_survives_admission_and_the_mapping_is_never_the_judges():
    node_of = {"C-" + "1" * 12: "MiCA:149.3", "C-" + "2" * 12: "answerability"}
    entries: list[dict[str, Any]] = [
        {
            "item_id": "C-" + "1" * 12,
            "grade": 2,
            "ambiguous": False,
            "answerability": None,
            "reason": "C",
        },
        {
            "item_id": "C-" + "2" * 12,
            "grade": None,
            "ambiguous": False,
            "answerability": "complete",
            "reason": "C",
        },
    ]

    class Item:
        def __init__(self, item_id: str, target: str) -> None:
            self.item_id, self.target = item_id, target

    items = [Item(i, t) for i, t in node_of.items()]
    assert ha.judge_c_problems({"items": entries}, items) == []
    for fault in ("wrong", "missing", "unknown", "duplicate", "conflicting_target"):
        broken = misbind(entries, fault, node_of)
        assert ha.judge_c_problems({"items": broken}, items), fault
    # the right ids in another order are the same decisions, bound to the same nodes
    reordered = misbind(entries, "reversed", node_of)
    assert ha.judge_c_problems({"items": reordered}, items) == []
    by_id = {e["item_id"]: e for e in reordered}
    assert {i.target: by_id[i.item_id]["grade"] for i in items} == {
        "MiCA:149.3": 2,
        "answerability": None,
    }
    # an id the orchestrator did not give is never resolved to a node, by position or otherwise
    with pytest.raises(KeyError):
        hj.judge_d_outcome(hj.Assembled(), [{**entries[1], "item_id": "C-" + "0" * 12}], node_of)
