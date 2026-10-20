"""The orchestrator of TEST end to end, on the real grid and corpus with synthetic replies
(Codex Gate 2, P1.2 to P1.6). No model runs, no retriever runs, and no TEST content exists:
the evidence of every need is the pipeline fixtures'."""

from __future__ import annotations

import dataclasses

import pytest

from ingest import holdout as ho
from ingest import holdout_ledger as hl
from ingest import holdout_packets as hp
from ingest import holdout_pipeline as hpl
from ingest import holdout_roles as hr
from ingest import holdout_state as hs
from tests.holdout_fakes import Backups, Scenario, load_base, make

pytestmark = pytest.mark.corpus


@pytest.fixture(scope="module")
def base():
    return load_base()


@pytest.fixture(scope="module")
def full(base, tmp_path_factory):
    backups = Backups()
    pipeline, executor = make(base, tmp_path_factory.mktemp("full") / "test", backup=backups)
    pipeline.build()
    return pipeline, executor, backups


def test_a_full_run_reaches_a_freezable_state(full):
    pipeline, _, backups = full
    assert set(pipeline.states.values()) == {"ready"}
    assert pipeline.freeze_problems() == []
    assert len(pipeline.ready) == 60 and len(pipeline.store.ready()) == 60
    assert backups.labels == ["checkpoint 20", "checkpoint 40", "checkpoint 60"]


def test_the_cells_are_built_in_the_frozen_order(base, full):
    pipeline, executor, _ = full
    first_seen = list(dict.fromkeys(cell for _, cell, _ in executor.calls))
    assert first_seen == ho.cell_order(base["grid"])
    assert first_seen != [c.cell_id for c in base["grid"].cells]
    assert ho.cell_order(base["grid"]) == ho.cell_order(base["grid"])
    assert ho.plan_record(base["grid"]) == ho.plan_record(base["grid"])


def test_every_stage_was_audited_before_it_was_consumed(full):
    pipeline, _, _ = full
    audited = {a["run_id"] for a in pipeline.audits if a["ok"]}
    assert pipeline.consumed and set(pipeline.consumed) <= audited
    admitted = [r for r in pipeline.store.runs() if r["status"] == "admitted"]
    assert {r["run_id"] for r in admitted} == set(pipeline.consumed)
    assert all('"ok":true' in r["audit"] for r in admitted)


def test_ready_needs_and_cells_are_one_to_one(full):
    pipeline, _, _ = full
    ready = pipeline.ready.values()
    assert sorted(r.cell_id for r in ready) == sorted(
        c for c, s in pipeline.states.items() if s == "ready"
    )
    accepted = [a for a in pipeline.ledger.attempts if a.outcome == "accepted"]
    assert sorted(a.need_id for a in accepted) == sorted(pipeline.ready)
    assert len({r.need.need_id for r in ready}) == 60
    rows = pipeline.store.ready()
    assert len({r["cell_id"] for r in rows}) == len({r["need_id"] for r in rows}) == 60


def test_the_omission_audit_runs_on_its_cells_only(base, full):
    pipeline, executor, _ = full
    searched = {cell for role, cell, _ in executor.calls if role == "omission_searcher"}
    assert searched == set(ho.omission_cells(base["grid"]))
    assert len(pipeline.omission_log) == 12


def test_every_need_was_locked_once_before_its_judges(full):
    pipeline, _, _ = full
    assert sorted(pipeline.locks) == sorted(pipeline.ready)
    assert len(set(pipeline.locks)) == len(pipeline.locks)
    events = pipeline.store.events()
    for r in pipeline.store.ready():
        pid = r["proposal_id"]
        order = [e.stage for e in events if e.proposal_id == pid and e.kind == "stage_admitted"]
        assert order.index("lock") < order.index("judge_a")


def test_no_person_reviews_a_statement(full):
    pipeline, executor, _ = full
    roles = {role for role, _, _ in executor.calls}
    assert "review" not in roles and not hasattr(pipeline, "reviewer")
    assert "statement_rejected_by_review" not in hl.Outcome.__args__  # type: ignore[attr-defined]


def test_a_methodological_rejection_retries_blindly_with_frame_a(base, tmp_path):
    cell = ho.cell_order(base["grid"])[0]
    scenario = Scenario(gate={cell: ["paraphrase", "independent"]})
    pipeline, executor = make(base, tmp_path / "test", scenario)
    pipeline.build_cell(cell)
    first = executor.packets[("concept_author", cell, 1)]
    second = executor.packets[("concept_author", cell, 2)]
    assert first != second
    assert hp.attempt_frame() in first and hp.attempt_frame() in second

    def strip(packet: str) -> str:
        return packet.replace(packet.split("YOUR CELL (")[1].split(")")[0], "REF")

    assert strip(first) == strip(second)
    assert "attempt" not in first.split("YOUR CELL")[1].lower()
    attempts = pipeline.ledger.of_cell(cell)
    assert attempts[0].outcome == "overlap_paraphrase" and attempts[0].against == "dev"
    assert attempts[1].outcome == "accepted" and pipeline.states[cell] == "ready"


def test_three_methodological_rejections_leave_the_cell_unfilled(base, tmp_path):
    cell = ho.cell_order(base["grid"])[0]
    scenario = Scenario(gate={cell: ["uncertain", "same_rule_superficial_change", "paraphrase"]})
    pipeline, _ = make(base, tmp_path / "test", scenario)
    pipeline.build_cell(cell)
    assert pipeline.states[cell] == "unfilled_after_attempt_limit"
    assert [a.outcome for a in pipeline.ledger.of_cell(cell)] == [
        "overlap_uncertain",
        "overlap_same_rule",
        "overlap_paraphrase",
    ]
    stored = [a["outcome"] for a in pipeline.store.attempts() if a["cell_id"] == cell]
    assert stored == ["overlap_uncertain", "overlap_same_rule", "overlap_paraphrase"]


@pytest.mark.parametrize(
    ("kind", "role"),
    [
        ("raise", "judge_b"),
        ("contaminate", "judge_b"),
        ("contaminate", "concept_author"),
        ("resource", "judge_a"),
        ("protocol", "judge_a"),
        ("malformed", "gatekeeper"),
        ("raise", "metadata_adjudicator"),
        ("raise", "qwen_diagnostic"),
    ],
)
def test_a_technical_failure_keeps_the_same_need_and_stops(base, tmp_path, kind, role):
    cell = ho.cell_order(base["grid"])[0]
    scenario = Scenario(technical={cell: role}, technical_kind=kind)
    pipeline, executor = make(base, tmp_path / "test", scenario)
    with pytest.raises(hpl.TechnicalStop):
        pipeline.build()
    assert pipeline.states[cell] == "pending_technical"
    assert pipeline.store.cells()[0]["state"] == "pending_technical"
    attempts = pipeline.ledger.of_cell(cell)
    assert len(attempts) == 1 and attempts[0].outcome in (None, "accepted")
    assert not any(a.outcome not in (None, "accepted") for a in pipeline.ledger.attempts)
    assert executor.calls[-1] == (role, cell, 1)  # nothing ran after the failing stage
    assert pipeline.technical is not None and pipeline.technical["cell_id"] == cell
    with pytest.raises(hs.StateError, match="resume"):
        pipeline.build()  # a second start is refused: only resume() goes on
    assert any("pending" in p for p in pipeline.freeze_problems())


def test_an_unaudited_output_cannot_be_consumed(base, tmp_path):
    pipeline, _ = make(base, tmp_path / "test")
    bad = hpl.Audited("judge_b", "R-x", {}, hr.RoleAudit("judge_b", "R-x", False, ["x"], {}))
    with pytest.raises(hpl.TechnicalFailure):
        pipeline._consume(bad)


def test_an_omission_of_grade_1_or_more_stops_the_construction(base, tmp_path):
    cell = ho.omission_cells(base["grid"])[0]
    pipeline, _ = make(base, tmp_path / "test", Scenario(extra_grade={cell: 1}))
    with pytest.raises(hpl.ConstructionStop, match="omission audit"):
        pipeline.build()
    assert pipeline.store.status() == "stopped"
    with pytest.raises(hpl.ConstructionStop, match="stopped"):
        pipeline.resume()  # a rule's stop is never resumed past


def test_a_grade_0_omission_is_only_a_diagnostic(base, tmp_path):
    cell = ho.omission_cells(base["grid"])[0]
    pipeline, _ = make(base, tmp_path / "test", Scenario(extra_grade={cell: 0}))
    pipeline.build_cell(cell)
    assert pipeline.states[cell] == "ready"
    logged = [o for o in pipeline.omission_log if o["cell_id"] == cell][0]
    assert logged["additional_final_grades"] == {"MiCA:143.3": 0}


def test_an_ambiguous_judge_d_makes_the_adjudication_uncertain(base, tmp_path):
    cell = ho.cell_order(base["grid"])[0]
    scenario = Scenario(qwen_crosses={cell}, d_answer={cell: {"ambiguous": True, "grade": 1}})
    pipeline, _ = make(base, tmp_path / "test", scenario)
    pipeline.build_cell(cell)
    assert pipeline.states[cell] == "unfilled_after_attempt_limit"
    assert [a.outcome for a in pipeline.ledger.of_cell(cell)] == ["adjudication_uncertain"] * 3


def test_judge_d_holding_the_consensus_keeps_the_need(base, tmp_path):
    cell = ho.cell_order(base["grid"])[0]
    pipeline, _ = make(base, tmp_path / "test", Scenario(qwen_crosses={cell}))
    pipeline.build_cell(cell)
    assert pipeline.states[cell] == "ready"
    assert set(pipeline.diagnostic_log[0]["judge_d"].values()) == {"consensus_held"}


def test_the_50_rule_stops_before_the_minimum_is_out_of_reach(base, tmp_path):
    order = ho.cell_order(base["grid"])
    scenario = Scenario(gate={c: ["paraphrase"] * 3 for c in order[:11]})
    pipeline, _ = make(base, tmp_path / "test", scenario)
    with pytest.raises(hl.StopForThePerson):
        pipeline.build()
    cells = {c.cell_id: c for c in base["grid"].cells}
    unfilled = [c for c, s in pipeline.states.items() if s == "unfilled_after_attempt_limit"]
    evaluable_lost = [c for c in unfilled if cells[c].answerability != "none"]
    assert len(evaluable_lost) == 5  # 54 planned evaluable cells - 5 = 49 < 50
    assert pipeline.store.status() == "stopped"


def test_the_freeze_refuses_every_invalid_state(base, full):
    pipeline, _, _ = full
    assert pipeline.freeze_problems() == []

    def problems_after(change):
        saved = (
            dict(pipeline.states),
            dict(pipeline.ready),
            list(pipeline.audits),
            list(pipeline.consumed),
            list(pipeline.omission_log),
        )
        try:
            change()
            return pipeline.freeze_problems()
        finally:
            (
                pipeline.states,
                pipeline.ready,
                pipeline.audits,
                pipeline.consumed,
                pipeline.omission_log,
            ) = saved

    a_need = next(iter(pipeline.ready))
    a_cell = pipeline.ready[a_need].cell_id

    def failed_audit():
        pipeline.consumed.append("R-x")
        pipeline.audits.append(
            {"role": "judge_b", "run_id": "R-x", "proposal_id": "x", "ok": False, "problems": []}
        )

    cases = {
        "a cell not terminal": lambda: pipeline.states.__setitem__(a_cell, "judging"),
        "pending technical": lambda: pipeline.states.__setitem__(a_cell, "pending_technical"),
        "a ready need without its cell": lambda: pipeline.states.__setitem__(
            a_cell, "unfilled_after_attempt_limit"
        ),
        "an admitted output with a failed audit": failed_audit,
        "an omission audit missing": lambda: pipeline.omission_log.clear(),
        "an ambiguity": lambda: pipeline.ready.__setitem__(
            a_need,
            dataclasses.replace(
                pipeline.ready[a_need],
                qrels=[pipeline.ready[a_need].qrels[0].model_copy(update={"ambiguous": True})],
            ),
        ),
        "incomplete queries": lambda: pipeline.ready.__setitem__(
            a_need,
            dataclasses.replace(pipeline.ready[a_need], queries=pipeline.ready[a_need].queries[:2]),
        ),
        "fewer than 50 evaluable": lambda: [
            pipeline.ready.pop(n) for n in list(pipeline.ready)[:11]
        ],
    }
    for name, change in cases.items():
        assert problems_after(change), name
    assert any("under 50" in p for p in problems_after(cases["fewer than 50 evaluable"]))


def test_the_freeze_refuses_a_changed_contract(base, tmp_path):
    pipeline, _ = make(base, tmp_path / "test")

    def changed():
        raise RuntimeError("the contract changed")

    pipeline.world = dataclasses.replace(pipeline.world, contract_check=changed)
    with pytest.raises(hs.IncompatibleState, match="contract"):
        pipeline.build()
    assert any("pre-authoring contract" in p for p in pipeline.freeze_problems())


def test_the_freeze_verifies_a_backup_before_the_commitment(full):
    pipeline, _, backups = full
    exported: list[int] = []
    committed: list[dict] = []

    def export(records, report):
        exported.append(len(records))
        return {"needs.jsonl": "0" * 64}

    backups.fail = True
    with pytest.raises(hpl.ConstructionStop, match="backup"):
        pipeline.freeze(export, committed.append)
    assert committed == [] and pipeline.store.status() == "frozen"  # recorded, not committed
    backups.fail = False
    record = pipeline.freeze(export, committed.append)  # the same freeze, run again
    assert len(committed) == 1 and exported == [60, 60] and record["needs"]
    assert set(pipeline.states.values()) == {"frozen"}
    assert {c["state"] for c in pipeline.store.cells()} == {"frozen"}
    assert all(r["frozen"] == 1 for r in pipeline.store.ready())
    assert [e.kind for e in pipeline.store.events()].count("frozen") == 1
