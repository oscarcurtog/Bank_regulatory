"""The orchestrator of TEST end to end, on the real grid and corpus with synthetic replies
(Codex Gate 2, P1.2 to P1.6), with exposure v2's audit, the development need exclusions and
D-061's review metrics (D-066). No model runs, no retriever runs, and no TEST content exists:
the evidence of every need is the pipeline fixtures'."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

from ingest import holdout as ho
from ingest import holdout_exclusions as hxe
from ingest import holdout_ledger as hl
from ingest import holdout_packets as hp
from ingest import holdout_pipeline as hpl
from ingest import holdout_roles as hr
from ingest import holdout_state as hs
from tests.holdout_fakes import (
    FAKE_EXCLUSIONS,
    Backups,
    Scenario,
    attach,
    load_base,
    make,
    raw_dir,
    world,
)

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
    searched = {cell for role, cell, _ in executor.calls if role == "audit_searcher"}
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
        assert order.index("lock") < order.index("search_1")


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
        ("resource", "searcher_1"),
        ("protocol", "searcher_3"),
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
    assert logged["additional_final_grades"] == {"MiCA:143.3": 0}  # no article expanded


# ---------------------------------------------------------------------------
# exposure v2 in the omission audit (D-066): the same rules, a global stop
# ---------------------------------------------------------------------------


def dora_28(base) -> list[str]:
    """DORA Article 28's nodes with their own text, in document order: none is evidence of the
    synthetic replies, nor of an article they list."""
    found = [
        f"{a}:{p}"
        for (a, p), text in base["texts"].items()
        if (a, p.split(".")[0]) == ("DORA", "28")
    ]
    return [n for n in base["corpus"].sort(found) if base["texts"][hp.Corpus.key(n)].strip()]


def test_v2_j_an_audit_node_only_a_search_page_named_graded_1_stops_everything(base, tmp_path):
    """A node the audit search's `search` page alone named, followed up and adjudicated grade 1:
    the whole construction stops; no replacement, no new attempt of the cell, no next cell."""
    order = ho.cell_order(base["grid"])
    cell = next(c for c in order if c in set(ho.omission_cells(base["grid"])))
    node = max(dora_28(base), key=lambda n: len(base["texts"][hp.Corpus.key(n)]))
    text = base["texts"][hp.Corpus.key(node)]
    pattern = re.escape(text[len(text) // 2 : len(text) // 2 + 60])
    scenario = Scenario(
        snippet={(cell, "audit_searcher"): pattern},
        pool_grade={(cell, "pool_judge_1"): {node: 1}, (cell, "pool_judge_2"): {node: 1}},
    )
    pipeline, executor = make(base, tmp_path / "test", scenario)
    with pytest.raises(hpl.ConstructionStop, match="outside the exposure pool") as stop:
        pipeline.build()
    assert (stop.value.reason, stop.value.cell_id) == ("omission_audit", cell)
    audit = pipeline.store.stage(f"{cell}-a1", "omission_audit")
    assert audit is not None
    assert audit["value"]["final_grades"][node] == 1
    assert audit["value"]["sources"][node] == ["snippet_follow_up"]
    # global: no cell after it was touched, and the cell has no other attempt
    assert {c for _, c, _ in executor.calls} <= set(order[: order.index(cell) + 1])
    assert [a["proposal_id"] for a in pipeline.store.attempts() if a["cell_id"] == cell] == [
        f"{cell}-a1"
    ]
    assert pipeline.store.status() == "stopped"
    with pytest.raises(hpl.ConstructionStop, match="stopped"):
        pipeline.resume()
    review = pipeline.report()["review_metrics"]["omission_audit_all_attempts"]
    assert review["grade_1"] >= 1 and review["by_category"]["snippet_follow_up"]["grade_1"] == 1


def test_v2_k_an_audit_node_reached_as_an_ancestor_graded_0_is_a_diagnostic(base, tmp_path):
    cell = ho.omission_cells(base["grid"])[0]
    point = next(
        n
        for n in dora_28(base)
        if base["texts"].get(hp.Corpus.key(n.rsplit(".", 1)[0]), "").strip() and n.count(".") >= 2
    )
    lead_in = point.rsplit(".", 1)[0]
    pipeline, _ = make(
        base, tmp_path / "test", Scenario(expose={(cell, "audit_searcher"): [point]})
    )
    pipeline.build_cell(cell)
    assert pipeline.states[cell] == "ready"
    logged = [o for o in pipeline.omission_log if o["cell_id"] == cell][0]
    assert logged["additional_final_grades"] == {lead_in: 0, point: 0}
    assert logged["sources"] == {lead_in: ["ancestor_closure"], point: ["full_text"]}
    review = pipeline.report()["review_metrics"]["omission_audit_ready"]
    assert (review["audit_new"], review["grade_0"], review["grade_1"]) == (2, 2, 0)
    assert review["by_category"]["ancestor_closure"]["grade_0"] == 1
    assert review["by_category"]["other_full_text_reach"]["grade_0"] == 1


# ---------------------------------------------------------------------------
# the development need exclusions (D-066): hidden references of the gate, never an author's
# ---------------------------------------------------------------------------

IDENTITY = {"schema": hxe.SCHEMA, "sha256": "e" * 64, "count": len(FAKE_EXCLUSIONS)}


def with_exclusions(base, directory: Path, scenario: Scenario | None = None):
    """A new construction whose world carries the development need exclusions."""
    w = dataclasses.replace(
        world(base), development_exclusions=FAKE_EXCLUSIONS, development=IDENTITY
    )
    store = hs.Store.create(
        directory, raw_dir(directory), w.identities, w.order, w.omission, runtime=w.runtime
    )
    return attach(w, store, scenario, None)


def test_v2_l_the_gate_compares_with_the_development_needs_and_no_author_sees_them(base, tmp_path):
    cell = ho.cell_order(base["grid"])[0]
    pipeline, executor = with_exclusions(base, tmp_path / "test")
    pipeline.build_cell(cell)
    gate = executor.packets[("gatekeeper", cell, 1)]
    # overlap_gate.v2: the four kinds of reference, described, and the rejection alike
    assert "You see them all, in four kinds:" in gate and "(DEVELOPMENT)" in gate
    assert "a proposal that repeats a DEVELOPMENT need is not new" in gate
    for e in FAKE_EXCLUSIONS:
        ref = ho.opaque_id(b"test-salt", f"gate-development:{cell}-a1", e.key)
        assert f"- {ref} (DEVELOPMENT): {e.statement}" in gate.splitlines()
        assert e.key not in gate  # an opaque id drawn for this proposal, never the key
    others = [p for role, _, _, p in executor.sent if role != "gatekeeper"]
    assert {r for r, _, _, _ in executor.sent} >= {"concept_author", "statement_writer"}
    assert not any(e.statement in p or e.key in p for p in others for e in FAKE_EXCLUSIONS)
    gate_run = next(r for r in pipeline.store.runs() if r["stage"] == "overlap_gate")
    assert json.loads(gate_run["inputs"])["development_exclusions"] == IDENTITY["sha256"]
    assert pipeline.store.verify()["cells"]  # the state verifies with that input
    assert pipeline.states[cell] == "ready"


def test_v2_l_a_proposal_repeating_a_development_need_is_an_overlap(base, tmp_path):
    cell = ho.cell_order(base["grid"])[0]
    ref = ho.opaque_id(b"test-salt", f"gate-development:{cell}-a1", FAKE_EXCLUSIONS[0].key)
    scenario = Scenario(gate={cell: ["paraphrase", "independent"]}, closest=ref)
    pipeline, executor = with_exclusions(base, tmp_path / "test", scenario)
    pipeline.build_cell(cell)
    first, second = pipeline.ledger.of_cell(cell)
    assert (first.outcome, first.against) == ("overlap_paraphrase", "development")
    assert hl.rejection_code(first) == "overlap_development"
    assert second.outcome == "accepted" and pipeline.states[cell] == "ready"
    # the next attempt's gate shows the same needs under ids drawn again
    assert ref not in executor.packets[("gatekeeper", cell, 2)]


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


def test_v2_n_the_freeze_refuses_a_ready_need_that_repeats_a_development_need(full):
    """A backstop to the gate: a READY need with a development need's statement, or its whole
    READY record, stops the freeze; the problem names the need's id, never a statement."""
    pipeline, _, _ = full
    ready = next(iter(pipeline.ready.values()))
    saved = pipeline.world
    repeats = {
        "statement": hxe.Exclusion("run-x/G09-a1", ready.need.statement),
        "record": hxe.Exclusion(
            "run-x/G09-a2",
            "A synthetic statement of its own.",
            record_sha256=hs.sha_value(ready.record()),
        ),
    }
    try:
        for kind, exclusion in repeats.items():
            pipeline.world = dataclasses.replace(saved, development_exclusions=(exclusion,))
            found = [p for p in pipeline.freeze_problems() if "development need" in p]
            assert found and all(p.startswith(ready.need.need_id) for p in found), kind
            assert not any(ready.need.statement in p for p in found)
    finally:
        pipeline.world = saved
    assert pipeline.freeze_problems() == []


def test_a_review_flag_and_a_low_per_search_recall_block_nothing(base, tmp_path):
    """Only search_3 finds a grade-2 node: the third search's increment is 1 of 1 need (above
    7%, a review flag) and search_1's and search_2's recall is 2/3 (below 0.90). The need is
    READY, its checkpoint passes, and no freeze problem names either metric."""
    cell = ho.cell_order(base["grid"])[0]
    scenario = Scenario(
        drop={(cell, "searcher_1"): "MiCA:149.3", (cell, "searcher_2"): "MiCA:149.3"}
    )
    pipeline, _ = make(base, tmp_path / "test", scenario)
    pipeline.build_cell(cell)
    assert pipeline.states[cell] == "ready" and pipeline.progress_problems([cell]) == []
    metrics = pipeline.report()["review_metrics"]
    third = metrics["third_search_incremental_g2"]
    assert (third["numerator"], third["denominator"], third["review_flag"]) == (1, 1, True)
    recall = metrics["per_search_recall_g2"]
    assert recall["min"] == pytest.approx(2 / 3) and recall["below_reference"] == 2
    assert not any(
        word in p for p in pipeline.freeze_problems() for word in ("third", "recall", "review")
    )


def test_the_review_metrics_are_reported_and_never_a_gate(full):
    pipeline, _, _ = full
    metrics = pipeline.report()["review_metrics"]
    third = metrics["third_search_incremental_g2"]
    # the synthetic searches list the same evidence: the third adds nothing
    assert (third["numerator"], third["denominator"], third["review_flag"]) == (0, 60, False)
    recall = metrics["per_search_recall_g2"]
    assert recall["denominator"] == 3 * (60 - recall["needs_without_grade_2"]) > 0
    assert (recall["min"], recall["below_reference"]) == (1.0, 0)
    assert metrics["omission_audit_ready"]["audit_new"] == 0
    assert metrics["omission_audit_all_attempts"]["audit_new"] == 0
    assert pipeline.freeze_problems() == []


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
