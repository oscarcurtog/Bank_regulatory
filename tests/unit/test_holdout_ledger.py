"""Attempts, need ids, the stop rule and the departures from the grid (S3, D-054)."""

from __future__ import annotations

import pytest

from ingest import benchmark as bm
from ingest import holdout as ho
from ingest import holdout_ledger as hl


def _cell(cell_id: str, answerability: str = "complete", **overrides) -> ho.Cell:
    data = {
        "cell_id": cell_id,
        "batch": 1,
        "acts": ["DORA"],
        "area": "Information sharing",
        "type": "direct" if answerability == "complete" else "partial_out_of_scope",
        "answerability": answerability,
        "difficulty": "easy",
        "number_or_date": False,
    }
    if answerability == "none":
        data["area"] = None
    return ho.Cell.model_validate(data | overrides)


def _grid(cells: list[ho.Cell]) -> ho.Grid:
    return ho.Grid(
        schema_version="holdout-grid.v1",
        seed=1,
        protocol_sha256="a" * 64,
        dev_benchmark_version="0.10.0",
        dev_needs_sha256="a" * 64,
        dev_qrels_sha256="a" * 64,
        area_order={},
        cells=cells,
    )


def test_three_blind_attempts_at_most():
    ledger = hl.Ledger()
    first = ledger.open_attempt("G01")
    assert first.proposal_id == "G01-a1"
    with pytest.raises(hl.LedgerError, match="in progress"):
        ledger.open_attempt("G01")
    ledger.close("G01-a1", "overlap_uncertain", "gate")
    ledger.open_attempt("G01")
    ledger.close("G01-a2", "concept_unfit", "concept")
    ledger.open_attempt("G01")
    ledger.close("G01-a3", "overlap_paraphrase", "gate", against="dev")
    assert ledger.unfilled("G01")
    with pytest.raises(hl.LedgerError, match=hl.UNFILLED):
        ledger.open_attempt("G01")


def test_need_ids_start_at_101_and_are_never_reused():
    ledger = hl.Ledger()
    ledger.open_attempt("G01")
    assert ledger.give_need_id("G01-a1") == "N101"
    ledger.close("G01-a1", "accepted", "gate")
    ledger.reopen_failed("G01-a1", "ambiguous", "judging", "a qrel was ambiguous")
    ledger.open_attempt("G01")
    assert ledger.give_need_id("G01-a2") == "N102"
    ledger.open_attempt("G02")
    assert ledger.give_need_id("G02-a1") == "N103"


def test_a_later_failure_counts_as_one_of_the_three_attempts():
    ledger = hl.Ledger()
    for number in (1, 2, 3):
        ledger.open_attempt("G01")
        ledger.give_need_id(f"G01-a{number}")
        ledger.close(f"G01-a{number}", "accepted", "gate")
        ledger.reopen_failed(f"G01-a{number}", "adjudication_uncertain", "diagnostic", "D")
    assert ledger.unfilled("G01")


def test_the_work_stops_when_50_evaluable_needs_are_out_of_reach():
    cells = [_cell(f"G{i:02d}") for i in range(1, 53)] + [_cell("G53", "none")]
    grid = _grid(cells)
    ledger = hl.Ledger()
    assert ledger.reachable_evaluable(grid, [], []) == 52
    ledger.check_minimum(grid, [], [])
    for cell in ("G01", "G02", "G03"):
        for number in (1, 2, 3):
            ledger.open_attempt(cell)
            ledger.close(f"{cell}-a{number}", "overlap_same_rule", "gate", against="dev")
    assert ledger.reachable_evaluable(grid, [], []) == 49
    with pytest.raises(hl.StopForThePerson):
        ledger.check_minimum(grid, [], [])


def test_a_final_need_that_is_not_evaluable_counts_zero():
    grid = _grid([_cell("G01"), _cell("G02")])
    ledger = hl.Ledger()
    ledger.open_attempt("G01")
    ledger.give_need_id("G01-a1")
    ledger.close("G01-a1", "accepted", "gate")
    assert ledger.reachable_evaluable(grid, ["N101"], []) == 1
    assert ledger.reachable_evaluable(grid, ["N101"], ["N101"]) == 2


def test_the_summary_counts_failures_by_area():
    grid = _grid([_cell("G01"), _cell("G02", area="Scope, definitions, proportionality")])
    ledger = hl.Ledger()
    ledger.open_attempt("G01")
    ledger.close("G01-a1", "overlap_uncertain", "gate")
    ledger.open_attempt("G02")
    ledger.give_need_id("G02-a1")
    ledger.close("G02-a1", "accepted", "gate")
    summary = ledger.summary(grid)
    assert summary["outcomes"] == {"accepted": 1, "overlap_uncertain": 1}
    assert summary["failed_attempts_by_area"] == {"DORA: Information sharing": 1}
    assert summary["cells_filled"] == 1 and summary[hl.UNFILLED] == []


def _need(**overrides) -> bm.Need:
    data = {
        "need_id": "N101",
        "split": "test",
        "statement": "A bank asks.",
        "type": "direct",
        "drivers": [],
        "difficulty": "easy",
        "acts": ["DORA"],
        "answerability": "complete",
        "status": "ready",
    }
    return bm.Need.model_validate(data | overrides)


def test_a_relabelled_need_is_kept_and_its_departure_recorded():
    cell = _cell("G01", "partial", number_or_date=True)
    need = _need()
    record = hl.deviations(cell, need)
    assert record["planned_type"] == "partial_out_of_scope"
    assert record["adjudicated_type"] == "direct" and record["type_deviation"]
    assert record["answerability_deviation"]
    assert record["planned_figure_date_not_realized"]
    summary = hl.plan_against_realized(_grid([cell]), [need], {"N101": "G01"})
    assert summary["deviations"]["type_deviation"] == 1
    assert summary["type_changes"] == {"partial_out_of_scope -> direct": 1}
    assert summary["planned"]["figure_or_date"] == 1 and summary["realized"]["figure_or_date"] == 0


def test_each_rejection_records_what_it_repeated():
    assert hl.gate_outcome("paraphrase", "DEV") == ("overlap_paraphrase", "dev")
    assert hl.gate_outcome("same_rule_superficial_change", "TEST") == (
        "overlap_same_rule",
        "test_existing",
    )
    assert hl.gate_outcome("uncertain", None) == ("overlap_uncertain", None)
    # a development need of an earlier, aborted construction (D-066)
    assert hl.gate_outcome("paraphrase", "DEVELOPMENT") == ("overlap_paraphrase", "development")
    with pytest.raises(hl.LedgerError, match="names no existing need"):
        hl.gate_outcome("same_legal_question", None)
    grid = _grid([_cell("G01"), _cell("G02"), _cell("G03"), _cell("G04")])
    ledger = hl.Ledger()
    for cell, verdict, source in (
        ("G01", "paraphrase", "DEV"),
        ("G02", "same_legal_question", "TEST"),
        ("G03", "uncertain", None),
        ("G04", "same_rule_superficial_change", "DEVELOPMENT"),
    ):
        ledger.open_attempt(cell)
        outcome, against = hl.gate_outcome(verdict, source)
        ledger.close(f"{cell}-a1", outcome, "gate", against=against)
    assert ledger.summary(grid)["rejections_by_comparison"] == {
        "overlap_dev": 1,
        "overlap_test_existing": 1,
        "overlap_uncertain": 1,
        "overlap_development": 1,
    }
    ledger.open_attempt("G01")
    with pytest.raises(hl.LedgerError, match="must say what it repeats"):
        ledger.close("G01-a2", "overlap_paraphrase", "gate")
