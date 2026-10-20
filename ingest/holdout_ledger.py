"""Every attempt to fill a cell of TEST, and what became of it (S3, D-054).

A cell gets at most three blind attempts. Each attempt is a new concept author, a new
writer and a new proposal id, with no access to the earlier attempts or their evidence,
whatever made them fail: an overlap verdict, an unfit cell, an ambiguity, or an
adjudication left uncertain by judge D. A cell that fails three times stays empty,
marked cell_unfilled_after_3_attempts; it is never filled by hand, given hints or let
through a looser gate. If the evaluable needs that TEST can still reach fall below 50,
the work stops and goes back to the person.

The ledger also records how each need departs from its cell. The grid is a target
distribution: a need keeps the type, the answerability and the drivers the evidence
gives it, and the departure is recorded, never corrected by selection.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ingest import benchmark as bm
from ingest import holdout as ho

LEDGER_FILE = "ledger.jsonl"
MAX_ATTEMPTS = 3
MIN_EVALUABLE = 50

# Methodological outcomes only: a technical failure never closes an attempt (Codex Gate 2,
# P1.3); it leaves the same need pending_technical (ingest/holdout_pipeline.py). No outcome
# comes from a person: the person does not review statements before the lock (2026-10-19).
Outcome = Literal[
    "accepted",
    "concept_unfit",
    "concept_invalid",  # CONSTRUCTION_AMENDMENT_1: the frozen anti-copy guard on the concept
    "statement_invalid",
    "overlap_paraphrase",
    "overlap_same_legal_question",
    "overlap_same_rule",
    "overlap_uncertain",
    "ambiguous",
    "adjudication_uncertain",
    "judge_problem",
]
VERDICT_OUTCOME: dict[str, Outcome] = {
    "paraphrase": "overlap_paraphrase",
    "same_legal_question": "overlap_same_legal_question",
    "same_rule_superficial_change": "overlap_same_rule",
    "uncertain": "overlap_uncertain",
}
UNFILLED = "cell_unfilled_after_3_attempts"
# What the rejected proposal was compared with when the gatekeeper dropped it (D-054)
Against = Literal["dev", "fixture", "test_existing"]
SOURCE_AGAINST: dict[str, Against] = {"DEV": "dev", "FIXTURE": "fixture", "TEST": "test_existing"}


class LedgerError(Exception):
    """An attempt the rules of D-054 do not allow."""


class StopForThePerson(LedgerError):
    """TEST can no longer reach the minimum of evaluable needs: a person decides."""


class Attempt(BaseModel):
    model_config = bm.STRICT

    proposal_id: str = Field(pattern=r"^G\d{2}-a[1-3]$")
    cell_id: str = Field(pattern=r"^G\d{2}$")
    number: int = Field(ge=1, le=MAX_ATTEMPTS)
    # given when the gatekeeper accepts the statement; retired with the attempt if it
    # fails later, and never reused
    need_id: str | None = Field(default=None, pattern=bm.NEED_ID_PATTERN)
    outcome: Outcome | None = None  # None while the attempt is in progress
    against: Against | None = None  # for an overlap: what the proposal repeated
    stage: str | None = None  # the pipeline stage where it ended (or overlap_gate, accepted)
    note: str | None = None


def gate_outcome(verdict: str, closest_source: str | None) -> tuple[Outcome, Against | None]:
    """The ledger's record of a gatekeeper verdict that drops a proposal: the kind of
    overlap and what it overlaps with. An uncertain verdict overlaps with nothing known."""
    outcome = VERDICT_OUTCOME[verdict]
    if verdict == "uncertain":
        return outcome, None
    if closest_source not in SOURCE_AGAINST:
        raise LedgerError(f"verdict {verdict} names no existing need it repeats")
    return outcome, SOURCE_AGAINST[closest_source]


def rejection_code(attempt: Attempt) -> str | None:
    """overlap_dev, overlap_fixture, overlap_test_existing or overlap_uncertain."""
    if attempt.outcome == "overlap_uncertain":
        return "overlap_uncertain"
    if attempt.against is not None:
        return f"overlap_{attempt.against}"
    return None


class Ledger:
    def __init__(self, attempts: Iterable[Attempt] = ()) -> None:
        self.attempts = list(attempts)

    @classmethod
    def load(cls, directory: Path) -> Ledger:
        path = directory / LEDGER_FILE
        return cls(bm.read_jsonl(path, Attempt) if path.exists() else [])

    def save(self, directory: Path) -> None:
        bm.write_jsonl(directory / LEDGER_FILE, self.attempts)

    def of_cell(self, cell_id: str) -> list[Attempt]:
        return [a for a in self.attempts if a.cell_id == cell_id]

    def get(self, proposal_id: str) -> Attempt:
        found = [a for a in self.attempts if a.proposal_id == proposal_id]
        if len(found) != 1:
            raise LedgerError(f"no single attempt {proposal_id}")
        return found[0]

    def filled(self, cell_id: str) -> bool:
        return any(a.outcome == "accepted" for a in self.of_cell(cell_id))

    def unfilled(self, cell_id: str) -> bool:
        tried = self.of_cell(cell_id)
        return len(tried) == MAX_ATTEMPTS and all(
            a.outcome not in (None, "accepted") for a in tried
        )

    def open_attempt(self, cell_id: str) -> Attempt:
        """The next blind attempt of a cell, or an error: one attempt at a time, none
        after an accepted one, never a fourth."""
        tried = self.of_cell(cell_id)
        if any(a.outcome is None for a in tried):
            raise LedgerError(f"{cell_id} has an attempt in progress")
        if self.filled(cell_id):
            raise LedgerError(f"{cell_id} is already filled")
        if len(tried) >= MAX_ATTEMPTS:
            raise LedgerError(f"{cell_id}: {UNFILLED}")
        number = len(tried) + 1
        attempt = Attempt(proposal_id=f"{cell_id}-a{number}", cell_id=cell_id, number=number)
        self.attempts.append(attempt)
        return attempt

    def next_need_id(self) -> str:
        used = [int(a.need_id[1:]) for a in self.attempts if a.need_id]
        return f"N{max(used, default=ho.FIRST_NEED_NUMBER - 1) + 1:03d}"

    def give_need_id(self, proposal_id: str) -> str:
        attempt = self.get(proposal_id)
        if attempt.need_id is not None:
            raise LedgerError(f"{proposal_id} already has {attempt.need_id}")
        attempt.need_id = self.next_need_id()
        return attempt.need_id

    def close(
        self,
        proposal_id: str,
        outcome: Outcome,
        stage: str,
        note: str | None = None,
        against: Against | None = None,
    ) -> None:
        attempt = self.get(proposal_id)
        if attempt.outcome is not None:
            raise LedgerError(f"{proposal_id} is already closed as {attempt.outcome}")
        if outcome.startswith("overlap_") and outcome != "overlap_uncertain" and against is None:
            raise LedgerError(f"{proposal_id}: an overlap must say what it repeats")
        attempt.outcome, attempt.stage, attempt.note = outcome, stage, note
        attempt.against = against

    def reopen_failed(self, proposal_id: str, outcome: Outcome, stage: str, note: str) -> None:
        """An accepted need that fails later (an ambiguity, an uncertain adjudication)
        closes its attempt as failed; its need id is retired with it."""
        attempt = self.get(proposal_id)
        if attempt.outcome != "accepted":
            raise LedgerError(f"{proposal_id} is not accepted")
        attempt.outcome, attempt.stage, attempt.note = outcome, stage, note

    def reachable_evaluable(
        self, grid: ho.Grid, final_needs: Iterable[str], evaluable_needs: Iterable[str]
    ) -> int:
        """The evaluable needs TEST can still reach, counted optimistically, so that the
        stop comes as soon as the minimum is out of reach and never later:
        - a filled cell whose need is final counts 1 if that need is evaluable;
        - a filled cell whose need is still being judged counts 1 if it plans an answer;
        - a cell still being attempted counts 1 if it plans an answer;
        - an unfilled cell counts 0."""
        final, evaluable = set(final_needs), set(evaluable_needs)
        total = 0
        for c in grid.cells:
            accepted = [a for a in self.of_cell(c.cell_id) if a.outcome == "accepted"]
            if accepted and accepted[0].need_id in final:
                total += accepted[0].need_id in evaluable
            elif not self.unfilled(c.cell_id):
                total += c.answerability != "none"
        return total

    def check_minimum(
        self, grid: ho.Grid, final_needs: Iterable[str], evaluable_needs: Iterable[str]
    ) -> None:
        reachable = self.reachable_evaluable(grid, final_needs, evaluable_needs)
        if reachable < MIN_EVALUABLE:
            raise StopForThePerson(
                f"TEST can reach at most {reachable} evaluable needs, under {MIN_EVALUABLE}: stop"
            )

    def summary(self, grid: ho.Grid) -> dict[str, Any]:
        """Counts only, for the freeze and the commitment."""
        cells = {c.cell_id: c for c in grid.cells}
        failed = [a for a in self.attempts if a.outcome not in (None, "accepted")]
        by_area: Counter[str] = Counter()
        for a in failed:
            cell = cells[a.cell_id]
            by_area[f"{'+'.join(cell.acts)}: {cell.area or 'no area'}"] += 1
        return {
            "attempts": len(self.attempts),
            "outcomes": dict(
                sorted(Counter(a.outcome or "in_progress" for a in self.attempts).items())
            ),
            "failed_attempts_by_area": dict(sorted(by_area.items())),
            "rejections_by_comparison": dict(
                sorted(Counter(c for a in self.attempts if (c := rejection_code(a))).items())
            ),
            "cells_filled": sum(self.filled(c) for c in cells),
            UNFILLED: sorted(c for c in cells if self.unfilled(c)),
            "attempts_per_filled_cell": dict(
                sorted(Counter(len(self.of_cell(c)) for c in cells if self.filled(c)).items())
            ),
        }


# ---------------------------------------------------------------------------
# planned against realized
# ---------------------------------------------------------------------------


def _figure(drivers: Sequence[str]) -> bool:
    return bool({"numeric", "date"} & set(drivers))


def deviations(cell: ho.Cell, need: bm.Need) -> dict[str, Any]:
    """How a ready need departs from its cell. The need is kept as the evidence made it."""
    record: dict[str, Any] = {
        "cell_id": cell.cell_id,
        "need_id": need.need_id,
        "planned_type": cell.type,
        "adjudicated_type": need.type,
        "type_deviation": need.type != cell.type,
        "planned_answerability": cell.answerability,
        "adjudicated_answerability": need.answerability,
        "answerability_deviation": need.answerability != cell.answerability,
        "planned_acts": cell.acts,
        "adjudicated_acts": need.acts,
        "acts_deviation": need.acts != cell.acts,
        "planned_figure_date": cell.number_or_date,
        "realized_figure_date": _figure(need.drivers),
    }
    record["planned_figure_date_not_realized"] = cell.number_or_date and not _figure(need.drivers)
    return record


def plan_against_realized(
    grid: ho.Grid, needs: Sequence[bm.Need], cell_of: dict[str, str]
) -> dict[str, Any]:
    """The planned distribution, the realized one and the departures, in counts. cell_of
    maps each need to its cell."""
    cells = {c.cell_id: c for c in grid.cells}
    records = [deviations(cells[cell_of[n.need_id]], n) for n in needs]

    def planned(attribute: str) -> dict[str, int]:
        return dict(sorted(Counter(str(getattr(c, attribute)) for c in grid.cells).items()))

    def realized(attribute: str) -> dict[str, int]:
        return dict(sorted(Counter(str(getattr(n, attribute)) for n in needs).items()))

    return {
        "planned": {
            "type": planned("type"),
            "answerability": planned("answerability"),
            "difficulty": planned("difficulty"),
            "figure_or_date": sum(c.number_or_date for c in grid.cells),
        },
        "realized": {
            "type": realized("type"),
            "answerability": realized("answerability"),
            "difficulty": realized("difficulty"),
            "figure_or_date": sum(_figure(n.drivers) for n in needs),
        },
        "deviations": {
            key: sum(bool(r[key]) for r in records)
            for key in (
                "type_deviation",
                "answerability_deviation",
                "acts_deviation",
                "planned_figure_date_not_realized",
            )
        },
        "type_changes": dict(
            sorted(
                Counter(
                    f"{r['planned_type']} -> {r['adjudicated_type']}"
                    for r in records
                    if r["type_deviation"]
                ).items()
            )
        ),
    }
