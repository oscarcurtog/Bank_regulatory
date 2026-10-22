"""CONSTRUCTION_AMENDMENT_1: the transition the frozen anti-copy guard lacked, and the classified
stop of every other failure of what the code builds before a model sees it. On the real grid and
corpus with synthetic replies; no model runs."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from ingest import holdout as ho
from ingest import holdout_judging as hj
from ingest import holdout_ledger as hl
from ingest import holdout_packets as hp
from ingest import holdout_pipeline as hpl
from tests.holdout_fakes import Scenario, copied, load_base, make

pytestmark = pytest.mark.corpus


@pytest.fixture(scope="module")
def base():
    return load_base()


@pytest.fixture(scope="module")
def first(base):
    return ho.cell_order(base["grid"])[0]


def attempts_of(pipeline: hpl.Pipeline, cell: str) -> list[tuple[int, str | None]]:
    return [(a["number"], a["outcome"]) for a in pipeline.store.attempts() if a["cell_id"] == cell]


def roles_of(executor: Any, cell: str, number: int) -> list[str]:
    return [role for role, c, n in executor.calls if (c, n) == (cell, number)]


# ---------------------------------------------------------------------------
# the anti-copy guard's transition
# ---------------------------------------------------------------------------


def test_a_concept_that_copies_dev_closes_its_attempt_methodologically(base, first, tmp_path):
    pipeline, executor = make(base, tmp_path / "test", Scenario(copied_concept={(first, 1)}))
    pipeline.build_cell(first)
    assert attempts_of(pipeline, first) == [(1, "concept_invalid"), (2, "accepted")]
    # never retried: the concept was asked once, and its statement writer never
    assert roles_of(executor, first, 1) == ["concept_author"]
    assert pipeline.store.summary()["technical_failures"] == {}
    closed = [e for e in pipeline.store.events() if e.key == f"{first}-a1/closed"]
    assert [(e.stage, e.payload["outcome"]) for e in closed] == [("statement", "concept_invalid")]


def test_the_next_attempt_learns_nothing_of_the_rejected_concept(base, first, tmp_path):
    pipeline, executor = make(base, tmp_path / "test", Scenario(copied_concept={(first, 1)}))
    pipeline.build_cell(first)
    packet = executor.packets[("concept_author", first, 2)]
    w = pipeline.world
    # exactly the frozen builder's packet, from the cell alone, in a fresh blind context
    fresh = hp.concept_packet(
        w.cell(first), w.forbidden, pipeline._ref("concept_author", f"{first}-a2")
    )
    assert packet == fresh
    assert copied(w.dev.needs) not in packet
    assert not any(word in packet for word in ("concept_invalid", "concept_no_dev_wording"))
    assert pipeline._ref("concept_author", f"{first}-a2") != pipeline._ref(
        "concept_author", f"{first}-a1"
    )


def test_three_copied_concepts_leave_the_cell_unfilled(base, first, tmp_path):
    scenario = Scenario(copied_concept={(first, 1), (first, 2), (first, 3)})
    pipeline, _ = make(base, tmp_path / "test", scenario)
    pipeline.build_cell(first)
    assert attempts_of(pipeline, first) == [(n, "concept_invalid") for n in (1, 2, 3)]
    assert pipeline.store.cells()[0]["state"] == "unfilled_after_attempt_limit"


# ---------------------------------------------------------------------------
# every other failure of what the code builds is an integrity stop, never a crash
# ---------------------------------------------------------------------------


def stopped(pipeline: hpl.Pipeline) -> dict[str, Any]:
    with pytest.raises(hpl.ConstructionStop) as stop:
        pipeline.build()
    summary = pipeline.store.summary()
    assert summary["status"] == "stopped"
    return {"reason": stop.value.reason, "stage": stop.value.stage, "summary": summary}


def test_the_guard_on_the_frozen_parts_alone_is_an_integrity_stop(
    base, first, tmp_path, monkeypatch
):
    def leaks(*args: Any, **kwargs: Any) -> str:
        raise hp.PacketError("the packet would leak")  # with or without the concept's words

    monkeypatch.setattr(hp, "writer_packet", leaks)
    pipeline, _ = make(base, tmp_path / "test")
    found = stopped(pipeline)
    assert (found["reason"], found["stage"]) == ("packet_construction", "statement")
    assert attempts_of(pipeline, first) == [(1, None)]  # no rejection, no new attempt
    assert pipeline.store.cells()[0]["state"] == "pending_technical_unresolved"


@pytest.mark.parametrize(
    ("target", "stage", "check"),
    [
        ("judge_b_packet", "judge_b", "packet_construction"),
        ("metadata_packet", "metadata", "packet_construction"),
        ("concept_packet", "concept", "packet_construction"),
    ],
)
def test_an_unrelated_packet_failure_is_an_integrity_stop(
    base, tmp_path, monkeypatch, target, stage, check
):
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise KeyError("a node the corpus lacks")

    monkeypatch.setattr(hp, target, broken)
    pipeline, _ = make(base, tmp_path / "test")
    found = stopped(pipeline)
    assert (found["reason"], found["stage"]) == (check, stage)
    assert "KeyError" in found["summary"]["stop"]["detail"]
    assert "a node the corpus lacks" not in found["summary"]["stop"]["detail"]  # no text


def test_a_record_that_cannot_be_assembled_is_an_integrity_stop(base, tmp_path, monkeypatch):
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("partial, but its judge wrote no gap")

    monkeypatch.setattr(hj, "assemble", broken)
    pipeline, _ = make(base, tmp_path / "test")
    found = stopped(pipeline)
    assert (found["reason"], found["stage"]) == ("record_assembly", "adjudication")


def test_a_ledger_refusal_is_an_integrity_stop(base, tmp_path, monkeypatch):
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise hl.LedgerError("a transition the ledger refuses")

    monkeypatch.setattr(hl.Ledger, "give_need_id", refuse)
    pipeline, _ = make(base, tmp_path / "test")
    found = stopped(pipeline)
    assert found["reason"] == "record_assembly"


def test_a_stop_for_the_person_is_still_its_own():
    assert issubclass(hl.StopForThePerson, hl.LedgerError)
    source = Path(hpl.__file__).read_text(encoding="utf-8")
    run = source[source.index("    def _run(") : source.index("    def _stopped(")]
    # the minimum's stop is caught before the ledger's other refusals
    assert run.index("hl.StopForThePerson") < run.index("hl.LedgerError")


# ---------------------------------------------------------------------------
# no packet or record the code builds is left without a classified failure
# ---------------------------------------------------------------------------

BUILDERS = {n for n in dir(hp) if n.endswith(("_packet", "_messages")) and callable(getattr(hp, n))}
ASSEMBLY = {
    ("bm", "Need"),
    ("bm", "render_query_prompt"),
    ("hj", "assemble"),
    ("hj", "assemble_audit"),
}
# a nested helper only _built calls: the pool batches' renderer is one
GUARDED_HELPERS = {"writer", "render", "_packet_guard", "_evidence_packet", "statement_problems"}


def test_no_packet_or_record_is_built_outside_a_classified_guard():
    """Every reference to a packet builder anywhere in the pipeline, and every record it
    assembles inside an attempt, sits inside _built, a helper only _built calls, or an
    admission (whose failures are classified technical already)."""
    tree = ast.parse(Path(hpl.__file__).read_text(encoding="utf-8"))
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def guarded(node: ast.AST) -> bool:
        while node in parents:
            node = parents[node]
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_built"
            ):
                return True
            if isinstance(node, ast.FunctionDef) and (
                node.name in GUARDED_HELPERS or node.name.startswith("admit")
            ):
                return True
        return False

    builders = [
        (node.attr, guarded(node))
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "hp"
        and node.attr in BUILDERS
    ]
    assert builders and all(ok for _, ok in builders), [b for b in builders if not b[1]]
    used = {name for name, _ in builders}
    assert {"concept_packet", "writer_packet", "gate_evidence_packet", "judge_d_packet"} <= used

    methods = {"attempt", "_gate", "_judge", "_evidence_gate"}
    assembled = []
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.FunctionDef) and fn.name in methods):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                owner = node.func.value
                name = (owner.id if isinstance(owner, ast.Name) else None, node.func.attr)
                query = isinstance(owner, ast.Attribute) and owner.attr == "Query"
                if name in ASSEMBLY or query:
                    assembled.append((fn.name, node.func.attr, guarded(node)))
    assert assembled and all(ok for _, _, ok in assembled), [a for a in assembled if not a[2]]


def test_the_amendment_classifies_every_exception_path_it_lists():
    record = json.loads(Path("config/holdout/construction-amendment-1.json").read_text())
    categories = (
        "technical_integrity stop",
        "methodological eligibility rejection",
        "diagnostic only",
        "technical retry",
    )
    paths = record["exception_paths"]
    assert len(paths) == 29 and record["exception_paths_unclassified_before"] == 18
    assert all(p["classification"].startswith(categories) for p in paths)
    assert sum(p["missing_transition"] for p in paths) == 18
    methodological = [p for p in paths if p["classification"].startswith(categories[1])]
    assert {p["source"] for p in methodological if p["missing_transition"]} == {
        "holdout_packets.writer_packet anti-copy guard"
    }
