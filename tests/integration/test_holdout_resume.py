"""The durable state of TEST's construction: resume, technical retries, crash recovery and
backup (the person's rules of 2026-10-19), on the real grid and corpus with synthetic
replies. No model runs and no TEST content exists.

A crash is simulated two ways: a Crash (a BaseException no handler of the construction
catches) raised at a chosen boundary, after which the store is closed as a dying process
would leave it; and, once, a real SIGKILL of a child process."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from ingest import holdout as ho
from ingest import holdout_packets as hp
from ingest import holdout_pipeline as hpl
from ingest import holdout_state as hs
from tests.holdout_fakes import (
    IDENTITIES,
    Crash,
    Scenario,
    load_base,
    make,
    raw_dir,
    reopen,
)

pytestmark = pytest.mark.corpus
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def base():
    return load_base()


@pytest.fixture(scope="module")
def order(base):
    return ho.cell_order(base["grid"])


def outcome(pipeline: hpl.Pipeline) -> dict[str, Any]:
    """What the construction decided, apart from how many executions it took."""
    store = pipeline.store
    return {
        "cells": {c["cell_id"]: c["state"] for c in store.cells()},
        "attempts": [
            (a["proposal_id"], a["number"], a["need_id"], a["outcome"], a["against"])
            for a in store.attempts()
        ],
        "ready": sorted((r["cell_id"], r["need_id"], r["record_sha256"]) for r in store.ready()),
        "stages": sorted((s["proposal_id"], s["stage"], s["value_sha256"]) for s in store.stages()),
    }


@pytest.fixture(scope="module")
def reference(base, tmp_path_factory):
    """The construction with no failure and no crash."""
    pipeline, _ = make(base, tmp_path_factory.mktemp("reference") / "test")
    pipeline.build()
    found = outcome(pipeline)
    pipeline.store.close()
    return found


def crash_on(store: hs.Store, kind: str, stage: str | None = None, before: bool = False) -> None:
    """The next event of this kind (and stage) kills the process just after it is committed,
    or just before it is written."""
    real = store.record
    armed = [True]
    target, at = kind, stage

    def record(kind: str, key: str | None, payload: Mapping[str, Any], **where: Any) -> bool:
        hit = armed[0] and kind == target and (at is None or where.get("stage") == at)
        if hit and before:
            armed[0] = False
            raise Crash(f"killed before {kind}")
        result = real(kind, key, payload, **where)
        if hit:
            armed[0] = False
            raise Crash(f"killed after {kind}")
        return result

    store.record = record  # type: ignore[method-assign]


def die(pipeline: hpl.Pipeline) -> None:
    """What a killed process leaves: its connection and its lock closed by the system."""
    pipeline.store.close()


def roles_on(executor: Any, cell: str) -> list[str]:
    return [role for role, c, _ in executor.calls if c == cell]


# ---------------------------------------------------------------------------
# resume: the same cell, need, stage and frozen inputs
# ---------------------------------------------------------------------------


def test_resume_in_a_new_process_goes_back_to_the_failed_stage_only(
    base, order, tmp_path, reference
):
    cell = order[1]
    directory = tmp_path / "test"
    pipeline, _ = make(base, directory, Scenario(technical={cell: "judge_a"}))
    with pytest.raises(hpl.TechnicalStop):
        pipeline.build()
    pid = f"{cell}-a1"
    need = pipeline.ledger.get(pid).need_id
    admitted = {(s["proposal_id"], s["stage"]) for s in pipeline.store.stages()}
    assert {(pid, s) for s in ("concept", "statement", "overlap_gate", "lock")} <= admitted
    die(pipeline)

    resumed, executor = reopen(base, directory, Scenario())  # the failure was transient
    resumed.resume()
    assert roles_on(executor, order[0]) == []  # a finished cell is replayed, never run
    assert roles_on(executor, cell)[0] == "judge_a"
    authors = {"concept_author", "statement_writer", "gatekeeper", "query_writer"}
    assert not authors & set(roles_on(executor, cell))
    attempts = [a for a in resumed.store.attempts() if a["cell_id"] == cell]
    assert [(a["number"], a["need_id"], a["outcome"]) for a in attempts] == [(1, need, "accepted")]
    runs = resumed.store.runs(pid, "judge_a")
    assert [(r["technical_run"], r["status"]) for r in runs] == [(1, "failed"), (2, "admitted")]
    assert len({r["packet_sha256"] for r in runs}) == 1 and len({r["run_id"] for r in runs}) == 2
    assert outcome(resumed) == reference


def test_technical_retries_never_count_as_attempts(base, order, tmp_path, reference):
    cell = order[0]
    directory = tmp_path / "test"
    failing = Scenario(technical={cell: "judge_b"})
    pipeline, _ = make(base, directory, failing)
    with pytest.raises(hpl.TechnicalStop):
        pipeline.build()
    die(pipeline)
    pipeline, _ = reopen(base, directory, failing)
    with pytest.raises(hpl.TechnicalStop):
        pipeline.resume()
    die(pipeline)
    pipeline, _ = reopen(base, directory, Scenario())
    pipeline.resume()
    runs = pipeline.store.runs(f"{cell}-a1", "judge_b")
    assert [r["status"] for r in runs] == ["failed", "failed", "admitted"]
    assert [a["number"] for a in pipeline.store.attempts() if a["cell_id"] == cell] == [1]
    assert outcome(pipeline) == reference


def test_three_technical_failures_leave_the_construction_stopped(base, order, tmp_path):
    cell = order[0]
    directory = tmp_path / "test"
    failing = Scenario(technical={cell: "metadata_adjudicator"})
    pipeline, _ = make(base, directory, failing)
    with pytest.raises(hpl.TechnicalStop):
        pipeline.build()
    for _ in range(2):
        die(pipeline)
        pipeline, _ = reopen(base, directory, failing)
        with pytest.raises(hpl.TechnicalStop):
            pipeline.resume()
    pid = f"{cell}-a1"
    runs = pipeline.store.runs(pid, "metadata")
    assert [(r["technical_run"], r["status"]) for r in runs] == [(i, "failed") for i in (1, 2, 3)]
    assert pipeline.store.status() == "pending_technical_unresolved"
    assert pipeline.store.cells()[0]["state"] == "pending_technical_unresolved"
    attempts = [a for a in pipeline.store.attempts() if a["cell_id"] == cell]
    assert len(attempts) == 1 and attempts[0]["outcome"] == "accepted"
    die(pipeline)
    pipeline, executor = reopen(base, directory, Scenario())  # even a healthy model
    events = len(pipeline.store.events())
    with pytest.raises(hpl.TechnicalStop, match="unresolved"):
        pipeline.resume()
    assert executor.calls == [] and len(pipeline.store.events()) == events
    assert len(pipeline.store.runs(pid, "metadata")) == 3
    assert any("pending" in p for p in pipeline.freeze_problems())


def test_a_valid_answer_is_never_executed_again(base, order, tmp_path):
    cell = order[0]
    directory = tmp_path / "test"
    pipeline, _ = make(base, directory, Scenario(gate={cell: ["paraphrase"]}))
    crash_on(pipeline.store, "stage_admitted", stage="overlap_gate")
    with pytest.raises(Crash):
        pipeline.build()
    die(pipeline)
    # the gatekeeper would now find the proposal independent: it is not asked again
    resumed, executor = reopen(base, directory, Scenario())
    resumed.resume()
    assert ("gatekeeper", cell, 1) not in executor.calls
    first = next(a for a in resumed.store.attempts() if a["proposal_id"] == f"{cell}-a1")
    assert first["outcome"] == "overlap_paraphrase" and first["against"] == "dev"
    assert resumed.states[cell] == "ready" and resumed.ledger.of_cell(cell)[1].number == 2
    with pytest.raises(hs.ReplayDiverged):
        resumed.store.record(
            "stage_admitted",
            f"{cell}-a1/overlap_gate/admitted",
            {"value": {"verdict": "independent"}},
            cell=cell,
            proposal=f"{cell}-a1",
            stage="overlap_gate",
        )


# ---------------------------------------------------------------------------
# crashes at every boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("boundary", "kind", "stage", "before"),
    [
        ("between two stages", "run_started", "judge_b", True),
        ("after a stage's admission, before the next", "stage_admitted", "judge_a", False),
        ("after the need id, before the queries", "need_id_given", None, False),
        ("after the lock, before the judges", "stage_admitted", "lock", False),
        ("after READY, before the next cell", "need_ready", None, False),
    ],
)
def test_a_crash_between_transitions_is_idempotent(
    base, order, tmp_path, reference, boundary, kind, stage, before
):
    directory = tmp_path / "test"
    pipeline, _ = make(base, directory)
    crash_on(pipeline.store, kind, stage=stage, before=before)
    with pytest.raises(Crash):
        pipeline.build()
    die(pipeline)
    resumed, _ = reopen(base, directory, Scenario())
    resumed.resume()
    assert outcome(resumed) == reference, boundary
    assert [a["number"] for a in resumed.store.attempts()] == [1] * 60
    keys = [e.key for e in resumed.store.events() if e.key is not None]
    assert len(keys) == len(set(keys))
    assert all(r["status"] in ("admitted",) for r in resumed.store.runs())


def test_a_crash_during_the_call_is_one_interrupted_execution(base, order, tmp_path, reference):
    cell = order[0]
    directory = tmp_path / "test"
    pipeline, _ = make(base, directory, Scenario(crash=(cell, "judge_b")))
    with pytest.raises(Crash):
        pipeline.build()
    die(pipeline)
    resumed, _ = reopen(base, directory, Scenario())
    summary = resumed.store.verify()
    assert summary["waiting"][0]["stage"] == "judge_b"
    resumed.resume()
    runs = resumed.store.runs(f"{cell}-a1", "judge_b")
    assert [(r["status"], r["failure_reason"]) for r in runs] == [
        ("failed", "interrupted"),
        ("admitted", None),
    ]
    assert outcome(resumed) == reference


@pytest.mark.parametrize("role", ["judge_b", "judge_a", "qwen_diagnostic"])
def test_a_response_spooled_before_the_orchestrator_dies_is_recovered(
    base, order, tmp_path, reference, role
):
    """The model answered and its output reached the spool; the orchestrator died before
    recording it. Resume records it from the spool, audits it and admits it: the model is not
    asked again, and the execution is not counted as failed."""
    cell = order[0]
    directory = tmp_path / "test"
    pipeline, _ = make(base, directory, Scenario(die_after=(cell, role)))
    with pytest.raises(Crash):
        pipeline.build()
    die(pipeline)
    resumed, executor = reopen(base, directory, Scenario())
    resumed.resume()
    assert role not in roles_on(executor, cell) and len(executor.recovered) == 1
    stage = {"judge_b": "judge_b", "judge_a": "judge_a", "qwen_diagnostic": "diagnostic"}[role]
    runs = resumed.store.runs(f"{cell}-a1", stage)
    assert [(r["technical_run"], r["status"]) for r in runs] == [(1, "admitted")]
    responded = [
        e.payload["recovered"]
        for e in resumed.store.events()
        if e.kind == "run_responded" and e.stage == stage and e.cell_id == cell
    ]
    assert responded == [True]
    assert outcome(resumed) == reference


def test_a_response_recorded_before_a_crash_is_audited_not_run_again(
    base, order, tmp_path, reference
):
    cell = order[0]
    directory = tmp_path / "test"
    pipeline, _ = make(base, directory)
    crash_on(pipeline.store, "run_responded", stage="judge_a")
    with pytest.raises(Crash):
        pipeline.build()
    die(pipeline)
    resumed, executor = reopen(base, directory, Scenario())
    resumed.resume()
    assert "judge_a" not in roles_on(executor, cell)
    runs = resumed.store.runs(f"{cell}-a1", "judge_a")
    assert [(r["technical_run"], r["status"]) for r in runs] == [(1, "admitted")]
    assert outcome(resumed) == reference
    # the same response delivered again changes nothing; another one is refused
    event = next(
        e for e in resumed.store.events() if e.kind == "run_responded" and e.stage == "judge_a"
    )
    where = {"cell": event.cell_id, "proposal": event.proposal_id, "stage": event.stage}
    assert not resumed.store.record(event.kind, event.key, event.payload, run=event.run_id, **where)
    other = {"response": {**event.payload["response"], "kept": "0" * 64}}
    with pytest.raises(hs.ReplayDiverged):
        resumed.store.record(event.kind, event.key, other, run=event.run_id, **where)


@pytest.mark.parametrize("spooled", [True, False])
def test_a_partial_artifact_write_recovers_the_spooled_response(
    base, tmp_path, reference, monkeypatch, spooled
):
    """The store's copy of a response is half written when the process dies. If the executor
    spooled the response while receiving it, resume records it from the spool and audits it,
    never asking again; without a spool the execution counts as interrupted."""
    directory = tmp_path / "test"
    pipeline, executor = make(base, directory)
    if not spooled:
        executor.spool = None
    real = hs.write_atomic
    calls = []

    def half_then_die(path: Path, data: bytes) -> None:
        calls.append(path)
        if len(calls) == 40:  # in the middle of the construction
            path.with_name(f".tmp-{path.name}-{os.getpid()}").write_bytes(data[: len(data) // 2])
            raise Crash("killed in the middle of a write")
        real(path, data)

    monkeypatch.setattr(hs, "write_atomic", half_then_die)
    with pytest.raises(Crash):
        pipeline.build()
    monkeypatch.setattr(hs, "write_atomic", real)
    die(pipeline)
    leftovers = list((directory / hs.STATE_DIR).rglob(".tmp-*")) + list(
        raw_dir(directory).rglob(".tmp-*")
    )
    assert len(leftovers) == 1
    resumed, again = reopen(base, directory, Scenario())
    resumed.resume()
    assert outcome(resumed) == reference
    failed = [r["failure_reason"] for r in resumed.store.runs() if r["status"] == "failed"]
    recovered = [e for e in resumed.store.events() if e.payload.get("recovered") is True]
    if spooled:
        assert failed == [] and len(recovered) == 1 and len(again.recovered) == 1
        run = next(r for r in resumed.store.runs() if r["run_id"] == recovered[0].run_id)
        assert (run["role"], recovered[0].cell_id, 1) not in again.calls
    else:
        assert failed == ["interrupted"] and recovered == []


def test_a_crash_at_a_checkpoint_redoes_its_backup_once(base, tmp_path, reference):
    directory = tmp_path / "test"
    pipeline, _ = make(base, directory)
    crash_on(pipeline.store, "checkpoint", before=True)  # backed up, not yet recorded
    with pytest.raises(Crash):
        pipeline.build()
    assert pipeline.backup.labels == ["checkpoint 20"]  # type: ignore[attr-defined]
    die(pipeline)
    resumed, _ = reopen(base, directory, Scenario())
    resumed.resume()
    assert resumed.backup.labels == [  # type: ignore[attr-defined]
        "checkpoint 20",
        "checkpoint 40",
        "checkpoint 60",
    ]
    assert outcome(resumed) == reference
    checkpoints = [e.key for e in resumed.store.events() if e.kind == "checkpoint"]
    assert checkpoints == ["checkpoint/20", "checkpoint/40", "checkpoint/60"]


def test_a_killed_child_process_resumes_from_its_state(base, order, tmp_path, reference):
    """A real SIGKILL in another process, with its own hash seed: the parent replays what the
    child admitted, renders every packet again with the same bytes, and goes on."""
    cell = order[1]
    directory = tmp_path / "test"
    code = textwrap.dedent(
        f"""
        from pathlib import Path
        from tests.holdout_fakes import Scenario, load_base, make
        scenario = Scenario(kill=({cell!r}, "judge_b"))
        pipeline, _ = make(load_base(), Path({str(directory)!r}), scenario)
        pipeline.build()
        """
    )
    env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONHASHSEED": "12345"}
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True
    )
    assert done.returncode == -signal.SIGKILL, done.stderr[-2000:]
    resumed, executor = reopen(base, directory, Scenario())
    waiting = resumed.store.verify()["waiting"]
    assert [(w["cell_id"], w["stage"], w["state"]) for w in waiting] == [
        (cell, "judge_b", "judging")
    ]
    resumed.resume()
    assert roles_on(executor, order[0]) == [] and roles_on(executor, cell)[0] == "judge_b"
    runs = resumed.store.runs(f"{cell}-a1", "judge_b")
    assert [r["failure_reason"] for r in runs] == ["interrupted", None]
    assert outcome(resumed) == reference


# ---------------------------------------------------------------------------
# duplicates, mismatches and corruption
# ---------------------------------------------------------------------------


def test_a_duplicate_resume_is_safe(base, order, tmp_path, reference):
    directory = tmp_path / "test"
    pipeline, _ = make(base, directory, Scenario(technical={order[0]: "judge_b"}))
    with pytest.raises(hpl.TechnicalStop):
        pipeline.build()
    with pytest.raises(hs.StateBusy):  # a second process while the first holds the state
        reopen(base, directory, Scenario())
    die(pipeline)
    resumed, _ = reopen(base, directory, Scenario())
    resumed.resume()
    before = outcome(resumed), len(resumed.store.runs())
    resumed.resume()  # the same command again, once the construction is done
    die(resumed)
    again, executor = reopen(base, directory, Scenario())
    again.resume()
    assert executor.calls == [] and (outcome(again), len(again.store.runs())) == before
    assert outcome(again) == reference
    assert [e.kind for e in again.store.events()].count("resumed") == 3


@pytest.mark.parametrize(
    "change",
    ["identity", "contract", "runtime", "admitted packet", "retried packet"],
)
def test_resume_refuses_anything_frozen_that_changed(base, order, tmp_path, monkeypatch, change):
    cell = order[0]
    directory = tmp_path / "test"
    pipeline, _ = make(base, directory, Scenario(technical={cell: "judge_b"}))
    with pytest.raises(hpl.TechnicalStop):
        pipeline.build()
    die(pipeline)
    changes: dict[str, Any] = {}

    def broken() -> None:
        raise RuntimeError(f"the {change} changed")

    if change == "identity":
        changes["identities"] = {**IDENTITIES, "analysis_plan": "0" * 64}
    elif change == "contract":
        changes["contract_check"] = broken
    elif change == "runtime":
        changes["runtime_check"] = broken
    elif change == "admitted packet":
        real_a = hp.judge_a_broker_packet
        monkeypatch.setattr(hp, "judge_a_broker_packet", lambda *a, **k: real_a(*a, **k) + " ")
    else:
        real_b = hp.judge_b_packet
        monkeypatch.setattr(hp, "judge_b_packet", lambda *a, **k: real_b(*a, **k) + " ")
    resumed, executor = reopen(base, directory, Scenario(), **changes)
    events = len(resumed.store.events())
    with pytest.raises(hs.IncompatibleState):
        resumed.resume()
    assert ("judge_b", cell, 1) not in executor.calls
    assert resumed.store.status() == "pending_technical"
    if change in ("identity", "contract", "runtime"):
        assert len(resumed.store.events()) == events  # refused before anything is written


def test_a_corrupted_artifact_or_projection_blocks_resume(base, order, tmp_path):
    directory = tmp_path / "test"
    pipeline, _ = make(base, directory, Scenario(technical={order[0]: "judge_b"}))
    with pytest.raises(hpl.TechnicalStop):
        pipeline.build()
    kept = sorted((directory / hs.STATE_DIR / hs.KEPT_DIR).rglob("*"))
    target = next(p for p in kept if p.is_file())
    original = target.read_bytes()
    target.write_bytes(original[:-1])
    die(pipeline)
    resumed, _ = reopen(base, directory, Scenario())
    with pytest.raises(hs.IntegrityError, match="does not hold its hash"):
        resumed.resume()
    target.write_bytes(original)
    resumed.store.db.execute("UPDATE cells SET state = 'ready' WHERE position = 1")
    with pytest.raises(hs.IntegrityError, match="cells: not what the event log builds"):
        resumed.resume()


# ---------------------------------------------------------------------------
# backup and restore
# ---------------------------------------------------------------------------


def test_backup_destroy_restore_and_resume_reach_the_same_final_state(
    base, order, tmp_path, reference
):
    directory = tmp_path / "test"
    directory.mkdir()
    ho.git(directory, "init", "-q", "-b", "main")
    ho.guard_private_history(directory)
    (directory / "README.md").write_text("# TEST (fixture)\n", encoding="utf-8")
    ho.git(directory, "add", "-A")
    ho.git(directory, "commit", "-q", "-m", "fixture: private history begins")
    folder = tmp_path / "backups"
    cell = order[3]  # three cells built before the stop
    pipeline, _ = make(base, directory, Scenario(technical={cell: "judge_b"}))
    with pytest.raises(hpl.TechnicalStop):
        pipeline.build()
    before = pipeline.store.verify()
    record = hs.backup_state(pipeline.store, tmp_path / "repo", folder, "after a technical stop")
    assert record["state"] == before and record["raw_artifacts"] > 0
    die(pipeline)

    # the working copy and the raw folder destroyed, each by its literal path
    shutil.rmtree(directory)
    shutil.rmtree(raw_dir(directory))
    restored = hs.restore(folder / record["bundle"], directory, raw_dir(directory))
    assert restored == before
    assert [(w["cell_id"], w["stage"], w["state"]) for w in restored["waiting"]] == [
        (cell, "judge_b", "pending_technical")
    ]
    assert (directory / ".git" / "hooks" / "pre-push").read_text() == ho.NO_PUSH_HOOK

    resumed, executor = reopen(base, directory, Scenario())
    resumed.resume()
    assert all(roles_on(executor, c) == [] for c in order[:3])
    assert roles_on(executor, cell)[0] == "judge_b"
    assert outcome(resumed) == reference
    ready = resumed.store.ready()
    assert len({r["cell_id"] for r in ready}) == len({r["need_id"] for r in ready}) == 60
