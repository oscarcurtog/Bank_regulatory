"""The durable state of TEST's construction, by itself: no corpus, no model, no TEST content.
The events here are synthetic; the pipeline's own use of them is tested on the corpus
(tests/integration/test_holdout_resume.py)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ingest import holdout as ho
from ingest import holdout_state as hs

CELLS = [f"G{i:02d}" for i in (3, 1, 2)]  # a frozen order that is not the grid's
IDENTITIES = {"pre_authoring_contract": "a" * 64, "binary": "b" * 64}


def new(tmp_path: Path) -> hs.Store:
    return hs.Store.create(tmp_path / "test", tmp_path / "raw", IDENTITIES, CELLS, ["G01"])


def open_attempt(store: hs.Store, cell: str = "G03", number: int = 1) -> str:
    pid = f"{cell}-a{number}"
    store.record("attempt_opened", f"{pid}/opened", {"number": number}, cell=cell, proposal=pid)
    return pid


def start_run(store: hs.Store, pid: str, run: str = "R-1", n: int = 1, stage: str = "concept"):
    store.record(
        "run_started",
        f"{run}/started",
        {
            "role": "concept_author",
            "ref": "R-ref",
            "technical_run": n,
            "packet_sha256": "p" * 64,
            "template": "concept_author.v1",
            "template_sha256": "t" * 64,
            "role_contract_sha256": "c" * 64,
            "inputs": {},
            "state": "authoring",
        },
        cell=pid.split("-")[0],
        proposal=pid,
        stage=stage,
        run=run,
    )


def ready(store: hs.Store, cell: str, need: str, pid: str) -> None:
    record = {"need": need}
    store.record(
        "need_ready",
        f"{cell}/ready",
        {"need_id": need, "record": record, "record_sha256": hs.sha_value(record)},
        cell=cell,
        proposal=pid,
        stage="ready",
    )


def test_a_new_state_holds_the_frozen_order_and_verifies(tmp_path):
    with new(tmp_path) as store:
        assert [c["cell_id"] for c in store.cells()] == CELLS
        assert [c["omission"] for c in store.cells()] == [0, 1, 0]
        summary = store.verify()
        assert summary["status"] == "building" and summary["cells"] == {"planned": 3}
        store.check_cells(CELLS, ["G01"])
        with pytest.raises(hs.IncompatibleState):
            store.check_cells(sorted(CELLS), ["G01"])
    with pytest.raises(hs.StateError, match="created once"):
        new(tmp_path)


def test_an_event_with_a_key_is_written_once_and_a_divergence_stops(tmp_path):
    with new(tmp_path) as store:
        open_attempt(store)
        assert not store.record(
            "attempt_opened", "G03-a1/opened", {"number": 1}, cell="G03", proposal="G03-a1"
        )
        with pytest.raises(hs.ReplayDiverged):
            store.record(
                "attempt_opened", "G03-a1/opened", {"number": 2}, cell="G03", proposal="G03-a1"
            )
        assert [e.kind for e in store.events()] == ["created", "attempt_opened"]
        assert len(store.attempts()) == 1


def test_a_transition_the_state_does_not_allow_writes_nothing(tmp_path):
    with new(tmp_path) as store:
        pid = open_attempt(store)
        events = len(store.events())
        with pytest.raises(hs.IntegrityError, match="does not allow"):
            store.record(  # a response for a run that never started
                "run_responded",
                "R-9/responded",
                {"response": {"raw": "r", "raw_bytes": 1, "kept": "k", "kept_bytes": 1}},
                cell="G03",
                proposal=pid,
                stage="concept",
                run="R-9",
            )
        assert len(store.events()) == events and store.verify()["runs"] == {}


def test_the_event_log_is_append_only_and_chained(tmp_path):
    with new(tmp_path) as store:
        open_attempt(store)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store.db.execute("UPDATE events SET payload = '{}' WHERE seq = 2")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store.db.execute("DELETE FROM events WHERE seq = 2")
        store.db.execute("DROP TRIGGER events_refuse_update")
        store.db.execute("UPDATE events SET payload = '{\"number\":1} ' WHERE seq = 2")
        with pytest.raises(hs.IntegrityError) as found:
            store.verify()
        assert "append-only" in str(found.value) and "chain breaks" in str(found.value)


def test_the_tables_are_exactly_what_the_event_log_builds(tmp_path):
    with new(tmp_path) as store:
        pid = open_attempt(store)
        start_run(store, pid)
        store.verify()
        rebuilt = hs.rebuild(store.events())
        for table, order in hs.PROJECTIONS.items():
            assert hs._rows(store.db, table, order) == hs._rows(rebuilt, table, order)
        store.db.execute("UPDATE runs SET status = 'admitted'")
        with pytest.raises(hs.IntegrityError, match="runs: not what the event log builds"):
            store.verify()


def test_ready_is_one_per_cell_and_one_per_need(tmp_path):
    with new(tmp_path) as store:
        a = open_attempt(store, "G03")
        b = open_attempt(store, "G01")
        ready(store, "G03", "N101", a)
        with pytest.raises(sqlite3.IntegrityError):  # the same need for another cell
            ready(store, "G01", "N101", b)
        with pytest.raises(sqlite3.IntegrityError):  # a second READY for the same cell
            store.record(
                "need_ready",
                "G03/ready-again",
                {"need_id": "N102", "record": {}, "record_sha256": hs.sha_value({})},
                cell="G03",
                proposal=b,
                stage="ready",
            )
        assert len(store.ready()) == 1 and store.verify()["ready"] == 1


def test_a_technical_run_number_above_three_is_refused(tmp_path):
    with new(tmp_path) as store:
        pid = open_attempt(store)
        with pytest.raises(sqlite3.IntegrityError):
            start_run(store, pid, run="R-4", n=hs.MAX_TECHNICAL_RUNS + 1)
        assert store.runs() == []


def test_artifacts_are_written_whole_and_checked_by_hash(tmp_path):
    with new(tmp_path) as store:
        digest, size = store.put("the response, as received", "raw")
        assert store.text(digest, "raw") == "the response, as received" and size == 25
        assert store.put("the response, as received", "raw") == (digest, size)
        path = store.artifact_path(digest, "raw")
        assert path.is_relative_to(tmp_path / "raw")
        assert oct(path.parent.stat().st_mode)[-3:] == "700"
        path.write_text("another text", encoding="utf-8")
        with pytest.raises(hs.IntegrityError, match="does not hold its hash"):
            store.text(digest, "raw")
        assert not list(path.parent.glob(".tmp-*"))


def test_one_process_at_a_time(tmp_path):
    store = new(tmp_path)
    with pytest.raises(hs.StateBusy):
        hs.Store.open(tmp_path / "test", tmp_path / "raw")
    with hs.Store.open(tmp_path / "test", tmp_path / "raw", readonly=True) as view:
        assert view.verify()["status"] == "building"
        with pytest.raises(hs.StateError, match="read-only"):
            view.record("resumed", None, {})
    store.close()
    with hs.Store.open(tmp_path / "test", tmp_path / "raw") as again:
        assert again.summary()["events"] == 1


def test_changed_identities_are_an_incompatibility(tmp_path):
    with new(tmp_path) as store:
        store.check_identities(IDENTITIES)
        with pytest.raises(hs.IncompatibleState, match="binary"):
            store.check_identities({**IDENTITIES, "binary": "0" * 64})
        with pytest.raises(hs.IncompatibleState, match="audit"):
            store.check_identities({**IDENTITIES, "audit": "0" * 64})


def test_the_summary_names_the_waiting_cell_and_stage_only(tmp_path):
    with new(tmp_path) as store:
        pid = open_attempt(store)
        start_run(store, pid)
        store.record(
            "run_failed",
            "R-1/ended",
            {"reason": "model_api", "detail": "529", "audit": None, "state": "pending_technical"},
            cell="G03",
            proposal=pid,
            stage="concept",
            run="R-1",
        )
        summary = store.verify()
        assert summary["status"] == "pending_technical"
        assert summary["waiting"] == [
            {
                "cell_id": "G03",
                "position": 1,
                "state": "pending_technical",
                "stage": "concept",
                "proposal_id": pid,
                "failed_technical_runs": 1,
                "last_failure": "model_api",
            }
        ]


def test_a_backup_restores_the_exact_state_into_a_new_copy(tmp_path):
    directory = tmp_path / "test"
    directory.mkdir()
    ho.git(directory, "init", "-q", "-b", "main")
    ho.guard_private_history(directory)
    (directory / "README.md").write_text("# fixture\n", encoding="utf-8")
    ho.git(directory, "add", "-A")
    ho.git(directory, "commit", "-q", "-m", "fixture")
    store = hs.Store.create(directory, tmp_path / "raw", IDENTITIES, CELLS, ["G01"])
    pid = open_attempt(store)
    start_run(store, pid)
    raw, raw_bytes = store.put('{"reply": "someone@example.org"}', "raw")
    kept, kept_bytes = store.put('{"reply": "[REDACTED]"}', "kept")
    store.record(
        "run_responded",
        "R-1/responded",
        {"response": {"raw": raw, "raw_bytes": raw_bytes, "kept": kept, "kept_bytes": kept_bytes}},
        cell="G03",
        proposal=pid,
        stage="concept",
        run="R-1",
    )
    folder = tmp_path / "backups"
    record = hs.backup_state(store, tmp_path / "repo", folder, "fixture")
    assert record["raw_artifacts"] == 1 and (folder / "raw" / raw[:2] / raw).exists()
    assert "someone@example.org" not in ho.git(directory, "log", "-p", "--all")
    before = store.verify()
    store.close()
    target = tmp_path / "restored"
    restored = hs.restore(folder / record["bundle"], target, tmp_path / "raw-restored")
    assert restored == before
    with pytest.raises(hs.StateError, match="never overwrites"):
        hs.restore(folder / record["bundle"], target, tmp_path / "raw-restored")
    with hs.Store.open(target, tmp_path / "raw-restored") as again:
        with pytest.raises(ho.HoldoutError, match="outside"):
            hs.backup_state(again, tmp_path, tmp_path / "inside")


def _responded(store: hs.Store, pid: str, run: str = "R-1") -> None:
    raw, raw_bytes = store.put("{}", "raw")
    kept, kept_bytes = store.put("{}", "kept")
    store.record(
        "run_responded",
        f"{run}/responded",
        {"response": {"raw": raw, "raw_bytes": raw_bytes, "kept": kept, "kept_bytes": kept_bytes}},
        cell="G03",
        proposal=pid,
        stage="concept",
        run=run,
    )


def _admit(store: hs.Store, pid: str, ok: bool, run: str = "R-1") -> bool:
    value = {"fit": "fits"}
    return store.record(
        "stage_admitted",
        f"{pid}/concept/admitted",
        {
            "inputs_sha256": "p" * 64,
            "value": value,
            "value_sha256": hs.sha_value(value),
            "audit": {"ok": ok, "problems": [] if ok else ["contaminated"]},
            "state": "authoring",
        },
        cell="G03",
        proposal=pid,
        stage="concept",
        run=run,
    )


def test_an_output_is_admitted_only_after_its_audit_passed(tmp_path):
    with new(tmp_path) as store:
        pid = open_attempt(store)
        start_run(store, pid)
        _responded(store, pid)
        with pytest.raises(hs.IntegrityError, match="only after its audit"):
            _admit(store, pid, ok=False)
        assert store.stage(pid, "concept") is None
        assert _admit(store, pid, ok=True) and store.consumption_problems() == []


def test_a_failed_run_never_has_a_consumer(tmp_path):
    with new(tmp_path) as store:
        pid = open_attempt(store)
        start_run(store, pid, run="R-1", n=1)
        store.record(
            "run_failed",
            "R-1/ended",
            {
                "reason": "isolation",
                "check": "role_audit",
                "detail": "x",
                "audit": {"ok": False, "problems": ["x"]},
                "state": "pending_technical",
            },
            cell="G03",
            proposal=pid,
            stage="concept",
            run="R-1",
        )
        start_run(store, pid, run="R-2", n=2)
        _responded(store, pid, run="R-2")
        _admit(store, pid, ok=True, run="R-2")
        assert store.consumption_problems() == []
        store.db.execute("UPDATE stages SET run_id = 'R-1' WHERE run_id = 'R-2'")
        found = store.consumption_problems()
        assert "R-1: a failed run has a consumer" in found
        assert "R-2: admitted with no stage" in found


def test_a_replay_that_only_reads_writes_nothing(tmp_path):
    with new(tmp_path) as store:
        pid = open_attempt(store)
        events = len(store.events())
        with store.no_writes():
            assert not store.writable
            assert not store.record(  # the same event again: nothing to write, allowed
                "attempt_opened", f"{pid}/opened", {"number": 1}, cell="G03", proposal=pid
            )
            with pytest.raises(hs.StateError, match="writes nothing"):
                open_attempt(store, "G01")
            with pytest.raises(hs.StateError, match="writes nothing"):
                store.record("resumed", None, {})
        assert store.writable and len(store.events()) == events


# ---------------------------------------------------------------------------
# a documented amendment (CONSTRUCTION_AMENDMENT_1)
# ---------------------------------------------------------------------------


def amendment(store: hs.Store, **changes: object) -> dict[str, object]:
    summary = store.verify()
    record: dict[str, object] = {
        "amendment": "CONSTRUCTION_AMENDMENT_1",
        "amendment_sha256": "c" * 64,
        "base_contract_sha256": IDENTITIES["pre_authoring_contract"],
        "contract_sha256": "d" * 64,
        "public_commit": "e" * 40,
        "after_event": {"seq": summary["events"], "sha256": summary["head_sha256"]},
        "changed_identities": ["pre_authoring_contract"],
        "first_transition": {"refused_write": "attempt_closed"},
        "from_identities": dict(IDENTITIES),
        "to_identities": {**IDENTITIES, "pre_authoring_contract": "d" * 64},
    }
    return record | changes


def test_an_amendment_moves_the_identities_at_one_point_of_the_log(tmp_path):
    store = new(tmp_path)
    open_attempt(store)
    before = store.verify()
    record = amendment(store)
    assert store.amend(record) is True
    assert store.amend(record) is False  # recorded once
    store.check_identities(record["to_identities"])  # type: ignore[arg-type]
    with pytest.raises(hs.IncompatibleState):
        store.check_identities(IDENTITIES)
    (found,) = store.verify()["amendments"]
    assert found["seq"] == before["events"] + 1 and found["after_event"] == record["after_event"]
    assert found["amendment"] == "CONSTRUCTION_AMENDMENT_1"
    assert "from_identities" not in found  # the log keeps them, the summary names the change
    store.close()
    with hs.Store.open(tmp_path / "test", tmp_path / "raw", readonly=True) as again:
        assert again.verify()["amendments"] == [found]  # the tables are what the log rebuilds


def test_an_amendment_from_identities_the_state_does_not_hold_is_refused(tmp_path):
    store = new(tmp_path)
    record = amendment(store, from_identities={**IDENTITIES, "binary": "0" * 64})
    with pytest.raises(hs.IntegrityError, match="does not hold"):
        store.amend(record)
    assert store.verify()["amendments"] == []


def test_a_stopped_construction_is_never_amended(tmp_path):
    store = new(tmp_path)
    store.record("construction_stopped", "stopped", {"reason": "rule", "detail": "x"})
    with pytest.raises(hs.IntegrityError, match="stopped"):
        store.amend(amendment(store))


def test_a_failure_of_the_medium_mid_transaction_writes_nothing(tmp_path, monkeypatch):
    """SQLite failing between the event's insert and its projection: the whole transaction is
    rolled back, the log and the tables stay what they were, and the state still verifies."""
    with new(tmp_path) as store:
        pid = open_attempt(store)
        before = store.verify()

        def disk_failure(db: sqlite3.Connection, e: hs.Event) -> None:
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(hs, "_apply", disk_failure)
        with pytest.raises(sqlite3.OperationalError):
            start_run(store, pid)
        monkeypatch.undo()
        assert store.verify() == before
        assert [e.kind for e in store.events()] == ["created", "attempt_opened"]
