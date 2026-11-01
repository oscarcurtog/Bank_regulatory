"""Operational controls with real imports, SQLite, role audit and synthetic sessions.

No DEV corpus, private TEST content, model, provider or process signal is used.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ingest import holdout_isolation as hi
from ingest import holdout_packets as hp
from ingest import holdout_pipeline as hpl
from ingest import holdout_roles as hr
from ingest import holdout_v2_engine as en
from ingest import holdout_v2_operations as ops
from ingest import holdout_v2_state as vs
from tests import holdout_fakes as hf
from tests import holdout_v2_fakes as f

IDENTITY = "a" * 64


class Clock:
    def __init__(self) -> None:
        self.value = 0.0
        self.origin = datetime.now(UTC)

    def monotonic(self) -> float:
        return self.value

    def utc(self) -> datetime:
        return self.origin + timedelta(seconds=self.value)

    def sleep(self, seconds: float) -> None:
        self.value += seconds
        time.sleep(0.0001)


def audit_context() -> en.AuditContext:
    """Real audit with public templates and runtime fixtures, without loading DEV."""
    return en.AuditContext(
        terms=hi.forbidden_terms(None, None, []),
        frozen_templates=hr.template_hashes(),
        forbidden=[],
        pin=hf.PIN,
        tmp_root=str(Path(tempfile.gettempdir()).resolve()),
        claude_effective=hf.FAKE_EFFECTIVE,
    )


def gate_job(number: int) -> en.Job:
    packet = hp.gate_packet(f"R-{number:04x}", "A synthetic firm on topic T001 asks.", [])

    def admit(reply: str) -> dict[str, Any]:
        value = en.reply_json(reply)
        if not isinstance(value, dict) or value.get("verdict") not in ("independent", "paraphrase"):
            raise en.Inadmissible("synthetic_gate_contract", ["verdict"])
        return value

    return en.Job(f"synthetic/{number}", "gatekeeper", packet, admit)


def operation(directory: Path, clock: Clock, interval: float = 0.0) -> ops.Operations:
    return ops.Operations(
        IDENTITY,
        directory,
        interval,
        clock=clock.monotonic,
        utc_now=clock.utc,
    )


def test_runtime_sqlite_audit_pacing_and_admitted_replay(tmp_path: Path) -> None:
    clock = Clock()
    namespace, raw = tmp_path / "state", tmp_path / "raw"
    fake = f.V2Fake(tmp_path / "spool")
    jobs = [gate_job(1), gate_job(2)]
    with vs.Store.create(namespace, raw, {"kind": "synthetic-operational-test"}) as store:
        engine = en.Engine(
            store,
            fake,
            audit_context(),
            workers=2,
            sleep=clock.sleep,
            operations=operation(tmp_path / "operations", clock, interval=2.0),
        )
        result = engine.run(jobs)
        assert all(not item.exhausted for item in result.values())
        assert clock.value >= 2.0
        assert fake.calls["gatekeeper"] == 2
        assert len(engine.ledger.admitted) == 2
        summary = store.verify()
        event_head = store.head()
        assert not [e for e in store.events() if e.kind == "run_failed"]

    # Open the actual SQLite log again, not a memory ledger or an AST substitute.
    again = f.V2Fake(tmp_path / "spool")
    with vs.Store.open(namespace, raw) as store:
        engine = en.Engine(
            store,
            again,
            audit_context(),
            sleep=clock.sleep,
            operations=operation(tmp_path / "operations", clock, interval=2.0),
        )
        replayed = engine.run(jobs)
        assert {key: value.value for key, value in replayed.items()} == {
            key: value.value for key, value in result.items()
        }
        assert not again.calls
        assert store.verify() == summary
        assert store.head() == event_head


def test_runtime_pause_drains_and_recovers_complete_spools(tmp_path: Path) -> None:
    clock = Clock()
    release = threading.Event()

    class Blocked(f.V2Fake):
        entered = 0
        entered_lock = threading.Lock()

        def headless(self, role: str, run_id: str, packet: str) -> hpl.Session:
            with self.entered_lock:
                self.entered += 1
            if not release.wait(5):
                raise RuntimeError("synthetic release timed out")
            return super().headless(role, run_id, packet)

    fake = Blocked(tmp_path / "spool")

    class PauseAfterTwo(ops.Operations):
        sent = False

        def poll(self) -> list[dict[str, Any]]:
            if not self.sent and fake.entered >= 2:
                rid = uuid.uuid4().hex
                ops.atomic_json(
                    self.directory / f"request-{rid}.json",
                    {
                        "schema": ops.REQUEST_SCHEMA,
                        "contract_sha256": IDENTITY,
                        "request_id": rid,
                        "action": "pause",
                        "created_at": clock.utc().isoformat(),
                    },
                )
                self.sent = True
                release.set()
            return super().poll()

    controls = PauseAfterTwo(
        IDENTITY,
        tmp_path / "operations",
        0,
        clock=clock.monotonic,
        utc_now=clock.utc,
    )
    namespace, raw = tmp_path / "state", tmp_path / "raw"
    jobs = [gate_job(i) for i in range(1, 5)]
    with vs.Store.create(namespace, raw, {"kind": "synthetic-operational-test"}) as store:
        engine = en.Engine(
            store,
            fake,
            audit_context(),
            workers=2,
            sleep=clock.sleep,
            operations=controls,
        )
        with pytest.raises(en.OperationalPause):
            engine.run(jobs)
        assert fake.calls["gatekeeper"] == 2
        assert {run["status"] for run in engine.ledger.runs.values()} == {"responded"}
        assert not [e for e in store.events() if e.kind == "run_failed"]
        store.verify()

    recovered_fake = f.V2Fake(tmp_path / "spool")
    with vs.Store.open(namespace, raw) as store:
        resumed = en.Engine(
            store,
            recovered_fake,
            audit_context(),
            workers=1,
            sleep=clock.sleep,
            operations=operation(tmp_path / "operations", clock),
        )
        result = resumed.run(jobs)
        assert all(not item.exhausted for item in result.values())
        assert len(result) == 4
        assert recovered_fake.calls["gatekeeper"] == 2
        assert len(resumed.ledger.runs) == 4
        assert len([e for e in store.events() if e.kind == "run_responded"]) == 4
        assert not [e for e in store.events() if e.kind == "run_failed"]
        store.verify()


def test_runtime_recovers_started_execution_from_complete_synthetic_spool(tmp_path: Path) -> None:
    clock = Clock()
    namespace, raw, spool = tmp_path / "state", tmp_path / "raw", tmp_path / "spool"
    fake = f.V2Fake(spool)
    wanted = gate_job(1)
    with vs.Store.create(namespace, raw, {"kind": "synthetic-operational-test"}) as store:
        engine = en.Engine(store, fake, audit_context())
        rid = engine._start(wanted)
        # Simulate a completed model call followed by coordinator death before
        # recording run_responded. V2Fake writes a full valid synthetic session.
        fake.headless(wanted.role, rid, wanted.packet)
        assert engine.ledger.runs[rid]["status"] == "started"
        store.verify()

    recovery = f.V2Fake(spool)
    with vs.Store.open(namespace, raw) as store:
        resumed = en.Engine(
            store,
            recovery,
            audit_context(),
            sleep=clock.sleep,
            operations=operation(tmp_path / "operations", clock),
        )
        result = resumed.run([wanted])
        assert not result[wanted.job_id].exhausted
        assert not recovery.calls
        assert resumed.ledger.runs[rid]["recovered"] is True
        assert len(resumed.ledger.runs) == 1
        assert not [e for e in store.events() if e.kind == "run_failed"]
        store.verify()


def test_runtime_operational_mode_preserves_real_isolation_stop(tmp_path: Path) -> None:
    class Leaky(f.V2Fake):
        def headless(self, role: str, run_id: str, packet: str) -> hpl.Session:
            session = super().headless(role, run_id, packet)
            session.session["transcript"] += "\n" + json.dumps(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "E5 reached Recall@10 0.8"}],
                    },
                }
            )
            return session

    clock = Clock()
    fake = Leaky(tmp_path / "spool")
    with vs.Store.create(tmp_path / "state", tmp_path / "raw", {"kind": "test"}) as store:
        engine = en.Engine(
            store,
            fake,
            audit_context(),
            workers=1,
            sleep=clock.sleep,
            operations=operation(tmp_path / "operations", clock),
        )
        with pytest.raises(en.V2Stop) as stopped:
            engine.run([gate_job(1), gate_job(2)])
        assert stopped.value.reason == "isolation"
        assert fake.calls["gatekeeper"] == 1
        assert not engine.ledger.admitted
        store.verify()
