"""Execution-control tests on the real Engine AST with synthetic boundaries.

The scientific admission/audit boundary is substituted. These tests exercise the
actual dispatch, pacing, drain and recovery code, not real models or private TEST.
They deliberately do not certify a full installed/runtime integration.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
import tempfile
import threading
import time
import types
import unittest
import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]


def load_file(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ops: Any = load_file("test_engine_operations_dependency", ROOT / "ingest/holdout_v2_operations.py")


@dataclass
class Session:
    reply: str
    session: dict[str, Any]


class TechnicalFailure(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        self.reason, self.detail = reason, detail
        super().__init__(detail)


class CleanupFailed(Exception):
    pass


@dataclass
class Event:
    seq: int
    key: str | None
    kind: str
    at: str
    payload: dict[str, Any]
    sha256: str = "0" * 64


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_engine() -> types.ModuleType:
    tree = ast.parse((ROOT / "ingest/holdout_v2_engine.py").read_text())
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "ingest"]
    tree.body = [n for n in tree.body if n not in imports]
    module = types.ModuleType("test_engine_real_ast")
    sys.modules[module.__name__] = module
    modules = {
        "holdout_v2_operations": ops,
        "holdout_v2_state": types.SimpleNamespace(Event=Event, sha_text=digest),
        "holdout_pipeline": types.SimpleNamespace(
            Session=Session, TechnicalFailure=TechnicalFailure
        ),
        "holdout_process": types.SimpleNamespace(CleanupFailed=CleanupFailed),
        "holdout_isolation": types.SimpleNamespace(
            redact=lambda text, identities: text,
            identity_values=lambda records: [],
            parse_records=lambda texts: [],
        ),
        "holdout_judging": types.SimpleNamespace(as_json=json.loads),
    }
    for node in imports:
        for alias in node.names:
            module.__dict__[alias.asname or alias.name] = modules.get(
                alias.name, types.SimpleNamespace()
            )
    exec(compile(tree, str(ROOT / "ingest/holdout_v2_engine.py"), "exec"), module.__dict__)
    return module


en: Any = load_engine()


class Clock:
    def __init__(self) -> None:
        self.value = 0.0
        self.origin = datetime(2026, 10, 27, 12, tzinfo=UTC)

    def monotonic(self) -> float:
        return self.value

    def utc(self) -> datetime:
        return self.origin + timedelta(seconds=self.value)

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise AssertionError("negative wait")
        self.value += seconds
        # Allow real ThreadPool workers to complete while keeping tests fast.
        time.sleep(0.0001)


class Store:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.log: list[Event] = []
        self.artifacts: dict[tuple[str, str], str] = {}

    def events(self) -> list[Event]:
        return list(self.log)

    def record(self, kind: str, key: str | None, payload: Any) -> Event:
        if key is not None:
            prior = [event for event in self.log if event.key == key]
            if prior:
                if prior[0].kind != kind or prior[0].payload != payload:
                    raise AssertionError("a replay changed an event")
                return prior[0]
        event = Event(
            len(self.log) + 1,
            key,
            kind,
            self.clock.utc().isoformat(),
            dict(payload),
            digest(str(len(self.log) + 1)),
        )
        self.log.append(event)
        return event

    def put(self, text: str, place: str) -> tuple[str, int]:
        sha = digest(text)
        self.artifacts[(sha, place)] = text
        return sha, len(text.encode())

    def text(self, sha: str, place: str) -> str:
        return self.artifacts[(sha, place)]

    def head(self) -> tuple[int, str]:
        return len(self.log), self.log[-1].sha256 if self.log else "0" * 64

    def verify(self) -> dict[str, Any]:
        seq, sha = self.head()
        return {"events": seq, "head_sha256": sha}


class Executor:
    def __init__(self, clock: Clock, release: threading.Event | None = None) -> None:
        self.clock = clock
        self.release = release
        self.calls: list[tuple[str, float]] = []
        self.saved: dict[str, Session] = {}
        self.lock = threading.Lock()

    def headless(self, role: str, run_id: str, packet: str) -> Session:
        with self.lock:
            self.calls.append((run_id, self.clock.monotonic()))
        if self.release is not None and not self.release.wait(5):
            raise RuntimeError("synthetic release timed out")
        value = Session(json.dumps({"value": packet}), {})
        self.saved[run_id] = value
        return value

    def recover_headless(self, role: str, run_id: str, packet: str) -> Session | None:
        return self.saved.get(run_id)


class Engine(en.Engine):
    def _decide(self, job: Any, run_id: str, document: Any) -> Any:
        # Substitute only scientific validation, leaving real record/replay intact.
        value = job.admit(document["reply"])
        self.record(
            "job_admitted",
            f"{job.job_id}/admitted",
            {
                "job_id": job.job_id,
                "run_id": run_id,
                "value": value,
            },
        )
        return value


def job(number: int) -> Any:
    return en.Job(f"judge/b1/synthetic/{number}", "v2_pool_judge_1", f"packet {number}", json.loads)


class EngineOperationsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="fundamento-engine-ops-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.clock = Clock()
        self.store = Store(self.clock)
        self.executor = Executor(self.clock)

    def operations(self, interval: float = 3.0, **kwargs: Any) -> Any:
        return ops.Operations(
            "a" * 64,
            self.directory / "ops",
            interval,
            clock=self.clock.monotonic,
            utc_now=self.clock.utc,
            **kwargs,
        )

    def engine(self, operational: Any = None, workers: int = 1, **kwargs: Any) -> Engine:
        audit = types.SimpleNamespace(claude_effective=None)
        return Engine(
            self.store,
            self.executor,
            audit,
            workers=workers,
            sleep=self.clock.sleep,
            operations=operational,
            **kwargs,
        )

    def test_dispatch_interval_is_global_across_run_calls(self) -> None:
        operation = self.operations()
        engine = self.engine(operation)
        engine.run([job(1), job(2)])
        engine.run([job(3)])
        starts = [at for _, at in self.executor.calls]
        self.assertEqual(len(starts), 3)
        self.assertTrue(all(b - a >= 3 for a, b in zip(starts, starts[1:], strict=False)), starts)
        self.assertEqual(len(engine.ledger.admitted), 3)
        self.assertFalse([e for e in self.store.log if e.kind == "run_failed"])

    def test_restart_reconstructs_delay_from_recorded_starts(self) -> None:
        first = self.engine(self.operations(10))
        first.run([job(1)])
        self.clock.value = 1
        second = self.engine(self.operations(10))
        second.run([job(2)])
        starts = [at for _, at in self.executor.calls]
        self.assertGreaterEqual(starts[1] - starts[0], 10)
        self.assertEqual(len(self.store.log), len({e.key for e in self.store.log if e.key}))

    def test_window_cap_waits_without_recording_false_failures(self) -> None:
        operation = self.operations(0, max_starts=2, window_seconds=10)
        engine = self.engine(operation)
        engine.run([job(1), job(2), job(3)])
        self.assertGreaterEqual(self.executor.calls[2][1], 10)
        self.assertFalse([e for e in self.store.log if e.kind in ("run_failed", "capacity_wait")])
        self.assertEqual(len([e for e in self.store.log if e.kind == "run_started"]), 3)

    def test_replay_does_not_generate_admitted_jobs_again(self) -> None:
        engine = self.engine(self.operations(0))
        first = engine.run([job(1)])
        again = self.engine(self.operations(0)).run([job(1)])
        self.assertEqual(first[job(1).job_id].value, again[job(1).job_id].value)
        self.assertEqual(len(self.executor.calls), 1)

    def test_complete_saved_response_is_recovered_without_generation(self) -> None:
        engine = self.engine(self.operations(0))
        wanted = job(1)
        rid = engine._start(wanted)
        self.executor.saved[rid] = Session('{"value": "saved"}', {})
        outcome = self.engine(self.operations(0)).run([wanted])
        self.assertEqual(outcome[wanted.job_id].value, {"value": "saved"})
        self.assertEqual(self.executor.calls, [])
        self.assertFalse([e for e in self.store.log if e.kind == "run_failed"])

    def test_cleanup_failure_during_drain_is_integrity_stop(self) -> None:
        engine = self.engine(self.operations(0))
        rid = engine._start(job(1))
        future: Future[Session] = Future()
        future.set_exception(CleanupFailed("synthetic process group remains"))
        with self.assertRaises(en.V2Stop):
            engine._settle(rid, future)
        self.assertEqual(engine.ledger.runs[rid]["reason"], "process_cleanup")

    def test_runtime_identity_failure_during_drain_is_integrity_stop(self) -> None:
        engine = self.engine(self.operations(0))
        rid = engine._start(job(1))
        future: Future[Session] = Future()
        future.set_exception(TechnicalFailure("runtime_identity", "synthetic mismatch"))
        with self.assertRaises(en.V2Stop):
            engine._settle(rid, future)
        self.assertEqual(engine.ledger.runs[rid]["reason"], "runtime_identity")

    def test_pause_stops_dispatch_and_preserves_running_responses(self) -> None:
        release = threading.Event()
        self.executor = Executor(self.clock, release)
        executor = self.executor
        clock = self.clock

        class PauseAfterTwo(ops.Operations):
            sent = False

            def poll(instance: Any) -> list[dict[str, Any]]:
                if not instance.sent and len(executor.calls) >= 2:
                    rid = uuid.uuid4().hex
                    ops.atomic_json(
                        instance.directory / f"request-{rid}.json",
                        {
                            "schema": ops.REQUEST_SCHEMA,
                            "contract_sha256": "a" * 64,
                            "request_id": rid,
                            "action": "pause",
                            "created_at": clock.utc().isoformat(),
                        },
                    )
                    instance.sent = True
                    release.set()
                return cast(list[dict[str, Any]], super().poll())

        operation = PauseAfterTwo(
            "a" * 64,
            self.directory / "ops",
            0,
            clock=clock.monotonic,
            utc_now=clock.utc,
        )
        engine = self.engine(operation, workers=2)
        pause_type = getattr(en, "OperationalPause", getattr(en, "OperationalPaused", None))
        self.assertIsNotNone(pause_type, "engine must distinguish operational and provider pause")
        assert pause_type is not None
        with self.assertRaises(pause_type):
            engine.run([job(i) for i in range(1, 6)])
        self.assertEqual(len(executor.calls), 2)
        self.assertEqual(len(executor.saved), 2)
        self.assertFalse([r for r in engine.ledger.runs.values() if r["status"] == "started"])
        self.assertFalse([e for e in self.store.log if e.kind == "run_failed"])
        recovered = self.engine(self.operations(0)).run([job(i) for i in range(1, 6)])
        self.assertEqual(len(recovered), 5)
        self.assertEqual(len(executor.calls), 5)

    def test_checkpoint_request_drains_before_backup_and_acknowledges(self) -> None:
        executor, clock = self.executor, self.clock

        class CheckpointAfterOne(ops.Operations):
            sent = False

            def poll(instance: Any) -> list[dict[str, Any]]:
                if not instance.sent and len(executor.calls) >= 1:
                    rid = uuid.uuid4().hex
                    ops.atomic_json(
                        instance.directory / f"request-{rid}.json",
                        {
                            "schema": ops.REQUEST_SCHEMA,
                            "contract_sha256": "a" * 64,
                            "request_id": rid,
                            "action": "checkpoint",
                            "created_at": clock.utc().isoformat(),
                        },
                    )
                    instance.sent = True
                return cast(list[dict[str, Any]], super().poll())

        operation = CheckpointAfterOne(
            "a" * 64,
            self.directory / "ops",
            0,
            clock=clock.monotonic,
            utc_now=clock.utc,
        )
        checkpoints: list[int] = []
        engine = None

        def checkpoint() -> dict[str, Any]:
            assert engine is not None
            self.assertEqual(engine._in_flight, set())
            checkpoints.append(len(executor.calls))
            return self.store.verify()

        engine = self.engine(operation, checkpoint=checkpoint)
        outcome = engine.run([job(i) for i in range(1, 4)])
        self.assertEqual(len(outcome), 3)
        self.assertEqual(checkpoints, [1])
        self.assertFalse(operation.checkpoint_requested)
        acknowledgements = list(operation.directory.glob("ack-*.json"))
        self.assertEqual(len(acknowledgements), 1)
        self.assertEqual(json.loads(acknowledgements[0].read_text())["status"], "completed")

    def test_capacity_refusal_preserves_retry_rules_with_pacing(self) -> None:
        clock = self.clock

        class CapacityOnce(Executor):
            refused = False

            def headless(instance: Any, role: str, run_id: str, packet: str) -> Session:
                if not instance.refused:
                    instance.refused = True
                    instance.calls.append((run_id, clock.monotonic()))
                    raise TechnicalFailure(en.CAPACITY, "synthetic usage limit")
                return super().headless(role, run_id, packet)

        self.executor = CapacityOnce(clock)
        engine = self.engine(self.operations(3))
        result = engine.run([job(1)])
        self.assertFalse(result[job(1).job_id].exhausted)
        self.assertEqual(engine.ledger.failures(job(1).job_id), 0)
        self.assertEqual(len(self.executor.calls), 2)
        self.assertGreaterEqual(self.executor.calls[1][1] - self.executor.calls[0][1], 300)
        reasons = [r.get("reason") for r in engine.ledger.runs.values() if r["status"] == "failed"]
        self.assertEqual(reasons, [en.CAPACITY])

    def test_sidecar_failure_with_calls_running_drains_without_new_dispatch(self) -> None:
        release = threading.Event()
        self.executor = Executor(self.clock, release)
        executor = self.executor

        class BrokenSnapshotAfterTwo(ops.Operations):
            broken = False

            def poll(instance: Any) -> list[dict[str, Any]]:
                if len(executor.calls) >= 2:
                    instance.broken = True
                    release.set()
                return cast(list[dict[str, Any]], super().poll())

            def publish(instance: Any, progress: Any) -> None:
                if instance.broken:
                    raise ops.OperationsError("synthetic informational write failure")
                super().publish(progress)

        operation = BrokenSnapshotAfterTwo(
            "a" * 64,
            self.directory / "ops",
            0,
            clock=self.clock.monotonic,
            utc_now=self.clock.utc,
        )
        engine = self.engine(operation, workers=2)
        with self.assertRaises(en.OperationalPause):
            engine.run([job(i) for i in range(1, 6)])
        self.assertEqual(len(executor.calls), 2)
        self.assertEqual(len(executor.saved), 2)
        self.assertEqual(engine._in_flight, set())
        self.assertFalse([r for r in engine.ledger.runs.values() if r["status"] == "started"])
        self.assertFalse([e for e in self.store.log if e.kind == "run_failed"])
        self.assertEqual(len([e for e in self.store.log if e.kind == "run_started"]), 2)

    def test_malformed_control_with_calls_running_drains_without_retry(self) -> None:
        release = threading.Event()
        self.executor = Executor(self.clock, release)
        executor = self.executor

        class InvalidControlAfterTwo(ops.Operations):
            sent = False

            def poll(instance: Any) -> list[dict[str, Any]]:
                if not instance.sent and len(executor.calls) >= 2:
                    ops.atomic_json(
                        instance.directory / f"request-{uuid.uuid4().hex}.json",
                        {
                            "schema": "malformed synthetic control",
                            "unexpected": True,
                        },
                    )
                    instance.sent = True
                    release.set()
                return cast(list[dict[str, Any]], super().poll())

        operation = InvalidControlAfterTwo(
            "a" * 64,
            self.directory / "ops",
            0,
            clock=self.clock.monotonic,
            utc_now=self.clock.utc,
        )
        engine = self.engine(operation, workers=2)
        with self.assertRaises(en.OperationalPause):
            engine.run([job(i) for i in range(1, 6)])
        self.assertEqual(len(executor.calls), 2)
        self.assertEqual(len(executor.saved), 2)
        self.assertEqual(engine._in_flight, set())
        self.assertTrue(all(r["status"] == "responded" for r in engine.ledger.runs.values()))
        self.assertFalse([e for e in self.store.log if e.kind == "run_failed"])
        self.assertFalse(engine.ledger.exhausted)

    def test_pacing_preserves_synthetic_results_and_exact_packets(self) -> None:
        jobs = [job(i) for i in range(1, 7)]

        class RecordingExecutor(Executor):
            def __init__(instance: Any, clock: Clock) -> None:
                super().__init__(clock)
                instance.packets: list[tuple[str, str]] = []

            def headless(instance: Any, role: str, run_id: str, packet: str) -> Session:
                with instance.lock:
                    instance.packets.append((role, packet))
                return super().headless(role, run_id, packet)

        def execute(pacing: bool) -> tuple[Any, Any, Any]:
            clock = Clock()
            store = Store(clock)
            executor = RecordingExecutor(clock)
            operational = (
                ops.Operations(
                    "a" * 64,
                    self.directory / "paced",
                    3,
                    clock=clock.monotonic,
                    utc_now=clock.utc,
                )
                if pacing
                else None
            )
            engine = Engine(
                store,
                executor,
                types.SimpleNamespace(claude_effective=None),
                workers=3,
                sleep=clock.sleep,
                operations=operational,
            )
            outcomes = engine.run(jobs)
            values = {key: (o.value, o.exhausted) for key, o in outcomes.items()}
            return values, sorted(executor.packets), executor.calls

        baseline, baseline_packets, baseline_calls = execute(False)
        paced, paced_packets, paced_calls = execute(True)
        self.assertEqual(paced, baseline)
        self.assertEqual(paced_packets, baseline_packets)
        self.assertEqual(len(paced_calls), len(baseline_calls))
        starts = [at for _, at in paced_calls]
        self.assertTrue(all(b - a >= 3 for a, b in zip(starts, starts[1:], strict=False)), starts)

    def test_checkpoint_seen_at_dispatch_gate_prevents_that_dispatch(self) -> None:
        clock = self.clock

        class CheckpointAtDispatchGate(ops.Operations):
            polls = 0

            def poll(instance: Any) -> list[dict[str, Any]]:
                instance.polls += 1
                # The outer loop already computed checkpoint_due=false. The request
                # arrives at the final pre-dispatch poll, before _start/submit.
                if instance.polls == 2:
                    rid = uuid.uuid4().hex
                    ops.atomic_json(
                        instance.directory / f"request-{rid}.json",
                        {
                            "schema": ops.REQUEST_SCHEMA,
                            "contract_sha256": "a" * 64,
                            "request_id": rid,
                            "action": "checkpoint",
                            "created_at": clock.utc().isoformat(),
                        },
                    )
                return cast(list[dict[str, Any]], super().poll())

        operation = CheckpointAtDispatchGate(
            "a" * 64,
            self.directory / "ops",
            0,
            clock=clock.monotonic,
            utc_now=clock.utc,
        )
        observed = []

        def checkpoint() -> dict[str, Any]:
            observed.append(len(self.executor.calls))
            self.assertFalse([e for e in self.store.log if e.kind == "run_started"])
            return self.store.verify()

        engine = self.engine(operation, workers=3, checkpoint=checkpoint)
        result = engine.run([job(i) for i in range(1, 4)])
        self.assertEqual(observed, [0])
        self.assertEqual(len(result), 3)
        self.assertEqual(len(self.executor.calls), 3)
        self.assertFalse(operation.checkpoint_requested)

    def test_snapshot_contains_aggregate_progress_not_packets_or_ids(self) -> None:
        operation = self.operations(0)
        engine = self.engine(operation)
        engine.run([job(1), job(2)])
        value = ops.read_snapshot(operation.directory, "a" * 64, now=self.clock.utc())
        text = json.dumps(value)
        self.assertNotIn("packet 1", text)
        self.assertNotIn("judge/b1/synthetic", text)
        self.assertIn("progress", value)
        self.assertIn("policy", value)


if __name__ == "__main__":
    unittest.main()
