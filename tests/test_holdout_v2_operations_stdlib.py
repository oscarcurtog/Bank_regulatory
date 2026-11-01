"""Pure stdlib operational tests: no corpus, model, SQLite or private TEST files."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ingest import holdout_v2_operations as ops

CONTRACT = "a" * 64
OTHER = "b" * 64


class Clock:
    def __init__(self) -> None:
        self.mono = 1000.0
        self.wall = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)

    def advance(self, seconds: float) -> None:
        self.mono += seconds
        self.wall += timedelta(seconds=seconds)


class OperationsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name) / "operations"
        self.clock = Clock()
        self.control = self.make()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make(self, **kwargs):
        return ops.Operations(
            CONTRACT,
            self.directory,
            clock=lambda: self.clock.mono,
            utc_now=lambda: self.clock.wall,
            **kwargs,
        )

    def snapshot(self, **progress) -> None:
        self.control.publish({"status": "running", **progress})

    def request(self, action="pause") -> str:
        self.snapshot()
        return ops.request(self.directory, CONTRACT, action, now=self.clock.wall)

    def test_default_interval_and_no_compensating_burst(self) -> None:
        self.assertEqual(self.control.seconds_until_start(), 0)
        self.control.started()
        self.assertEqual(self.control.seconds_until_start(), 300)
        self.clock.advance(299)
        self.assertEqual(self.control.seconds_until_start(), 1)
        self.clock.advance(10000)
        self.assertEqual(self.control.seconds_until_start(), 0)
        self.control.started()
        self.assertEqual(self.control.seconds_until_start(), 300)

    def test_seed_restores_interval_and_window_from_log(self) -> None:
        control = self.make(max_starts=2, window_seconds=1000)
        wall = self.clock.wall
        control.seed([(wall - timedelta(seconds=400)).isoformat(), wall.isoformat()])
        self.assertEqual(control.seconds_until_start(), 600)
        self.clock.advance(601)
        self.assertEqual(control.seconds_until_start(), 0)
        control.started()
        self.assertEqual(control.seconds_until_start(), 399)

    def test_window_counts_excess_old_concurrency_before_dispatching(self) -> None:
        control = self.make(min_interval_seconds=0, max_starts=2, window_seconds=1000)
        control.seed(
            [(self.clock.wall - timedelta(seconds=age)).isoformat() for age in (500, 400, 300)]
        )
        self.assertEqual(control.seconds_until_start(), 600)

    def test_future_wall_time_is_conservative_after_clock_rollback(self) -> None:
        self.control.seed([(self.clock.wall + timedelta(hours=1)).isoformat()])
        self.assertEqual(self.control.seconds_until_start(), 300)
        self.clock.wall += timedelta(days=20)
        self.assertEqual(self.control.seconds_until_start(), 300)
        self.clock.advance(300)
        self.assertEqual(self.control.seconds_until_start(), 0)

    def test_monotonic_clock_regression_does_not_grant_capacity(self) -> None:
        self.control.started()
        self.clock.advance(100)
        self.assertEqual(self.control.seconds_until_start(), 200)
        self.clock.mono -= 200
        self.assertEqual(self.control.seconds_until_start(), 200)

    def test_start_history_uses_real_dispatch_and_is_pruned(self) -> None:
        self.assertEqual(self.control.export_history(), [])
        self.control.started()
        self.assertEqual(len(self.control.export_history()), 1)
        self.clock.advance(86400)
        self.assertEqual(self.control.export_history(), [])

    def test_invalid_policies_fail_closed(self) -> None:
        for kwargs in (
            {"min_interval_seconds": -1},
            {"min_interval_seconds": float("nan")},
            {"min_interval_seconds": True},
            {"window_seconds": 0},
            {"max_starts": 0},
            {"max_starts": 1.5},
            {"max_starts": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ops.OperationsError):
                self.make(**kwargs)

    def test_pause_is_pending_until_acknowledged_across_restart(self) -> None:
        rid = self.request()
        delivered = self.control.poll()
        self.assertEqual([r["request_id"] for r in delivered], [rid])
        self.assertTrue(self.control.pause_requested)
        self.assertFalse(self.control.checkpoint_requested)
        self.assertEqual([r["request_id"] for r in self.control.poll()], [rid])
        self.control.acknowledge(rid)
        self.control.acknowledge(rid)
        self.assertEqual(self.control.poll(), [])
        again = self.make()
        self.assertEqual(again.poll(), [])
        self.assertFalse(again.pause_requested)

    def test_checkpoint_is_not_completed_merely_by_polling(self) -> None:
        rid = self.request("checkpoint")
        self.control.poll()
        self.assertTrue(self.control.checkpoint_requested)
        again = self.make()
        self.assertEqual([r["request_id"] for r in again.poll()], [rid])
        again.acknowledge(rid, "failed")
        self.assertFalse(again.checkpoint_requested)
        self.assertEqual(self.make().poll(), [])

    def test_acknowledged_old_request_does_not_expire_on_restart(self) -> None:
        rid = self.request()
        self.control.poll()
        self.control.acknowledge(rid)
        self.clock.advance(8 * 86400)
        self.assertEqual(self.make().poll(), [])

    def test_request_bound_to_other_construction_fails_closed(self) -> None:
        rid = self.request()
        path = self.directory / f"request-{rid}.json"
        value = json.loads(path.read_text())
        value["contract_sha256"] = OTHER
        ops.atomic_json(path, value)
        with self.assertRaises(ops.OperationsError):
            self.control.poll()
        self.assertFalse(self.control.pause_requested)

    def test_invalid_queue_is_validated_before_any_delivery(self) -> None:
        self.request()
        (self.directory / "request-malformed.json").write_text("{}")
        with self.assertRaises(ops.OperationsError):
            self.control.poll()
        self.assertFalse(self.control.pause_requested)

    def test_duplicate_keys_unknown_fields_expiry_and_oversize_are_rejected(self) -> None:
        rid = self.request()
        path = self.directory / f"request-{rid}.json"
        original = path.read_text()
        replacements = [
            original.replace('"action": "pause"', '"action": "pause", "action": "checkpoint"'),
            original[:-2] + ', "unexpected": 1}\n',
            original.replace(
                self.clock.wall.isoformat(timespec="microseconds"),
                (self.clock.wall - timedelta(days=8)).isoformat(timespec="microseconds"),
            ),
            " " * 2049,
        ]
        for text in replacements:
            path.write_text(text)
            with self.subTest(text=text[:40]), self.assertRaises(ops.OperationsError):
                self.make().poll()

    def test_unknown_or_corrupt_ack_is_rejected(self) -> None:
        rid = self.request()
        ops.atomic_json(self.directory / f"ack-{rid}.json", {})
        with self.assertRaises(ops.OperationsError):
            self.control.poll()

    def test_atomic_snapshot_is_readable_and_informational(self) -> None:
        self.snapshot(planned=10, admitted=3, in_flight=1, pending=6)
        snapshot = ops.read_snapshot(self.directory, CONTRACT, now=self.clock.wall)
        self.assertEqual(snapshot["progress"]["admitted"], 3)
        self.assertEqual(snapshot["policy"]["min_interval_seconds"], 300)
        self.assertFalse(snapshot["snapshot_stale"])
        self.assertEqual(list(self.directory.glob(".ops-*")), [])
        self.clock.advance(121)
        snapshot = ops.read_snapshot(self.directory, CONTRACT, now=self.clock.wall)
        self.assertTrue(snapshot["snapshot_stale"])
        with self.assertRaises(ops.OperationsError):
            ops.read_snapshot(self.directory, OTHER, now=self.clock.wall)

    def test_snapshot_stale_or_terminal_refuses_new_requests(self) -> None:
        self.snapshot()
        self.clock.advance(121)
        with self.assertRaises(ops.OperationsError):
            ops.request(self.directory, CONTRACT, "pause", now=self.clock.wall)
        self.snapshot(status="paused")
        with self.assertRaises(ops.OperationsError):
            ops.request(self.directory, CONTRACT, "pause", now=self.clock.wall)

    def test_engine_terminal_modes_refuse_new_requests(self) -> None:
        for mode in (
            "operational_paused",
            "provider_paused",
            "integrity_stop",
            "finished",
            "complete",
        ):
            self.snapshot(mode=mode)
            with self.subTest(mode=mode), self.assertRaises(ops.OperationsError):
                ops.request(self.directory, CONTRACT, "pause", now=self.clock.wall)

    def test_non_finite_snapshot_does_not_replace_previous_file(self) -> None:
        self.snapshot(admitted=3)
        before = (self.directory / ops.SNAPSHOT_FILE).read_bytes()
        with self.assertRaises(ops.OperationsError):
            self.snapshot(admitted=float("nan"))
        self.assertEqual((self.directory / ops.SNAPSHOT_FILE).read_bytes(), before)

    def test_cli_show_has_no_external_dependencies(self) -> None:
        self.snapshot(admitted=3)
        # Use today's time so that the CLI can independently report its snapshot age.
        self.control.utc_now = lambda: datetime.now(UTC)
        self.snapshot(admitted=3)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = ops.main(["show", "--directory", str(self.directory), "--contract", CONTRACT])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(out.getvalue())["progress"]["admitted"], 3)

    def test_request_directory_symlink_is_refused(self) -> None:
        linked = Path(self.temp.name) / "linked"
        linked.symlink_to(self.directory)
        with self.assertRaises(ops.OperationsError):
            ops.Operations(CONTRACT, linked)


if __name__ == "__main__":
    unittest.main()
