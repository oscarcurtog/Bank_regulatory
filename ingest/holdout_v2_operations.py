"""Local operational controls for TEST v2; no model, corpus or SQLite access.

The coordinator is the only writer of progress and acknowledgements. An external
operator can read snapshot.json or publish one construction-bound request. The
scientific event log remains the source of truth: seed() reconstructs pacing from
its run_started timestamps after a restart, rather than trusting a sidecar.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA = "holdout-v2-operations.v1"
REQUEST_SCHEMA = "holdout-v2-operation-request.v1"
ACK_SCHEMA = "holdout-v2-operation-ack.v1"
SNAPSHOT_FILE = "snapshot.json"
SHA = re.compile(r"[0-9a-f]{64}")
REQUEST_NAME = re.compile(r"request-([0-9a-f]{32})\.json")
MAX_REQUEST_BYTES = 2048
MAX_REQUEST_AGE_SECONDS = 7 * 86400
FUTURE_TOLERANCE_SECONDS = 300


class OperationsError(Exception):
    """An invalid operational policy, request or snapshot: stop dispatching."""


def _utc(value: str | datetime) -> datetime:
    try:
        result = datetime.fromisoformat(value) if isinstance(value, str) else value
        if not isinstance(result, datetime) or result.utcoffset() is None:
            raise ValueError
        return result.astimezone(UTC)
    except (ValueError, TypeError, OverflowError) as error:
        raise OperationsError("invalid operational timestamp") from error


def _stamp(value: datetime) -> str:
    return _utc(value).isoformat(timespec="microseconds")


def _contract(value: str) -> str:
    if not isinstance(value, str) or SHA.fullmatch(value) is None:
        raise OperationsError("invalid construction identity")
    return value


def _json(text: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise OperationsError("duplicate operational JSON key")
            result[key] = value
        return result

    def constant(_: str) -> None:
        raise OperationsError("non-finite operational JSON value")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise OperationsError("invalid operational JSON") from error


def atomic_json(path: Path, value: Any) -> None:
    """Publish a complete file, with the temporary file on the same filesystem."""
    path = Path(path)
    data = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=".ops-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class Policy:
    min_interval_seconds: float = 300.0
    max_starts: int | None = None
    window_seconds: float = 86400.0

    def __post_init__(self) -> None:
        for value, positive in (
            (self.min_interval_seconds, False),
            (self.window_seconds, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or (value <= 0 if positive else value < 0)
            ):
                raise OperationsError("invalid pacing policy")
        if self.max_starts is not None and (
            type(self.max_starts) is not int or self.max_starts < 1
        ):
            raise OperationsError("invalid execution-window limit")

    def record(self) -> dict[str, Any]:
        return {
            "min_interval_seconds": self.min_interval_seconds,
            "max_starts": self.max_starts,
            "window_seconds": self.window_seconds,
        }


class Operations:
    """Coordinator-owned pacing, validated requests and an informational snapshot."""

    def __init__(
        self,
        contract_sha256: str,
        directory: Path,
        min_interval_seconds: float = 300.0,
        max_starts: int | None = None,
        window_seconds: float = 86400.0,
        clock: Callable[[], float] = time.monotonic,
        utc_now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.contract_sha256 = _contract(contract_sha256)
        self.directory = Path(directory)
        if self.directory.is_symlink():
            raise OperationsError("the operational directory must not be a symbolic link")
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.policy = Policy(min_interval_seconds, max_starts, window_seconds)
        self.clock = clock
        self.utc_now = utc_now
        self._monotonic_floor = float(clock())
        if not math.isfinite(self._monotonic_floor):
            raise OperationsError("invalid operational clock")
        self._history: deque[tuple[float, str]] = deque()
        self._last_start: float | None = None
        self._delivered: dict[str, dict[str, Any]] = {}
        self._acknowledged: dict[str, str] = {}
        self._last_ack: dict[str, str] | None = None

    def _now(self) -> float:
        value = float(self.clock())
        if not math.isfinite(value):
            raise OperationsError("invalid operational clock")
        # A misbehaving/replaced clock cannot grant a new burst.
        self._monotonic_floor = max(self._monotonic_floor, value)
        return self._monotonic_floor

    def seed(self, starts: Sequence[str]) -> None:
        """Rebuild from scientific log timestamps, including refusals and failed attempts."""
        now, wall = self._now(), _utc(self.utc_now())
        converted = []
        for value in starts:
            stamp = _utc(value)
            age = max(0.0, (wall - stamp).total_seconds())
            converted.append((now - age, _stamp(stamp)))
        converted.sort(key=lambda item: item[0])
        self._last_start = max((at for at, _ in converted), default=None)
        self._history = deque(
            item for item in converted if now - item[0] < self.policy.window_seconds
        )

    def _prune(self, now: float) -> None:
        while self._history and now - self._history[0][0] >= self.policy.window_seconds:
            self._history.popleft()

    def seconds_until_start(self) -> float:
        now = self._now()
        self._prune(now)
        wait = (
            0.0
            if self._last_start is None
            else max(0.0, self.policy.min_interval_seconds - (now - self._last_start))
        )
        maximum = self.policy.max_starts
        if maximum is not None and len(self._history) >= maximum:
            releases_at = self._history[len(self._history) - maximum][0]
            wait = max(wait, releases_at + self.policy.window_seconds - now)
        return max(0.0, wait)

    def started(self, at_utc: str | None = None) -> None:
        """Record a real dispatch after the coordinator has durably recorded run_started."""
        now = self._now()
        stamp = _stamp(_utc(at_utc) if at_utc is not None else _utc(self.utc_now()))
        self._last_start = now
        self._history.append((now, stamp))
        self._prune(now)

    def export_history(self) -> list[str]:
        self._prune(self._now())
        return [stamp for _, stamp in self._history]

    def _read_request(self, path: Path, now: datetime) -> dict[str, Any]:
        match = REQUEST_NAME.fullmatch(path.name)
        try:
            if match is None or path.is_symlink() or path.stat().st_size > MAX_REQUEST_BYTES:
                raise OperationsError("invalid operational control request")
            value = _json(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or set(value) != {
                "schema",
                "contract_sha256",
                "request_id",
                "action",
                "created_at",
            }:
                raise OperationsError("invalid operational control request")
            if (
                value["schema"] != REQUEST_SCHEMA
                or value["contract_sha256"] != self.contract_sha256
                or value["request_id"] != match.group(1)
                or value["action"] not in ("pause", "checkpoint")
                or not isinstance(value["created_at"], str)
            ):
                raise OperationsError("invalid operational control request")
            _utc(value["created_at"])
            return value
        except (OSError, UnicodeError, TypeError, ValueError) as error:
            raise OperationsError("invalid operational control request") from error

    def _read_ack(self, request: Mapping[str, Any]) -> str | None:
        path = self.directory / f"ack-{request['request_id']}.json"
        if not path.exists():
            return None
        try:
            if path.is_symlink() or path.stat().st_size > MAX_REQUEST_BYTES:
                raise OperationsError("invalid operational acknowledgement")
            value = _json(path.read_text(encoding="utf-8"))
            if (
                not isinstance(value, dict)
                or set(value)
                != {"schema", "contract_sha256", "request_id", "action", "status", "at"}
                or value["schema"] != ACK_SCHEMA
                or value["contract_sha256"] != self.contract_sha256
                or value["request_id"] != request["request_id"]
                or value["action"] != request["action"]
                or value["status"] not in ("completed", "failed")
            ):
                raise OperationsError("invalid operational acknowledgement")
            _utc(value["at"])
            return str(value["status"])
        except (OSError, UnicodeError, TypeError, ValueError) as error:
            raise OperationsError("invalid operational acknowledgement") from error

    def poll(self) -> list[dict[str, Any]]:
        """Validate the entire queue before delivering anything; malformed input fails closed."""
        wall = _utc(self.utc_now())
        # A malformed request-* filename is not silently ignored.
        requests = [self._read_request(p, wall) for p in self.directory.glob("request-*")]
        requests.sort(key=lambda value: (value["created_at"], value["request_id"]))
        acknowledgements = [(r, self._read_ack(r)) for r in requests]
        result = []
        for request, status in acknowledgements:
            rid = str(request["request_id"])
            if status is not None:
                self._acknowledged[rid] = status
                continue
            age = (wall - _utc(request["created_at"])).total_seconds()
            if age < -FUTURE_TOLERANCE_SECONDS or age > MAX_REQUEST_AGE_SECONDS:
                raise OperationsError("expired operational control request")
            self._delivered[rid] = request
            result.append(dict(request))
        return result

    @property
    def pause_requested(self) -> bool:
        return any(
            r["action"] == "pause" and rid not in self._acknowledged
            for rid, r in self._delivered.items()
        )

    @property
    def checkpoint_requested(self) -> bool:
        return any(
            r["action"] == "checkpoint" and rid not in self._acknowledged
            for rid, r in self._delivered.items()
        )

    def acknowledge(self, request_id: str, status: str = "completed") -> None:
        if request_id not in self._delivered or status not in ("completed", "failed"):
            raise OperationsError("invalid operational acknowledgement")
        if request_id in self._acknowledged:
            if self._acknowledged[request_id] != status:
                raise OperationsError("operational acknowledgement changed")
            return
        request = self._delivered[request_id]
        record = {
            "schema": ACK_SCHEMA,
            "contract_sha256": self.contract_sha256,
            "request_id": request_id,
            "action": request["action"],
            "status": status,
            "at": _stamp(self.utc_now()),
        }
        atomic_json(self.directory / f"ack-{request_id}.json", record)
        self._acknowledged[request_id] = status
        self._last_ack = {"request_id": request_id, "action": request["action"], "status": status}

    def publish(self, progress: Mapping[str, Any]) -> None:
        """Informational only. The caller must supply counters, never scientific content."""
        wait = self.seconds_until_start()
        wall = _utc(self.utc_now())
        snapshot = {
            "schema": SCHEMA,
            "contract_sha256": self.contract_sha256,
            "updated_at": _stamp(wall),
            "policy": self.policy.record(),
            "control": {
                "pause_requested": self.pause_requested,
                "checkpoint_requested": self.checkpoint_requested,
                "last_acknowledgement": self._last_ack,
                "seconds_until_next_start": wait,
                "next_dispatch_at": _stamp(wall + timedelta(seconds=wait)),
            },
            "progress": dict(progress),
        }
        try:
            atomic_json(self.directory / SNAPSHOT_FILE, snapshot)
        except (OSError, ValueError, TypeError, OverflowError) as error:
            raise OperationsError("invalid operational snapshot") from error


def request(
    directory: Path,
    contract_sha256: str,
    action: str,
    *,
    now: datetime | None = None,
) -> str:
    """Publish a request without opening or locking scientific state."""
    identity = _contract(contract_sha256)
    if action not in ("pause", "checkpoint"):
        raise OperationsError("invalid operational action")
    directory = Path(directory)
    if directory.is_symlink():
        raise OperationsError("the operational directory must not be a symbolic link")
    # Require a coordinator-owned identity; do not create a misleading empty control area.
    snapshot = read_snapshot(directory, identity, now=now)
    if snapshot["snapshot_stale"]:
        raise OperationsError("the operational snapshot is stale")
    mode = snapshot["progress"].get("mode", snapshot["progress"].get("status"))
    if mode in (
        "finished",
        "complete",
        "paused",
        "stopped",
        "paused_provider",
        "paused_operational",
        "operational_paused",
        "provider_paused",
        "integrity_stop",
    ):
        raise OperationsError("the construction is not accepting operational requests")
    rid = uuid.uuid4().hex
    value = {
        "schema": REQUEST_SCHEMA,
        "contract_sha256": identity,
        "request_id": rid,
        "action": action,
        "created_at": _stamp(now or datetime.now(UTC)),
    }
    try:
        atomic_json(directory / f"request-{rid}.json", value)
    except (OSError, ValueError, TypeError) as error:
        raise OperationsError("the operational request could not be published") from error
    return rid


def read_snapshot(
    directory: Path,
    contract_sha256: str,
    *,
    now: datetime | None = None,
    stale_after_seconds: float = 120.0,
) -> dict[str, Any]:
    identity = _contract(contract_sha256)
    path = Path(directory) / SNAPSHOT_FILE
    try:
        if path.is_symlink():
            raise OperationsError("invalid operational snapshot")
        value = _json(path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema") != SCHEMA
            or value.get("contract_sha256") != identity
            or not isinstance(value.get("progress"), dict)
        ):
            raise OperationsError("invalid operational snapshot")
        age = (_utc(now or datetime.now(UTC)) - _utc(value["updated_at"])).total_seconds()
        value["snapshot_age_seconds"] = max(0.0, age)
        value["snapshot_stale"] = age < -FUTURE_TOLERANCE_SECONDS or age > stale_after_seconds
        return value
    except (OSError, UnicodeError, KeyError, TypeError, ValueError) as error:
        raise OperationsError("the operational snapshot is unavailable or invalid") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("show", "request"):
        part = sub.add_parser(command)
        part.add_argument("--directory", type=Path, required=True)
        part.add_argument("--contract", required=True)
        if command == "request":
            part.add_argument("action", choices=("pause", "checkpoint"))
    args = parser.parse_args(argv)
    try:
        if args.command == "show":
            value = read_snapshot(args.directory, args.contract)
        else:
            value = {"request_id": request(args.directory, args.contract, args.action)}
        print(json.dumps(value, indent=1, sort_keys=True))
        return 0
    except OperationsError as error:
        print(f"OPERATIONS STOP: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
