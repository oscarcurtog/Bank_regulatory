"""The unattended controller of TEST's construction (S3, P1 #4, #5 and #7 of the review before
the clean restart): the one supervisor that may run START or RESUME without a person at each
step, under the person's standing authorization for one construction, given by typing the
pinned contract's SHA-256 and this policy's id.

It runs the launcher's own commands as children (scripts/holdout.py start | resume) and reads
only the durable state and the launcher's validate(). Its whole policy:

- before every invocation: validate() clean (the contract and every frozen file, the
  runtimes and the host, the public commit, the state verified), or STOP;
- the first invocation: START only where no state exists and when asked; RESUME only when
  asked; nothing else;
- after every invocation it reads the state, and goes on only if the invocation produced
  observable progress (new events) and ended on ONE new technical failure of the waiting
  stage, recorded by that same invocation (its run_failed event comes after the state's head
  before it ran), never used before, with executions left, and either
  * a technical reason the protocol retries (RETRIED), or
  * isolation whose only finding lies inside the authorized metadata (a reasoning block's
    signature, an ISO timestamp, the cost fields): the frozen audit, run again on the
    recorded run, must reproduce the recorded problem exactly, and find nothing at all once
    only those values are neutralized;
- anything else STOPS: the omission audit, any construction stop, an integrity or identity
  problem, an unresolved stage, an isolation finding anywhere else (reasoning text,
  attachments, any content), a failure no invocation produced (stale), an invocation that
  recorded nothing (no progress), a construction complete (the freeze is the person's), and
  any exit with no recorded outcome.

Each invocation runs in a process group of its own (ingest/holdout_process.py): with no new
event for STALL_SECONDS, longer than any execution's own deadline, or past
INVOCATION_SECONDS, the group is ended, reaped and verified gone, and the controller STOPS.
After every invocation, however it ended, the groups the open runs registered in their spools
are ended and verified gone. It never freezes and never runs S4 or a retriever. It prints and
logs codes, counts and field names only.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from ingest import holdout_pipeline as hpl
from ingest import holdout_process as hproc
from ingest import holdout_roles as hr
from ingest import holdout_state as hs

POLICY = "holdout-controller.v1"
# technical reasons the protocol retries while executions remain; isolation has its own rule,
# and runtime_identity, process_cleanup or any other reason is a hard stop
RETRIED = frozenset(
    {
        "transport",
        "broker",
        "malformed_session",
        "model_api",
        "resource",
        "infrastructure",
        "protocol",
        "interrupted",
        "timeout",
    }
)
FORBIDDEN = "forbidden content outside the authorized input: "
STALL_SECONDS = 5400  # longer than any execution's deadline (holdout_process.py)
INVOCATION_SECONDS = 48 * 3600
POLL_SECONDS = 30
MAX_CYCLES = 500
TERMINATION_GRACE = 120.0  # for the orchestrator to end its own children on SIGTERM
LOCK_FILE = "holdout-controller.lock"
LOG_DIR = Path.home() / ".fundamento-private" / "holdout-logs"
ISO = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?(Z|[+-]\d\d:\d\d)?$")
COSTS = ("costUSD", "totalCostUSD", "total_cost_usd")
REASONING = ("thinking", "redacted_thinking")


class ControllerStop(Exception):
    """The controller stops and goes back to the person, with a code."""


# ---------------------------------------------------------------------------
# what the controller reads of the state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Failure:
    seq: int
    run_id: str
    proposal_id: str
    stage: str
    reason: str


@dataclass(frozen=True)
class Snapshot:
    exists: bool
    events: int = 0
    head: str = hs.GENESIS
    status: str = ""
    waiting: tuple[Mapping[str, Any], ...] = ()
    terminal: bool = False
    stop: Mapping[str, Any] | None = None
    failures: tuple[Failure, ...] = ()
    open_runs: tuple[str, ...] = ()

    @classmethod
    def read(cls, test: Path, raw: Path) -> Snapshot:
        """The state, verified (an integrity problem raises), as the controller needs it."""
        if not (test / hs.STATE_DIR / hs.DB_FILE).exists():
            return cls(False)
        with hs.Store.open(test, raw, readonly=True) as store:
            summary = store.verify()
            failures = tuple(
                Failure(e.seq, str(e.run_id), str(e.proposal_id), str(e.stage), e.payload["reason"])
                for e in store.events()
                if e.kind == "run_failed"
            )
            cells = store.cells()
            open_runs = tuple(r["run_id"] for r in store.runs() if r["status"] == "started")
        terminal = all(c["state"] in hpl.TERMINAL for c in cells)
        return cls(
            True,
            summary["events"],
            summary["head_sha256"],
            summary["status"],
            tuple(summary["waiting"]),
            terminal,
            summary["stop"],
            failures,
            open_runs,
        )

    def counts(self) -> str:
        w = self.waiting[0] if self.waiting else {}
        return (
            f"status={self.status} events={self.events} waiting={w.get('cell_id')}/"
            f"{w.get('stage')}/{w.get('state')}/failed={w.get('failed_technical_runs')}/"
            f"{w.get('last_failure')}"
        )


@dataclass(frozen=True)
class Decision:
    resume: bool
    code: str  # a short code: why it goes on, or why it stops
    run_id: str | None = None


def decide(
    before: Snapshot,
    after: Snapshot,
    used: Sequence[str],
    isolation_check: Callable[[str], tuple[bool, str]],
) -> Decision:
    """After one invocation: RESUME again, or STOP. Pure but for the isolation re-audit."""
    if not after.exists:
        return Decision(False, "no_state")
    if after.events <= before.events:
        return Decision(False, "no_progress: the invocation recorded no event")
    if after.status == "stopped":
        return Decision(False, f"construction_stopped: {(after.stop or {}).get('reason')}")
    if after.status in ("pending_technical_unresolved", "frozen"):
        return Decision(False, after.status)
    if after.status == "building":
        if after.terminal:
            return Decision(False, "complete: every cell terminal; the freeze is the person's")
        return Decision(False, "no_recorded_outcome: the invocation ended mid-construction")
    if after.status != "pending_technical" or len(after.waiting) != 1:
        return Decision(False, f"unexpected: {after.status} with {len(after.waiting)} waiting")
    waiting = after.waiting[0]
    new = [f for f in after.failures if f.seq > before.events]
    if not new:
        return Decision(False, "stale: no failure produced by this invocation")
    last = new[-1]
    if (last.proposal_id, last.stage, last.reason) != (
        waiting["proposal_id"],
        waiting["stage"],
        waiting["last_failure"],
    ):
        return Decision(False, "stale: the newest failure is not the waiting stage's")
    if last.run_id in used:
        return Decision(False, "stale: a failure already used to resume")
    if waiting["failed_technical_runs"] >= hs.MAX_TECHNICAL_RUNS:
        return Decision(False, "no_execution_left")
    if last.reason in RETRIED:
        return Decision(True, f"technical_retry: {last.reason}", last.run_id)
    if last.reason == "isolation":
        ok, why = isolation_check(last.run_id)
        if ok:
            return Decision(True, f"isolation_metadata_only: {why}", last.run_id)
        return Decision(False, f"isolation: {why}")
    return Decision(False, f"hard_stop: {last.reason}")


# ---------------------------------------------------------------------------
# isolation: the metadata-only false positive, and nothing else
# ---------------------------------------------------------------------------


def neutralized(texts: Mapping[str, str]) -> tuple[dict[str, str], int]:
    """The session with only the authorized metadata values neutralized, and how many: a
    reasoning block's signature, an ISO timestamp, the three cost fields. Everything else,
    content, reasoning text and attachments included, is left exactly as it is."""
    count = 0

    def walk(x: Any) -> None:
        nonlocal count
        if isinstance(x, dict):
            for k, v in list(x.items()):
                if k == "signature" and x.get("type") in REASONING and isinstance(v, str) and v:
                    x[k] = ""
                    count += 1
                elif k == "timestamp" and isinstance(v, str) and ISO.match(v):
                    x[k] = ""
                    count += 1
                elif k in COSTS and isinstance(v, (int, float)) and not isinstance(v, bool):
                    x[k] = 0.0
                    count += 1
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    out = dict(texts)
    for source in ("transcript", "stream"):
        lines = []
        for line in texts.get(source, "").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                lines.append(line)
                continue
            walk(record)
            lines.append(json.dumps(record, ensure_ascii=False))
        out[source] = "\n".join(lines)
    return out, count


def packet_of(row: Mapping[str, Any], doc: Mapping[str, Any]) -> str | None:
    """The packet a recorded run received, found by the hash its run recorded."""
    if doc["kind"] == "broker":
        sent = doc["run"]["sent"]
        return str(sent[0]) if sent else None
    target = row["packet_sha256"]
    for source in ("transcript", "stream"):
        for line in doc["session"].get(source, "").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            stack: list[Any] = [record]
            while stack:
                x = stack.pop()
                if isinstance(x, dict):
                    stack += x.values()
                elif isinstance(x, list):
                    stack += x
                elif isinstance(x, str) and hashlib.sha256(x.encode()).hexdigest() == target:
                    return x
    return None


Audit = Callable[[hr.RoleRun], hr.RoleAudit]


def metadata_only_isolation(test: Path, raw: Path, run_id: str, audit: Audit) -> tuple[bool, str]:
    """Whether a failed isolation run is the authorized case: one forbidden-content problem,
    reproduced by the frozen audit on the recorded run, and none at all once only the
    authorized metadata values are neutralized."""
    with hs.Store.open(test, raw, readonly=True) as store:
        rows = [r for r in store.runs() if r["run_id"] == run_id]
        if len(rows) != 1:
            return False, "no such run"
        row = dict(rows[0])
        if row["status"] != "failed" or row["failure_reason"] != "isolation":
            return False, f"run {row['status']} / {row['failure_reason']}"
        if not row["response"]:
            return False, "the run has no recorded response"
        doc = json.loads(store.text(json.loads(row["response"])["raw"], "raw"))
    problems = json.loads(row["audit"] or "{}").get("problems") or []
    if len(problems) != 1 or not str(problems[0]).startswith(FORBIDDEN):
        return False, f"{len(problems)} audit problems, not the single forbidden-content one"
    if doc.get("kind") not in ("headless", "broker"):
        return False, "not a Claude session"
    packet = packet_of(row, doc)
    if packet is None:
        return False, "the packet cannot be identified by its hash"
    run = hpl.role_run(row["role"], run_id, packet, doc)
    again = audit(run)
    if again.problems != problems:
        return False, "the frozen audit does not reproduce the recorded problem"
    session, count = neutralized(run.session)
    if not count:
        return False, "the session has no authorized metadata field"
    cleared = copy.copy(run)
    cleared.session = session
    without = audit(cleared)
    if not without.ok or without.problems:
        return False, f"{len(without.problems)} problems remain with only metadata neutralized"
    categories = sorted(set(re.findall(r"'(\w+)'", str(problems[0]))))
    return True, f"{count} metadata values, categories {categories}"


# ---------------------------------------------------------------------------
# the invocations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ended:
    code: int | None
    how: str  # exited | stalled | deadline


def supervise(
    command: Sequence[str],
    log: TextIO,
    progress: Callable[[], int | None],
    stall: float = STALL_SECONDS,
    deadline: float = INVOCATION_SECONDS,
    poll: float = POLL_SECONDS,
    grace: float = TERMINATION_GRACE,
    cwd: Path | None = None,
) -> Ended:
    """One invocation in a group of its own: until it exits, or stalls (no new event for
    stall seconds), or passes its deadline; then its group ended, reaped and verified gone."""
    process = hproc.spawn(command, stdout=log, stderr=log, cwd=cwd)
    started = last_change = time.monotonic()
    last = progress()
    how = "exited"
    try:
        while True:
            try:
                process.wait(timeout=poll)
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            found = progress()
            if found is not None and found != last:
                last, last_change = found, now
            if now - last_change >= stall:
                how = "stalled"
                hproc.end_group(process, grace=grace)
                break
            if now - started >= deadline:
                how = "deadline"
                hproc.end_group(process, grace=grace)
                break
    except BaseException:
        hproc.end_group(process, grace=grace)
        raise
    code = hproc.finish(process) if how == "exited" else process.returncode
    return Ended(code, how)


def end_open_runs(test: Path, raw: Path, spool: Path) -> int:
    """The groups the open runs registered, ended and verified gone (CleanupFailed if one
    cannot be): what an orchestrator that was ended could not end itself."""
    if not (test / hs.STATE_DIR / hs.DB_FILE).exists():
        return 0
    with hs.Store.open(test, raw, readonly=True) as store:
        open_runs = [r["run_id"] for r in store.runs() if r["status"] == "started"]
    return sum(hproc.end_registered(spool / run_id) for run_id in open_runs)


def event_count(test: Path) -> int | None:
    """The number of events, read without verifying (it runs while the child writes)."""
    path = test / hs.STATE_DIR / hs.DB_FILE
    if not path.exists():
        return 0
    import sqlite3

    try:
        db = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5)
        try:
            row = db.execute("SELECT max(seq) FROM events").fetchone()
        finally:
            db.close()
    except sqlite3.Error:
        return None
    return int(row[0] or 0)


@dataclass
class Controller:
    """One unattended construction: the loop, its log, and what it stopped on."""

    test: Path
    raw: Path
    contract: str
    validate: Callable[[], list[str]]
    audit: Audit
    command: Callable[[str], list[str]]
    logs: Path
    stall: float = STALL_SECONDS
    deadline: float = INVOCATION_SECONDS
    poll: float = POLL_SECONDS
    grace: float = TERMINATION_GRACE
    max_cycles: int = MAX_CYCLES
    cwd: Path | None = None
    lines: list[str] = field(default_factory=list)

    @property
    def spool(self) -> Path:
        return self.raw / "spool"

    def say(self, *parts: Any) -> None:
        line = " ".join([datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), *map(str, parts)])
        self.lines.append(line)
        print(line, flush=True)
        with self._log_path().open("a", encoding="utf-8") as out:
            out.write(line + "\n")

    def _log_path(self) -> Path:
        return self.logs / "controller.log"

    def _invoke(self, action: str) -> Ended:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = self.logs / f"{action}-{stamp}.log"
        with path.open("w", encoding="utf-8") as log:
            path.chmod(0o600)
            ended = supervise(
                self.command(action),
                log,
                lambda: event_count(self.test),
                self.stall,
                self.deadline,
                self.poll,
                self.grace,
                self.cwd,
            )
        found = end_open_runs(self.test, self.raw, self.spool)
        if found:
            self.say(f"cleanup: {found} process groups of open runs ended and verified gone")
        return ended

    def _checked(self) -> None:
        problems = self.validate()
        if problems:
            raise ControllerStop(f"validate: {len(problems)} problems: {problems[:3]}")

    def run(self, first: str) -> str:
        """The loop, from a first action the person asked for (start or resume). Returns the
        code it stopped on; it never raises for a stop."""
        self.logs.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock = os.open(self.logs / LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(lock)
            return "busy: another controller runs"
        previous = None
        if threading.current_thread() is threading.main_thread():
            previous = hproc.install_termination()  # SIGTERM: its child's group ended first
        try:
            return self._loop(first)
        except ControllerStop as stop:
            self.say("STOP:", stop)
            return str(stop)
        except (hs.StateError, hproc.CleanupFailed) as error:
            self.say("STOP:", type(error).__name__, error)
            return f"{type(error).__name__}: {error}"
        except (hproc.Terminated, KeyboardInterrupt) as error:
            # the invocation's group was ended by supervise; the open runs' groups now
            self._cleanup_quietly()
            self.say("STOP: interrupted:", type(error).__name__)
            return f"interrupted: {type(error).__name__}"
        except Exception as error:
            self._cleanup_quietly()
            self.say("STOP: unexpected:", type(error).__name__, str(error)[:200])
            return f"unexpected: {type(error).__name__}"
        finally:
            if previous is not None:
                signal.signal(signal.SIGTERM, previous)
            os.close(lock)
            self.say("CONTROLLER END")

    def _cleanup_quietly(self) -> None:
        try:
            end_open_runs(self.test, self.raw, self.spool)
        except Exception as error:
            self.say("cleanup of the open runs failed:", type(error).__name__)

    def _loop(self, first: str) -> str:
        before = Snapshot.read(self.test, self.raw)
        if first == "start":
            if before.exists:
                raise ControllerStop("start: a state exists; START never starts it again")
        elif first == "resume":
            if not before.exists:
                raise ControllerStop("resume: no state")
        else:
            raise ControllerStop(f"unknown first action {first!r}")
        action = first
        used: list[str] = []
        for cycle in range(1, self.max_cycles + 1):
            self._checked()
            self.say(f"cycle {cycle}: {action}; {before.counts() if before.exists else 'no state'}")
            ended = self._invoke(action)
            if ended.how != "exited":
                raise ControllerStop(f"{ended.how}: the invocation was ended, its group gone")
            after = Snapshot.read(self.test, self.raw)
            decision = decide(before, after, used, self._isolation)
            self.say(f"cycle {cycle}: exit {ended.code}; {after.counts()}; {decision.code}")
            if not decision.resume:
                raise ControllerStop(decision.code)
            assert decision.run_id is not None
            used.append(decision.run_id)
            before, action = after, "resume"
        raise ControllerStop("cycle_limit")

    def _isolation(self, run_id: str) -> tuple[bool, str]:
        return metadata_only_isolation(self.test, self.raw, run_id, self.audit)


def child_command(contract: str, repo: Path, paths: Any) -> Callable[[str], list[str]]:
    """The launcher's own command for an action, on the same private paths, kept awake on
    macOS."""

    def command(action: str) -> list[str]:
        base = [
            sys.executable,
            str(repo / "scripts" / "holdout.py"),
            "--dir",
            str(paths.test),
            "--raw-dir",
            str(paths.raw),
            "--backup-dir",
            str(paths.backups),
            action,
            "--contract",
            contract,
        ]
        awake = shutil.which("caffeinate")
        return [awake, "-is", *base] if awake else base

    return command
