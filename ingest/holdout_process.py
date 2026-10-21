"""Bounded executions and verified cleanup of the processes TEST's construction starts (S3,
P1 #7 of the review before the clean restart).

Every process started for a model (a headless Claude role, the stream-json judge behind the
evidence broker, the MLX diagnostic and its token count) and every invocation the unattended
controller starts runs in a process group of its own, under a deadline. At the deadline, or
when its caller stops for any reason, the group is ended in a fixed order:

1. SIGTERM to the whole group;
2. a grace period for the leader to exit;
3. SIGKILL to the whole group;
4. the leader waited for (reaped);
5. the group checked gone, with SIGKILL again while any member is left.

A process that ends by itself is reaped and its group checked gone in the same way, so that
no child outlives its execution. Only a group verified gone lets the caller record the
failure and allow another technical execution. A group that cannot be verified gone is
CleanupFailed: no execution may follow it, and the construction stops for a person
(ingest/holdout_checks.py process_cleanup).

The leader's pid, which is also its group's id, is written into the run's spool as soon as
the process starts (REGISTRY), so that a supervisor that had to end the orchestrator can find
and end the groups the orchestrator could not (end_registered).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

GRACE_SECONDS = 10.0
KILL_WAIT_SECONDS = 10.0
REGISTRY = "process.json"
# The deadlines of the construction's executions, fixed before authoring: a headless role, a
# whole broker run and each of its turns, the diagnostic's generation and its token count.
HEADLESS_SECONDS = 900
BROKER_TURN_SECONDS = 600
BROKER_RUN_SECONDS = 3600
LOCAL_SECONDS = 1800
TOKEN_COUNT_SECONDS = 300
OPERATION_SECONDS = 60  # one corpus operation of the broker, in this process


class ExecutionTimeout(Exception):
    """A process reached its deadline: its group was ended and verified gone."""


class CleanupFailed(Exception):
    """A process group could not be verified gone after SIGKILL: no execution may follow."""


class OperationTimeout(Exception):
    """An operation in this process (a corpus operation of the broker) passed its deadline."""


class Terminated(BaseException):
    """The orchestrator was asked to stop (SIGTERM). A BaseException, like the death of the
    process: no handler of the construction treats it as a failure of a run, and every
    executor ends its child's group on the way out."""


def raise_terminated(signum: int, frame: Any) -> None:
    raise Terminated(f"signal {signum}")


def install_termination() -> Any:
    """SIGTERM ends the orchestrator through Terminated, so that its children are ended.
    Returns the handler it replaced."""
    return signal.signal(signal.SIGTERM, raise_terminated)


@contextmanager
def alarm(seconds: float) -> Iterator[None]:
    """A deadline for an operation run in this process, on its main thread: OperationTimeout
    when it passes (SIGALRM, which interrupts even a regular expression's backtracking). Off
    the main thread, where no signal can be delivered, the operation runs unbounded."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def expire(signum: int, frame: Any) -> None:
        raise OperationTimeout(f"the deadline of {seconds:g} s")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


# Every process of these groups is this user's: the construction started them. So the kernel
# refusing a signal to the group (EPERM) cannot mean a live member owned by someone else; on
# macOS it is what killpg answers when only zombies are left, dead processes waiting to be
# reaped by their parent, which run nothing and hold nothing but their exit status.


def _signal(pgid: int, signum: int) -> None:
    try:
        os.killpg(pgid, signum)
    except (ProcessLookupError, PermissionError):
        pass


def group_alive(pgid: int) -> bool:
    """Whether any live process of the group is left."""
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _write_registry(path: Path, pid: int, command: Sequence[str]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as out:
        out.write(json.dumps({"pid": pid, "pgid": pid, "program": Path(command[0]).name}))
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)


def spawn(
    command: Sequence[str], registry: Path | None = None, **popen: Any
) -> subprocess.Popen[Any]:
    """The process, leader of a new process group (start_new_session), registered at once."""
    if "start_new_session" in popen or "preexec_fn" in popen or "process_group" in popen:
        raise ValueError("the process group is this module's to set")
    process: subprocess.Popen[Any] = subprocess.Popen(
        list(command), start_new_session=True, **popen
    )
    if registry is not None:
        try:
            _write_registry(registry, process.pid, command)
        except BaseException:
            # never a live child that no registry names
            end_group(process)
            raise
    return process


def _verify_gone(pgid: int, kill_wait: float) -> None:
    deadline = time.monotonic() + kill_wait
    while group_alive(pgid):
        if time.monotonic() > deadline:
            raise CleanupFailed(f"process group {pgid} is still alive after SIGKILL")
        _signal(pgid, signal.SIGKILL)
        time.sleep(0.05)


def end_group(
    process: subprocess.Popen[Any],
    grace: float = GRACE_SECONDS,
    kill_wait: float = KILL_WAIT_SECONDS,
) -> None:
    """SIGTERM to the group, the grace period, SIGKILL to the group, the leader reaped, the
    group verified gone. CleanupFailed if it cannot be."""
    pgid = process.pid
    _signal(pgid, signal.SIGTERM)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    _signal(pgid, signal.SIGKILL)
    try:
        process.wait(timeout=kill_wait)
    except subprocess.TimeoutExpired:
        raise CleanupFailed(f"process {pgid} did not end after SIGKILL") from None
    _verify_gone(pgid, kill_wait)


def finish(process: subprocess.Popen[Any], kill_wait: float = KILL_WAIT_SECONDS) -> int:
    """After a process ended by itself: the leader reaped, and any member of its group that
    outlived it killed and verified gone. Returns the leader's exit code."""
    try:
        code = process.wait(timeout=kill_wait)
    except subprocess.TimeoutExpired:
        end_group(process, kill_wait=kill_wait)
        code = process.returncode
    _signal(process.pid, signal.SIGKILL)
    _verify_gone(process.pid, kill_wait)
    return int(code)


def run(
    command: Sequence[str],
    timeout: float,
    registry: Path | None = None,
    stdin_text: str | None = None,
    grace: float = GRACE_SECONDS,
    kill_wait: float = KILL_WAIT_SECONDS,
    **popen: Any,
) -> int:
    """One bounded execution: the process within its deadline, its group verified gone
    whatever happens. Returns its exit code; ExecutionTimeout at the deadline, after the group
    was ended; the caller's own interruption, after the group was ended too."""
    if stdin_text is not None:
        popen["stdin"] = subprocess.PIPE
        popen.setdefault("text", True)
    process = spawn(command, registry, **popen)
    try:
        process.communicate(input=stdin_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        end_group(process, grace, kill_wait)
        raise ExecutionTimeout(f"the deadline of {timeout:g} s") from None
    except BaseException:
        end_group(process, grace, kill_wait)
        raise
    return finish(process, kill_wait)


def end_registered(spool: Path, kill_wait: float = KILL_WAIT_SECONDS) -> bool:
    """A supervisor's cleanup of the group a run's spool registered, when the orchestrator
    that started it is gone: SIGKILL to the group while it lives, then verified gone.
    Returns whether a live group was found. CleanupFailed if it cannot be ended."""
    path = spool / REGISTRY
    if not path.exists():
        return False
    pgid = int(json.loads(path.read_text(encoding="utf-8"))["pgid"])
    if not group_alive(pgid):
        return False
    _signal(pgid, signal.SIGKILL)
    _verify_gone(pgid, kill_wait)
    return True


def close_quietly(*files: IO[Any] | None) -> None:
    for f in files:
        if f is not None and not f.closed:
            f.close()
