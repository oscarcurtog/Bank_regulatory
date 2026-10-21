"""Bounded executions and verified cleanup (P1 #7 of the review before the clean restart), with
real processes: a child that ignores SIGTERM, a grandchild in its group, a group that cannot be
verified gone, and an orchestrator asked to stop. No model runs."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ingest import holdout_process as hproc

# a leader that starts a grandchild of its own group, writes both pids, and waits
FAMILY = """
import os, subprocess, sys, time, signal
if {ignore_term}:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
open({pids!r}, "w").write(f"{{os.getpid()}} {{child.pid}}")
time.sleep({sleep})
"""


def family(tmp_path: Path, sleep: float = 300, ignore_term: bool = False) -> tuple[list[str], Path]:
    pids = tmp_path / "pids"
    code = FAMILY.format(ignore_term=ignore_term, pids=str(pids), sleep=sleep)
    return [sys.executable, "-c", code], pids


def wait_for(path: Path) -> list[int]:
    for _ in range(200):
        if path.exists() and path.read_text().strip():
            return [int(p) for p in path.read_text().split()]
        time.sleep(0.05)
    raise AssertionError(f"{path} never written")


def gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_a_process_is_the_leader_of_its_own_group_and_registered(tmp_path):
    command, pids = family(tmp_path, sleep=0.2)
    registry = tmp_path / "process.json"
    process = hproc.spawn(command, registry)
    leader, child = wait_for(pids)
    assert os.getpgid(leader) == leader == process.pid != os.getpgrp()
    assert os.getpgid(child) == leader  # the grandchild shares the group
    assert json.loads(registry.read_text())["pgid"] == leader
    hproc.end_group(process, grace=1)
    assert gone(leader) and gone(child) and not hproc.group_alive(leader)


def test_at_its_deadline_the_whole_group_is_ended_and_verified_gone(tmp_path):
    command, pids = family(tmp_path, ignore_term=True)  # only SIGKILL ends it
    started = time.monotonic()
    with pytest.raises(hproc.ExecutionTimeout):
        hproc.run(command, timeout=1, registry=tmp_path / "process.json", grace=1)
    assert time.monotonic() - started < 15
    leader, child = wait_for(pids)
    assert gone(leader) and gone(child) and not hproc.group_alive(leader)


def test_a_process_that_ends_by_itself_leaves_no_member_of_its_group(tmp_path):
    """The leader exits at once; the grandchild it left behind is ended with the group."""
    command, pids = family(tmp_path, sleep=0)
    code = hproc.run(command, timeout=30)
    assert code == 0
    leader, child = wait_for(pids)
    assert gone(child) and not hproc.group_alive(leader)


def test_the_callers_interruption_ends_the_group_on_the_way_out(tmp_path):
    command, pids = family(tmp_path)

    def interrupt(signum, frame):
        raise hproc.Terminated("signal")

    previous = signal.signal(signal.SIGALRM, interrupt)
    signal.alarm(1)
    try:
        with pytest.raises(hproc.Terminated):
            hproc.run(command, timeout=60, grace=1)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    leader, child = wait_for(pids)
    assert gone(leader) and gone(child)


def test_a_group_that_cannot_be_verified_gone_is_cleanup_failed(tmp_path, monkeypatch):
    command, pids = family(tmp_path, sleep=0.1)
    process = hproc.spawn(command)
    wait_for(pids)
    monkeypatch.setattr(hproc, "group_alive", lambda pgid: True)  # it never seems to go
    with pytest.raises(hproc.CleanupFailed):
        hproc.end_group(process, grace=0.2, kill_wait=0.3)
    monkeypatch.undo()
    hproc.end_group(process, grace=0.2)


def test_a_supervisor_ends_the_groups_an_orchestrator_registered(tmp_path):
    command, pids = family(tmp_path)
    spool = tmp_path / "spool"
    spool.mkdir()
    process = hproc.spawn(command, spool / hproc.REGISTRY)
    leader, child = wait_for(pids)
    assert hproc.end_registered(spool) is True
    process.wait(timeout=10)
    assert gone(child) and not hproc.group_alive(leader)
    assert hproc.end_registered(spool) is False  # nothing left to end
    assert hproc.end_registered(tmp_path / "no-spool") is False


def test_sigterm_reaches_the_orchestrator_as_terminated(tmp_path):
    script = (
        "import time\n"
        "from ingest import holdout_process as hproc\n"
        "hproc.install_termination()\n"
        "try:\n"
        "    time.sleep(60)\n"
        "except hproc.Terminated:\n"
        "    print('terminated', flush=True)\n"
        "    raise SystemExit(7)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(Path.cwd())}
    process = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True, env=env
    )
    time.sleep(1.5)
    process.send_signal(signal.SIGTERM)
    out, _ = process.communicate(timeout=20)
    assert process.returncode == 7 and "terminated" in out


def test_spawn_refuses_to_have_its_group_set_by_the_caller():
    with pytest.raises(ValueError):
        hproc.spawn([sys.executable, "-c", "pass"], start_new_session=False)


def test_a_child_whose_registry_cannot_be_written_is_ended_at_once(tmp_path):
    pidfile = tmp_path / "pid"
    code = f"import os, time; open({str(pidfile)!r}, 'w').write(str(os.getpid())); time.sleep(300)"
    with pytest.raises(OSError):
        hproc.spawn([sys.executable, "-c", code], tmp_path / "no-such-folder" / "process.json")
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text():
            break
        time.sleep(0.05)
    if pidfile.exists() and pidfile.read_text():
        assert gone(int(pidfile.read_text()))


def test_an_operation_in_this_process_has_its_deadline():
    with pytest.raises(hproc.OperationTimeout):
        with hproc.alarm(0.3):
            time.sleep(5)
    with hproc.alarm(5):
        pass  # within it: nothing raised, and the timer is cleared
    time.sleep(0.1)
