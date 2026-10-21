"""The unattended controller (ingest/holdout_controller.py; P1 #4, #5 and #7 of the review
before the clean restart): its decision after each invocation, its supervision of real child
processes, the metadata-only isolation case with the frozen audit, and its whole loop over real
construction processes (tests/holdout_child.py) on the stand-in construction. No model runs."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from ingest import holdout_controller as hct
from ingest import holdout_process as hproc
from ingest import holdout_roles as hr
from tests.holdout_fakes import load_base, raw_dir, world
from tests.unit.test_holdout_process import gone

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# the decision after one invocation, pure
# ---------------------------------------------------------------------------


def snap(**fields: Any) -> hct.Snapshot:
    return hct.Snapshot(True, **fields)


WAITING = {
    "cell_id": "G03",
    "proposal_id": "G03-a1",
    "stage": "judge_b",
    "state": "pending_technical",
    "failed_technical_runs": 1,
    "last_failure": "model_api",
}
BEFORE = snap(events=10, status="building")


def failed(seq: int, reason: str = "model_api", stage: str = "judge_b", run: str = "R-1"):
    return hct.Failure(seq, run, "G03-a1", stage, reason)


def pending(*failures: hct.Failure, **waiting: Any) -> hct.Snapshot:
    w = {**WAITING, **waiting}
    return snap(events=20, status="pending_technical", waiting=(w,), failures=failures)


def no_isolation(run_id: str) -> tuple[bool, str]:
    raise AssertionError("no isolation check expected")


def test_a_new_retried_failure_of_the_waiting_stage_resumes():
    decision = hct.decide(BEFORE, pending(failed(15)), [], no_isolation)
    assert (decision.resume, decision.run_id) == (True, "R-1")
    assert decision.code == "technical_retry: model_api"


@pytest.mark.parametrize(
    ("after", "code"),
    [
        (snap(events=10, status="pending_technical"), "no_progress"),
        (pending(failed(9)), "stale: no failure produced by this invocation"),
        (pending(failed(15, stage="judge_c")), "stale: the newest failure is not the waiting"),
        (pending(failed(15, reason="transport")), "stale: the newest failure is not the waiting"),
        (pending(failed(15), failed_technical_runs=3), "no_execution_left"),
        (
            pending(failed(15, reason="runtime_identity"), last_failure="runtime_identity"),
            "hard_stop",
        ),
        (
            pending(failed(15, reason="process_cleanup"), last_failure="process_cleanup"),
            "hard_stop",
        ),
        (pending(failed(15, reason="no_such_reason"), last_failure="no_such_reason"), "hard_stop"),
        (
            snap(events=20, status="stopped", stop={"reason": "omission_audit"}),
            "construction_stopped: omission_audit",
        ),
        (
            snap(events=20, status="stopped", stop={"reason": "made_up_stop"}),
            "construction_stopped: made_up_stop",
        ),
        (snap(events=20, status="pending_technical_unresolved"), "pending_technical_unresolved"),
        (snap(events=20, status="building", terminal=True), "complete"),
        (snap(events=20, status="building", terminal=False), "no_recorded_outcome"),
        (hct.Snapshot(False), "no_state"),
    ],
)
def test_everything_else_stops(after, code):
    decision = hct.decide(BEFORE, after, [], no_isolation)
    assert not decision.resume and decision.code.startswith(code), decision.code


def test_a_failure_already_used_never_resumes_again():
    """P1 #5: the authorization is bound to the failure the invocation just produced."""
    decision = hct.decide(BEFORE, pending(failed(15)), ["R-1"], no_isolation)
    assert decision == hct.Decision(False, "stale: a failure already used to resume")


def test_isolation_resumes_only_for_the_metadata_only_case():
    after = pending(failed(15, reason="isolation"), last_failure="isolation")
    yes = hct.decide(BEFORE, after, [], lambda run: (True, "1 metadata values"))
    assert yes.resume and yes.code.startswith("isolation_metadata_only")
    no = hct.decide(BEFORE, after, [], lambda run: (False, "content"))
    assert not no.resume and no.code == "isolation: content"


# ---------------------------------------------------------------------------
# only the authorized metadata is ever neutralized
# ---------------------------------------------------------------------------


def test_only_signatures_timestamps_and_costs_are_neutralized():
    records = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "nDCG in the reasoning",
                        "signature": "xxnDCGxx",
                    },
                    {"type": "text", "text": "nDCG", "signature": "kept: not a reasoning block"},
                ]
            },
        },
        {"timestamp": "2026-10-21T01:02:03.456Z", "costUSD": 0.824, "note": "2026-10-21"},
        {"attachment": {"type": "x", "content": "0.824"}},
    ]
    texts = {"transcript": "\n".join(json.dumps(r) for r in records), "stream": "", "stderr": "s"}
    out, count = hct.neutralized(texts)
    assert count == 3
    lines = [json.loads(line) for line in out["transcript"].splitlines()]
    blocks = lines[0]["message"]["content"]
    assert blocks[0] == {"type": "thinking", "thinking": "nDCG in the reasoning", "signature": ""}
    assert blocks[1]["signature"] == "kept: not a reasoning block"
    assert lines[1] == {"timestamp": "", "costUSD": 0.0, "note": "2026-10-21"}
    assert lines[2] == records[2] and out["stderr"] == "s"


# ---------------------------------------------------------------------------
# supervision of a real invocation
# ---------------------------------------------------------------------------


def child(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_an_invocation_that_exits_is_reaped_with_its_group(tmp_path):
    with (tmp_path / "log").open("w") as log:
        ended = hct.supervise(child("raise SystemExit(3)"), log, lambda: 0, poll=0.1)
    assert ended == hct.Ended(3, "exited")


def test_a_stalled_invocation_is_ended_and_its_group_verified_gone(tmp_path):
    pids = tmp_path / "pids"
    code = (
        "import os, subprocess, sys, time\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
        f"open({str(pids)!r}, 'w').write(f'{{os.getpid()}} {{c.pid}}')\n"
        "time.sleep(300)\n"
    )
    started = time.monotonic()
    with (tmp_path / "log").open("w") as log:
        ended = hct.supervise(child(code), log, lambda: 7, stall=1.5, poll=0.2, grace=1)
    assert ended.how == "stalled" and time.monotonic() - started < 30
    leader, grandchild = (int(p) for p in pids.read_text().split())
    assert not hproc.group_alive(leader)
    assert gone(grandchild)


def test_an_invocation_past_its_deadline_is_ended_even_while_it_progresses(tmp_path):
    ticks = iter(range(10_000))
    with (tmp_path / "log").open("w") as log:
        ended = hct.supervise(
            child("import time; time.sleep(300)"),
            log,
            lambda: next(ticks),  # always new events: never a stall
            deadline=1.5,
            poll=0.2,
            grace=1,
        )
    assert ended.how == "deadline"


# ---------------------------------------------------------------------------
# the whole loop, over real construction processes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base():
    return load_base()


def audit_of(base: dict[str, Any]) -> hct.Audit:
    w = world(base)

    def audit(run: hr.RoleRun) -> hr.RoleAudit:
        return hr.audit_role_run(
            run,
            w.terms,
            w.frozen_templates,
            w.forbidden,
            pin=w.pin,
            tmp_root=w.tmp_root,
            tools=w.tools,
        )

    return audit


def invocations(directory: Path, scenarios: list[dict[str, Any] | str]):
    """The command of each invocation in turn: a real construction process with its scenario,
    or a literal Python program in its place."""
    calls: list[str] = []

    def command(action: str) -> list[str]:
        item = scenarios[min(len(calls), len(scenarios) - 1)]
        calls.append(action)
        if isinstance(item, str):
            return child(item)
        args = [str(directory), action, json.dumps(item)]
        return [sys.executable, "-m", "tests.holdout_child", *args]

    return command, calls


def controller(base, tmp_path, scenarios, validate=lambda: []) -> tuple[hct.Controller, list[str]]:
    directory = tmp_path / "test"
    command, calls = invocations(directory, scenarios)
    run = hct.Controller(
        directory,
        raw_dir(directory),
        "sha",
        validate,
        audit_of(base),
        command,
        tmp_path / "logs",
        stall=600,
        poll=0.5,
        grace=5,
        cwd=REPO,
    )
    return run, calls


FIRST = "G03"
SIGNED = [
    {
        "type": "assistant",
        "message": {"content": [{"type": "thinking", "thinking": "", "signature": "xxnDCGxx"}]},
    }
]
LEAK = [{"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": "nDCG"}]}}]


@pytest.mark.corpus
def test_the_loop_retries_a_new_technical_failure_and_stops_at_completion(base, tmp_path):
    failing = {"technical": {FIRST: "judge_b"}, "technical_kind": "raise"}
    run, calls = controller(base, tmp_path, [failing, {}])
    code = run.run("start")
    assert calls == ["start", "resume"]
    assert code.startswith("complete"), code
    assert any("technical_retry: model_api" in line for line in run.lines)


@pytest.mark.corpus
def test_the_loop_stops_on_an_invocation_that_records_nothing(base, tmp_path):
    failing = {"technical": {FIRST: "judge_b"}, "technical_kind": "raise"}
    run, calls = controller(base, tmp_path, [failing, "raise SystemExit(1)"])
    code = run.run("start")
    assert calls == ["start", "resume"] and code.startswith("no_progress"), code


@pytest.mark.corpus
def test_the_loop_resumes_a_metadata_only_isolation_finding(base, tmp_path):
    signed = {
        "technical": {FIRST: "judge_b"},
        "technical_kind": "records",
        "technical_times": 1,
        "records": {f"{FIRST}|judge_b": SIGNED},
    }
    run, calls = controller(base, tmp_path, [signed, {}])
    code = run.run("start")
    assert calls == ["start", "resume"] and code.startswith("complete"), code
    assert any("isolation_metadata_only" in line for line in run.lines)


@pytest.mark.parametrize(
    "where",
    [
        pytest.param({"records": {f"{FIRST}|judge_b": LEAK}}, id="in_reasoning"),
        pytest.param({"technical_kind": "contaminate"}, id="in_a_reminder"),
        pytest.param(
            {
                "records": {
                    f"{FIRST}|judge_b": [
                        {"type": "attachment", "attachment": {"type": "x", "content": "nDCG"}}
                    ]
                }
            },
            id="in_an_attachment",
        ),
    ],
)
@pytest.mark.corpus
def test_the_loop_stops_on_a_real_isolation_finding(base, tmp_path, where):
    leaking = {"technical": {FIRST: "judge_b"}, "technical_kind": "records", **where}
    run, calls = controller(base, tmp_path, [leaking, {}])
    code = run.run("start")
    assert calls == ["start"] and code.startswith("isolation:"), code


@pytest.mark.corpus
def test_the_loop_stops_before_any_invocation_when_validate_finds_a_problem(base, tmp_path):
    run, calls = controller(base, tmp_path, [{}], validate=lambda: ["the binary changed"])
    code = run.run("start")
    assert calls == [] and code.startswith("validate: 1 problems"), code


@pytest.mark.corpus
def test_the_loop_stops_on_a_construction_stop_and_never_freezes(base, tmp_path):
    from ingest import holdout as ho

    audited = ho.omission_cells(base["grid"])
    cell = next(c for c in ho.cell_order(base["grid"]) if c in audited)
    run, calls = controller(base, tmp_path, [{"extra_grade": {cell: 2}}])
    code = run.run("start")
    assert calls == ["start"] and code == "construction_stopped: omission_audit", code


def test_a_second_controller_is_refused_while_one_runs(tmp_path):
    import fcntl

    logs = tmp_path / "logs"
    logs.mkdir()
    held = os.open(logs / hct.LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(held, fcntl.LOCK_EX)
    try:
        run = hct.Controller(
            tmp_path / "test",
            tmp_path / "raw",
            "sha",
            lambda: [],
            lambda r: None,  # type: ignore[arg-type,return-value]
            lambda a: [],
            logs,
        )
        assert run.run("start").startswith("busy")
    finally:
        os.close(held)


def test_the_controller_never_runs_freeze_or_anything_but_start_and_resume():
    source = Path(hct.__file__).read_text(encoding="utf-8")
    assert '"freeze"' not in source and "s4" not in source.lower().replace("s4 or", "")
    commands = hct.child_command(
        "sha", REPO, type("P", (), {"test": Path("t"), "raw": Path("r"), "backups": Path("b")})()
    )
    for action in ("start", "resume"):
        found = commands(action)
        assert found[found.index(str(REPO / "scripts" / "holdout.py")) + 7] == action


def test_an_unexpected_failure_of_the_controller_logs_a_stop(tmp_path):
    def validate() -> list[str]:
        raise RuntimeError("the grid cannot be read")

    run = hct.Controller(
        tmp_path / "test",
        tmp_path / "raw",
        "sha",
        validate,
        lambda r: None,  # type: ignore[arg-type,return-value]
        lambda a: [],
        tmp_path / "logs",
    )
    code = run.run("start")
    assert code == "unexpected: RuntimeError"
    assert any(line.split(" ", 1)[1].startswith("STOP: unexpected") for line in run.lines)


def test_sigterm_to_the_controller_ends_its_invocation_and_logs_a_stop(tmp_path):
    import signal

    pids = tmp_path / "pids"
    program = (
        "import os, subprocess, sys, time\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
        f"open({str(pids)!r}, 'w').write(f'{{os.getpid()}} {{c.pid}}')\n"
        "time.sleep(300)\n"
    )
    run = hct.Controller(
        tmp_path / "test",
        tmp_path / "raw",
        "sha",
        lambda: [],
        lambda r: None,  # type: ignore[arg-type,return-value]
        lambda a: child(program),
        tmp_path / "logs",
        poll=0.2,
        grace=1,
    )
    previous = signal.signal(signal.SIGALRM, lambda s, f: os.kill(os.getpid(), signal.SIGTERM))
    signal.setitimer(signal.ITIMER_REAL, 2)
    try:
        code = run.run("start")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert code == "interrupted: Terminated"
    leader, grandchild = (int(p) for p in pids.read_text().split())
    assert not hproc.group_alive(leader)
    assert gone(grandchild)
    assert signal.getsignal(signal.SIGTERM) is not hproc.raise_terminated  # restored
