"""The real executor and its spools, with a stand-in for the Claude Code binary: a shell script
that prints what the real one prints in stream-json and never calls a model. No model runs."""

from __future__ import annotations

import copy
import json
import os
import re
import shlex
import signal
import stat
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from ingest import holdout_broker as hb
from ingest import holdout_executor as hx
from ingest import holdout_isolation as hi
from ingest import holdout_pipeline as hpl
from ingest import holdout_runtime as hrt
from ingest import holdout_spool as hsp
from tests.holdout_fakes import FAKE_MLX
from tests.unit.test_holdout_broker import FINAL, FakeTools, req

SCRIPT = """#!/bin/sh
if [ "$1" = "--version" ]; then echo "2.1.301 (Claude Code)"; exit 0; fi
cat > /dev/null
echo '{"type": "system", "subtype": "init", "session_id": "s-1", "tools": []}'
echo "a warning" >&2
echo '{"type": "result", "result": "%s", "is_error": %s}'
exit %d
"""


def fake_binary(folder: Path, answer: str = '{\\\\"ok\\\\": 1}', error: bool = False) -> Path:
    path = folder / "claude"
    path.write_text(SCRIPT % (answer, "true" if error else "false", 1 if error else 0))
    path.chmod(0o755)
    return path


def executor(tmp_path: Path, binary: Path, live: dict | None = None) -> hx.RealExecutor:
    pin = hi.binary_identity(binary)
    found = FAKE_MLX if live is None else live
    return hx.RealExecutor(
        binary, "m", pin, [], tmp_path / "raw" / "spool", FAKE_MLX, lambda: found
    )


def test_a_headless_run_writes_its_output_into_a_private_spool(tmp_path):
    run = executor(tmp_path, fake_binary(tmp_path))
    result = run.headless("judge_b", "R-0", "PACKET")
    assert json.loads(result.reply) == {"ok": 1}
    spool = tmp_path / "raw" / "spool" / "R-0"
    context = result.session["_gate1_context"]
    spool_meta = json.loads((spool / "meta.json").read_text())
    assert context["launcher_workdir"] == spool_meta["workdir"]
    assert context["clock_start"] == spool_meta["clock_start"]
    assert context["clock_end"] == json.loads((spool / "exit.json").read_text())["clock_end"]
    assert stat.S_IMODE(spool.stat().st_mode) == 0o700
    assert '"subtype": "init"' in (spool / "stdout").read_text()
    assert (spool / "stderr").read_text() == "a warning\n"
    exit_record = json.loads((spool / "exit.json").read_text())
    assert exit_record["returncode"] == 0
    assert set(exit_record) == {"returncode", "clock_end"}
    assert hi._clock_observation_date(exit_record["clock_end"])[0] is not None
    with pytest.raises(hpl.TechnicalFailure, match="belongs to one run"):  # never reused
        run.headless("judge_b", "R-0", "PACKET")


def test_a_spooled_answer_is_recovered_whole_or_not_at_all(tmp_path):
    run = executor(tmp_path, fake_binary(tmp_path))
    run.headless("judge_b", "R-1", "PACKET")
    again = run.recover_headless("judge_b", "R-1", "PACKET")
    assert again is not None and json.loads(again.reply) == {"ok": 1}
    assert (
        again.session["_gate1_context"]["launcher_workdir"]
        == json.loads((tmp_path / "raw" / "spool" / "R-1" / "meta.json").read_text())["workdir"]
    )
    assert run.recover_headless("judge_b", "R-none", "PACKET") is None
    spool = tmp_path / "raw" / "spool" / "R-1"
    (spool / "exit.json").unlink()  # the orchestrator died before it saw the process end
    found = hsp.recover_headless(spool)
    assert found is not None and found.returncode == hsp.UNOBSERVED
    assert run.recover_headless("judge_b", "R-1", "PACKET") is not None
    text = (spool / "stdout").read_text()
    (spool / "stdout").write_text(text[: text.rindex('"is_error"')])  # a line cut by the crash
    assert run.recover_headless("judge_b", "R-1", "PACKET") is None


def test_a_failed_run_is_technical_and_recovers_as_nothing(tmp_path):
    run = executor(tmp_path, fake_binary(tmp_path, answer="overloaded", error=True))
    with pytest.raises(hpl.TechnicalFailure) as failure:
        run.headless("concept_author", "R-2", "PACKET")
    assert failure.value.reason == "model_api"
    assert run.recover_headless("concept_author", "R-2", "PACKET") is None


def _stream(spool: Path, replies: Sequence[object]) -> None:
    hsp.begin(spool, spool.parent / "gone", "broker")
    lines = [json.dumps({"type": "system", "subtype": "init", "session_id": "s-2"})]
    for reply in replies:
        text = reply if isinstance(reply, str) else json.dumps(reply)
        lines.append(json.dumps({"type": "result", "result": text, "is_error": False}))
    (spool / "stream").write_text("\n".join(lines) + "\n")


def test_a_judge_a_run_is_rebuilt_from_its_spooled_replies(tmp_path):
    replies = [req("titles", act="DORA"), req("article", act="DORA", article="16"), FINAL]
    live = hb.run_judge_a(_Scripted(replies), "PACKET", FakeTools())
    spool = tmp_path / "spool" / "R-3"
    _stream(spool, replies)
    found = hb.recover_conversation(spool, "PACKET", FakeTools())
    assert found is not None
    run, used, session = found
    assert (
        session["_gate1_context"]["clock_start"]
        == json.loads((spool / "meta.json").read_text())["clock_start"]
    )
    assert session["_gate1_context"]["clock_end"] is None
    assert (run.outcome, run.final, run.sent, run.log) == (
        live.outcome,
        live.final,
        live.sent,
        live.log,
    )
    assert len(used) == 3 and session["returncode"] == hsp.UNOBSERVED
    cut = tmp_path / "spool" / "R-4"
    _stream(cut, replies[:2])  # the final answer never arrived
    assert hb.recover_conversation(cut, "PACKET", FakeTools()) is None


def test_real_broker_execution_and_recovery_preserve_auditable_launcher_context(
    tmp_path, monkeypatch
):
    """Exercise the actual process, spool, executor projection and recovery; no model.
    The transcript imitates the runtime's cwd envelopes, including the old nDCG collision."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    cwd = tmp_path / "tmp.abnDCGxy"
    cwd.mkdir()
    monkeypatch.setattr(hb.tempfile, "mkdtemp", lambda **kwargs: str(cwd))
    replies = [req("titles", act="DORA"), req("article", act="DORA", article="16"), FINAL]
    program = tmp_path / "stand-in.py"
    program.write_text(
        "import json,os,re,sys\nfrom pathlib import Path\nfrom datetime import datetime\n"
        "if '--version' in sys.argv:\n print('2.1.301 (Claude Code)'); sys.exit(0)\n"
        "cwd=os.getcwd()\n"
        "folder=Path.home()/'.claude'/'projects'/re.sub(r'[^A-Za-z0-9]', '-', cwd)\n"
        "folder.mkdir(parents=True)\n"
        "transcript=(folder/'s-broker.jsonl').open('w')\n"
        "def keep(r):\n transcript.write(json.dumps(r)+'\\n'); transcript.flush()\n"
        f"plugins={list(hi.BUILTIN_PLUGINS)!r}\n"
        "init={'type':'system','subtype':'init','tools':[],'cwd':cwd,'original_cwd':cwd,"
        "'plugins':plugins,'session_id':'s-broker','claude_code_version':'2.1.301'}\n"
        "print(json.dumps(init),flush=True)\n"
        "keep({'type':'attachment','cwd':cwd,'attachment':"
        "{'type':'date','date':datetime.now().date().isoformat()}})\n"
        f"replies={replies!r}\n"
        "for line,reply in zip(sys.stdin,replies):\n"
        " r=json.loads(line); r['cwd']=cwd; keep(r)\n"
        " keep({'type':'assistant','cwd':cwd,'message':{'content':json.dumps(reply)}})\n"
        " print(json.dumps({'type':'result','result':json.dumps(reply),'is_error':False}),"
        "flush=True)\n"
    )
    binary = tmp_path / "claude"
    binary.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(program))} "$@"\n'
    )
    binary.chmod(0o755)
    real = executor(tmp_path, binary)
    live = real.broker("search_1", "R-live", "PACKET", FakeTools())
    monkeypatch.setattr(hi, "runtime_clock_observation", _never_observed_again)
    recovered = real.recover_broker("search_1", "R-live", "PACKET", FakeTools())
    assert recovered is not None
    assert recovered.session == live.session
    assert live.session["_gate1_context"]["launcher_workdir"] == str(cwd.resolve())
    assert not cwd.exists()
    for session in (live, recovered):
        report = hi.audit_run(
            session.session,
            [hi._literal("metric", "nDCG")],
            expected_cwd=str(cwd.resolve()),
            runtime_context=session.session["_gate1_context"],
        )
        assert report.ok and report.cwd_validated and report.date_validated, report.problems
    # the document the pipeline records, read back as the run its audit reads
    document = {
        "kind": "broker",
        "run": hpl._broker_record(live.run),
        "replies": list(live.replies),
        "session": dict(live.session),
    }
    run = hpl.role_run("search_1", "R-live", "PACKET", json.loads(json.dumps(document)))
    assert run.session["_gate1_context"] == live.session["_gate1_context"]


def _never_observed_again() -> dict[str, str]:
    raise AssertionError("a recovery observed the clock again instead of reading the spool")


HEADLESS_STAND_IN = """import json, os, re, sys
from datetime import datetime
from pathlib import Path
if "--version" in sys.argv:
    print("2.1.301 (Claude Code)"); sys.exit(0)
packet = sys.stdin.read()
cwd = os.getcwd()
folder = Path.home() / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", cwd)
folder.mkdir(parents=True)
day = datetime.now().date().isoformat()
environment = "# Environment\\n - Primary working directory: " + cwd
def reminder(text):
    return [{"content": "<system-reminder>\\n" + text + "\\n</system-reminder>"}]
records = [
    {"type": "attachment", "cwd": cwd, "rendered": reminder(environment),
     "attachment": {"type": "environment", "snapshot": {**SNAPSHOT, "workingDirectory": cwd}}},
    {"type": "attachment", "cwd": cwd, "rendered": reminder("Today's date is " + day + "."),
     "attachment": {"type": "date", "date": day}},
    {"type": "user", "cwd": cwd, "message": {"role": "user", "content": packet}},
    {"type": "assistant", "cwd": cwd, "message": {"content": [{"type": "text", "text": "{}"}]}},
]
(folder / "s-headless.jsonl").write_text("\\n".join(map(json.dumps, records)) + "\\n")
init = {"type": "system", "subtype": "init", "session_id": "s-headless", "tools": [],
        "cwd": cwd, "original_cwd": cwd, "plugins": PLUGINS, "claude_code_version": "2.1.301"}
print(json.dumps(init))
print(json.dumps({"type": "result", "result": "{}", "is_error": False}))
"""


def test_real_headless_execution_and_recovery_keep_the_original_launcher_context(
    tmp_path, monkeypatch
):
    """The actual process, spool, executor and recovery, no model: the launcher's directory and
    clock reach the session, a recovery reads them back from the spool and never observes
    again, and the run the audit reads carries them. The directory's suffix holds a term."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    cwd = tmp_path / "tmp.k_ndcg4x"
    cwd.mkdir()
    monkeypatch.setattr(hsp.tempfile, "mkdtemp", lambda **kwargs: str(cwd))
    program = tmp_path / "stand-in.py"
    program.write_text(
        f"SNAPSHOT = {dict(hi._SNAPSHOT)!r}\nPLUGINS = {list(hi.BUILTIN_PLUGINS)!r}\n"
        + HEADLESS_STAND_IN
    )
    binary = tmp_path / "claude"
    binary.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(program))} "$@"\n'
    )
    binary.chmod(0o755)
    real = executor(tmp_path, binary)
    live = real.headless("judge_b", "R-h", "PACKET")
    spool = tmp_path / "raw" / "spool" / "R-h"
    context = live.session["_gate1_context"]
    assert context == {
        "launcher_workdir": str(cwd.resolve()),
        "clock_start": json.loads((spool / "meta.json").read_text())["clock_start"],
        "clock_end": json.loads((spool / "exit.json").read_text())["clock_end"],
    }
    monkeypatch.setattr(hi, "runtime_clock_observation", _never_observed_again)
    recovered = real.recover_headless("judge_b", "R-h", "PACKET")
    assert recovered is not None and recovered.session == live.session
    (spool / "exit.json").unlink()  # the orchestrator died before it saw the process end
    unobserved = real.recover_headless("judge_b", "R-h", "PACKET")
    assert unobserved is not None
    assert unobserved.session["_gate1_context"] == {**context, "clock_end": None}
    for session in (live.session, recovered.session, unobserved.session):
        report = hi.audit_run(
            session,
            [hi._literal("metric", "nDCG")],
            expected_cwd=str(cwd.resolve()),
            runtime_context=session["_gate1_context"],
        )
        assert report.ok and report.cwd_validated and report.date_validated, report.problems
    # the document the pipeline records, read back as the run its audit reads
    document = {"kind": "headless", "reply": live.reply, "session": dict(live.session)}
    run = hpl.role_run("judge_b", "R-h", "PACKET", json.loads(json.dumps(document)))
    assert run.session["_gate1_context"] == context


def _changed(path: tuple[str, ...], value: object) -> dict:
    other = copy.deepcopy(FAKE_MLX)
    place = other
    for key in path[:-1]:
        place = place[key]
    place[path[-1]] = value
    return other


MLX_STAND_IN = """import argparse, json, sys, time
parser = argparse.ArgumentParser()
for flag in ("--config", "--prompts", "--out", "--max-new-tokens"):
    parser.add_argument(flag)
args = parser.parse_args()
with open(%(log)r, "a") as log:
    log.write(%(name)r + "\\n")
time.sleep(%(sleep)r)
row = json.loads(open(args.prompts).read().splitlines()[0])
line = %(line)s
with open(args.out, "a") as out:
    out.write(json.dumps(line) + "\\n")
"""


def _generator(
    monkeypatch, tmp_path: Path, versions: dict, tokens: int = 1000, sleep: float = 0
) -> Path:
    """scripts/mlx_prompt_tokens.py and scripts/mlx_generate.py, stood in for by two real
    processes: the counter appends the prompt's tokens, the generator its output, each with
    its versions. Returns the log of the processes that ran, one name per line."""
    log = tmp_path / "mlx-calls.log"
    log.touch()
    counter_line = (
        f'{{"id": row["id"], "prompt_sha256": "p", "prompt_tokens": {tokens}, '
        f'"max_prompt_tokens": 16384, "versions": {versions!r}}}'
    )
    generator_line = (
        f'{{"id": row["id"], "prompt_sha256": "p", "output": "{{}}", "versions": {versions!r}}}'
    )
    for name, line, wait in (
        ("counter", counter_line, 0),
        ("generator", generator_line, sleep),
    ):
        script = tmp_path / f"stand-in-{name}.py"
        script.write_text(
            MLX_STAND_IN % {"log": str(log), "name": name, "sleep": wait, "line": line}
        )
        monkeypatch.setattr(hx, "TOKEN_COUNTER" if name == "counter" else "GENERATOR", str(script))
    monkeypatch.setattr(hx, "MLX_PYTHON", Path(sys.executable))
    return log


def _ran(log: Path) -> list[str]:
    return log.read_text().split()


@pytest.mark.parametrize(
    "path",
    [
        ("python", "build"),
        ("packages", "mlx"),
        ("packages", "mlx-lm"),
        ("packages", "tokenizers"),
        ("host", "macos_product_version"),
        ("host", "macos_build_version"),
        ("host", "hardware_model"),
    ],
)
def test_the_diagnostic_never_runs_under_another_mlx_runtime(tmp_path, monkeypatch, path):
    run = executor(tmp_path, fake_binary(tmp_path), live=_changed(path, "9.9.9"))
    log = _generator(monkeypatch, tmp_path, hrt.generator_versions(FAKE_MLX))
    with pytest.raises(hpl.TechnicalFailure, match="another") as failure:
        run.local("qwen_diagnostic", "R-6", [{"role": "user", "content": "PACKET"}])
    assert failure.value.reason == "runtime_identity" and _ran(log) == []
    assert not (tmp_path / "raw" / "spool" / "R-6").exists()


def test_the_diagnostic_keeps_the_runtime_that_produced_it(tmp_path, monkeypatch):
    run = executor(tmp_path, fake_binary(tmp_path))
    log = _generator(monkeypatch, tmp_path, hrt.generator_versions(FAKE_MLX))
    messages = [{"role": "user", "content": "PACKET"}]
    record = run.local("qwen_diagnostic", "R-5", messages)
    assert _ran(log) == ["counter", "generator"] and record["runtime"] == FAKE_MLX
    assert record["preflight"] == {"prompt_tokens": 1000, "max_prompt_tokens": 16384, "fits": True}
    assert record["versions"] == hrt.generator_versions(FAKE_MLX)
    spool = tmp_path / "raw" / "spool" / "R-5"
    assert json.loads((spool / "runtime.json").read_text()) == FAKE_MLX  # before the run
    again = run.recover_local("qwen_diagnostic", "R-5", messages)
    assert again == record and hrt.local_problems(again, FAKE_MLX) == []
    # an output whose spool says another runtime produced it keeps that runtime
    hsp.put(spool, "runtime.json", _changed(("packages", "mlx"), "9.9.9"))
    produced = run.recover_local("qwen_diagnostic", "R-5", messages)
    assert produced is not None and produced["runtime"]["packages"]["mlx"] == "9.9.9"
    assert hrt.local_problems(produced, FAKE_MLX) == ["produced under another packages.mlx"]
    text = (spool / "outputs.jsonl").read_text()
    (spool / "outputs.jsonl").write_text(text[:20])
    assert run.recover_local("qwen_diagnostic", "R-5", messages) is None
    assert run.recover_local("qwen_diagnostic", "R-none", messages) is None


class _Scripted:
    def __init__(self, replies: Sequence[object]) -> None:
        self.replies = [r if isinstance(r, str) else json.dumps(r) for r in replies]

    def send(self, text: str) -> str:
        return self.replies.pop(0)


def test_only_the_launcher_builds_the_real_executor_or_an_orchestrator():
    """No other module, script or command can assemble a construction with real models, and
    so none can skip the launcher's checks."""
    builders: dict[str, set[str]] = {"RealExecutor(": set(), "Pipeline(": set()}
    for path in [*Path("ingest").glob("*.py"), *Path("scripts").glob("*.py")]:
        text = path.read_text(encoding="utf-8")
        for name in builders:
            if re.search(rf"(?<![A-Za-z_]){re.escape(name)}", text.replace("class " + name, "")):
                builders[name].add(path.as_posix())
    assert builders == {
        "RealExecutor(": {"ingest/holdout_launcher.py"},
        "Pipeline(": {"ingest/holdout_launcher.py"},
    }


# ---------------------------------------------------------------------------
# P1 #6: the diagnostic's preflight; P1 #7: every execution bounded, its group verified gone
# ---------------------------------------------------------------------------

MESSAGES = [{"role": "user", "content": "PACKET"}]
# a stand-in binary that never answers: it starts a child of its own group and waits
HANG = """#!/bin/sh
if [ "$1" = "--version" ]; then echo "2.1.301 (Claude Code)"; exit 0; fi
sleep 300 &
echo $! > "%s"
cat > /dev/null
sleep 300
"""


def hanging_binary(folder: Path) -> tuple[Path, Path]:
    path, child = folder / "claude", folder / "child.pid"
    path.write_text(HANG % child)
    path.chmod(0o755)
    return path, child


def gone(pid: int, within: float = 5.0) -> bool:
    """Ended and reaped: a killed process the system has not reaped yet (a zombie, which runs
    nothing) is given a moment to disappear."""
    import time as _time

    deadline = _time.monotonic() + within
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        if _time.monotonic() > deadline:
            return False
        _time.sleep(0.05)


def test_a_prompt_over_the_frozen_limit_is_counted_and_never_generated(tmp_path, monkeypatch):
    run = executor(tmp_path, fake_binary(tmp_path))
    log = _generator(monkeypatch, tmp_path, hrt.generator_versions(FAKE_MLX), tokens=20000)
    record = run.local("qwen_diagnostic", "R-7", MESSAGES)
    assert _ran(log) == ["counter"]  # one deterministic count, no generation, no retry
    assert record["output"] is None
    assert record["preflight"] == {
        "prompt_tokens": 20000,
        "max_prompt_tokens": 16384,
        "fits": False,
    }
    assert hrt.local_problems(record, FAKE_MLX) == []  # the counter writes the frozen versions
    assert run.recover_local("qwen_diagnostic", "R-7", MESSAGES) == record


def test_a_generation_past_its_deadline_is_ended_and_verified_gone(tmp_path, monkeypatch):
    run = executor(tmp_path, fake_binary(tmp_path))
    _generator(monkeypatch, tmp_path, hrt.generator_versions(FAKE_MLX), sleep=60)
    monkeypatch.setattr(hx.hproc, "LOCAL_SECONDS", 1)
    monkeypatch.setattr(hx.hproc, "GRACE_SECONDS", 1)
    with pytest.raises(hpl.TechnicalFailure) as failure:
        run.local("qwen_diagnostic", "R-8", MESSAGES)
    assert (failure.value.reason, failure.value.check) == ("timeout", "execution_deadline")
    registry = json.loads((tmp_path / "raw" / "spool" / "R-8" / "process.json").read_text())
    assert not hx.hproc.group_alive(registry["pgid"])


def test_a_headless_run_past_its_deadline_leaves_no_process_of_its_group(tmp_path, monkeypatch):
    binary, child = hanging_binary(tmp_path)
    run = executor(tmp_path, binary)
    monkeypatch.setattr(hx.hproc, "HEADLESS_SECONDS", 2)
    monkeypatch.setattr(hx.hproc, "GRACE_SECONDS", 1)
    with pytest.raises(hpl.TechnicalFailure) as failure:
        run.headless("judge_b", "R-9", "PACKET")
    assert (failure.value.reason, failure.value.check) == ("timeout", "execution_deadline")
    registry = json.loads((tmp_path / "raw" / "spool" / "R-9" / "process.json").read_text())
    assert not hx.hproc.group_alive(registry["pgid"])
    assert gone(int(child.read_text()))  # the child it started, ended with its group


def test_a_broker_turn_past_its_deadline_ends_the_judge_and_its_children(tmp_path, monkeypatch):
    binary, child = hanging_binary(tmp_path)
    run = executor(tmp_path, binary)
    monkeypatch.setattr(hx.hproc, "BROKER_TURN_SECONDS", 2)
    monkeypatch.setattr(hx.hproc, "GRACE_SECONDS", 1)
    with pytest.raises(hpl.TechnicalFailure) as failure:
        run.broker("judge_a", "R-10", "PACKET", FakeTools())
    assert (failure.value.reason, failure.value.check) == ("timeout", "execution_deadline")
    registry = json.loads((tmp_path / "raw" / "spool" / "R-10" / "process.json").read_text())
    assert not hx.hproc.group_alive(registry["pgid"])
    assert gone(int(child.read_text()))


def test_a_judge_that_will_not_exit_is_ended_at_close(tmp_path, monkeypatch):
    """A broker run whose judge answered and then does not exit when its input ends."""
    script = tmp_path / "claude"
    script.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo "2.1.301 (Claude Code)"; exit 0; fi\n'
        "trap '' TERM\n"  # it ignores SIGTERM: only SIGKILL ends it
        "read line\n"
        'echo \'{"type": "result", "result": "{\\"final\\": {}}", "is_error": false}\'\n'
        "while true; do sleep 1; done\n"
    )
    script.chmod(0o755)
    monkeypatch.setattr(hb, "CLOSE_SECONDS", 1)
    monkeypatch.setattr(hx.hproc, "GRACE_SECONDS", 1)
    pin = hi.binary_identity(script)
    conversation = hb.StreamJsonConversation(script, "m", pin, spool=tmp_path / "spool")
    assert json.loads(conversation.send("PACKET")) == {"final": {}}
    session = conversation.close()
    assert session["returncode"] == -signal.SIGKILL
    assert not hx.hproc.group_alive(conversation.process.pid)


def test_a_changed_binary_is_a_runtime_identity_mismatch_never_a_transport(tmp_path):
    binary = fake_binary(tmp_path)
    pin = {**hi.binary_identity(binary), "sha256": "0" * 64}
    run = hx.RealExecutor(
        binary, "m", pin, [], tmp_path / "raw" / "spool", FAKE_MLX, lambda: FAKE_MLX
    )
    for call in (
        lambda: run.headless("judge_b", "R-11", "PACKET"),
        lambda: run.broker("judge_a", "R-12", "PACKET", FakeTools()),
    ):
        with pytest.raises(hpl.TechnicalFailure) as failure:
            call()
        assert failure.value.reason == "runtime_identity"
    assert not (tmp_path / "raw" / "spool" / "R-11").exists()


@pytest.mark.parametrize(
    ("raised", "reason"),
    [
        (lambda: hb.hproc.OperationTimeout("x"), "timeout"),
        (lambda: hb.BrokerFailure("x"), "broker"),
    ],
)
def test_a_broker_failure_is_classified_and_its_judge_ended(tmp_path, monkeypatch, raised, reason):
    binary, child = hanging_binary(tmp_path)
    run = executor(tmp_path, binary)
    monkeypatch.setattr(hx.hproc, "GRACE_SECONDS", 1)

    def failing(*args, **kwargs):
        child_pid = None
        for _ in range(100):
            if child.exists() and child.read_text().strip():
                child_pid = int(child.read_text())
                break
            __import__("time").sleep(0.05)
        assert child_pid is not None
        raise raised()

    monkeypatch.setattr(hx.hb, "run_judge_a", failing)
    with pytest.raises(hpl.TechnicalFailure) as failure:
        run.broker("judge_a", "R-13", "PACKET", FakeTools())
    assert failure.value.reason == reason
    assert gone(int(child.read_text()))


def test_a_recovered_judge_that_exited_with_an_error_gives_no_answer(tmp_path):
    run = executor(tmp_path, fake_binary(tmp_path))
    spool = tmp_path / "raw" / "spool" / "R-14"
    _stream(spool, [FINAL])
    hsp.end(spool, 1)
    with pytest.raises(hpl.TechnicalFailure) as failure:
        run.recover_broker("judge_a", "R-14", "PACKET", FakeTools())
    assert failure.value.reason == "transport"
    unobserved = tmp_path / "raw" / "spool" / "R-15"
    _stream(unobserved, [FINAL])  # the orchestrator died: no exit was observed
    assert run.recover_broker("judge_a", "R-15", "PACKET", FakeTools()) is not None
