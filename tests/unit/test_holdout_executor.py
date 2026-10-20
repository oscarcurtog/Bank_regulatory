"""The real executor and its spools, with a stand-in for the Claude Code binary: a shell script
that prints what the real one prints in stream-json and never calls a model. No model runs."""

from __future__ import annotations

import copy
import json
import re
import stat
import subprocess
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
    assert stat.S_IMODE(spool.stat().st_mode) == 0o700
    assert '"subtype": "init"' in (spool / "stdout").read_text()
    assert (spool / "stderr").read_text() == "a warning\n"
    assert json.loads((spool / "exit.json").read_text()) == {"returncode": 0}
    with pytest.raises(hpl.TechnicalFailure, match="belongs to one run"):  # never reused
        run.headless("judge_b", "R-0", "PACKET")


def test_a_spooled_answer_is_recovered_whole_or_not_at_all(tmp_path):
    run = executor(tmp_path, fake_binary(tmp_path))
    run.headless("judge_b", "R-1", "PACKET")
    again = run.recover_headless("judge_b", "R-1", "PACKET")
    assert again is not None and json.loads(again.reply) == {"ok": 1}
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


def _changed(path: tuple[str, ...], value: object) -> dict:
    other = copy.deepcopy(FAKE_MLX)
    place = other
    for key in path[:-1]:
        place = place[key]
    place[path[-1]] = value
    return other


def _generator(monkeypatch, versions: dict) -> list[list[str]]:
    """scripts/mlx_generate.py, stood in for: it appends its output with its versions."""
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(command)
        out = Path(command[command.index("--out") + 1])
        line = {"id": out.parent.name, "prompt_sha256": "p", "output": "{}", "versions": versions}
        out.write_text(json.dumps(line) + "\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(hx.subprocess, "run", run)
    return calls


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
    calls = _generator(monkeypatch, hrt.generator_versions(FAKE_MLX))
    with pytest.raises(hpl.TechnicalFailure, match="another") as failure:
        run.local("qwen_diagnostic", "R-6", [{"role": "user", "content": "PACKET"}])
    assert failure.value.reason == "runtime_identity" and calls == []
    assert not (tmp_path / "raw" / "spool" / "R-6").exists()


def test_the_diagnostic_keeps_the_runtime_that_produced_it(tmp_path, monkeypatch):
    run = executor(tmp_path, fake_binary(tmp_path))
    calls = _generator(monkeypatch, hrt.generator_versions(FAKE_MLX))
    messages = [{"role": "user", "content": "PACKET"}]
    record = run.local("qwen_diagnostic", "R-5", messages)
    assert len(calls) == 1 and record["runtime"] == FAKE_MLX
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
