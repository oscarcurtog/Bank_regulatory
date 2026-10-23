"""Each model execution's raw output on disk while it is received (S3, D-055).

The isolation policy remains Gate 1's: the same pinned binary, command, minimal environment
and fresh empty directory (ingest/holdout_isolation.py). The spool now also records the
launcher-created path and host-clock observations so Gate 1 can validate the exact directory
and a legitimate date rollover. Instead of a pipe this process holds in memory, the run writes
its standard output and error straight into the files of a private folder, its spool, which are
flushed to disk when it ends. If the orchestrator dies during or just after the call, the
response is still there: recover_headless reads it back when it is complete, and the
orchestrator records, audits and admits it like any other (ingest/holdout_pipeline.py). A
spooled response is never an admitted one.

A spool lives under the raw folder outside git, readable by this user only, and is never
reused: one run, one spool.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ingest import holdout_isolation as hi
from ingest import holdout_process as hproc

UNOBSERVED = -1  # the exit code of a process that ended after the orchestrator died


def put(spool: Path, name: str, data: Any) -> None:
    """A JSON file of the spool, written whole or not at all."""
    tmp = spool / f".{name}.tmp"
    with open(tmp, "w", encoding="utf-8") as out:
        out.write(json.dumps(data, sort_keys=True))
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, spool / name)


def begin(spool: Path, workdir: Path, kind: str) -> None:
    """A run's spool, before its process starts: a new private folder, and a note of the run's
    directory and the host's local clock written atomically."""
    clock_start = hi.runtime_clock_observation()
    if spool.exists():
        raise RuntimeError(f"{spool} exists: a spool belongs to one run")
    spool.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    spool.mkdir(mode=0o700)
    note = {
        "kind": kind,
        "workdir": str(workdir.resolve()),
        "workdir_created": str(workdir),
        "clock_start": clock_start,
    }
    put(spool, "meta.json", note)


def end(spool: Path, returncode: int | None) -> None:
    """The process ended and was observed: its exit code and local clock, written last."""
    clock_end = hi.runtime_clock_observation()
    put(spool, "exit.json", {"returncode": returncode, "clock_end": clock_end})


def note(spool: Path) -> dict[str, Any] | None:
    path = spool / "meta.json"
    found: dict[str, Any] | None = json.loads(path.read_text("utf-8")) if path.exists() else None
    return found


def exit_code(spool: Path) -> int:
    path = spool / "exit.json"
    if not path.exists():
        return UNOBSERVED
    return int(json.loads(path.read_text(encoding="utf-8"))["returncode"])


def gate1_context(spool: Path, workdir: str | Path) -> dict[str, Any]:
    """Launcher-owned path and clock evidence, carried beside the raw model artifacts."""
    started = note(spool)
    if started is None:
        raise RuntimeError("the spool has no launcher metadata")
    recorded = started.get("workdir")
    created = started.get("workdir_created")
    if (
        not isinstance(recorded, str)
        or not isinstance(created, str)
        or str(Path(recorded).resolve(strict=False)) != recorded
        or str(Path(created).resolve(strict=False)) != recorded
        or str(Path(workdir).resolve(strict=False)) != recorded
    ):
        raise RuntimeError("the spool and launcher disagree about the working directory")
    exit_path = spool / "exit.json"
    ended = json.loads(exit_path.read_text(encoding="utf-8")) if exit_path.exists() else {}
    return {
        "launcher_workdir": recorded,
        "clock_start": started.get("clock_start"),
        "clock_end": ended.get("clock_end"),
    }


def text(spool: Path, name: str) -> str:
    path = spool / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def session_transcript(init: Mapping[str, Any], workdirs: Sequence[str]) -> tuple[str | None, str]:
    """The transcript Claude Code wrote for a session, found by its id and its directory, as the
    session names it or as it was created and resolved."""
    session = init.get("session_id")
    if not session:
        return None, ""
    for cwd in dict.fromkeys([str(init.get("cwd") or workdirs[0]), *workdirs]):
        candidate = (
            Path.home() / ".claude" / "projects" / hi.encoded_project_dir(cwd) / f"{session}.jsonl"
        )
        if candidate.exists():
            return str(candidate), candidate.read_text(encoding="utf-8")
    return None, ""


def remove_empty(workdir: Path) -> None:
    for child in workdir.iterdir():
        raise RuntimeError(f"the run left {child} in its directory")
    workdir.rmdir()


def _collected(
    workdirs: Sequence[str],
    returncode: int,
    stdout: str,
    stderr: str,
    runtime_context: Mapping[str, Any],
) -> hi.HeadlessRun:
    events = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
    init: dict[str, Any] = next((e for e in events if e.get("subtype") == "init"), {})
    result: dict[str, Any] = next((e for e in reversed(events) if e.get("type") == "result"), {})
    transcript_path, transcript = session_transcript(init, workdirs)
    return hi.HeadlessRun(
        workdirs[0],
        returncode,
        stdout,
        stderr,
        init.get("session_id"),
        transcript_path,
        transcript,
        result.get("result"),
        dict(runtime_context),
    )


def run_headless(
    packet: str,
    model: str,
    binary: Path,
    pinned: Mapping[str, str],
    spool: Path,
    extra_env: Sequence[str] = (),
    timeout: float | None = None,
) -> hi.HeadlessRun:
    """ingest/holdout_isolation.py run_headless, with the output written into the spool while
    the run lasts: the same checks, command, environment and directory. The process runs in a
    group of its own under its deadline, and its group is verified gone whatever happens
    (ingest/holdout_process.py): ExecutionTimeout at the deadline, CleanupFailed if it cannot
    be ended."""
    hi.check_pinned(binary, pinned)
    workdir = Path(tempfile.mkdtemp(prefix="tmp."))
    if any(workdir.iterdir()):
        raise RuntimeError(f"{workdir} is not empty")
    found = hi.agents_md_problems(workdir)
    if found:
        workdir.rmdir()
        raise RuntimeError(f"an AGENTS.md could reach the author: {found}")
    try:
        begin(spool, workdir, "headless")
        with (
            open(spool / "stdout", "w", encoding="utf-8") as out,
            open(spool / "stderr", "w", encoding="utf-8") as err,
        ):
            try:
                returncode = hproc.run(
                    hi.headless_command(binary, model),
                    hproc.HEADLESS_SECONDS if timeout is None else timeout,
                    registry=spool / hproc.REGISTRY,
                    stdin_text=packet,
                    cwd=workdir,
                    env=hi.clean_environment(extra_env),
                    stdout=out,
                    stderr=err,
                )
            finally:
                for f in (out, err):
                    f.flush()
                    os.fsync(f.fileno())
        end(spool, returncode)
    except BaseException:
        # the run's own failure is what is recorded: a directory it left is not
        try:
            remove_empty(workdir)
        except (RuntimeError, OSError):
            pass
        raise
    remove_empty(workdir)
    workdirs = [str(workdir.resolve()), str(workdir)]
    return _collected(
        workdirs,
        returncode,
        text(spool, "stdout"),
        text(spool, "stderr"),
        gate1_context(spool, workdir),
    )


def recover_headless(spool: Path) -> hi.HeadlessRun | None:
    """The run a spool holds, if its process gave a complete answer: a final result that is not
    an error, every line whole. None otherwise: the orchestrator counts the execution
    interrupted and nobody reads what there was."""
    found = note(spool)
    stdout = text(spool, "stdout")
    if found is None or not stdout:
        return None
    events = []
    for line in stdout.splitlines():
        if line.startswith("{"):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                return None  # a line cut by the crash
    result = next((e for e in reversed(events) if e.get("type") == "result"), None)
    if result is None or result.get("is_error"):
        return None
    workdir = Path(found["workdir"])
    if workdir.exists():
        remove_empty(workdir)
    workdirs = [found["workdir"], found["workdir_created"]]
    return _collected(
        workdirs,
        exit_code(spool),
        stdout,
        text(spool, "stderr"),
        gate1_context(spool, workdir),
    )
