"""The evidence broker of TEST's judge A (S3, D-054): the judge reasons, the broker reads.

Judge A runs isolated, with no tool (ingest/holdout_isolation.py). It asks for one corpus
operation at a time, as JSON; the broker validates the request against a closed schema,
runs the operation in this process through ingest/corpus_tools.py, never a shell, and
sends the result back. The operations are the four of DEV batch 2 (titles, article,
nodes, search): structural and literal reading of the parsed corpus, no retriever, no
ranking, no embedding. The judge sees its packet and the broker's messages, nothing else.

The limits are fixed before any need exists (the user's decisions of 2026-10-19, revised by
Codex Gate 2, P1.3). They are technical guards, never a judgment of the need:
- 16 evidence requests, every page included, are the normal run; past them the run goes on
  for the SAME need with resource_escalation set, up to a technical ceiling of 32. A 33rd
  request ends the run as pending_technical_resource: the need is neither rejected nor
  replaced, and the pipeline stops;
- at most 8,000 characters of corpus text per message, in deterministic pages cut at line
  ends; the broker never summarizes;
- an invalid reply gets a fixed schema error and nothing else; the third ends the run as
  judge_a_protocol_failure.
Either end is one failed technical execution of the stage (ingest/holdout_pipeline.py): the
need stays pending_technical and the process stops; resume() runs it again in a new blind
context with the same packet, up to three executions in all (the person's rule of
2026-10-19, which replaces the single second run after a protocol failure).
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TextIO

from ingest import holdout_isolation as hi
from ingest import holdout_process as hproc
from ingest import holdout_spool as hsp
from ingest.corpus_tools import CorpusTools

ESCALATION_AT = 16
MAX_REQUESTS = 32  # the technical ceiling; ESCALATION_AT is the normal run
MAX_PAYLOAD = 8000
MAX_VIOLATIONS = 2
MAX_REFS = 20
MAX_PATTERN = 200
# how long a judge may take to exit once its input has ended, before its group is ended
CLOSE_SECONDS = 120
ACTS = ("DORA", "MiCA", "PSD2")
_ARTICLE = re.compile(r"^\d{1,3}[a-z]?$")
_NODE = re.compile(r"^(DORA|MiCA|PSD2):\d{1,3}[a-z]?(\.[A-Za-z0-9_]{1,12}){0,6}$")
# a group with a quantifier inside, quantified again: the shape of catastrophic backtracking
_NESTED_QUANTIFIER = re.compile(r"\([^()]*[+*][^()]*\)\s*[+*{]")

SCHEMA: dict[str, dict[str, Any]] = {
    "titles": {"act": ACTS, "from": int, "to": int, "page": int},
    "article": {"act": ACTS, "article": _ARTICLE, "page": int},
    "nodes": {"refs": list, "page": int},
    "search": {"scope": (*ACTS, "ALL"), "pattern": str, "titles": bool, "page": int},
}
REQUIRED = {
    "titles": {"act"},
    "article": {"act", "article"},
    "nodes": {"refs"},
    "search": {"scope", "pattern"},
}

ERRORS = {
    "E1": "the reply is not exactly one JSON object",
    "E2": 'the object has neither exactly "request" nor exactly "final"',
    "E3": "the operation is not one of titles, article, nodes, search",
    "E4": "the arguments do not match the operation's schema",
    "E5": "the page does not exist",
}
Outcome = Literal["final", "pending_technical_resource", "judge_a_protocol_failure"]


class BrokerFailure(RuntimeError):
    """A corpus operation that a valid request made fail: the broker's, never the judge's."""


class RequestError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def error_message(code: str) -> str:
    """Fixed text for each kind of invalid reply: what the schema rejects, never a hint
    about the evidence."""
    return (
        f"PROTOCOL ERROR {code}: {ERRORS[code]}. Reply with exactly one JSON object, "
        '{"request": {...}} or {"final": {...}}, as your instructions define.'
    )


BUDGET_NOTICE = (
    f"That was evidence request {MAX_REQUESTS} of {MAX_REQUESTS}: none is left. "
    'Your next reply must be {"final": {...}}.'
)


@dataclass(frozen=True)
class Request:
    op: str
    args: dict[str, Any]
    page: int


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0]
    return stripped.strip()


def parse_reply(text: str) -> tuple[str, Any]:
    """("request", Request) or ("final", the judge's answer), or a RequestError."""
    try:
        data = json.loads(_strip_fence(text))
    except json.JSONDecodeError:
        raise RequestError("E1") from None
    if not isinstance(data, dict) or len(data) != 1 or set(data) not in ({"request"}, {"final"}):
        raise RequestError("E2")
    if "final" in data:
        if not isinstance(data["final"], dict):
            raise RequestError("E2")
        return "final", data["final"]
    return "request", validate(data["request"])


def _valid(value: Any, rule: Any) -> bool:
    if rule is int:
        return type(value) is int and 0 <= value <= 10_000
    if rule is bool:
        return type(value) is bool
    if rule is str:
        return isinstance(value, str)
    if rule is list:
        return isinstance(value, list)
    if isinstance(rule, tuple):
        return value in rule
    if isinstance(rule, re.Pattern):
        return isinstance(value, str) and rule.fullmatch(value) is not None
    return False


def validate(request: Any) -> Request:
    if not isinstance(request, dict) or set(request) - {"op", "args"} or "op" not in request:
        raise RequestError("E2")
    op = request["op"]
    if op not in SCHEMA:
        raise RequestError("E3")
    args = request.get("args", {})
    if not isinstance(args, dict) or set(args) - set(SCHEMA[op]) or REQUIRED[op] - set(args):
        raise RequestError("E4")
    if not all(_valid(v, SCHEMA[op][k]) for k, v in args.items()):
        raise RequestError("E4")
    if op == "titles" and (("from" in args) != ("to" in args)):
        raise RequestError("E4")
    if op == "nodes":
        refs = args["refs"]
        if not 1 <= len(refs) <= MAX_REFS or not all(
            isinstance(r, str) and _NODE.fullmatch(r) for r in refs
        ):
            raise RequestError("E4")
    if op == "search":
        pattern = args["pattern"]
        if not pattern or len(pattern) > MAX_PATTERN or _NESTED_QUANTIFIER.search(pattern):
            raise RequestError("E4")
        try:
            re.compile(pattern)
        except (re.error, OverflowError, RecursionError, ValueError):
            raise RequestError("E4") from None
    page = args.get("page", 1)
    if page < 1:
        raise RequestError("E5")
    return Request(op, {k: v for k, v in args.items() if k != "page"}, page)


class Tools(Protocol):
    def titles(self, act: str, lo: int = 0, hi: int = 10**6) -> str: ...
    def article(self, act: str, number: str) -> str | None: ...
    def nodes(self, refs: Sequence[str]) -> str: ...
    def search(self, scope: str, pattern: str, in_titles: bool) -> str: ...


def execute(tools: Tools, request: Request) -> str:
    """The operation's whole text, exactly as the DEV tool printed it."""
    a = request.args
    if request.op == "titles":
        if "from" in a:
            return tools.titles(a["act"], a["from"], a["to"])
        return tools.titles(a["act"])
    if request.op == "article":
        found = tools.article(a["act"], a["article"])
        return found if found is not None else f"no article {a['act']} {a['article']}\n"
    if request.op == "nodes":
        return tools.nodes(a["refs"])
    return tools.search(a["scope"], a["pattern"], a.get("titles", False))


def paginate(text: str, limit: int = MAX_PAYLOAD) -> list[str]:
    """Deterministic pages of at most `limit` characters, cut at line ends; a single line
    longer than the limit is cut into pieces of the limit. Nothing is dropped or rewritten:
    the pages joined give the text back."""
    pages: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:
            if current:
                pages.append(current)
                current = ""
            pages.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) > limit:
            pages.append(current)
            current = ""
        current += line
    if current or not pages:
        pages.append(current)
    return pages


def result_message(number: int, request: Request, page: str, pages: int) -> str:
    args = json.dumps(request.args, ensure_ascii=False, sort_keys=True)
    return (
        f"RESULT {number}: {request.op} {args}, page {request.page} of {pages}, "
        f"{len(page)} characters\n{page}"
    )


class Conversation(Protocol):
    def send(self, text: str) -> str: ...


@dataclass
class BrokerRun:
    outcome: Outcome
    final: dict[str, Any] | None
    log: list[dict[str, Any]] = field(default_factory=list)
    violations: list[dict[str, Any]] = field(default_factory=list)
    sent: list[str] = field(default_factory=list)  # every message the judge received
    resource_escalation: bool = False

    def consumption(self) -> dict[str, Any]:
        """The evidence the run consumed, to judge later whether the ceiling was reasonable."""
        nodes = {n for e in self.log for n in e.get("nodes", [])}
        return {
            "requests": len(self.log),
            "pages_beyond_the_first": sum(e["page"] > 1 for e in self.log),
            "characters": sum(e["chars"] for e in self.log),
            "distinct_nodes_delivered": len(nodes),
            "resource_escalation": self.resource_escalation,
        }


_NODE_IN_RESULT = re.compile(r"\[((?:DORA|MiCA|PSD2):[^\]\s]+)\]")


def run_judge_a(
    conversation: Conversation,
    packet: str,
    tools: Tools,
    max_requests: int = MAX_REQUESTS,
    max_payload: int = MAX_PAYLOAD,
    max_violations: int = MAX_VIOLATIONS,
    escalation_at: int = ESCALATION_AT,
) -> BrokerRun:
    """One run of judge A: the packet, then a result or an error for each reply, until the
    judge gives its final answer or a technical limit ends the run. A transport failure
    raises: it is technical, and the caller keeps the need pending."""
    run = BrokerRun("final", None)
    requests = 0

    def send(text: str) -> str:
        run.sent.append(text)
        return conversation.send(text)

    reply = send(packet)
    while True:
        try:
            kind, value = parse_reply(reply)
            if kind == "final":
                run.final = value
                return run
            if requests >= max_requests:
                run.outcome = "pending_technical_resource"
                return run
            try:
                # a corpus operation in this process, under its own deadline
                with hproc.alarm(hproc.OPERATION_SECONDS):
                    pages = paginate(execute(tools, value), max_payload)
            except (RequestError, hproc.OperationTimeout):
                raise
            except Exception as error:
                raise BrokerFailure(
                    f"the corpus operation failed: {type(error).__name__}"
                ) from error
            if value.page > len(pages):
                raise RequestError("E5")
        except RequestError as error:
            run.violations.append({"after_requests": requests, "code": error.code})
            if len(run.violations) > max_violations:
                run.outcome = "judge_a_protocol_failure"
                return run
            reply = send(error_message(error.code))
            continue
        requests += 1
        if requests > escalation_at:
            run.resource_escalation = True
        page = pages[value.page - 1]
        run.log.append(
            {
                "n": requests,
                "op": value.op,
                "args": value.args,
                "page": value.page,
                "pages": len(pages),
                "chars": len(page),
                "nodes": sorted(set(_NODE_IN_RESULT.findall(page))),
            }
        )
        message = result_message(requests, value, page, len(pages))
        if requests == max_requests:
            message += "\n" + BUDGET_NOTICE
        reply = send(message)


def foreign_user_messages(transcript: str, sent: Sequence[str]) -> list[str]:
    """User messages in a session that the broker did not send: there must be none."""
    out = []
    for record in hi.parse_records({"transcript": transcript}):
        if record.get("type") != "user":
            continue
        content = (record.get("message") or {}).get("content")
        texts = (
            [content]
            if isinstance(content, str)
            else [c.get("text", "") for c in content or [] if isinstance(c, dict)]
        )
        out += [t for t in texts if t and t not in sent]
    return out


# ---------------------------------------------------------------------------
# the transport: one isolated process, its conversation over stream-json
# ---------------------------------------------------------------------------


class StreamJsonConversation:
    """One isolated judge in one Claude Code process that reads user messages as stream-json
    on its standard input and answers each with a turn on its standard output. Same binary,
    environment and directory rules as the isolated authors.

    With a spool, every line of the stream is written to a private file and flushed as it is
    read, and the standard error goes straight to another: the judge's replies survive the
    death of this process, and the broker's side follows from them (recover_conversation).

    The process is the leader of a group of its own (ingest/holdout_process.py), registered in
    the spool. Each turn has a deadline, and so has the whole run: at either, the group is
    ended, reaped and verified gone, and ExecutionTimeout is raised. close() and abort() leave
    no member of the group alive, or raise CleanupFailed."""

    def __init__(
        self,
        binary: Path,
        model: str,
        pinned: Mapping[str, str],
        extra_env: Sequence[str] = (),
        timeout: float | None = None,
        spool: Path | None = None,
        deadline: float | None = None,
        command: Sequence[str] | None = None,
    ) -> None:
        if command is None:
            hi.check_pinned(binary, pinned)
        self.workdir = Path(tempfile.mkdtemp(prefix="tmp."))
        found = hi.agents_md_problems(self.workdir)
        if found:
            self.workdir.rmdir()
            raise RuntimeError(f"an AGENTS.md could reach the judge: {found}")
        self.timeout = hproc.BROKER_TURN_SECONDS if timeout is None else timeout
        self.ends_at = time.monotonic() + (
            hproc.BROKER_RUN_SECONDS if deadline is None else deadline
        )
        self.lines: list[str] = []
        self.spool = spool
        self.launcher_workdir = str(self.workdir.resolve())
        self._stream: TextIO | None = None
        stderr: Any = subprocess.PIPE
        if spool is not None:
            hsp.begin(spool, self.workdir, "broker")
            self.clock_start = (hsp.note(spool) or {}).get("clock_start")
            self._stream = open(spool / "stream", "w", encoding="utf-8")  # noqa: SIM115
            stderr = open(spool / "stderr", "w", encoding="utf-8")  # noqa: SIM115
        else:
            self.clock_start = hi.runtime_clock_observation()
        self._stderr = stderr
        # a stand-in command only in the tests of the transport's deadlines and cleanup
        self.command = list(
            command or [*hi.headless_command(binary, model), "--input-format", "stream-json"]
        )
        self.process = hproc.spawn(
            self.command,
            None if spool is None else spool / hproc.REGISTRY,
            cwd=self.workdir,
            env=hi.clean_environment(extra_env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
        )
        self._queue: queue.Queue[str | None] = queue.Queue()
        try:
            threading.Thread(target=self._read, daemon=True).start()
        except BaseException:
            hproc.end_group(self.process)
            raise

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            if self._stream is not None:
                self._stream.write(line)
                self._stream.flush()
                os.fsync(self._stream.fileno())
            self._queue.put(line)
        self._queue.put(None)

    def send(self, text: str) -> str:
        assert self.process.stdin is not None
        message = {
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": text}]},
        }
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()
        while True:
            wait = min(self.timeout, self.ends_at - time.monotonic())
            try:
                if wait <= 0:
                    raise queue.Empty
                line = self._queue.get(timeout=wait)
            except queue.Empty:
                hproc.end_group(self.process)
                raise hproc.ExecutionTimeout("a turn or the run reached its deadline") from None
            if line is None:
                raise RuntimeError("the judge's process ended in the middle of a turn")
            self.lines.append(line)
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "result":
                if event.get("is_error"):
                    raise RuntimeError(f"the turn failed: {str(event.get('result'))[:200]}")
                return str(event.get("result") or "")

    def abort(self) -> None:
        """After a failure: the group ended and verified gone (CleanupFailed otherwise), the
        files closed and the directory removed if it is empty. Nothing is collected."""
        try:
            hproc.end_group(self.process)
        finally:
            hproc.close_quietly(self.process.stdin, self._stream)
            if self._stderr is not subprocess.PIPE:
                hproc.close_quietly(self._stderr)
            try:
                hsp.remove_empty(self.workdir)
            except (RuntimeError, OSError):
                pass

    def close(self) -> dict[str, Any]:
        """End the process and collect what it left; the directory is removed. A judge that
        does not exit within the grace period after its input ends is ended; either way its
        group is verified gone before anything is read."""
        assert self.process.stdin is not None
        try:
            self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=CLOSE_SECONDS)
        except subprocess.TimeoutExpired:
            hproc.end_group(self.process)
        hproc.finish(self.process)
        try:
            while (line := self._queue.get(timeout=self.timeout)) is not None:
                self.lines.append(line)
        except queue.Empty:
            raise RuntimeError("the judge's output did not end with its process") from None
        if self.spool is not None:
            assert self._stream is not None
            for f in (self._stream, self._stderr):
                f.flush()
                os.fsync(f.fileno())
                f.close()
            hsp.end(self.spool, self.process.returncode)
            stderr = (self.spool / "stderr").read_text(encoding="utf-8")
        else:
            stderr = self.process.stderr.read() if self.process.stderr else ""
        if self.spool is not None:
            runtime_context = hsp.gate1_context(self.spool, self.workdir)
        else:
            runtime_context = {
                "launcher_workdir": self.launcher_workdir,
                "clock_start": self.clock_start,
                "clock_end": hi.runtime_clock_observation(),
            }
        found = _session(
            self.lines,
            [str(self.workdir.resolve()), str(self.workdir)],
            runtime_context,
        )
        left = list(self.workdir.iterdir())
        if left:
            raise RuntimeError(f"the judge left {left[0]} in its directory")
        self.workdir.rmdir()
        return found | {
            "returncode": self.process.returncode,
            "stderr": stderr,
            "command": self.command,
        }


def _session(
    lines: Sequence[str],
    workdirs: Sequence[str],
    runtime_context: Mapping[str, Any],
) -> dict[str, Any]:
    events = [json.loads(x) for x in lines if x.startswith("{")]
    init: dict[str, Any] = next((e for e in events if e.get("subtype") == "init"), {})
    path, transcript = hsp.session_transcript(init, workdirs)
    return {
        "stream": "".join(lines),
        "session_id": init.get("session_id"),
        "transcript_path": path,
        "transcript": transcript,
        "_gate1_context": dict(runtime_context),
    }


class _Exhausted(Exception):
    """The spooled replies end before the judge's final answer."""


class _Replay:
    """The judge's spooled replies, in order, as a conversation."""

    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = list(replies)
        self.used = 0

    def send(self, text: str) -> str:
        if self.used >= len(self.replies):
            raise _Exhausted
        self.used += 1
        return self.replies[self.used - 1]


def recover_conversation(
    spool: Path, packet: str, tools: Tools
) -> tuple[BrokerRun, list[str], dict[str, Any]] | None:
    """Judge A's run from its spool, after this process died during or just after it. The
    broker is deterministic, so the judge's replies alone give back every message the broker
    sent, by running the same broker again on them. None unless the spool holds every reply up
    to an end of the run (a final answer, the ceiling or a protocol failure)."""
    found = hsp.note(spool)
    if found is None or not (spool / "stream").exists():
        return None
    lines = hsp.text(spool, "stream").splitlines(keepends=True)
    replies = []
    for line in lines:
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return None  # a line cut by the crash
        if event.get("type") == "result":
            if event.get("is_error"):
                return None
            replies.append(str(event.get("result") or ""))
    replay = _Replay(replies)
    try:
        run = run_judge_a(replay, packet, tools)
    except _Exhausted:
        return None
    workdir = Path(found["workdir"])
    if workdir.exists():
        hsp.remove_empty(workdir)
    session = _session(
        lines,
        [found["workdir"], found["workdir_created"]],
        hsp.gate1_context(spool, workdir),
    )
    extra = {"returncode": hsp.exit_code(spool), "stderr": hsp.text(spool, "stderr")}
    return run, replies[: replay.used], session | extra


# A task on the corpus that is neither DEV nor TEST, to test the transport alone: it needs
# two requests, and the second and the answer depend on what the first returned.
SYNTHETIC_PACKET = """IMPORTANT: this message is your whole input. You have no tools: do not try
to use any.

You are testing a protocol. You can read a legal corpus (the EU acts DORA, MiCA and PSD2) only
by asking an operator, one request per reply. Each reply must be exactly one JSON object and
nothing else:
- to ask: {"request": {"op": OP, "args": {...}}}
- to finish: {"final": {...}}

Operations:
- titles: {"act": "DORA" or "MiCA" or "PSD2"}, optionally with "from" and "to" (article numbers):
  the article numbers and titles of the act.
- article: {"act": ..., "article": "16"}: every node of the article, with its path and own text.
- nodes: {"refs": ["DORA:16.1"]}: nodes in full, with their article heading.
- search: {"scope": "DORA" or "MiCA" or "PSD2" or "ALL", "pattern": "a regular expression",
  "titles": false}: the nodes whose own text matches.
Any operation may take "page": N when a result says that it has more than one page.

You have at most 16 requests. The operator answers each request with a message that starts with
RESULT, or with PROTOCOL ERROR if a reply breaks these rules.

Task: find the article of DORA whose title is about a simplified ICT risk management framework.
Then read that article and find its first node whose own text is not empty. Finish with:
{"final": {"article": "the article number", "first_node": "ACT:path of that node",
"requests_made": the number of requests you made}}
"""
SYNTHETIC_ANSWER = {"article": "16", "first_node": "DORA:16.1"}


def corpus_tools() -> CorpusTools:
    from ingest import benchmark as bm

    return CorpusTools(bm.load_corpus())
