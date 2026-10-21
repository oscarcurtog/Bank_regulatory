"""The real executor of TEST's orchestrator (S3, D-054): each role in its isolated process.

Headless roles run as `claude -p --safe-mode` with no tool, through the binary pinned for S3
(ingest/holdout_isolation.py run_headless); judge A and the second searcher run in one
stream-json process with the evidence broker (ingest/holdout_broker.py); the diagnostic judge
runs locally with MLX (scripts/mlx_generate.py, config/retrieval/qwen3-8b-mlx.json, no
thinking).

Every execution has a spool, a private folder named by its run under the raw folder outside
git, created before its process starts. The process writes its raw output there while it runs
(the headless and MLX processes straight into files, the stream-json judge line by line as
it is read), and the files are flushed to disk when it ends. If the orchestrator dies during or
just after the call, recover_* reads back a complete response, which the orchestrator then
records, audits and admits like any other; an incomplete one is an interrupted execution. A
spooled response is never an admitted one: it is admitted only after its audit, through the
durable state (ingest/holdout_state.py).

Claude authenticates through the macOS keychain, as the person authorized on 2026-10-19 for
every isolated role of S3 that runs headless or through the broker, and for nothing else: no
credential is created, changed, deleted or read beyond that authentication. The local
diagnostic needs no authentication.

Every Claude command asks for the model the contract freezes; whether the session ran on it is
checked by the orchestrator before admission (ingest/holdout_runtime.py claude_problems). The
diagnostic runs only if the MLX runtime is, at that moment, the frozen one; the identity it
runs under is written into the spool before the run, and the output keeps it with the versions
the generator records, so that recovery returns the identity that produced it.

Only the launcher builds this executor (ingest/holdout_launcher.py), for START or RESUME. The
tests use a stand-in.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ingest import holdout_broker as hb
from ingest import holdout_isolation as hi
from ingest import holdout_pipeline as hpl
from ingest import holdout_process as hproc
from ingest import holdout_runtime as hrt
from ingest import holdout_spool as hsp

MLX_PYTHON = hrt.MLX_PYTHON
DIAGNOSTIC_CONFIG = hrt.DIAGNOSTIC_CONFIG
DIAGNOSTIC_MAX_NEW_TOKENS = hrt.DIAGNOSTIC_MAX_NEW_TOKENS
# the diagnostic's two processes: the preflight's token count, then the generation (P1 #6)
TOKEN_COUNTER = "scripts/mlx_prompt_tokens.py"
GENERATOR = "scripts/mlx_generate.py"
TOKENS_FILE = "tokens.jsonl"
OUTPUTS_FILE = "outputs.jsonl"


class _Recording:
    """A conversation that keeps the judge's replies, for the audit."""

    def __init__(self, conversation: hb.StreamJsonConversation) -> None:
        self.conversation = conversation
        self.replies: list[str] = []

    def send(self, text: str) -> str:
        reply = self.conversation.send(text)
        self.replies.append(reply)
        return reply


def _texts(session: Mapping[str, Any]) -> dict[str, str]:
    return {k: str(session[k]) for k in ("transcript", "stream", "stderr")}


class RealExecutor:
    def __init__(
        self,
        binary: Path,
        model: str,
        pin: dict[str, str],
        extra_env: Sequence[str],
        spool: Path,
        local_runtime: Mapping[str, Any],
        runtime_probe: Callable[[], Mapping[str, Any]] = hrt.mlx_runtime,
    ) -> None:
        self.binary = binary
        self.model = model  # the requested model the contract freezes
        self.pin = pin
        self.extra_env = list(extra_env)
        self.spool = spool
        self.local_runtime = dict(local_runtime)
        self.runtime_probe = runtime_probe

    def _spool(self, run_id: str) -> Path:
        return self.spool / run_id

    def _answer(self, role: str, run: hi.HeadlessRun) -> hpl.Session:
        """A completed run's answer; -1 is the exit code of a process that ended unobserved,
        after the orchestrator died, and whose spool holds a final result that is no error."""
        texts = {"transcript": run.transcript, "stream": run.stream, "stderr": run.stderr}
        if run.returncode not in (0, hsp.UNOBSERVED) or run.answer is None:
            raise hpl.TechnicalFailure("model_api", f"{role}: exit {run.returncode}")
        return hpl.Session(run.answer, texts)

    def _pinned(self, role: str) -> None:
        """The pinned binary, checked before its process starts: a change is a runtime identity
        mismatch, never a transport or infrastructure failure."""
        try:
            hi.check_pinned(self.binary, self.pin)
        except (RuntimeError, OSError) as error:
            raise hpl.TechnicalFailure("runtime_identity", f"{role}: {error}") from error

    def headless(self, role: str, run_id: str, packet: str) -> hpl.Session:
        self._pinned(role)
        try:
            run = hsp.run_headless(
                packet, self.model, self.binary, self.pin, self._spool(run_id), self.extra_env
            )
        except hproc.ExecutionTimeout as error:
            raise hpl.technical("execution_deadline", f"{role}: {error}") from error
        except RuntimeError as error:
            raise hpl.TechnicalFailure("infrastructure", f"{role}: {error}") from error
        return self._answer(role, run)

    def recover_headless(self, role: str, run_id: str, packet: str) -> hpl.Session | None:
        run = hsp.recover_headless(self._spool(run_id))
        return None if run is None else self._answer(role, run)

    def broker(self, role: str, run_id: str, packet: str, tools: hb.Tools) -> hpl.BrokerSession:
        self._pinned(role)
        conversation = hb.StreamJsonConversation(
            self.binary, self.model, self.pin, self.extra_env, spool=self._spool(run_id)
        )
        recording = _Recording(conversation)
        try:
            run = hb.run_judge_a(recording, packet, tools)
        except BaseException as error:
            # the judge's group ended and verified gone before the failure goes up
            conversation.abort()
            if isinstance(error, (hproc.ExecutionTimeout, hproc.OperationTimeout)):
                raise hpl.technical("execution_deadline", f"{role}: {error}") from error
            if isinstance(error, hb.BrokerFailure):
                raise hpl.TechnicalFailure("broker", f"{role}: {error}") from error
            raise
        try:
            session = conversation.close()
        except BaseException:
            conversation.abort()  # no member of the group survives a failed close
            raise
        if session["returncode"] != 0:
            raise hpl.TechnicalFailure("transport", f"{role}: exit {session['returncode']}")
        return hpl.BrokerSession(run, recording.replies, _texts(session))

    def recover_broker(
        self, role: str, run_id: str, packet: str, tools: hb.Tools
    ) -> hpl.BrokerSession | None:
        found = hb.recover_conversation(self._spool(run_id), packet, tools)
        if found is None:
            return None
        run, replies, session = found
        # as live: a judge that exited with an error gives no answer; one that ended unobserved,
        # after the orchestrator died, gives the answer its spool holds whole
        if session["returncode"] not in (0, hsp.UNOBSERVED):
            raise hpl.TechnicalFailure("transport", f"{role}: exit {session['returncode']}")
        return hpl.BrokerSession(run, replies, _texts(session))

    def local(self, role: str, run_id: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        try:
            runtime = dict(self.runtime_probe())
        except hrt.RuntimeIdentityError as error:
            raise hpl.TechnicalFailure("runtime_identity", f"{role}: {error}") from error
        differ = hrt.runtime_differences(runtime, self.local_runtime)
        if differ:  # not run at all under another runtime
            raise hpl.TechnicalFailure("runtime_identity", f"{role}: another {differ[:3]}")
        spool = self._spool(run_id)
        hsp.begin(spool, spool, "local")
        hsp.put(spool, "runtime.json", runtime)  # the identity it runs under, before it runs
        prompts = spool / "prompts.jsonl"
        prompts.write_text(json.dumps({"id": run_id, "messages": messages}) + "\n", "utf-8")
        # 1. the preflight (P1 #6): the prompt's tokens by the frozen tokenizer and template,
        # before any generation; a prompt over the frozen limit is never generated
        self._bounded(role, spool, TOKEN_COUNTER, TOKENS_FILE, hproc.TOKEN_COUNT_SECONDS)
        preflight = _preflight(spool, run_id)
        if preflight is None:
            raise hpl.TechnicalFailure("model_api", f"{role}: the preflight gave no count")
        if preflight["fits"]:
            # 2. the generation, bounded
            code = self._bounded(role, spool, GENERATOR, OUTPUTS_FILE, hproc.LOCAL_SECONDS)
            hsp.end(spool, code)
            if code != 0:
                tail = (spool / "stderr").read_text(encoding="utf-8")[-300:]
                raise hpl.TechnicalFailure("model_api", f"{role}: {tail}")
        else:
            hsp.end(spool, 0)
        record = self.recover_local(role, run_id, messages)
        if record is None:
            raise hpl.TechnicalFailure("model_api", f"{role}: no output for this run")
        return record

    def _bounded(self, role: str, spool: Path, script: str, out: str, timeout: float) -> int:
        """One of the diagnostic's processes, under its deadline, its group verified gone."""
        command = [
            str(MLX_PYTHON),
            script,
            "--config",
            str(DIAGNOSTIC_CONFIG),
            "--prompts",
            str(spool / "prompts.jsonl"),
            "--out",
            str(spool / out),
            "--max-new-tokens",
            str(DIAGNOSTIC_MAX_NEW_TOKENS),
        ]
        with open(spool / "stderr", "a", encoding="utf-8") as err:
            try:
                return hproc.run(
                    command,
                    timeout,
                    registry=spool / hproc.REGISTRY,
                    stdout=subprocess.DEVNULL,
                    stderr=err,
                )
            except hproc.ExecutionTimeout as error:
                raise hpl.technical("execution_deadline", f"{role}: {error}") from error
            finally:
                err.flush()
                os.fsync(err.fileno())

    def recover_local(
        self, role: str, run_id: str, messages: list[dict[str, str]]
    ) -> dict[str, Any] | None:
        """The diagnostic's record, from its spool: its preflight, and its output when the
        prompt fits. The token counter and the generator each append their line whole when
        they end, so a line that does not parse or names another run is no answer. It keeps
        the runtime the spool recorded before the run and the versions the generator (or, for
        a prompt over the limit, the counter) wrote: the orchestrator admits it only under the
        frozen ones."""
        spool = self._spool(run_id)
        preflight = _preflight(spool, run_id)
        if preflight is None:
            return None
        identity = spool / "runtime.json"
        runtime = json.loads(identity.read_text(encoding="utf-8")) if identity.exists() else None
        base = {
            "messages": messages,
            "prompt_sha256": preflight["prompt_sha256"],
            "preflight": {k: preflight[k] for k in ("prompt_tokens", "max_prompt_tokens", "fits")},
            "runtime": runtime,
        }
        if not preflight["fits"]:
            return {**base, "output": None, "versions": preflight["versions"]}
        record = _line(spool / OUTPUTS_FILE, run_id)
        if record is None:
            return None
        return {
            **base,
            "prompt_sha256": record["prompt_sha256"],
            "output": record["output"],
            "versions": record.get("versions"),
        }


def _line(path: Path, run_id: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            return None
        if record.get("id") == run_id:
            return record
    return None


def _preflight(spool: Path, run_id: str) -> dict[str, Any] | None:
    """The token counter's line for this run, with whether the prompt fits."""
    found = _line(spool / TOKENS_FILE, run_id)
    if found is None:
        return None
    tokens, limit = found.get("prompt_tokens"), found.get("max_prompt_tokens")
    if not isinstance(tokens, int) or not isinstance(limit, int):
        return None
    return {**found, "fits": tokens <= limit}
