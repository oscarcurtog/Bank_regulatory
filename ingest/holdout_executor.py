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
from ingest import holdout_runtime as hrt
from ingest import holdout_spool as hsp

MLX_PYTHON = hrt.MLX_PYTHON
DIAGNOSTIC_CONFIG = hrt.DIAGNOSTIC_CONFIG
DIAGNOSTIC_MAX_NEW_TOKENS = hrt.DIAGNOSTIC_MAX_NEW_TOKENS


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

    def headless(self, role: str, run_id: str, packet: str) -> hpl.Session:
        try:
            run = hsp.run_headless(
                packet, self.model, self.binary, self.pin, self._spool(run_id), self.extra_env
            )
        except RuntimeError as error:
            raise hpl.TechnicalFailure("infrastructure", f"{role}: {error}") from error
        return self._answer(role, run)

    def recover_headless(self, role: str, run_id: str, packet: str) -> hpl.Session | None:
        run = hsp.recover_headless(self._spool(run_id))
        return None if run is None else self._answer(role, run)

    def broker(self, role: str, run_id: str, packet: str, tools: hb.Tools) -> hpl.BrokerSession:
        conversation = hb.StreamJsonConversation(
            self.binary, self.model, self.pin, self.extra_env, spool=self._spool(run_id)
        )
        recording = _Recording(conversation)
        try:
            run = hb.run_judge_a(recording, packet, tools)
        finally:
            session = conversation.close()
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
        prompts, outputs = spool / "prompts.jsonl", spool / "outputs.jsonl"
        prompts.write_text(json.dumps({"id": run_id, "messages": messages}) + "\n", "utf-8")
        with open(spool / "stderr", "w", encoding="utf-8") as err:
            done = subprocess.run(
                [
                    str(MLX_PYTHON),
                    "scripts/mlx_generate.py",
                    "--config",
                    str(DIAGNOSTIC_CONFIG),
                    "--prompts",
                    str(prompts),
                    "--out",
                    str(outputs),
                    "--max-new-tokens",
                    str(DIAGNOSTIC_MAX_NEW_TOKENS),
                ],
                stdout=subprocess.DEVNULL,
                stderr=err,
                text=True,
                check=False,
            )
            err.flush()
            os.fsync(err.fileno())
        hsp.end(spool, done.returncode)
        record = self.recover_local(role, run_id, messages)
        if done.returncode != 0 or record is None:
            tail = (spool / "stderr").read_text(encoding="utf-8")[-300:]
            raise hpl.TechnicalFailure("model_api", f"{role}: {tail}")
        return record

    def recover_local(
        self, role: str, run_id: str, messages: list[dict[str, str]]
    ) -> dict[str, Any] | None:
        """The diagnostic's output, from its spool: the generator appends it whole when the
        generation ends, so a line that does not parse or names another run is no answer. It
        keeps the runtime the spool recorded before the run and the versions the generator
        wrote: the orchestrator admits it only under the frozen ones."""
        spool = self._spool(run_id)
        outputs = spool / "outputs.jsonl"
        if not outputs.exists():
            return None
        identity = spool / "runtime.json"
        runtime = json.loads(identity.read_text(encoding="utf-8")) if identity.exists() else None
        for line in outputs.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                return None
            if record.get("id") == run_id:
                return {
                    "messages": messages,
                    "prompt_sha256": record["prompt_sha256"],
                    "output": record["output"],
                    "versions": record.get("versions"),
                    "runtime": runtime,
                }
        return None
