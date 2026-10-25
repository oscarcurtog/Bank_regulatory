"""TEST v2's execution engine (D-067): one model job at a time per worker, every state change
recorded by one writer.

A job is one scientific task: a role, a frozen packet and the admission its answer must pass. It
gets at most MAX_RUNS technical executions (the earlier construction's rule): each one recorded
as started before its process runs, its response recorded before it is read, then checked, in
this order, for the frozen model (an integrity stop otherwise), for the isolation of its context
(the role's contract, ingest/holdout_roles.py), and for its answer's admission. A valid answer is
never run again. A retry is the same packet in a new context: the same task, never a new one.

What may be retried, and what stops:
- a technical failure of the call (transport, model API, deadline, malformed output, an answer
  that breaks its contract): retried while executions remain; after the last, the job is
  exhausted and its caller moves on by the frozen priority;
- an isolation failure whose only problem is forbidden content that disappears once a reasoning
  block's signature, the ISO timestamps and the three cost fields are neutralized: a metadata
  false positive (the earlier controller's rule, holdout_controller.metadata_only_isolation),
  retried like a technical failure;
- any other isolation finding, a model other than the frozen one, a process group that cannot be
  verified gone, or a state that does not hold what it says: an integrity stop (V2Stop).

A call the provider refuses for its capacity (a usage or rate limit, an overload) is not a
failure of the job that meets it: the executor classifies it (reason CAPACITY) from the final
result the CLI writes, never from a guess, and it never counts toward MAX_RUNS. The engine
waits (CAPACITY_COOLDOWN_SECONDS, doubling to an hour, once per episode: calls already in flight
do not wait again) and runs the job again; capacity refusals with no response between them for
longer than OUTAGE_MAX_SECONDS pause the construction (Paused), with nothing decided, for a
resume. Every other failure counts, whatever its kind, so a job that fails the same way every
time is exhausted after MAX_RUNS executions. Each exhaustion is recorded as an event, and what
counts is a function of the recorded reasons alone: a resume decides exactly as the run it
continues.

Workers only run executor calls; the coordinating thread alone writes the state, so the order in
which calls finish never decides anything but the order of the log.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

from ingest import holdout_controller as hct
from ingest import holdout_isolation as hi
from ingest import holdout_judging as hj
from ingest import holdout_pipeline as hpl
from ingest import holdout_process as hproc
from ingest import holdout_roles as hr
from ingest import holdout_runtime as hrt
from ingest import holdout_v2_state as vs

MAX_RUNS = 3
CAPACITY = "provider_capacity"  # the provider refused the call for its capacity: never counted
CAPACITY_COOLDOWN_SECONDS = 300.0
OUTAGE_MAX_SECONDS = 6 * 3600.0
INTEGRITY = (
    "contract",
    "corpus",
    "retrieval_identity",
    "model_identity",
    "runtime",
    "isolation",
    "process_cleanup",
    "state",
    "topic_mutation",
    "artifact",
    "unclassified",
)


# the pinned CLI's own words for a call the provider refused for its capacity (read from the
# 2.1.301 binary: "Claude AI usage limit reached", rate_limit_error, overloaded_error)
CAPACITY_MARKERS = re.compile(r"usage limit|rate_limit_error|rate limit|overloaded", re.I)


def capacity_refusal(stream: str, stderr: str) -> bool:
    """Whether a failed call was refused for the provider's capacity: read from the final error
    result the CLI wrote (stream-json) and from its stderr, never from the packet or a reply."""
    told = stderr
    for line in stream.splitlines():
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "result" and event.get("is_error"):
            fields = ("subtype", "result", "error", "api_error_status")
            told += " " + json.dumps({k: event.get(k) for k in fields}, default=str)
    return bool(CAPACITY_MARKERS.search(told))


class V2Stop(Exception):
    """An integrity failure: the construction stops for a person, and nothing is decided."""

    def __init__(self, reason: str, detail: str) -> None:
        if reason not in INTEGRITY:
            raise ValueError(f"{reason!r} is not an integrity reason")
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


class Paused(Exception):
    """The provider has refused every call for longer than OUTAGE_MAX_SECONDS: the construction
    pauses with nothing decided, and a resume goes on from the same state."""


class Inadmissible(Exception):
    """An answer that breaks its role's contract: a failed technical execution, never a label."""

    def __init__(self, check: str, problems: Sequence[str]) -> None:
        super().__init__(f"{check}: {list(problems)[:3]}")
        self.check = check
        self.problems = list(problems)


def reply_json(text: str) -> Any:
    try:
        return hj.as_json(text)
    except (json.JSONDecodeError, IndexError):
        raise Inadmissible(
            "reply_is_json", ["a reply is not the JSON its packet asks for"]
        ) from None


@dataclass(frozen=True)
class Job:
    job_id: str
    role: str
    packet: str
    admit: Callable[[str], Any]
    inputs: Mapping[str, str] = field(default_factory=dict)


@dataclass
class Ledger:
    """What the log says about the jobs, folded in the coordinating thread."""

    runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    jobs: dict[str, list[str]] = field(default_factory=dict)
    admitted: dict[str, Any] = field(default_factory=dict)
    keyed: dict[str, dict[str, Any]] = field(default_factory=dict)
    stops: list[dict[str, Any]] = field(default_factory=list)
    exhausted: set[str] = field(default_factory=set)  # jobs recorded as exhausted

    def apply(self, event: vs.Event) -> None:
        p = event.payload
        if event.key is not None:
            self.keyed[event.key] = p
        if event.kind == "run_started":
            self.runs[p["run_id"]] = {**p, "status": "started"}
            self.jobs.setdefault(p["job_id"], []).append(p["run_id"])
        elif event.kind == "run_responded":
            self.runs[p["run_id"]].update(
                status="responded", raw=p["raw"], recovered=p["recovered"]
            )
        elif event.kind == "run_failed":
            self.runs[p["run_id"]].update(status="failed", reason=p["reason"])
        elif event.kind == "job_admitted":
            self.runs[p["run_id"]]["status"] = "admitted"
            self.admitted[p["job_id"]] = p["value"]
        elif event.kind == "stopped":
            self.stops.append(p)
        elif event.kind == "job_exhausted":
            self.exhausted.add(p["job_id"])

    @classmethod
    def fold(cls, events: Sequence[vs.Event]) -> Ledger:
        ledger = cls()
        for e in events:
            ledger.apply(e)
        return ledger

    def failures(self, job_id: str) -> int:
        """The failed executions that count toward MAX_RUNS: every one but a capacity refusal."""
        return sum(
            self.runs[r]["status"] == "failed" and self.runs[r].get("reason") != CAPACITY
            for r in self.jobs.get(job_id, [])
        )


@dataclass(frozen=True)
class AuditContext:
    terms: Sequence[hi.Term]
    frozen_templates: Mapping[str, str]
    forbidden: Sequence[str]
    pin: Mapping[str, str] | None
    tmp_root: str | None
    claude_effective: Mapping[str, Any] | None


@dataclass(frozen=True)
class Outcome:
    job_id: str
    value: Any = None
    exhausted: bool = False


class Engine:
    def __init__(
        self,
        store: vs.Store,
        executor: Any,
        audit: AuditContext,
        workers: int = 1,
        backoff_seconds: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.store = store
        self.executor = executor
        self.audit = audit
        self.workers = max(1, workers)
        self.backoff_seconds = backoff_seconds
        self.sleep = sleep
        self.ledger = Ledger.fold(store.events())

    # -----------------------------------------------------------------------
    # recording
    # -----------------------------------------------------------------------

    def record(self, kind: str, key: str | None, payload: Mapping[str, Any]) -> vs.Event:
        event = self.store.record(kind, key, payload)
        self.ledger.apply(event)
        return event

    def stop(self, stop: V2Stop) -> V2Stop:
        self.record("stopped", None, {"reason": stop.reason, "detail": stop.detail[:500]})
        return stop

    # -----------------------------------------------------------------------
    # one execution
    # -----------------------------------------------------------------------

    def _start(self, job: Job) -> str:
        n = len(self.ledger.jobs.get(job.job_id, [])) + 1
        run_id = f"{job.job_id}#{n}"
        packet_sha, _ = self.store.put(job.packet, "kept")
        self.record(
            "run_started",
            f"{run_id}/started",
            {
                "job_id": job.job_id,
                "run_id": run_id,
                "n": n,
                "role": job.role,
                "packet_sha256": packet_sha,
                "inputs": dict(job.inputs),
            },
        )
        return run_id

    def _responded(self, run_id: str, session: hpl.Session, recovered: bool) -> dict[str, Any]:
        texts: dict[str, Any] = dict(session.session)
        document: dict[str, Any] = {"kind": "headless", "reply": session.reply, "session": texts}
        try:
            raw = json.dumps(document, ensure_ascii=False)
            raw.encode("utf-8")
        except (TypeError, ValueError):
            self._failed(run_id, "malformed_session", "the response cannot be recorded as text")
            return {}
        try:
            kept = hi.redact(raw, hi.identity_values(hi.parse_records(texts)))
        except Exception as error:  # the session cannot be redacted: never kept unredacted
            self._failed(run_id, "infrastructure", f"redaction: {type(error).__name__}")
            return {}
        raw_sha, _ = self.store.put(raw, "raw")
        kept_sha, _ = self.store.put(kept, "kept")
        self.record(
            "run_responded",
            f"{run_id}/responded",
            {"run_id": run_id, "raw": raw_sha, "kept": kept_sha, "recovered": recovered},
        )
        return document

    def _failed(self, run_id: str, reason: str, detail: str, problems: Sequence[str] = ()) -> None:
        self.record(
            "run_failed",
            f"{run_id}/failed",
            {
                "run_id": run_id,
                "reason": reason,
                "detail": detail[:300],
                "problems": list(problems)[:20],
            },
        )

    def _metadata_only(self, run: hr.RoleRun, audit: hr.RoleAudit) -> bool:
        """The earlier controller's rule: exactly one forbidden-content problem, and none at all
        once only a reasoning block's signature, the ISO timestamps and the costs are neutral."""
        if len(audit.problems) != 1 or not str(audit.problems[0]).startswith(hct.FORBIDDEN):
            return False
        session, count = hct.neutralized(run.session)
        if not count:
            return False
        cleared = hr.RoleRun(
            run.role,
            run.run_id,
            run.packet,
            run.packet_sha256,
            run.sent,
            run.replies,
            session=session,
        )
        again = self._audit(cleared)
        return again.ok and not again.problems

    def _audit(self, run: hr.RoleRun) -> hr.RoleAudit:
        a = self.audit
        return hr.audit_role_run(
            run, a.terms, a.frozen_templates, a.forbidden, pin=a.pin, tmp_root=a.tmp_root
        )

    def _decide(self, job: Job, run_id: str, document: Mapping[str, Any]) -> Any:
        """The response's checks, in order. Returns the admitted value, or raises V2Stop; a
        technical failure is recorded and returns _RETRY."""
        session, reply = document["session"], document["reply"]
        if self.audit.claude_effective is not None:
            problems = hrt.claude_problems(session, self.audit.claude_effective)
            if problems:
                self._failed(run_id, "model_identity", "; ".join(problems[:3]))
                raise self.stop(V2Stop("model_identity", f"{job.role}: {problems[:2]}"))
        packet_sha = vs.sha_text(job.packet)
        run = hr.RoleRun(
            job.role, run_id, job.packet, packet_sha, [job.packet], [reply], session=dict(session)
        )
        audit = self._audit(run)
        if not audit.ok:
            if self._metadata_only(run, audit):
                self._failed(
                    run_id, "isolation_metadata_only", "metadata values only", audit.problems
                )
                return _RETRY
            self._failed(run_id, "isolation", "audit problems", audit.problems)
            raise self.stop(V2Stop("isolation", f"{job.role} {run_id}: {audit.problems[:3]}"))
        try:
            value = job.admit(reply)
        except Inadmissible as error:
            self._failed(run_id, "admission", error.check, error.problems)
            return _RETRY
        self.record(
            "job_admitted",
            f"{job.job_id}/admitted",
            {"job_id": job.job_id, "run_id": run_id, "value": value},
        )
        return value

    # -----------------------------------------------------------------------
    # many jobs
    # -----------------------------------------------------------------------

    def _open_run(self, job: Job) -> tuple[str, str] | None:
        """The last execution of a job, if it never got its decision: ('started', id) for one
        whose call never returned, ('responded', id) for one whose response was recorded."""
        runs = self.ledger.jobs.get(job.job_id, [])
        if not runs:
            return None
        last = self.ledger.runs[runs[-1]]
        if last["status"] in ("started", "responded"):
            return last["status"], runs[-1]
        return None

    def run(self, jobs: Sequence[Job]) -> dict[str, Outcome]:
        """Every job to its outcome: admitted, or exhausted after MAX_RUNS failed executions."""
        out: dict[str, Outcome] = {}
        queue: list[Job] = []
        for job in jobs:
            if job.job_id in self.ledger.admitted:
                admitted = self.ledger.keyed[f"{job.job_id}/admitted"]["run_id"]
                if self.ledger.runs[admitted]["packet_sha256"] != vs.sha_text(job.packet):
                    raise self.stop(
                        V2Stop("state", f"{job.job_id}: its packet changed since its admission")
                    )
                out[job.job_id] = Outcome(job.job_id, self.ledger.admitted[job.job_id])
                continue
            if job.job_id in self.ledger.exhausted:
                out[job.job_id] = Outcome(job.job_id, exhausted=True)
                continue
            opened = self._open_run(job)
            if opened is not None:
                status, run_id = opened
                if status == "responded":
                    doc = json.loads(self.store.text(self.ledger.runs[run_id]["raw"], "raw"))
                    value = self._decide(job, run_id, doc)
                    if value is not _RETRY:
                        out[job.job_id] = Outcome(job.job_id, value)
                        continue
                else:
                    recovered = self._recover(job, run_id)
                    if recovered is not None and recovered is not _RETRY:
                        out[job.job_id] = Outcome(job.job_id, recovered)
                        continue
            if self.ledger.failures(job.job_id) >= MAX_RUNS:
                out[job.job_id] = self._exhaust(job)
                continue
            queue.append(job)
        if queue:
            out |= self._execute(queue)
        return out

    def _exhaust(self, job: Job) -> Outcome:
        self.record(
            "job_exhausted",
            f"{job.job_id}/exhausted",
            {"job_id": job.job_id, "failures": self.ledger.failures(job.job_id)},
        )
        return Outcome(job.job_id, exhausted=True)

    def _recover(self, job: Job, run_id: str) -> Any:
        """A call that never returned before the orchestrator died: its spool, if it holds a
        complete answer, is recorded and decided now; otherwise one interrupted execution."""
        try:
            session = self.executor.recover_headless(job.role, run_id, job.packet)
        except Exception as error:  # the spool cannot be read
            self._failed(
                run_id, "infrastructure", f"the spool cannot be read: {type(error).__name__}"
            )
            return None
        if session is None:
            self._failed(run_id, "interrupted", "the process stopped before a complete response")
            return None
        document = self._responded(run_id, session, recovered=True)
        if not document:
            return None
        return self._decide(job, run_id, document)

    def _call(self, job: Job, run_id: str) -> hpl.Session:
        return self.executor.headless(job.role, run_id, job.packet)  # type: ignore[no-any-return]

    def _capacity_wait(self, episode: int, waited: float) -> float:
        """One wait of a capacity episode: CAPACITY_COOLDOWN_SECONDS doubling to an hour.
        Returns the episode's total wait; past OUTAGE_MAX_SECONDS, a pause."""
        cooldown = float(min(CAPACITY_COOLDOWN_SECONDS * float(2 ** (episode - 1)), 3600.0))
        if waited + cooldown > OUTAGE_MAX_SECONDS:
            self.record("paused", None, {"waited_seconds": waited, "episode": episode})
            raise Paused(f"the provider refused every call for its capacity for {waited:.0f} s")
        self.record("capacity_wait", None, {"cooldown_seconds": cooldown, "episode": episode})
        self.sleep(cooldown)
        return waited + cooldown

    def _execute(self, queue: list[Job]) -> dict[str, Outcome]:
        out: dict[str, Outcome] = {}
        pending = list(queue)
        running: dict[Future[hpl.Session], tuple[Job, str, int]] = {}
        backoff = 0
        episode, waited, waits = 0, 0.0, 0  # the capacity episode since the last response
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            try:
                while pending or running:
                    while pending and len(running) < self.workers:
                        job = pending.pop(0)
                        run_id = self._start(job)
                        running[pool.submit(self._call, job, run_id)] = (job, run_id, waits)
                    done, _ = wait(list(running), return_when=FIRST_COMPLETED)
                    for future in done:
                        job, run_id, started_after = running.pop(future)
                        value = self._finish(job, run_id, future)
                        run = self.ledger.runs[run_id]
                        if value is _RETRY and run.get("reason") == CAPACITY:
                            if started_after == waits:  # it started after the last wait
                                episode += 1
                                waited = self._capacity_wait(episode, waited)
                                waits += 1
                            pending.append(job)
                            continue
                        if "raw" in run:  # a response: the provider answers again
                            episode, waited = 0, 0.0
                        if value is not _RETRY:
                            backoff = 0
                            out[job.job_id] = Outcome(job.job_id, value)
                        elif self.ledger.failures(job.job_id) >= MAX_RUNS:
                            out[job.job_id] = self._exhaust(job)
                        else:
                            if "raw" not in run:  # no response: a short wait before the retry
                                backoff += 1
                                self.sleep(self.backoff_seconds * min(2 ** (backoff - 1), 8))
                            pending.append(job)
            except (V2Stop, Paused):
                # nothing more starts; what is running ends and its response is recorded, never
                # decided: a person reads the stop first, and a resume decides it
                for future in list(running):
                    job, run_id, _ = running.pop(future)
                    try:
                        session = future.result()
                    except BaseException:
                        continue
                    self._responded(run_id, session, recovered=False)
                raise
        return out

    def _finish(self, job: Job, run_id: str, future: Future[hpl.Session]) -> Any:
        try:
            session = future.result()
        except hproc.CleanupFailed as error:
            self._failed(run_id, "process_cleanup", str(error))
            raise self.stop(V2Stop("process_cleanup", f"{job.role} {run_id}")) from error
        except hpl.TechnicalFailure as error:
            self._failed(run_id, error.reason, error.detail)
            if error.reason == "runtime_identity":  # the pinned binary changed: never a retry
                raise self.stop(
                    V2Stop("runtime", f"{job.role} {run_id}: {error.detail}")
                ) from error
            return _RETRY
        except Exception as error:  # the model, the API, the process or the transport
            self._failed(run_id, "model_api", f"{type(error).__name__}: {error}")
            return _RETRY
        document = self._responded(run_id, session, recovered=False)
        if not document:
            return _RETRY
        return self._decide(job, run_id, document)


class _Retry:
    def __repr__(self) -> str:
        return "<retry>"


_RETRY = _Retry()
