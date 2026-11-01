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
from collections import Counter
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
from ingest import holdout_v2_operations as ops
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


# Provider error vocabulary in the pinned 2.1.301 binary, including its rendered
# subscription windows. Only the final error result and final stderr line are read.
CAPACITY_MARKERS = re.compile(
    r"usage (?:credit )?limit|session limit|weekly limit|(?:Opus|Sonnet|Fable) limit"
    r"|spend limit|you've hit your (?:limit\b|team's shared budget)"
    r"|you're out of (?:usage credits|extra usage)|your org is out of usage\b"
    r"|rate_limit_error|rate limit|overloaded|(?:Opus|Fable) is experiencing high load"
    r"|server is temporarily limiting requests|request rejected \(429\)|API is at capacity",
    re.I,
)


def capacity_refusal(stream: str, stderr: str) -> bool:
    """Whether a failed call was refused for the provider's capacity: read from the final error
    result the CLI wrote (stream-json) and from the last line of its stderr (an earlier notice,
    such as the CLI's own retry message, decides nothing), never from the packet or a reply."""
    last = [line for line in stderr.splitlines() if line.strip()]
    told = last[-1] if last else ""
    result = None
    for line in stream.splitlines():
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            result = event
    if result is not None and result.get("is_error") is True:
        # Structured status is evidence only in an error result, never a reply,
        # request, earlier stream event or an arbitrary occurrence of the digits.
        status = result.get("api_error_status")
        if type(status) in (int, str) and status in (429, 529, "429", "529"):
            return True
        fields = ("subtype", "result", "error")
        told += " " + json.dumps({k: result.get(k) for k in fields}, default=str)
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


class OperationalPause(Exception):
    """A cooperative operator pause; no job is rejected and no attempt is charged."""


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
        operations: ops.Operations | None = None,
        checkpoint: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.store = store
        self.executor = executor
        self.audit = audit
        self.workers = max(1, workers)
        self.backoff_seconds = backoff_seconds
        self.sleep = sleep
        events = store.events()
        self.ledger = Ledger.fold(events)
        self.operations = operations
        self.checkpoint = checkpoint
        self.phase = "initializing"
        self.stage = "none"
        self.wave: int | None = None
        self._stage_jobs: tuple[str, ...] = ()
        self._in_flight: set[str] = set()
        self._mode = "initializing"
        self.target_needs: int | None = None
        self._stage_started = operations.clock() if operations is not None else 0.0
        self._stage_initial_resolved = 0
        self._last_publish = float("-inf")
        self._last_checkpoint: Mapping[str, Any] | None = None
        self._last_admission: str | None = None
        self._last_progress: str | None = None
        self._wait_collect: Callable[[], None] | None = None
        if operations is not None:
            operations.seed([e.at for e in events if e.kind == "run_started"])
            admissions = [e.at for e in events if e.kind == "job_admitted"]
            if admissions:
                self._last_admission = admissions[-1]
            useful = [e.at for e in events if e.kind in ("job_admitted", "job_exhausted")]
            if useful:
                self._last_progress = useful[-1]
            self.publish(force=True)

    # -----------------------------------------------------------------------
    # recording
    # -----------------------------------------------------------------------

    def record(self, kind: str, key: str | None, payload: Mapping[str, Any]) -> vs.Event:
        event = self.store.record(kind, key, payload)
        self.ledger.apply(event)
        if kind == "job_admitted":
            self._last_admission = event.at
        if kind in ("job_admitted", "job_exhausted"):
            self._last_progress = event.at
        return event

    def stop(self, stop: V2Stop) -> V2Stop:
        self.record("stopped", None, {"reason": stop.reason, "detail": stop.detail[:500]})
        return stop

    def set_context(self, phase: str, wave: int | None = None) -> None:
        self.phase, self.wave = phase, wave
        self.publish(force=True)

    def publish(self, force: bool = False) -> None:
        """Only aggregates; the event log, never this sidecar, remains authoritative."""
        op = self.operations
        if op is None:
            return
        now = op.clock()
        if not force and now - self._last_publish < 30:
            return
        ids = set(self._stage_jobs)
        admitted = len(ids & self.ledger.admitted.keys())
        exhausted = len(ids & self.ledger.exhausted)
        flight = len(ids & self._in_flight)
        responded = sum(
            self.ledger.runs[runs[-1]]["status"] == "responded"
            for job, runs in self.ledger.jobs.items()
            if job in ids and runs
        )
        # A completed response can still be held by a future during a capacity wait.
        responded_waiting = sum(
            self.ledger.runs[runs[-1]]["status"] == "responded" and job not in self._in_flight
            for job, runs in self.ledger.jobs.items()
            if job in ids and runs
        )
        failed = Counter(
            str(r.get("reason", "unknown"))
            for r in self.ledger.runs.values()
            if r["status"] == "failed"
        )
        stage_terminal = admitted + exhausted
        delay = op.seconds_until_start()
        total = len(ids)
        grouped: dict[str, list[str]] = {}
        for job_id in ids:
            parts = job_id.split("/")
            if (
                len(parts) >= 4
                and parts[0] in ("judge", "completeness")
                and parts[1] in ("b1", "b2", "c")
            ):
                grouped.setdefault("/".join(parts[:-1]), []).append(job_id)
        groups_admitted = sum(
            all(j in self.ledger.admitted for j in group) for group in grouped.values()
        )
        groups_exhausted = sum(
            any(j in self.ledger.exhausted for j in group) for group in grouped.values()
        )
        elapsed = max(0.0, now - self._stage_started)
        newly_resolved = max(0, stage_terminal - self._stage_initial_resolved)
        observed_rate = newly_resolved / elapsed if elapsed >= 60 and newly_resolved >= 5 else None
        remaining = max(0, total - stage_terminal)
        stage_estimate = remaining / observed_rate if observed_rate else None
        judged = [v for k, v in self.ledger.keyed.items() if k.startswith("judged/")]
        need_stages = Counter(
            str(v.get("stage"))
            for k, v in self.ledger.keyed.items()
            if k.startswith("judge/") and "/" in k and v.get("stage") in ("b1", "b2", "c")
        )
        event_head = self.store.head()
        op.publish(
            {
                "phase": self.phase,
                "stage": self.stage,
                "wave": self.wave,
                "mode": self._mode,
                "workers": self.workers,
                "stage_counts": {
                    "planned_jobs": total,
                    "admitted_jobs": admitted,
                    "exhausted_jobs": exhausted,
                    "in_flight_jobs": flight,
                    "responses_waiting_decision": responded,
                    "pending_jobs": max(0, total - stage_terminal - flight - responded_waiting),
                    "resolved_jobs": stage_terminal,
                    "resolved_fraction": stage_terminal / total if total else None,
                },
                "stage_need_groups": {
                    "planned": len(grouped) if grouped else None,
                    "fully_admitted": groups_admitted if grouped else None,
                    "with_exhaustion": groups_exhausted if grouped else None,
                    "pending": len(grouped) - groups_admitted - groups_exhausted
                    if grouped
                    else None,
                },
                "total_counts": {
                    "execution_starts": len(self.ledger.runs),
                    "admitted_jobs": len(self.ledger.admitted),
                    "exhausted_jobs": len(self.ledger.exhausted),
                    "capacity_refusals": failed.get(CAPACITY, 0),
                    "counted_failures": sum(v for k, v in failed.items() if k != CAPACITY),
                    "failures_by_reason": dict(sorted(failed.items())),
                    "responded_not_admitted": sum(
                        r["status"] == "responded" for r in self.ledger.runs.values()
                    ),
                },
                "construction_counts": {
                    "target_needs": self.target_needs,
                    "topics_frozen": self.ledger.keyed.get("topics", {}).get("topics"),
                    "needs_judged": len(judged),
                    "evaluable_needs": sum(bool(v.get("evaluable")) for v in judged),
                    "need_stages_completed": dict(sorted(need_stages.items())),
                    "answerability_completed": sum(
                        k.startswith("answerability/") and "need_id" in v and "answerability" in v
                        for k, v in self.ledger.keyed.items()
                    ),
                    "metadata_completed": sum(
                        k.startswith("metadata/") and "need_id" in v
                        for k, v in self.ledger.keyed.items()
                    ),
                    "completeness_screens_completed": sum(
                        k.startswith("completeness/screen/") and "screened" in v
                        for k, v in self.ledger.keyed.items()
                    ),
                    "completeness_followups_completed": sum(
                        k.startswith("completeness/")
                        and k not in ("completeness/selection",)
                        and not k.startswith(
                            (
                                "completeness/screen/",
                                "completeness/b1/",
                                "completeness/b2/",
                                "completeness/c/",
                            )
                        )
                        and "followed_nodes" in v
                        for k, v in self.ledger.keyed.items()
                    ),
                },
                "last_admission_at": self._last_admission,
                "last_useful_progress_at": self._last_progress,
                "next_dispatch_in_seconds": delay,
                "event_sequence": event_head[0],
                "event_head_sha256": event_head[1],
                "last_verified_checkpoint": self._last_checkpoint,
                "eta": {
                    "stage_minimum_dispatch_seconds": max(
                        0, total - stage_terminal - flight - responded_waiting - 1
                    )
                    * op.policy.min_interval_seconds
                    + (delay if total > stage_terminal + flight + responded_waiting else 0),
                    "stage_minimum_note": (
                        "Dispatch-only lower bound; excludes calls, retries, "
                        "future window limits and provider waits"
                    ),
                    "stage_estimated_seconds": stage_estimate,
                    "stage_estimate_observations": newly_resolved,
                    "stage_estimate_elapsed_seconds": elapsed,
                    "stage_estimate_note": (
                        "Observed resolutions per wall time; assumes the current stage rate "
                        "continues, not a confidence interval"
                    ),
                    "future_stage_jobs": None,
                    "overall_seconds": None,
                    "note": (
                        "Later stages are conditional; no global percentage or completion "
                        "time is inferred"
                    ),
                },
            }
        )
        self._last_publish = now

    def _publish_during_drain(self) -> None:
        # A broken informational sidecar must never prevent draining model calls.
        try:
            self.publish(force=True)
        except (ops.OperationsError, OSError):
            pass

    def _controls(self) -> None:
        if self.operations is None:
            return
        self.operations.poll()
        if self.operations.pause_requested:
            raise OperationalPause("operator request")

    def _checkpoint(self) -> None:
        op = self.operations
        if op is None or not op.checkpoint_requested:
            return
        if self.checkpoint is None:
            raise OperationalPause("checkpoint callback unavailable")
        previous_mode = self._mode
        self._mode = "checkpoint"
        self.publish(force=True)
        try:
            result = self.checkpoint()
        except Exception as error:
            raise self.stop(
                V2Stop("state", "operational checkpoint verification failed")
            ) from error
        # Do not publish private paths or arbitrary callback fields.
        self._last_checkpoint = {
            "verified": True,
            "events": result.get("events"),
            "head_sha256": result.get("head_sha256"),
        }
        for request in op.poll():
            if request["action"] == "checkpoint":
                op.acknowledge(request["request_id"])
        self._mode = previous_mode
        self.publish(force=True)

    def _idle(self, seconds: float) -> None:
        """An operational wait outside any executor deadline; fakes can replace sleep."""
        if self.operations is None:
            self.sleep(seconds)
            return
        remaining = seconds
        while remaining > 0:
            self._controls()
            if self._wait_collect is not None:
                self._wait_collect()
            self.publish()
            step = min(1.0, remaining)
            self.sleep(step)
            remaining -= step
        self._controls()

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
        self._stage_jobs = tuple(j.job_id for j in jobs)
        self._stage_started = self.operations.clock() if self.operations is not None else 0.0
        stage_ids = set(self._stage_jobs)
        self._stage_initial_resolved = len(stage_ids & self.ledger.admitted.keys()) + len(
            stage_ids & self.ledger.exhausted
        )
        roles = {j.role for j in jobs}
        self.stage = next(iter(roles)) if len(roles) == 1 else "mixed" if roles else "empty"
        self._mode = "replaying"
        self.publish(force=True)
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
        self._mode = "stage_complete"
        self.publish(force=True)
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
            # a spool whose final result is a capacity refusal is that refusal, never counted;
            # otherwise the process stopped before a complete response
            probe = getattr(self.executor, "recover_failure", None)
            try:
                reason = probe(job.role, run_id) if probe is not None else None
            except Exception:
                reason = None
            if reason == CAPACITY:
                self._failed(run_id, CAPACITY, "a capacity refusal, recovered from the spool")
            else:
                self._failed(
                    run_id, "interrupted", "the process stopped before a complete response"
                )
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
        self._mode = "provider_wait"
        self.publish(force=True)
        self._idle(cooldown)
        return waited + cooldown

    def _execute(self, queue: list[Job]) -> dict[str, Outcome]:
        out: dict[str, Outcome] = {}
        pending = list(queue)
        running: dict[Future[hpl.Session], tuple[Job, str, int]] = {}
        staged: set[Future[hpl.Session]] = set()
        backoff = 0
        episode, waited, waits = 0, 0.0, 0
        op = self.operations

        def collect() -> None:
            # Preserve completed calls promptly during a provider or pacing wait.
            # Their scientific decisions are made in the normal order below.
            for future, (job, run_id, _) in list(running.items()):
                if future.done() and future not in staged:
                    self._settle(run_id, future)
                    self._in_flight.discard(job.job_id)
                    staged.add(future)

        self._wait_collect = collect
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                try:
                    while pending or running:
                        self._controls()
                        checkpoint_due = op is not None and op.checkpoint_requested
                        if checkpoint_due and not running:
                            self._checkpoint()
                            checkpoint_due = False
                        while pending and len(running) < self.workers and not checkpoint_due:
                            if op is not None and op.seconds_until_start() > 0:
                                break
                            self._controls()
                            if op is not None and op.checkpoint_requested:
                                break
                            job = pending.pop(0)
                            run_id = self._start(job)
                            if op is not None:
                                op.started()
                            self._in_flight.add(job.job_id)
                            running[pool.submit(self._call, job, run_id)] = (job, run_id, waits)
                            self._mode = "running"
                            self.publish(force=True)
                        if not running:
                            self._mode = "pacing_wait"
                            self.publish()
                            self._idle(
                                min(1.0, op.seconds_until_start()) if op is not None else 0.0
                            )
                            continue
                        done, _ = wait(
                            list(running),
                            return_when=FIRST_COMPLETED,
                            timeout=1.0 if op is not None else None,
                        )
                        for future in done:
                            job, run_id, started_after = running.pop(future)
                            staged.discard(future)
                            self._in_flight.discard(job.job_id)
                            value = self._finish(job, run_id, future)
                            run = self.ledger.runs[run_id]
                            self.publish(force=True)
                            if value is _RETRY and run.get("reason") == CAPACITY:
                                if started_after == waits:
                                    episode += 1
                                    waited = self._capacity_wait(episode, waited)
                                    waits += 1
                                pending.append(job)
                                continue
                            if "raw" in run:
                                episode, waited = 0, 0.0
                            if value is not _RETRY:
                                backoff = 0
                                out[job.job_id] = Outcome(job.job_id, value)
                            elif self.ledger.failures(job.job_id) >= MAX_RUNS:
                                out[job.job_id] = self._exhaust(job)
                            else:
                                if "raw" not in run:
                                    backoff += 1
                                    self._idle(self.backoff_seconds * min(2 ** (backoff - 1), 8))
                                pending.append(job)
                        self.publish()
                    self._checkpoint()
                except (V2Stop, Paused, OperationalPause, ops.OperationsError) as stopped:
                    self._mode = "draining"
                    self._publish_during_drain()
                    severe: V2Stop | None = stopped if isinstance(stopped, V2Stop) else None
                    for future in list(running):
                        job, run_id, _ = running.pop(future)
                        self._in_flight.discard(job.job_id)
                        try:
                            self._settle(run_id, future)
                        except V2Stop as error:
                            severe = error
                    if severe is not None:
                        self._mode = "integrity_stop"
                        self._publish_during_drain()
                        raise severe from stopped
                    if isinstance(stopped, (OperationalPause, ops.OperationsError)):
                        self.record(
                            "paused", None, {"reason": "operational", "pending_jobs": len(pending)}
                        )
                        self._mode = "operational_paused"
                        if op is not None and not isinstance(stopped, ops.OperationsError):
                            self._checkpoint()
                            for request in op.poll():
                                if request["action"] == "pause":
                                    op.acknowledge(request["request_id"])
                        self._publish_during_drain()
                        raise OperationalPause(
                            "operational pause; resume requires the operator"
                        ) from stopped
                    self._mode = (
                        "provider_paused" if isinstance(stopped, Paused) else "integrity_stop"
                    )
                    self._publish_during_drain()
                    raise
        finally:
            self._wait_collect = None
        return out

    def _settle(self, run_id: str, future: Future[hpl.Session]) -> None:
        """A call that ends after a stop or a pause: its response or its failure, with the
        reason _finish would give it, recorded; nothing decided."""
        try:
            session = future.result()
        except hproc.CleanupFailed as error:
            self._failed(run_id, "process_cleanup", str(error))
            raise self.stop(
                V2Stop("process_cleanup", "a draining process group could not be verified gone")
            ) from error
        except hpl.TechnicalFailure as error:
            self._failed(run_id, error.reason, error.detail)
            if error.reason == "runtime_identity":
                raise self.stop(V2Stop("runtime", "runtime changed during drain")) from error
            return
        except Exception as error:
            self._failed(run_id, "model_api", f"{type(error).__name__}: {error}")
            return
        self._responded(run_id, session, recovered=False)

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
