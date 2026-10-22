"""The orchestrator of TEST's construction (S3, D-054, Codex Gate 2 P1.6), on a durable state.

Cell by cell, in the order frozen before authoring, each cell goes through these stages; a
model's output is consumed only after it passes its role's audit (ingest/holdout_roles.py):

    concept (blind concept author) -> statement (blind statement writer) -> overlap_gate (DEV,
    fixtures, ready TEST) -> statement_validation (the existing rules, by code) ->
    query_formulation (blind query writer) -> lock (statement and queries) -> judge_a (search
    A, through the evidence broker) -> second_search (search B, blind and independent, for
    every need) -> omission_search (the audit's search C, blind, in the omission audit's 12
    cells) -> judge_b -> judge_c (on every disagreement) -> adjudication -> diagnostic
    (Qwen3-8B) -> judge_d (on a material discrepancy, or on every candidate and the
    answerability when the diagnostic is unavailable) -> metadata (blind adjudicator) ->
    evidence_gate (each DEV need that shares grade-3 evidence) -> validators -> READY

The evidence of every need is discovered the same way (config/holdout/evidence-discovery.md):
search A and search B, each a fresh isolated context with the same frozen packet
(judge_a.v2) built from the locked need alone, neither seeing the other's candidates, path,
labels or output; their candidates joined into one set; one adjudication on it. In the 12
cells of the omission audit a third blind search, C, looks for what that set lacks: a node new
to C goes through the same adjudication, and one adjudicated grade >= 1 stops the construction
before the freeze. A search that declares a problem: A or B, the need is not judgeable
(judge_problem); C, the audit cannot be completed, and the construction stops for a person.

No person reads or approves a statement before the lock (the person's decision of
2026-10-19): the person knows E, DEV and their failure modes, and any wording of theirs would
bring a model-aware source into the holdout. The statement is reviewed by code and by the
blind roles only.

The state is durable (ingest/holdout_state.py): every admitted output, every run, attempt and
decision is an event in SQLite, written in one transaction with the state it implies. The
construction always runs by replaying it from the first cell: a stage already admitted returns
its recorded output, after its packet has been rendered again and found identical, and is
never executed again; only the first stage without an admitted output runs.

Two kinds of failure, never mixed:
- a METHODOLOGICAL rejection (an overlap, an unfit cell, an invalid statement, a genuine
  ambiguity, an uncertain adjudication, a need the protocol cannot judge) closes the attempt,
  and the cell gets a new blind attempt, up to three;
- a TECHNICAL failure (transport, broker, malformed session, isolation, model or API, the
  broker's resource ceiling, protocol, an interrupted process) is one failed execution of the
  stage, before any admissible output. The SAME need stays pending_technical and the process
  stops. resume() verifies everything frozen again and executes that stage once more, in a new
  isolated context with the same packet. A stage has at most three executions; after the third
  failure the need is pending_technical_unresolved and the construction stays stopped. Never a
  rejection, an unfilled cell, a replacement or a new question. A valid answer is never
  executed again to obtain another.

The diagnostic's own answer is neither (diagnostic_admissible, after the first START was
aborted on 2026-10-20 with no ready need). A run of the local model that ended normally under
the frozen runtime, but answered something other than one JSON object grading every candidate
once with the protocol's labels, makes the diagnostic unavailable: greedy decoding would give
the same answer again, so it is not executed again. The answer stays with its run, never read
again, with no vote and no label; judge D, blind, grades every candidate and the answerability
in its place, by the rule of a discrepancy. A failure of the process or of the runtime (a
crash, memory, an interruption, another runtime or host) is still technical.

An omission found by the audit's search C and adjudicated grade >= 1 stops the construction
before the freeze; so do fewer than 50 reachable evaluable needs. After the 20th, 40th and 60th
cell of the order a checkpoint verifies the state, makes a verified backup and records a report
of counts: it changes nothing, and a problem it finds stops the construction.

Every answer is admitted only after its complete contract holds (ingest/holdout_admission.py):
nothing admitted can fail later in a consumer for a reason of shape. Every execution is
bounded (ingest/holdout_process.py): at its deadline its process group is ended, reaped and
verified gone before the failure is recorded; a group that cannot be verified gone stops the
construction. Any other failure the construction does not classify is recorded as a stop for a
person (unclassified_failure) before the process ends, never a silent death; only a failure of
the state itself (its integrity, its identities, its lock or its medium) leaves it unwritten.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import sqlite3
import traceback
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

from ingest import benchmark as bm
from ingest import holdout as ho
from ingest import holdout_admission as ha
from ingest import holdout_broker as hb
from ingest import holdout_checks as hk
from ingest import holdout_isolation as hi
from ingest import holdout_judging as hj
from ingest import holdout_ledger as hl
from ingest import holdout_packets as hp
from ingest import holdout_process as hproc
from ingest import holdout_roles as hr
from ingest import holdout_runtime as hrt
from ingest import holdout_state as hs

CellState = Literal[
    "planned",
    "authoring",
    "candidate",
    "locked",
    "judging",
    "pending_technical",
    "pending_technical_unresolved",
    "rejected_methodological",
    "unfilled_after_attempt_limit",
    "ready",
    "frozen",
]
TERMINAL: tuple[CellState, ...] = ("ready", "unfilled_after_attempt_limit")
READY: tuple[CellState, ...] = ("ready", "frozen")
TechnicalReason = Literal[
    "transport",
    "broker",
    "malformed_session",
    "isolation",
    "model_api",
    "resource",
    "infrastructure",
    "protocol",
    "interrupted",
    "runtime_identity",
    "timeout",
    "process_cleanup",
]
VARIANTS = [("en", "formal"), ("en", "natural"), ("es", "natural")]
T = TypeVar("T")
# The state a cell is in while each stage runs; every later stage is judging
STAGE_STATE: dict[str, CellState] = {
    "concept": "authoring",
    "statement": "authoring",
    "overlap_gate": "candidate",
    "statement_validation": "candidate",
    "query_formulation": "candidate",
    "lock": "locked",
}
# What a failure of the executor itself is, by mechanism, when it does not say
EXECUTOR_FAILURE: dict[str, TechnicalReason] = {
    "headless": "model_api",
    "headless_broker": "transport",
    "local_mlx": "model_api",
}


# The checks whose failure can be a frozen semantic invariant of a well-formed answer, and the
# stages where three executions violating the same one reject the attempt (the person's rule of
# 2026-10-21). Search C keeps the frozen technical exhaustion: the audit's cells get no
# rejection the others do not get.
SEMANTIC_CHECKS = ("metadata_answer", "judge_answer_complete")
SEMANTIC_STAGES = ("judge_a", "second_search", "metadata")
# The searches of a need's evidence, in their order: search A, search B (every need) and the
# omission audit's search C (its 12 cells); every later stage is built from what they admitted
SEARCHES = ("judge_a", "second_search", "omission_search")
# The check that covers an answer of a shape its stage cannot read, by stage
SHAPE_CHECK = {
    "concept": "answer_fields",
    "statement": "answer_fields",
    "overlap_gate": "answer_fields",
    "evidence_gate": "answer_fields",
    "query_formulation": "query_formulations",
    "metadata": "metadata_answer",
}


def stage_state(stage: str) -> CellState:
    return STAGE_STATE.get(stage, "judging")


def stage_kind(stage: str) -> str:
    """The stage's name without the DEV need an evidence gate is about."""
    return stage.split(":", 1)[0]


class TechnicalFailure(Exception):
    def __init__(self, reason: TechnicalReason, detail: str, check: str | None = None) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail
        self.check = check


class SemanticFailure(TechnicalFailure):
    """A well-formed answer that violates frozen semantic invariants of its stage: one failed
    technical execution, which carries the invariants' codes (holdout_admission.Verdict)."""

    def __init__(self, reason: TechnicalReason, detail: str, check: str, violations: Sequence[str]):
        super().__init__(reason, detail, check)
        self.violations = tuple(sorted(set(violations)))


def semantic(check: str, violations: Sequence[str]) -> SemanticFailure:
    found = hk.check(check)
    if not found.retriable:
        raise ValueError(f"{check} is not a check on a model's output")
    detail = f"semantic invariants violated: {sorted(set(violations))}"
    return SemanticFailure(found.consequence, detail, check, violations)  # type: ignore[arg-type]


def semantic_exhaustion(failures: Sequence[Mapping[str, Any]]) -> tuple[str, list[str]] | None:
    """Whether a stage's failed executions show the need or its adjudication unable to give an
    admissible answer under the frozen rules: all three ended normally, passed their audit,
    failed the same semantic check, and share at least one violated invariant. The identity is
    the check and its codes, never the answers' text. Returns the check and the invariants
    every execution violated, or None."""
    if len(failures) < hs.MAX_TECHNICAL_RUNS:
        return None
    checks = {f.get("check") for f in failures}
    if len(checks) != 1 or not checks <= set(SEMANTIC_CHECKS):
        return None
    if any((f.get("audit") or {}).get("ok") is not True for f in failures):
        return None
    shared = set(failures[0].get("violations") or [])
    for f in failures[1:]:
        shared &= set(f.get("violations") or [])
    if not shared:
        return None
    return str(checks.pop()), sorted(shared)


def technical(check: str, detail: str) -> TechnicalFailure:
    """A failed check on a model's output, with the consequence the frozen table gives it:
    one failed technical execution of the stage (ingest/holdout_checks.py)."""
    found = hk.check(check)
    if not found.retriable:
        raise ValueError(f"{check} is not a check on a model's output")
    return TechnicalFailure(found.consequence, detail, check)  # type: ignore[arg-type]


class TechnicalStop(Exception):
    """The construction stopped with a need pending_technical: resume() retries the stage, up
    to three executions in all."""


class ConstructionStop(Exception):
    """The construction stopped before the freeze by a rule fixed in advance: a person decides."""

    def __init__(
        self,
        message: str,
        reason: str = "rule",
        cell_id: str | None = None,
        proposal_id: str | None = None,
        stage: str | None = None,
        cell_state: CellState | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.cell_id = cell_id
        self.proposal_id = proposal_id
        self.stage = stage
        self.cell_state = cell_state


class Rejected(Exception):
    """A methodological rejection of the current attempt."""

    def __init__(
        self,
        outcome: hl.Outcome,
        stage: str,
        against: hl.Against | None = None,
        note: str | None = None,
    ) -> None:
        super().__init__(outcome)
        self.outcome = outcome
        self.stage = stage
        self.against = against
        self.note = note


class _ConceptLeaks(Exception):
    """The frozen anti-copy guard fires on the concept's own words: the statement writer's
    packet passes without them (CONSTRUCTION_AMENDMENT_1)."""


# what is already classified, and passes through an integrity guard as it is
CLASSIFIED: tuple[type[BaseException], ...] = (
    TechnicalFailure,
    TechnicalStop,
    ConstructionStop,
    Rejected,
    _ConceptLeaks,
    hs.StateError,
    hl.LedgerError,
)


def _packet_guard(writer: Callable[[str, list[str]], str], words: str, facts: list[str]) -> str:
    """The statement writer's packet under the frozen anti-copy guard (hp.author_leaks), as it
    is. When the guard fires, the packet is built again without the concept's words: if that
    passes, the concept carries the leak (_ConceptLeaks, methodological); if it fires again,
    the frozen parts do, and its PacketError is the packet's own (packet_construction)."""
    try:
        return writer(words, facts)
    except hp.PacketError:
        writer("", [])
        raise _ConceptLeaks() from None


def _evidence_packet(
    corpus: hp.Corpus,
    ref: str,
    statement: str,
    pair: Any,
    dev_by_id: Mapping[str, bm.Need],
    others: Mapping[str, list[str]],
) -> str:
    return hp.gate_evidence_packet(
        corpus,
        ref,
        statement,
        pair.dev_need,
        dev_by_id[pair.dev_need].statement,
        pair.shared_nodes,
        others,
    )


@dataclass
class Session:
    reply: str
    session: dict[str, str] = field(default_factory=dict)


@dataclass
class BrokerSession:
    run: hb.BrokerRun
    replies: list[str]
    session: dict[str, str] = field(default_factory=dict)


class Executor(Protocol):
    """The model boundary. Each call spools its raw output to a durable file while it is
    received; recover_* returns that output, complete, after the orchestrator died during or
    just after the call, or None if the spool holds no complete response."""

    def headless(self, role: str, run_id: str, packet: str) -> Session: ...
    def broker(self, role: str, run_id: str, packet: str, tools: hb.Tools) -> BrokerSession: ...
    def local(self, role: str, run_id: str, messages: list[dict[str, str]]) -> dict[str, Any]: ...
    def recover_headless(self, role: str, run_id: str, packet: str) -> Session | None: ...
    def recover_broker(
        self, role: str, run_id: str, packet: str, tools: hb.Tools
    ) -> BrokerSession | None: ...
    def recover_local(
        self, role: str, run_id: str, messages: list[dict[str, str]]
    ) -> dict[str, Any] | None: ...


@dataclass
class World:
    """Everything the construction reads and never writes. identities are the hashes of
    everything frozen the state was created with; resume() refuses if one differs."""

    grid: ho.Grid
    order: list[str]
    omission: list[str]
    dev: bm.Benchmark
    fixtures: bm.Benchmark
    corpus: hp.Corpus
    tools: hb.Tools
    texts: Mapping[bm.NodeKey, str]
    excluded_psd2: frozenset[str]
    terms: Sequence[hi.Term]
    frozen_templates: Mapping[str, str]
    pin: Mapping[str, str]
    tmp_root: str
    salt: bytes
    identities: Mapping[str, str]
    contract_check: Callable[[], None]
    runtime_check: Callable[[], None]
    # the frozen model runtimes: {"claude_runtime": ..., "mlx_runtime": ...}
    # (ingest/holdout_runtime.py); an output is admitted only if produced under them
    runtime: Mapping[str, Mapping[str, Any]]

    @property
    def forbidden(self) -> list[str]:
        return [n.statement for n in self.dev.needs + self.fixtures.needs]

    @property
    def claude_effective(self) -> Mapping[str, Any]:
        effective: Mapping[str, Any] = self.runtime["claude_runtime"]["effective"]
        return effective

    @property
    def local_runtime(self) -> Mapping[str, Any]:
        return self.runtime["mlx_runtime"]

    def cell(self, cell_id: str) -> ho.Cell:
        return next(c for c in self.grid.cells if c.cell_id == cell_id)


@dataclass
class ReadyNeed:
    cell_id: str
    proposal_id: str
    need: bm.Need
    queries: list[bm.Query]
    qrels: list[bm.Qrel]
    adjudications: list[bm.Adjudication]
    metadata: hj.Metadata
    provenance: dict[str, Any]

    def record(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "proposal_id": self.proposal_id,
            "need": self.need.model_dump(mode="json"),
            "queries": [q.model_dump(mode="json") for q in self.queries],
            "qrels": [q.model_dump(mode="json") for q in self.qrels],
            "adjudications": [a.model_dump(mode="json") for a in self.adjudications],
            "metadata": dataclasses.asdict(self.metadata),
            "provenance": self.provenance,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ReadyNeed:
        return cls(
            record["cell_id"],
            record["proposal_id"],
            bm.Need.model_validate(record["need"]),
            [bm.Query.model_validate(q) for q in record["queries"]],
            [bm.Qrel.model_validate(q) for q in record["qrels"]],
            [bm.Adjudication.model_validate(a) for a in record["adjudications"]],
            hj.Metadata(**record["metadata"]),
            dict(record["provenance"]),
        )


@dataclass
class Audited:
    role: str
    run_id: str
    value: Any
    audit: hr.RoleAudit


def _json(text: str) -> Any:
    try:
        return hj.as_json(text)
    except (json.JSONDecodeError, IndexError):
        raise technical("reply_is_json", "a reply is not the JSON its packet asks for") from None


def parse_queries(text: str, need_id: str) -> list[bm.Query]:
    """The three formulations, one per line as '<lang> <style>: <question>'."""
    found: dict[tuple[str, str], str] = {}
    for line in text.strip().splitlines():
        head, _, question = line.partition(":")
        parts = head.strip().split()
        if len(parts) == 2 and question.strip():
            found[(parts[0], parts[1])] = question.strip()
    if set(found) != set(VARIANTS):
        raise technical("query_formulations", "the queries are not the three formulations")
    # the dataset's validators refuse two queries of the same text (bm.dataset_problems)
    if len({bm._normalized(q) for q in found.values()}) != len(found):
        raise technical("query_formulations", "two formulations have the same text")
    numbers: Counter[str] = Counter()
    queries = []
    for lang, style in VARIANTS:
        numbers[lang] += 1
        queries.append(
            bm.Query(
                query_id=f"{need_id}-{lang}-{numbers[lang]}",
                need_id=need_id,
                lang=lang,  # type: ignore[arg-type]
                style=style,  # type: ignore[arg-type]
                origin="llm_from_need",
                template=bm.QUERY_TEMPLATE_ID,
                text=found[(lang, style)],
            )
        )
    return queries


def statement_problems(statement: str, forbidden: Sequence[str]) -> dict[str, list[str]]:
    """The two methodological eligibility checks of a statement, by their frozen names: no
    reference the query renderer refuses (article, paragraph, point, structural unit, node
    path, CELEX), and no legal wording or run of eight words of DEV or the fixtures."""
    cites = []
    try:
        bm.render_query_prompt(statement, [("en", "formal")])
    except bm.BenchmarkError as error:
        # the frozen query template itself must render: if it does not, that is the packet's
        # failure (packet_construction), never the statement's
        bm.render_query_prompt("A firm asks a question.", [("en", "formal")])
        cites.append(str(error))
    return {
        "statement_cites_no_provision": cites,
        "statement_no_dev_wording": hp.author_leaks(statement, forbidden),
    }


def _broker_record(run: hb.BrokerRun) -> dict[str, Any]:
    return {
        "outcome": run.outcome,
        "final": run.final,
        "log": run.log,
        "violations": run.violations,
        "sent": run.sent,
        "resource_escalation": run.resource_escalation,
    }


def _broker_run(record: Mapping[str, Any]) -> hb.BrokerRun:
    return hb.BrokerRun(
        record["outcome"],
        record["final"],
        list(record["log"]),
        list(record["violations"]),
        list(record["sent"]),
        bool(record["resource_escalation"]),
    )


def _assembled_record(final: hj.Assembled) -> dict[str, Any]:
    return {
        "qrels": [q.model_dump(mode="json") for q in final.qrels],
        "adjudications": [a.model_dump(mode="json") for a in final.adjudications],
        "answerability": final.answerability,
        "gap_note": final.gap_note,
    }


def _assembled(record: Mapping[str, Any]) -> hj.Assembled:
    return hj.Assembled(
        [bm.Qrel.model_validate(q) for q in record["qrels"]],
        [bm.Adjudication.model_validate(a) for a in record["adjudications"]],
        record["answerability"],
        record["gap_note"],
    )


def _admitted(check: str, problems: Callable[[Any], list[str]]) -> Callable[[Any], Any]:
    """An answer admitted only if its complete contract holds (ingest/holdout_admission.py);
    otherwise one failed technical execution of its stage, by the check named."""

    def admit(value: Any) -> Any:
        found = problems(value)
        if found:
            raise technical(check, f"the answer breaks its contract: {found[:3]}")
        return value

    return admit


def role_run(role: str, run_id: str, packet: str, doc: Mapping[str, Any]) -> hr.RoleRun:
    """What one recorded run received and returned, in the shape its role's audit reads: the
    pipeline's admission and the controller's re-audit build it the same way."""
    sha = _sha(packet)
    if doc["kind"] == "headless":
        return hr.RoleRun(
            role, run_id, packet, sha, [packet], [doc["reply"]], session=doc["session"]
        )
    if doc["kind"] == "broker":
        run = doc["run"]
        return hr.RoleRun(
            role,
            run_id,
            packet,
            sha,
            list(run["sent"]),
            list(doc["replies"]),
            session=doc["session"],
            broker_log=list(run["log"]),
        )
    record = doc["record"]
    return hr.RoleRun(
        role, run_id, packet, sha, [packet], [record.get("output") or ""], local=record
    )


class Pipeline:
    def __init__(
        self,
        world: World,
        executor: Executor,
        store: hs.Store,
        backup: Callable[[str], Mapping[str, Any]],
    ) -> None:
        self.world = world
        self.executor = executor
        self.store = store
        self.backup = backup
        self.role_contracts = {r: hs.sha_value(c) for r, c in hr.contracts_record().items()}
        self._reset()

    def _reset(self) -> None:
        """The construction's view in memory, rebuilt by every run from the durable state."""
        self.states: dict[str, CellState] = {c.cell_id: "planned" for c in self.world.grid.cells}
        self.ledger = hl.Ledger()
        self.ready: dict[str, ReadyNeed] = {}
        self.audits: list[dict[str, Any]] = []
        self.gate_log: list[dict[str, Any]] = []
        self.omission_log: list[dict[str, Any]] = []
        self.diagnostic_log: list[dict[str, Any]] = []
        self.consumed: list[str] = []  # run ids, in the order their outputs were consumed
        self.locks: list[str] = []  # need ids, in the order they were locked
        self.resuming = False

    @property
    def technical(self) -> dict[str, Any] | None:
        """The need pending_technical, from the durable state."""
        waiting = self.store.summary()["waiting"]
        pending = [w for w in waiting if str(w["state"]).startswith("pending_technical")]
        return pending[0] if pending else None

    # -- running -----------------------------------------------------------------

    def _verify(self) -> dict[str, Any]:
        """Everything frozen, again: the pre-authoring contract (code, systems, S4 plan, corpus,
        snapshot and chunk layer), the binary and runtime, the identities the state was
        created with (contract, plan, systems, role contracts, templates, order, corpus,
        binary, audit), the cells, and the state itself."""
        w = self.world
        for name, check in (
            ("the pre-authoring contract", w.contract_check),
            ("the runtime", w.runtime_check),
        ):
            try:
                check()
            except Exception as error:
                raise hs.IncompatibleState(f"{name}: {error}") from error
        self.store.check_identities(w.identities, w.runtime)
        self.store.check_cells(w.order, w.omission)
        return self.store.verify()

    def replay(self) -> None:
        """The construction rebuilt in memory from the durable state, verified, with every
        write refused: no model runs, no event is recorded. A stage without its admitted
        output makes it fail, before anything is written (the freeze reads the state so)."""
        self._verify()
        with self.store.no_writes():
            self._run()

    def build(self) -> None:
        """The construction from its first cell. Once it has begun, only resume() goes on."""
        self._verify()
        if self.store.has_progress():
            raise hs.StateError("the construction has begun: resume() continues it")
        self._run()

    def resume(self) -> None:
        """The same cell, the same need, the same stage, the same frozen inputs. Everything
        frozen is verified again; every admitted output is replayed, its packet rendered again
        and compared, never executed; only the incomplete stage runs, once more if it failed
        technically. A stopped or unresolved construction stays stopped."""
        verified = self._verify()
        status = self.store.status()
        if status == "frozen":
            return
        if status == "pending_technical_unresolved":
            raise TechnicalStop(
                f"pending_technical_unresolved: {verified['waiting']}: no replacement, no new "
                "need, no change of protocol; a person decides"
            )
        if status == "stopped":
            stop = verified["stop"]
            raise ConstructionStop(f"the construction stopped: {stop}: a person decides")
        self.store.record("resumed", None, {"status": status, "waiting": verified["waiting"]})
        self._run(resuming=True)

    def _run(self, resuming: bool = False) -> None:
        self._reset()
        self.resuming = resuming
        try:
            for position, cell_id in enumerate(self.world.order, start=1):
                self.build_cell(cell_id)
                if position in hs.CHECKPOINTS:
                    self._checkpoint(position)
        except (ConstructionStop, hl.StopForThePerson) as stop:
            self._stopped(stop)
            raise
        except hl.LedgerError as error:
            # the ledger refuses a transition: an invariant of what the code assembles
            refused = self._integrity("record_assembly", "ready", None, error)
            self._stopped(refused)
            raise refused from error
        except (TechnicalStop, hs.StateError, sqlite3.Error):
            # a failed execution, recorded already; or the state itself (its integrity, its
            # identities, its lock, a write refused by a replay, its medium: hs.MediumError and
            # SQLite's own errors): nothing more can be written
            raise
        except Exception as error:
            # no failure kills the construction unrecorded: a stop for a person, naming the
            # kind of failure and where, never its text (unclassified_failure)
            unclassified = self._unclassified(error)
            self._stopped(unclassified)
            raise unclassified from error

    def _unclassified(self, error: Exception) -> ConstructionStop:
        cell_id = proposal_id = stage = None
        if self.ledger.attempts:
            last = self.ledger.attempts[-1]
            cell_id, proposal_id = last.cell_id, last.proposal_id
            row = next((c for c in self.store.cells() if c["cell_id"] == cell_id), None)
            stage = row["stage"] if row else None
        frames = traceback.extract_tb(error.__traceback__)
        place = f"{Path(frames[-1].filename).name}:{frames[-1].name}" if frames else "?"
        return ConstructionStop(
            f"unclassified_failure at {stage}: {type(error).__name__} in {place}",
            reason="unclassified_failure",
            cell_id=cell_id,
            proposal_id=proposal_id,
            stage=stage,
        )

    def _stopped(self, stop: Exception) -> None:
        detail: dict[str, Any] = {"reason": "minimum", "detail": str(stop), "cell_state": None}
        where: dict[str, str | None] = {}
        if isinstance(stop, ConstructionStop):
            detail |= {"reason": stop.reason, "cell_state": stop.cell_state}
            where = {"cell": stop.cell_id, "proposal": stop.proposal_id, "stage": stop.stage}
        self.store.record("construction_stopped", "stopped", detail, **where)

    def build_cell(self, cell_id: str) -> None:
        cell = self.world.cell(cell_id)
        while True:
            if self.ledger.unfilled(cell_id):
                self.states[cell_id] = "unfilled_after_attempt_limit"
                self._check_minimum()
                return
            attempt = self._open_attempt(cell_id)
            try:
                ready = self.attempt(cell, attempt)
            except Rejected as rejection:
                self._reject(attempt, rejection)
                self._check_minimum()
                if self.ledger.unfilled(cell_id):
                    return  # the third failure: the cell was terminal before the check
                continue
            self._ready(ready)
            return

    def _check_minimum(self) -> None:
        final = list(self.ready)
        evaluable = [n for n, r in self.ready.items() if r.need.answerability != "none"]
        self.ledger.check_minimum(self.world.grid, final, evaluable)

    @staticmethod
    def _ineligible(check: str, stage: str, note: str | None = None) -> Rejected:
        """A failed methodological eligibility check: the rejection the frozen table names."""
        found = hk.check(check)
        if found.kind != "methodological_eligibility":
            raise ValueError(f"{check} is not a methodological eligibility check")
        return Rejected(found.consequence, stage, note=note)  # type: ignore[arg-type]

    def _exhausted(self, pid: str, stage: str) -> Rejected | None:
        """The rejection a stage's three recorded failures call for, if they are a semantic
        admissibility failure (semantic_exhaustion): the same answer to it live and on replay."""
        if stage not in SEMANTIC_STAGES:
            return None
        failed = [r["run_id"] for r in self.store.runs(pid, stage) if r["status"] == "failed"]
        found = semantic_exhaustion(self.store.failure_payloads(failed))
        if found is None:
            return None
        check, shared = found
        note = f"{check}: {', '.join(shared)} in {len(failed)} executions"
        return self._ineligible("semantic_admissibility", stage, note)

    def _integrity(
        self, check: str, stage: str, pid: str | None, error: Exception
    ) -> ConstructionStop:
        """A technical_integrity check of what the code builds or assembles, with the stop the
        frozen table gives it: no execution can change it (CONSTRUCTION_AMENDMENT_1). It names
        the kind of failure, never the text that caused it."""
        found = hk.check(check)
        if found.kind != "technical_integrity" or found.consequence != "stop":
            raise ValueError(f"{check} is not an integrity check that stops")
        cell_id = self.ledger.get(pid).cell_id if pid is not None else None
        return ConstructionStop(
            f"{check} at {stage}: {type(error).__name__}",
            reason=check,
            cell_id=cell_id,
            proposal_id=pid,
            stage=stage,
            cell_state="pending_technical_unresolved" if cell_id is not None else None,
        )

    def _built(self, check: str, stage: str, pid: str, build: Callable[[], T]) -> T:
        """What the code builds before a model sees it, or assembles from admitted outputs: a
        failure is the integrity stop of check, never a crash (CONSTRUCTION_AMENDMENT_1). What
        is already classified passes through as it is."""
        try:
            return build()
        except CLASSIFIED:
            raise
        except Exception as error:
            raise self._integrity(check, stage, pid, error) from error

    # -- the ledger, durable -----------------------------------------------------

    def _open_attempt(self, cell_id: str) -> hl.Attempt:
        attempt = self.ledger.open_attempt(cell_id)
        self.states[cell_id] = "authoring"
        self.store.record(
            "attempt_opened",
            f"{attempt.proposal_id}/opened",
            {"number": attempt.number},
            cell=cell_id,
            proposal=attempt.proposal_id,
        )
        return attempt

    def _accept(self, pid: str) -> str:
        cell_id = self.ledger.get(pid).cell_id
        need_id = self.ledger.give_need_id(pid)
        self.store.record(
            "need_id_given",
            f"{pid}/need_id",
            {"need_id": need_id},
            cell=cell_id,
            proposal=pid,
            stage="statement_validation",
        )
        self.ledger.close(pid, "accepted", "overlap_gate")
        self.store.record(
            "attempt_closed",
            f"{pid}/closed",
            {"outcome": "accepted", "against": None, "note": None, "state": "candidate"},
            cell=cell_id,
            proposal=pid,
            stage="overlap_gate",
        )
        return need_id

    def _reject(self, attempt: hl.Attempt, rejection: Rejected) -> None:
        pid, cell_id = attempt.proposal_id, attempt.cell_id
        payload: dict[str, Any]
        if attempt.need_id is not None and attempt.outcome == "accepted":
            note = rejection.note or "after the gate"
            self.ledger.reopen_failed(pid, rejection.outcome, rejection.stage, note)
            kind, key = "attempt_failed_after_gate", f"{pid}/failed_after_gate"
            payload = {"outcome": rejection.outcome, "note": note}
        else:
            self.ledger.close(pid, rejection.outcome, rejection.stage, against=rejection.against)
            kind, key = "attempt_closed", f"{pid}/closed"
            payload = {"outcome": rejection.outcome, "against": rejection.against, "note": None}
        state: CellState = (
            "unfilled_after_attempt_limit"
            if self.ledger.unfilled(cell_id)
            else "rejected_methodological"
        )
        self.states[cell_id] = state
        payload["state"] = state
        self.store.record(kind, key, payload, cell=cell_id, proposal=pid, stage=rejection.stage)

    def _ready(self, ready: ReadyNeed) -> None:
        record = ready.record()
        self.store.record(
            "need_ready",
            f"{ready.cell_id}/ready",
            {
                "need_id": ready.need.need_id,
                "record": record,
                "record_sha256": hs.sha_value(record),
            },
            cell=ready.cell_id,
            proposal=ready.proposal_id,
            stage="ready",
        )
        self.ready[ready.need.need_id] = ready
        self.states[ready.cell_id] = "ready"

    # -- stages ------------------------------------------------------------------

    def _ref(self, role: str, proposal_id: str) -> str:
        return ho.opaque_id(self.world.salt, f"ref:{role}", proposal_id)

    def _item_ids(self, pid: str, stage: str, targets: Sequence[str]) -> dict[str, str]:
        """target -> the opaque id a judge answers by, for one stage of one attempt: the
        orchestrator's own mapping, which only it ever resolves (ingest/holdout.py item_id)."""
        ids = {t: ho.item_id(self.world.salt, f"{pid}:{stage}", t) for t in targets}
        if len(set(ids.values())) != len(ids):
            raise ValueError("two items of one stage under one id")
        return ids

    def _run_id(self, role: str, proposal_id: str, stage: str, n: int) -> str:
        return ho.opaque_id(self.world.salt, f"run:{role}", f"{proposal_id}:{stage}#{n}")

    def _inputs(self, pid: str, *stages: str) -> dict[str, str]:
        """The admitted outputs a stage is built from, by the hash of each."""
        out = {}
        for stage in stages:
            row = self.store.stage(pid, stage)
            if row is not None:
                out[stage] = row["value_sha256"]
        return out

    def _audit(self, run: hr.RoleRun, proposal_id: str) -> hr.RoleAudit:
        audit = hr.audit_role_run(
            run,
            self.world.terms,
            self.world.frozen_templates,
            self.world.forbidden,
            pin=self.world.pin,
            tmp_root=self.world.tmp_root,
            tools=self.world.tools,
        )
        self.audits.append(
            {
                "role": run.role,
                "run_id": run.run_id,
                "proposal_id": proposal_id,
                "packet_sha256": run.packet_sha256,
                "ok": audit.ok,
                "problems": audit.problems,
            }
        )
        return audit

    def _replayed_audits(self, runs: Sequence[Mapping[str, Any]]) -> None:
        """The audits of recorded runs, in the order they were made, as the live run kept them."""
        for r in runs:
            if r["audit"] is not None:
                audit = json.loads(r["audit"])
                self.audits.append(
                    {
                        "role": r["role"],
                        "run_id": r["run_id"],
                        "proposal_id": r["proposal_id"],
                        "packet_sha256": r["packet_sha256"],
                        "ok": audit["ok"],
                        "problems": audit["problems"],
                    }
                )

    def _consume(self, out: Audited) -> Any:
        """The only way a model's output reaches the next stage: after its audit passed."""
        if not out.audit.ok:
            raise TechnicalFailure(
                "isolation", f"{out.role}: an unaudited output was offered", "role_audit"
            )
        self.consumed.append(out.run_id)
        return out.value

    def _model(
        self,
        role: str,
        pid: str,
        stage: str,
        packet: str,
        ref: str,
        inputs: Mapping[str, str],
        call: Callable[[str], dict[str, Any]],
        recall: Callable[[str], dict[str, Any] | None],
        admit: Callable[[dict[str, Any]], Any],
    ) -> Any:
        """A stage a model answers. Its admitted output, if the state has one; otherwise one
        technical execution: recorded as started, its raw output spooled to disk while it is
        received, the response recorded before it is read, then audited and admitted, or
        failed, each in one transaction."""
        store = self.store
        cell_id = self.ledger.get(pid).cell_id
        packet_sha = _sha(packet)
        runs = store.runs(pid, stage)
        if any(r["packet_sha256"] != packet_sha for r in runs):
            raise hs.IncompatibleState(
                f"{cell_id} {stage}: the packet rendered now is not the one its runs received"
            )
        admitted = store.stage(pid, stage)
        if admitted is not None:
            self._replayed_audits(runs)
            self.consumed.append(admitted["run_id"])
            self.states[cell_id] = stage_state(stage)
            return admitted["value"]
        if sum(r["status"] == "failed" for r in runs) >= hs.MAX_TECHNICAL_RUNS:
            # three recorded failures that are a semantic admissibility failure: the rejection,
            # decided again from the records alone, on a replay that writes nothing too
            exhausted = self._exhausted(pid, stage)
            if exhausted is not None:
                raise exhausted
        if not store.writable:
            raise hs.StateError(f"{cell_id} {stage} has no admitted output: this replay runs none")
        self._replayed_audits([r for r in runs if r["status"] == "failed"])
        open_runs = [r for r in runs if r["status"] in ("started", "responded")]
        if open_runs and open_runs[0]["status"] == "responded":
            # a crash after the response was recorded: it is audited now, never run again
            return self._audit_and_admit(open_runs[0], role, packet, admit)
        if open_runs:
            # a crash during or just after the call: a complete response in the spool is
            # recorded and audited now, never asked again; otherwise one failed execution
            # whose answer nobody read
            try:
                recovered = recall(open_runs[0]["run_id"])
            except TechnicalFailure as failure:
                raise self._failed(
                    open_runs[0], failure.reason, failure.detail, None, failure.check
                ) from failure
            except Exception as error:
                detail = f"the spool cannot be read: {type(error).__name__}: {error}"
                raise self._failed(open_runs[0], "infrastructure", detail, None) from error
            if recovered is not None:
                self._responded(open_runs[0], recovered, recovered=True)
                return self._audit_and_admit(store.runs(pid, stage)[-1], role, packet, admit)
            self._failed(
                open_runs[0],
                "interrupted",
                "the process stopped before a complete response reached the spool",
                None,
            )
            runs = store.runs(pid, stage)
        failed = [r for r in runs if r["status"] == "failed"]
        if len(failed) >= hs.MAX_TECHNICAL_RUNS:
            exhausted = self._exhausted(pid, stage)
            if exhausted is not None:
                raise exhausted
            raise TechnicalStop(
                f"{cell_id}: {stage} failed {len(failed)} technical executions: "
                "pending_technical_unresolved; a person decides"
            )
        if failed and not self.resuming:
            raise TechnicalStop(f"{cell_id} is pending_technical at {stage}: resume() retries it")
        n = len(runs) + 1
        run_id = self._run_id(role, pid, stage, n)
        contract = hr.ROLE_CONTRACTS[role]
        state = stage_state(stage)
        store.record(
            "run_started",
            f"{run_id}/started",
            {
                "role": role,
                "ref": ref,
                "technical_run": n,
                "packet_sha256": packet_sha,
                "template": contract.template,
                "template_sha256": self.world.frozen_templates.get(contract.template, ""),
                "role_contract_sha256": self.role_contracts[role],
                "inputs": dict(inputs),
                "state": state,
            },
            cell=cell_id,
            proposal=pid,
            stage=stage,
            run=run_id,
        )
        self.states[cell_id] = state
        self.resuming = False  # one technical execution per resume
        run = store.runs(pid, stage)[-1]
        try:
            document = call(run_id)
        except TechnicalFailure as failure:
            raise self._failed(
                run, failure.reason, failure.detail, None, failure.check
            ) from failure
        except hproc.CleanupFailed as error:
            # a process group that could not be verified gone: no execution may follow
            raise self._cleanup_stop(run, role, error) from error
        except Exception as error:  # the model, the API, the process or the transport
            reason = EXECUTOR_FAILURE[contract.mechanism]
            raise self._failed(run, reason, f"{role}: {error}", None) from error
        self._responded(run, document)
        return self._audit_and_admit(store.runs(pid, stage)[-1], role, packet, admit)

    def _responded(
        self, run: Mapping[str, Any], document: dict[str, Any], recovered: bool = False
    ) -> None:
        """The response, as received and redacted, into its two places of the store, then the
        event that names both. recovered: it was read back from the executor's spool after a
        crash, not returned by the call."""
        # as received, key order included: an audit hashes some of it (the local prompt)
        try:
            raw = json.dumps(document, ensure_ascii=False)
            raw.encode("utf-8")
        except (TypeError, ValueError) as error:  # a lone surrogate, a value JSON cannot hold
            detail = f"the response cannot be recorded as text: {type(error).__name__}"
            raise self._failed(run, "malformed_session", detail, None) from error
        session = document.get("session") or {}
        try:
            kept = hi.redact(raw, hi.identity_values(hi.parse_records(session)))
        except Exception as error:
            detail = f"the session cannot be redacted: {type(error).__name__}: {error}"
            raise self._failed(run, "infrastructure", detail, None) from error
        raw_sha, raw_bytes = self.store.put(raw, "raw")
        kept_sha, kept_bytes = self.store.put(kept, "kept")
        self.store.record(
            "run_responded",
            f"{run['run_id']}/responded",
            {
                "response": {
                    "raw": raw_sha,
                    "raw_bytes": raw_bytes,
                    "kept": kept_sha,
                    "kept_bytes": kept_bytes,
                },
                "recovered": recovered,
            },
            cell=self.ledger.get(run["proposal_id"]).cell_id,
            proposal=run["proposal_id"],
            stage=run["stage"],
            run=run["run_id"],
        )

    def _role_run(self, role: str, run_id: str, packet: str, doc: Mapping[str, Any]) -> hr.RoleRun:
        return role_run(role, run_id, packet, doc)

    def _audit_and_admit(
        self,
        run: Mapping[str, Any],
        role: str,
        packet: str,
        admit: Callable[[dict[str, Any]], Any],
    ) -> Any:
        pid, stage = run["proposal_id"], run["stage"]
        document = json.loads(self.store.text(json.loads(run["response"])["raw"], "raw"))
        try:
            audit = self._audit(self._role_run(role, run["run_id"], packet, document), pid)
        except Exception as error:
            # a run its audit cannot read is never admitted: fail closed, as a failed audit
            detail = f"{role}: the audit could not read the run: {type(error).__name__}"
            raise self._failed(run, "isolation", detail, None, "role_audit") from error
        audit_record = {"ok": audit.ok, "problems": audit.problems}
        if not audit.ok:
            detail = f"{role}: {audit.problems[:3]}"
            raise self._failed(run, "isolation", detail, audit_record, "role_audit")
        try:
            value = admit(document)
        except TechnicalFailure as failure:
            violations = failure.violations if isinstance(failure, SemanticFailure) else ()
            raise self._failed(
                run,
                failure.reason,
                failure.detail,
                audit_record,
                failure.check,
                violations=violations,
            ) from failure
        except Exception as error:  # an answer of a shape the stage cannot read
            shape = technical(
                SHAPE_CHECK.get(stage_kind(stage), "judge_answer_complete"),
                f"{role}: {type(error).__name__}: {error}",
            )
            raise self._failed(
                run, shape.reason, shape.detail, audit_record, shape.check
            ) from error
        try:
            normalized = json.loads(hs.canonical(value))
            hs.canonical(normalized).encode("utf-8")  # what the state will write
        except (TypeError, ValueError) as error:  # a lone surrogate from a JSON escape
            shape = technical(
                SHAPE_CHECK.get(stage_kind(stage), "judge_answer_complete"),
                f"{role}: the answer cannot be recorded as text: {type(error).__name__}",
            )
            raise self._failed(
                run, shape.reason, shape.detail, audit_record, shape.check
            ) from error
        normalized = self._consume(Audited(role, run["run_id"], normalized, audit))
        self.store.record(
            "stage_admitted",
            f"{pid}/{stage}/admitted",
            {
                "inputs_sha256": run["packet_sha256"],
                "value": normalized,
                "value_sha256": hs.sha_value(normalized),
                "audit": audit_record,
                "state": stage_state(stage),
            },
            cell=self.ledger.get(pid).cell_id,
            proposal=pid,
            stage=stage,
            run=run["run_id"],
        )
        return normalized

    def _failed(
        self,
        run: Mapping[str, Any],
        reason: TechnicalReason,
        detail: str,
        audit: Mapping[str, Any] | None,
        check: str | None = None,
        unresolved: bool = False,
        violations: Sequence[str] = (),
    ) -> Exception:
        """One failed technical execution, recorded with the state it leaves: pending_technical,
        or pending_technical_unresolved after the third (or at once, when unresolved: no
        execution may follow it). Nothing of it is ever consumed. When the third failure makes
        the stage's executions a semantic admissibility failure (semantic_exhaustion), the
        attempt is rejected methodologically instead: the cell goes on, and the three failures
        stay recorded with their invariants as provenance. Returns what the caller raises."""
        pid, stage = run["proposal_id"], run["stage"]
        cell_id = self.ledger.get(pid).cell_id
        before = [r["run_id"] for r in self.store.runs(pid, stage) if r["status"] == "failed"]
        failed = len(before) + 1
        payload = {
            "reason": reason,
            "check": check,
            "detail": detail[:500],
            "audit": None if audit is None else dict(audit),
            "violations": sorted(set(violations)),
        }
        exhausted = None
        if not unresolved and failed >= hs.MAX_TECHNICAL_RUNS and stage in SEMANTIC_STAGES:
            exhausted = semantic_exhaustion([*self.store.failure_payloads(before), payload])
        state: CellState = (
            "judging"  # the attempt is rejected next, by its own event
            if exhausted is not None
            else "pending_technical_unresolved"
            if unresolved or failed >= hs.MAX_TECHNICAL_RUNS
            else "pending_technical"
        )
        self.store.record(
            "run_failed",
            f"{run['run_id']}/ended",
            {**payload, "state": state},
            cell=cell_id,
            proposal=pid,
            stage=stage,
            run=run["run_id"],
        )
        self.states[cell_id] = state
        if exhausted is not None:
            rejection = self._exhausted(pid, stage)
            assert rejection is not None  # the same decision, now from the recorded failures
            return rejection
        return TechnicalStop(
            f"{cell_id} is {state} at {stage}, technical execution {failed} of "
            f"{hs.MAX_TECHNICAL_RUNS}: {reason}: {detail[:200]}"
        )

    def _cleanup_stop(
        self, run: Mapping[str, Any], role: str, error: hproc.CleanupFailed
    ) -> ConstructionStop:
        """The execution whose process group could not be verified gone: recorded failed and
        unresolved, and the construction stops for a person (process_cleanup)."""
        self._failed(run, "process_cleanup", f"{role}: {error}", None, "process_cleanup", True)
        cell_id = self.ledger.get(run["proposal_id"]).cell_id
        return ConstructionStop(
            f"process_cleanup at {run['stage']}: {role}'s process group was not verified gone",
            reason="process_cleanup",
            cell_id=cell_id,
            proposal_id=run["proposal_id"],
            stage=run["stage"],
            cell_state="pending_technical_unresolved",
        )

    def _decide(self, pid: str, stage: str, inputs: Mapping[str, str], value: Any) -> Any:
        """A stage the code decides from admitted outputs: recorded once; a replay must decide
        exactly the same from exactly the same inputs."""
        normalized = json.loads(hs.canonical(value))
        inputs_sha = hs.sha_value(dict(inputs))
        cell_id = self.ledger.get(pid).cell_id
        recorded = self.store.stage(pid, stage)
        if recorded is not None:
            if (recorded["inputs_sha256"], recorded["value_sha256"]) != (
                inputs_sha,
                hs.sha_value(normalized),
            ):
                raise hs.ReplayDiverged(f"{cell_id} {stage}: the replay decides differently")
        else:
            self.store.record(
                "stage_admitted",
                f"{pid}/{stage}/admitted",
                {
                    "inputs_sha256": inputs_sha,
                    "value": normalized,
                    "value_sha256": hs.sha_value(normalized),
                    "audit": None,
                    "state": stage_state(stage),
                },
                cell=cell_id,
                proposal=pid,
                stage=stage,
            )
        self.states[cell_id] = stage_state(stage)
        return normalized

    def _headless(
        self,
        role: str,
        pid: str,
        stage: str,
        packet: str,
        ref: str,
        inputs: Mapping[str, str],
        admit: Callable[[Any], Any],
        parse: bool = True,
    ) -> Any:
        def document(result: Session | None) -> dict[str, Any] | None:
            if result is None:
                return None
            return {"kind": "headless", "reply": result.reply, "session": dict(result.session)}

        def call(run_id: str) -> dict[str, Any]:
            found = document(self.executor.headless(role, run_id, packet))
            assert found is not None
            return found

        def recall(run_id: str) -> dict[str, Any] | None:
            return document(self.executor.recover_headless(role, run_id, packet))

        def check(doc: dict[str, Any]) -> Any:
            self._claude_identity(role, doc)
            return admit(_json(doc["reply"]) if parse else doc["reply"])

        return self._model(role, pid, stage, packet, ref, inputs, call, recall, check)

    def _claude_identity(self, role: str, doc: Mapping[str, Any]) -> None:
        """Before any of its output is read: the session ran on the frozen model."""
        problems = hrt.claude_problems(doc["session"], self.world.claude_effective)
        if problems:
            raise technical("model_identity", f"{role}: {problems[:3]}")

    def _broker(
        self,
        role: str,
        pid: str,
        stage: str,
        packet: str,
        ref: str,
        inputs: Mapping[str, str],
        admit: Callable[[Any], Any],
    ) -> Any:
        """Judge A or the second searcher. Every outcome but a final answer is a technical
        failure of this execution: the ceiling (pending_technical_resource) or the protocol."""

        def document(result: BrokerSession | None) -> dict[str, Any] | None:
            if result is None:
                return None
            return {
                "kind": "broker",
                "run": _broker_record(result.run),
                "replies": list(result.replies),
                "session": dict(result.session),
            }

        def call(run_id: str) -> dict[str, Any]:
            found = document(self.executor.broker(role, run_id, packet, self.world.tools))
            assert found is not None
            return found

        def recall(run_id: str) -> dict[str, Any] | None:
            tools = self.world.tools
            return document(self.executor.recover_broker(role, run_id, packet, tools))

        def check(doc: dict[str, Any]) -> Any:
            self._claude_identity(role, doc)
            run = _broker_run(doc["run"])
            if run.outcome == "pending_technical_resource":
                raise technical("broker_ceiling", f"{role}: {run.consumption()}")
            if run.outcome != "final":
                raise technical("broker_protocol", f"{role}: {run.outcome}")
            return admit({"final": run.final, "consumption": run.consumption()})

        return self._model(role, pid, stage, packet, ref, inputs, call, recall, check)

    def _local(
        self,
        role: str,
        pid: str,
        stage: str,
        messages: list[dict[str, str]],
        ref: str,
        inputs: Mapping[str, str],
        admit: Callable[[Any], Any],
    ) -> Any:
        def call(run_id: str) -> dict[str, Any]:
            return {"kind": "local", "record": self.executor.local(role, run_id, messages)}

        def recall(run_id: str) -> dict[str, Any] | None:
            record = self.executor.recover_local(role, run_id, messages)
            return None if record is None else {"kind": "local", "record": record}

        def check(doc: dict[str, Any]) -> Any:
            # before its output is read: produced under the frozen MLX runtime
            problems = hrt.local_problems(doc["record"], self.world.local_runtime)
            if problems:
                raise technical("runtime_identity", f"{role}: {problems[:3]}")
            # the record with its preflight and raw answer: what it is worth is the stage's to
            # decide (diagnostic_admissible)
            return admit(doc["record"])

        packet = messages[0]["content"]
        return self._model(role, pid, stage, packet, ref, inputs, call, recall, check)

    # -- one attempt -------------------------------------------------------------

    def attempt(self, cell: ho.Cell, attempt: hl.Attempt) -> ReadyNeed:
        w, pid = self.world, attempt.proposal_id
        ref = self._ref("concept_author", pid)
        concept = self._headless(
            "concept_author",
            pid,
            "concept",
            self._built(
                "packet_construction",
                "concept",
                pid,
                lambda: hp.concept_packet(cell, w.forbidden, ref),
            ),
            ref,
            {},
            _admitted("answer_fields", ha.concept_problems),
        )
        if str(concept.get("fit", "")).strip().lower() != "fits":
            raise Rejected("concept_unfit", "concept")
        ref = self._ref("statement_writer", pid)
        words, facts = str(concept["concept"]), [str(f) for f in concept["facts"]]

        def writer(text: str, items: list[str]) -> str:
            return hp.writer_packet(cell, ref, text, items, w.forbidden)

        try:
            writer_packet = self._built(
                "packet_construction", "statement", pid, lambda: _packet_guard(writer, words, facts)
            )
        except _ConceptLeaks:
            # CONSTRUCTION_AMENDMENT_1: the frozen anti-copy guard fires on the concept's own
            # words. A methodological rejection of this attempt: never a technical retry, and
            # the next blind attempt receives nothing of this one
            raise self._ineligible("concept_no_dev_wording", "statement") from None
        written = self._headless(
            "statement_writer",
            pid,
            "statement",
            writer_packet,
            ref,
            self._inputs(pid, "concept"),
            _admitted("answer_fields", ha.statement_problems),
        )
        statement = str(written["primary"]).strip()
        self._gate(pid, statement)
        found = self._built(
            "packet_construction",
            "statement_validation",
            pid,
            lambda: statement_problems(statement, w.forbidden),
        )
        validation = self._decide(
            pid, "statement_validation", self._inputs(pid, "statement"), {"problems": found}
        )
        for name, found in sorted(validation["problems"].items()):
            if found:
                raise self._ineligible(name, "statement_validation")
        need_id = self._accept(pid)
        formulated = self._headless(
            "query_writer",
            pid,
            "query_formulation",
            self._built(
                "packet_construction",
                "query_formulation",
                pid,
                lambda: bm.render_query_prompt(statement, VARIANTS),
            ),
            "",
            self._inputs(pid, "statement"),
            lambda text: [q.model_dump(mode="json") for q in parse_queries(str(text), need_id)],
            parse=False,
        )
        queries = self._built(
            "record_assembly", "lock", pid, lambda: [bm.Query.model_validate(q) for q in formulated]
        )
        draft = self._built(
            "record_assembly",
            "lock",
            pid,
            lambda: bm.Need(
                need_id=need_id,
                split="test",
                statement=statement,
                type=cell.type,
                drivers=[],
                difficulty=cell.difficulty,
                acts=list(cell.acts),
                answerability=cell.answerability,
                gap_note="planned" if cell.answerability != "complete" else None,
                status="draft",
            ),
        )
        # the lock: statement and queries fixed, and recorded with their hash, before any
        # judge sees the evidence
        self._decide(
            pid,
            "lock",
            self._inputs(pid, "statement", "query_formulation"),
            {"need": draft.model_dump(mode="json"), "queries": formulated},
        )
        self.locks.append(need_id)
        return self._judge(cell, attempt, need_id, draft, queries)

    def _gate(self, pid: str, statement: str) -> None:
        w = self.world
        existing = [hp.Existing(n.need_id, "DEV", n.statement) for n in w.dev.needs]
        existing += [hp.Existing(n.need_id, "FIXTURE", n.statement) for n in w.fixtures.needs]
        test_refs = {}
        for need_id, r in self.ready.items():
            ref = ho.opaque_id(w.salt, f"gate-existing:{pid}", need_id)
            test_refs[ref] = need_id
            existing.append(hp.Existing(ref, "TEST", r.need.statement))
        source = {e.need_id: e.source for e in existing}
        ready_test = hs.sha_value([hs.sha_value(r.record()) for r in self.ready.values()])

        def admit(verdict: Any) -> Any:
            return _admitted("answer_fields", lambda v: ha.gate_problems(v, list(source)))(verdict)

        ref = self._ref("gatekeeper", pid)
        verdict = self._headless(
            "gatekeeper",
            pid,
            "overlap_gate",
            self._built(
                "packet_construction",
                "overlap_gate",
                pid,
                lambda: hp.gate_packet(ref, statement, existing),
            ),
            ref,
            {**self._inputs(pid, "statement"), "ready_test": ready_test},
            admit,
        )
        closest = verdict.get("closest")
        self.gate_log.append(
            {
                "proposal_id": pid,
                "stage": "statement",
                "verdict": verdict["verdict"],
                "closest_source": source.get(closest),
                "closest_test_need": test_refs.get(closest),
            }
        )
        if verdict["verdict"] != "independent":
            outcome, against = hl.gate_outcome(verdict["verdict"], source.get(closest))
            raise Rejected(outcome, "overlap_gate", against)

    def _judge(
        self,
        cell: ho.Cell,
        attempt: hl.Attempt,
        need_id: str,
        draft: bm.Need,
        queries: list[bm.Query],
    ) -> ReadyNeed:
        w, pid = self.world, attempt.proposal_id

        def a_file_of(value: Mapping[str, Any]) -> dict[str, Any]:
            found = dict(value["final"])
            found["need_id"] = need_id
            return found

        def admit_search(value: Any) -> Any:
            # the complete contract of a search's answer, or of the problem it declares; then
            # the semantic invariants tying its answerability to its own grades
            verdict = ha.search_verdict(a_file_of(value), w.texts, w.excluded_psd2)
            if verdict.problems:
                detail = f"a search's answer: {verdict.problems[:3]}"
                raise technical("judge_answer_complete", detail)
            if verdict.violations:
                raise semantic("judge_answer_complete", verdict.violations)
            return value

        def search(role: str, stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
            """One blind search: a fresh context, the frozen packet built from the locked
            need alone (never another search's candidates, path, labels or output)."""
            ref = self._ref(role, pid)
            value = self._broker(
                role,
                pid,
                stage,
                self._built(
                    "packet_construction",
                    stage,
                    pid,
                    lambda: hp.judge_a_broker_packet(draft, cell, ref=ref),
                ),
                ref,
                self._inputs(pid, "lock"),
                admit_search,
            )
            return value, a_file_of(value)

        # search A, then search B: a declared problem is the need's, before any candidate
        judged_a, a_file = search("judge_a", "judge_a")
        if ha.declared_problem(a_file) is not None:
            raise Rejected("judge_problem", "judge_a")
        searched_b, b_file = search("second_searcher", "second_search")
        if ha.declared_problem(b_file) is not None:
            raise Rejected("judge_problem", "second_search")
        c_file = None
        if cell.cell_id in w.omission:
            _, c_file = search("omission_searcher", "omission_search")
            if ha.declared_problem(c_file) is not None:
                raise ConstructionStop(
                    f"the omission audit's search of {cell.cell_id} declares a problem: the "
                    "audit cannot be completed; a person decides",
                    reason="omission_audit_incomplete",
                    cell_id=cell.cell_id,
                    proposal_id=pid,
                    stage="omission_search",
                )
        discovery = self._built(
            "record_assembly",
            "evidence_union",
            pid,
            lambda: hj.evidence_union(a_file, b_file, c_file),
        )
        a, scope, additional = discovery.first, discovery.scope, discovery.additional
        candidates = self._built(
            "record_assembly", "evidence_union", pid, lambda: w.corpus.sort(list(discovery.nodes))
        )

        b_ref = self._ref("judge_b", pid)
        b_ids = self._built(
            "packet_construction",
            "judge_b",
            pid,
            lambda: self._item_ids(pid, "judge_b", candidates),
        )
        b_by_id = {v: k for k, v in b_ids.items()}
        b_answer = self._headless(
            "judge_b",
            pid,
            "judge_b",
            self._built(
                "packet_construction",
                "judge_b",
                pid,
                lambda: hp.judge_b_packet(w.corpus, draft, candidates, scope, b_ids, ref=b_ref),
            ),
            b_ref,
            self._inputs(pid, "lock", *SEARCHES),
            _admitted("judge_answer_complete", lambda v: ha.judge_b_problems(v, b_by_id)),
        )
        b = self._built(
            "record_assembly", "adjudication", pid, lambda: hj.b_judgments(b_answer, b_by_id)
        )
        disputed = self._built(
            "record_assembly", "adjudication", pid, lambda: hj.disagreements(candidates, a, b)
        )
        c_answers: dict[str, Any] = {}
        if disputed.nodes or disputed.answerability:
            c_ref = self._ref("judge_c", pid)
            c_targets = [*disputed.nodes, *([bm.ANSWERABILITY_TARGET] * disputed.answerability)]
            packet, items = self._built(
                "packet_construction",
                "judge_c",
                pid,
                lambda: hp.judge_c_packet(
                    w.corpus,
                    draft,
                    candidates,
                    scope,
                    a,
                    b,
                    disputed.nodes,
                    disputed.answerability,
                    self._item_ids(pid, "judge_c", c_targets),
                    ref=c_ref,
                ),
            )
            admit_c = _admitted("judge_answer_complete", lambda v: ha.judge_c_problems(v, items))

            c_reply = self._headless(
                "judge_c",
                pid,
                "judge_c",
                packet,
                c_ref,
                self._inputs(pid, "lock", *SEARCHES, "judge_b"),
                admit_c,
            )
            # each decision bound to its item by the opaque id alone, never by position
            by_item = {i["item_id"]: i for i in c_reply.get("items", [])}
            for item in items:
                c_answers[item.target] = by_item[item.item_id]
        assembled = self._built(
            "record_assembly",
            "adjudication",
            pid,
            lambda: hj.assemble(
                need_id, candidates, a, b, c_answers, a_file.get("gap_note"), w.texts
            ),
        )
        final = _assembled(
            self._decide(
                pid,
                "adjudication",
                self._inputs(pid, *SEARCHES, "judge_b", "judge_c"),
                _assembled_record(assembled),
            )
        )
        if any(q.ambiguous for q in final.qrels):
            raise self._ineligible("final_qrels_unambiguous", "adjudication")
        omitted = [q for q in final.qrels if f"{q.act}:{q.path}" in additional]
        if cell.cell_id in w.omission:
            # bound to this attempt: the audit of the attempt that becomes READY is the one a
            # checkpoint and the freeze read; an earlier attempt's stays as provenance
            self.omission_log.append(
                {
                    "cell_id": cell.cell_id,
                    "proposal_id": pid,
                    "need_id": need_id,
                    "additional_nodes": additional,
                    "additional_final_grades": {f"{q.act}:{q.path}": q.grade for q in omitted},
                }
            )
            if any(q.grade >= 1 for q in omitted):
                raise ConstructionStop(
                    f"the omission audit of {cell.cell_id} found evidence the union of searches "
                    "A and B missed, adjudicated grade >= 1: stop before the freeze and review "
                    "the procedure",
                    reason="omission_audit",
                    cell_id=cell.cell_id,
                    proposal_id=pid,
                    stage="adjudication",
                )

        limit = w.local_runtime["generation"]["max_prompt_tokens"]
        q_ids = self._built(
            "packet_construction",
            "diagnostic",
            pid,
            lambda: self._item_ids(pid, "diagnostic", candidates),
        )
        q_by_id = {v: k for k, v in q_ids.items()}

        def admit_q(record: Mapping[str, Any]) -> Any:
            # P1 #6: the prompt's length, counted by the frozen tokenizer before generation. A
            # prompt that does not fit the frozen limit is never generated, truncated or run
            # again: the diagnostic is unavailable (diagnostic_admissible), deterministically
            preflight = record.get("preflight")
            tokens = preflight.get("prompt_tokens") if isinstance(preflight, Mapping) else None
            if (
                not isinstance(tokens, int)
                or isinstance(tokens, bool)
                or preflight.get("max_prompt_tokens") != limit  # type: ignore[union-attr]
            ):
                raise technical("diagnostic_preflight", "no token count against the frozen limit")
            output = record.get("output")
            if tokens > limit:
                if output is not None:
                    raise technical("diagnostic_preflight", "a generation over the limit")
                return hj.unavailable_diagnostic(["input_exceeds_context"])
            if not isinstance(output, str):
                raise technical("diagnostic_preflight", "a prompt that fits, and no generation")
            # every expected candidate, from the stage's own input, and only those. An answer
            # that is not that, from a run that ended normally, is no diagnostic: never read
            # again, no vote, no label, no replacement; the run keeps it, and judge D grades
            # the need in its place (diagnostic_admissible). Executing the greedy model again
            # would give the same answer, so it is no technical failure.
            defects = hj.diagnostic_defects(q_by_id, output)
            if defects:
                return hj.unavailable_diagnostic(defects)
            diagnostic = hj.as_json(output)
            hj.material_discrepancies(q_by_id, final, diagnostic)
            return diagnostic

        q_ref = self._ref("qwen_diagnostic", pid)
        diagnostic = self._local(
            "qwen_diagnostic",
            pid,
            "diagnostic",
            self._built(
                "packet_construction",
                "diagnostic",
                pid,
                lambda: hp.diagnostic_messages(
                    w.corpus, draft, candidates, scope, q_ids, ref=q_ref
                ),
            ),
            q_ref,
            self._inputs(pid, "lock", *SEARCHES),
            admit_q,
        )
        unavailable = hj.diagnostic_unavailable(diagnostic)
        discrepancies: list[str] = []
        if unavailable:
            # Qwen unavailable is not Qwen agreeing: judge D, blind, grades every candidate and
            # the answerability, and holds the consensus or not by the rule of a discrepancy
            targets = [*candidates, bm.ANSWERABILITY_TARGET]
        else:
            discrepancies = hj.material_discrepancies(q_by_id, final, diagnostic).discrepancies
            targets = discrepancies
        d_outcome: dict[str, str] = {}
        if targets:
            # every candidate under its id, so that an item names the candidate it grades
            d_ids = self._built(
                "packet_construction",
                "judge_d",
                pid,
                lambda: self._item_ids(pid, "judge_d", [*candidates, bm.ANSWERABILITY_TARGET]),
            )
            d_by_id = {d_ids[t]: t for t in targets}
            admit_d = _admitted("judge_answer_complete", lambda v: ha.judge_d_problems(v, d_by_id))

            d_ref = self._ref("judge_d", pid)
            d_packet = self._built(
                "packet_construction",
                "judge_d",
                pid,
                lambda: hp.judge_d_packet(
                    w.corpus, draft, candidates, scope, targets, d_ids, ref=d_ref
                ),
            )
            d_reply = self._headless(
                "judge_d",
                pid,
                "judge_d",
                d_packet,
                d_ref,
                self._inputs(pid, "lock", "adjudication", "diagnostic"),
                admit_d,
            )
            d_outcome = hj.judge_d_outcome(final, d_reply.get("items", []), d_by_id)
        self.diagnostic_log.append(
            {
                "need_id": need_id,
                "status": "unavailable" if unavailable else "ok",
                "defects": list(diagnostic["defects"]) if unavailable else [],
                "discrepancies": discrepancies,
                "judge_d": d_outcome,
            }
        )
        if "adjudication_uncertain" in d_outcome.values():
            raise Rejected("adjudication_uncertain", "judge_d")

        def admit_m(reply: Any) -> Any:
            verdict = ha.metadata_verdict(reply, final.qrels, final.answerability)
            if verdict.problems:
                raise technical(
                    "metadata_answer", f"the answer breaks its contract: {verdict.problems[:3]}"
                )
            if verdict.violations:
                raise semantic("metadata_answer", verdict.violations)
            hj.final_metadata(reply, final.qrels, final.answerability, cell.acts)
            return reply

        m_ref = self._ref("metadata_adjudicator", pid)
        m_reply = self._headless(
            "metadata_adjudicator",
            pid,
            "metadata",
            self._built(
                "packet_construction",
                "metadata",
                pid,
                lambda: hp.metadata_packet(
                    w.corpus, draft, final.qrels, final.answerability, final.gap_note, ref=m_ref
                ),
            ),
            m_ref,
            self._inputs(pid, "lock", "adjudication"),
            admit_m,
        )
        meta = hj.final_metadata(m_reply, final.qrels, final.answerability, cell.acts)
        need = self._built(
            "record_assembly",
            "metadata",
            pid,
            lambda: bm.Need(
                need_id=need_id,
                split="test",
                statement=draft.statement,
                type=meta.type,  # type: ignore[arg-type]
                drivers=meta.drivers,  # type: ignore[arg-type]
                difficulty=meta.difficulty,  # type: ignore[arg-type]
                acts=meta.acts,  # type: ignore[arg-type]
                answerability=final.answerability,  # type: ignore[arg-type]
                gap_note=final.gap_note,
                status="ready",
            ),
        )
        self._evidence_gate(pid, need, final.qrels)
        others = list(self.ready.values())
        refused = {
            "dataset_records": self._built(
                "dataset_records",
                "validators",
                pid,
                lambda: bm.dataset_problems(
                    [*(r.need for r in others), need],
                    [*(q for r in others for q in r.queries), *queries],
                    [*(q for r in others for q in r.qrels), *final.qrels],
                    "test",
                    w.excluded_psd2,
                ),
            ),
            "qrel_texts": self._built(
                "qrel_texts", "validators", pid, lambda: bm.qrel_problems(final.qrels, w.texts)
            ),
        }
        for name, problems in refused.items():
            if problems:
                # a technical_integrity check of records the code assembled: no execution can
                # change them, so the need stays as it is and the construction stops
                assert hk.check(name).consequence == "stop"
                raise ConstructionStop(
                    f"{name}: the validators refuse the need: {problems[:3]}",
                    reason=name,
                    cell_id=cell.cell_id,
                    proposal_id=pid,
                    stage="validators",
                    cell_state="pending_technical_unresolved",
                )
        self._decide(
            pid,
            "validators",
            self._inputs(pid, "lock", "adjudication", "metadata"),
            {"problems": []},
        )
        return self._built(
            "record_assembly",
            "ready",
            pid,
            lambda: ReadyNeed(
                cell.cell_id,
                pid,
                need,
                queries,
                final.qrels,
                final.adjudications,
                meta,
                {
                    "cell_id": cell.cell_id,
                    "proposal_id": pid,
                    "attempt": attempt.number,
                    "judge_a_consumption": judged_a["consumption"],
                    "second_search_consumption": searched_b["consumption"],
                    # which search found each candidate, and the counts of the union
                    "evidence_discovery": discovery.record(),
                    "omission_audit": cell.cell_id in w.omission,
                    "deviations": hl.deviations(cell, need),
                    "metadata_source": meta.source,
                },
            ),
        )

    def _evidence_gate(self, pid: str, need: bm.Need, qrels: Sequence[bm.Qrel]) -> None:
        w = self.world
        dev_by_id = {n.need_id: n for n in w.dev.needs}
        pairs = self._built(
            "record_assembly",
            "evidence_gate",
            pid,
            functools.partial(ho.evidence_overlap, qrels, w.dev.qrels),
        )
        for pair in pairs:
            if not pair.shared_nodes:
                continue
            others = {
                "the held-out need": [
                    f"{q.act}:{q.path}"
                    for q in qrels
                    if q.grade == 3 and f"{q.act}:{q.path}" not in pair.shared_nodes
                ],
                pair.dev_need: [
                    f"{q.act}:{q.path}"
                    for q in w.dev.qrels_of(pair.dev_need)
                    if q.grade == 3
                    and not q.ambiguous
                    and f"{q.act}:{q.path}" not in pair.shared_nodes
                ],
            }
            ref = self._ref("gatekeeper_evidence", pid)
            packet = self._built(
                "packet_construction",
                "evidence_gate",
                pid,
                functools.partial(
                    _evidence_packet, w.corpus, ref, need.statement, pair, dev_by_id, others
                ),
            )

            admit = _admitted("answer_fields", ha.evidence_gate_problems)

            verdict = self._headless(
                "gatekeeper_evidence",
                pid,
                f"evidence_gate:{pair.dev_need}",
                packet,
                ref,
                self._inputs(pid, "lock", "adjudication", "metadata"),
                admit,
            )
            self.gate_log.append(
                {
                    "proposal_id": pid,
                    "stage": "evidence",
                    "verdict": verdict["verdict"],
                    "closest_source": "DEV",
                }
            )
            if verdict["verdict"] != "independent":
                outcome, against = hl.gate_outcome(verdict["verdict"], "DEV")
                raise Rejected(outcome, f"evidence_gate:{pair.dev_need}", against)

    # -- checkpoints and the freeze ---------------------------------------------

    def _checkpoint(self, position: int) -> None:
        """An operational checkpoint of the frozen order: the state verified, a verified
        backup, a report of counts. It changes nothing; a problem it finds stops."""
        key = f"checkpoint/{position}"
        if self.store.has_event(key):
            return
        problems = self.progress_problems(self.world.order[:position])
        if problems:
            raise ConstructionStop(
                f"checkpoint {position}: " + "; ".join(problems[:8]), reason="checkpoint"
            )
        try:
            backup = dict(self.backup(f"checkpoint {position}"))
        except Exception as error:
            raise ConstructionStop(
                f"checkpoint {position}: the backup could not be verified: {error}",
                reason="checkpoint",
            ) from error
        try:
            report = self.report()
        except Exception as error:
            raise ConstructionStop(
                f"checkpoint {position}: the report could not be made: {type(error).__name__}",
                reason="checkpoint",
            ) from error
        self.store.record(
            "checkpoint", key, {"position": position, "report": report, "backup": backup}
        )

    def progress_problems(self, cells: Sequence[str]) -> list[str]:
        """The invariants that hold for the cells built so far, checked at every checkpoint and
        at the freeze: terminal cells, nothing pending, admitted outputs audited, identities one
        to one, omission audits complete, the ledger in sequence, and the durable state verified
        and equal to what the replay rebuilt."""
        problems = []
        try:
            summary = self.store.verify()
        except hs.IntegrityError as error:
            return [str(error)]
        not_terminal = [c for c in cells if self.states.get(c) not in (*TERMINAL, "frozen")]
        if not_terminal:
            problems.append(f"cells not terminal: {not_terminal}")
        if summary["status"] not in ("building", "frozen") or any(
            str(s).startswith("pending_technical") for s in self.states.values()
        ):
            problems.append(f"the construction is {summary['status']}: a need is pending")
        ready_cells = sorted(c for c in cells if self.states.get(c) in READY)
        accepted = [a for a in self.ledger.attempts if a.outcome == "accepted"]
        if sorted(r.cell_id for r in self.ready.values()) != ready_cells:
            problems.append("ready cells and ready needs are not one-to-one")
        if sorted(a.cell_id for a in accepted) != ready_cells or len(
            {a.cell_id for a in accepted}
        ) != len(accepted):
            problems.append("accepted attempts and ready cells are not one-to-one")
        if sorted(a.need_id or "" for a in accepted) != sorted(self.ready):
            problems.append("accepted attempts and ready needs do not carry the same ids")
        for r in self.ready.values():
            if r.need.status != "ready":
                problems.append(f"{r.need.need_id}: status {r.need.status}")
            if r.need.ambiguity_note is not None or any(q.ambiguous for q in r.qrels):
                problems.append(f"{r.need.need_id}: an unresolved ambiguity")
            if sorted((q.lang, q.style) for q in r.queries) != sorted(VARIANTS):
                problems.append(f"{r.need.need_id}: incomplete query coverage")
            if ({q.need_id for q in r.queries} | {q.need_id for q in r.qrels}) - {r.need.need_id}:
                problems.append(f"{r.need.need_id}: records of another need")
            targets = {a.target for a in r.adjudications}
            if (
                bm.ANSWERABILITY_TARGET not in targets
                or {f"{q.act}:{q.path}" for q in r.qrels} - targets
            ):
                problems.append(f"{r.need.need_id}: judging outputs missing")
        problems += self._audit_problems()
        # the omission audit of the attempt that became READY (P1 #3): an earlier attempt's
        # audit of the same cell is provenance and never stands for it, or against it
        ready_attempt = {r.cell_id: r.proposal_id for r in self.ready.values()}
        for cell_id in self.world.omission:
            if cell_id not in ready_attempt:
                continue
            logged = [o for o in self.omission_log if o["proposal_id"] == ready_attempt[cell_id]]
            if len(logged) != 1:
                problems.append(f"{cell_id}: the omission audit of its READY attempt is not one")
            elif any(g >= 1 for g in logged[0]["additional_final_grades"].values()):
                problems.append(f"{cell_id}: its omission audit did not pass")
        problems += self._ledger_problems()
        gate_ids = {g["proposal_id"] for g in self.gate_log if g["stage"] == "statement"}
        if {a.proposal_id for a in accepted} - gate_ids:
            problems.append("an accepted attempt has no gate verdict")
        if {g["proposal_id"] for g in self.gate_log} - {
            a.proposal_id for a in self.ledger.attempts
        }:
            problems.append("a gate verdict belongs to no attempt")
        problems += self._consumption_problems()
        problems += self._diagnostic_coverage_problems()
        problems += self._durable_problems()
        return problems

    def _diagnostic_coverage_problems(self) -> list[str]:
        """Every admitted diagnostic covers exactly its attempt's candidates, as the attempt's
        adjudication records them, with the protocol's labels; an unavailable one names its
        classes of defect, and judge D graded every candidate and the answerability instead."""
        problems = []
        for row in self.store.stages():
            if row["stage"] != "diagnostic":
                continue
            pid = row["proposal_id"]
            diagnostic = self.store.stage(pid, "diagnostic")
            adjudication = self.store.stage(pid, "adjudication")
            if diagnostic is None or adjudication is None:
                problems.append(f"{pid}: a diagnostic without its adjudication")
                continue
            candidates = [
                a["target"]
                for a in adjudication["value"]["adjudications"]
                if a["target"] != bm.ANSWERABILITY_TARGET
            ]
            if hj.diagnostic_unavailable(diagnostic["value"]):
                judge_d = self.store.stage(pid, "judge_d")
                d_by_id = {
                    v: k
                    for k, v in self._item_ids(
                        pid, "judge_d", [*candidates, bm.ANSWERABILITY_TARGET]
                    ).items()
                }
                answered = {
                    d_by_id.get(i.get("item_id"))
                    for i in (judge_d or {"value": {}})["value"].get("items", [])
                }
                if answered != {*candidates, bm.ANSWERABILITY_TARGET}:
                    problems.append(
                        f"{pid}: an unavailable diagnostic without judge D on every candidate "
                        "and the answerability"
                    )
                continue
            q_by_id = {v: k for k, v in self._item_ids(pid, "diagnostic", candidates).items()}
            found = hj.diagnostic_problems(q_by_id, diagnostic["value"])
            if found:
                problems.append(f"{pid}: an admitted diagnostic is incomplete: {found[:3]}")
        return problems

    def _consumption_problems(self) -> list[str]:
        """Rejected technical outputs have zero downstream consumers: no output of a failed run
        was consumed, and every consumed output is that of an admitted run whose audit passed.
        The durable state checks the same from its side (ingest/holdout_state.py)."""
        runs = {r["run_id"]: r for r in self.store.runs()}
        consumed = set(self.consumed)
        problems = [
            f"{run_id}: the output of a failed run was consumed"
            for run_id in sorted(consumed)
            if runs.get(run_id, {}).get("status") == "failed"
        ]
        for run_id in sorted(consumed):
            run = runs.get(run_id)
            audit = json.loads(run["audit"]) if run and run["audit"] else {}
            if run is None or run["status"] != "admitted" or audit.get("ok") is not True:
                problems.append(f"{run_id}: consumed without an admitted run that passed its audit")
        admitted = {i for i, r in runs.items() if r["status"] == "admitted"}
        if admitted - consumed:
            unread = sorted(admitted - consumed)[:3]
            problems.append(f"admitted runs the replay did not consume: {unread}")
        return problems

    def _durable_problems(self) -> list[str]:
        """The durable state against the replay's view: the same cell states, attempts and
        ready records."""
        problems = []
        frozen = self.store.status() == "frozen"
        stored = {
            c["cell_id"]: ("ready" if frozen and c["state"] == "frozen" else c["state"])
            for c in self.store.cells()
        }
        replayed = {c: ("ready" if s == "frozen" else s) for c, s in self.states.items()}
        if stored != replayed:
            differ = sorted(c for c in stored if stored[c] != replayed.get(c))
            problems.append(f"the durable cell states differ from the replay: {differ[:5]}")
        attempts = {
            a["proposal_id"]: (a["cell_id"], a["number"], a["need_id"], a["outcome"], a["against"])
            for a in self.store.attempts()
        }
        ledger = {
            a.proposal_id: (a.cell_id, a.number, a.need_id, a.outcome, a.against)
            for a in self.ledger.attempts
        }
        if attempts != ledger:
            problems.append("the durable attempts differ from the replay")
        ready = {r["need_id"]: r["record_sha256"] for r in self.store.ready()}
        if ready != {n: hs.sha_value(r.record()) for n, r in self.ready.items()}:
            problems.append("the durable ready needs differ from the replay")
        return problems

    def freeze_problems(self, required_minimum: int = hl.MIN_EVALUABLE) -> list[str]:
        """Every invariant the freeze needs, checked at once (Codex Gate 2, P1.6)."""
        w = self.world
        problems = []
        for name, check in (
            ("the pre-authoring contract", w.contract_check),
            ("the runtime", w.runtime_check),
        ):
            try:
                check()
            except Exception as error:
                problems.append(f"{name}: {error}")
        try:
            self.store.check_identities(w.identities)
        except hs.IncompatibleState as error:
            problems.append(str(error))
        current = hr.template_hashes()
        problems += [
            f"template {t} changed since its freeze"
            for t, h in current.items()
            if w.frozen_templates.get(t) != h
        ]
        if len(self.states) != len(w.grid.cells) or set(self.states) != {
            c.cell_id for c in w.grid.cells
        }:
            problems.append("the states do not cover exactly the grid's cells")
        problems += self.progress_problems(w.order)
        evaluable = sum(r.need.answerability != "none" for r in self.ready.values())
        if evaluable < required_minimum:
            problems.append(f"{evaluable} evaluable needs, under {required_minimum}")
        return problems

    def _audit_problems(self) -> list[str]:
        problems = []
        consumed = set(self.consumed)
        admitted = [a for a in self.audits if a["run_id"] in consumed]
        if any(not a["ok"] for a in admitted) or len({a["run_id"] for a in admitted}) != len(
            consumed
        ):
            problems.append("an admitted output has no passing audit")
        by_proposal: dict[str, set[str]] = {}
        for a in admitted:
            by_proposal.setdefault(a["proposal_id"], set()).add(a["role"])
        for r in self.ready.values():
            roles = by_proposal.get(r.proposal_id, set())
            needed = {
                "concept_author",
                "statement_writer",
                "gatekeeper",
                "query_writer",
                "judge_a",
                "second_searcher",
                "judge_b",
                "qwen_diagnostic",
                "metadata_adjudicator",
            }
            if r.cell_id in self.world.omission:
                needed.add("omission_searcher")
            missing = needed - roles
            if missing:
                problems.append(f"{r.need.need_id}: no audited run of {sorted(missing)}")
        return problems

    def _ledger_problems(self) -> list[str]:
        problems = []
        by_cell: dict[str, list[hl.Attempt]] = {}
        for a in self.ledger.attempts:
            by_cell.setdefault(a.cell_id, []).append(a)
        for cell_id, attempts in by_cell.items():
            if [a.number for a in attempts] != list(range(1, len(attempts) + 1)):
                problems.append(f"{cell_id}: attempts out of sequence")
            if any(a.outcome is None for a in attempts):
                problems.append(f"{cell_id}: an attempt still open")
            if sum(a.outcome == "accepted" for a in attempts) > 1:
                problems.append(f"{cell_id}: more than one accepted attempt")
            state = self.states.get(cell_id)
            if state == "unfilled_after_attempt_limit" and not self.ledger.unfilled(cell_id):
                problems.append(f"{cell_id}: unfilled without three failed attempts")
        ids = [a.need_id for a in self.ledger.attempts if a.need_id]
        if len(ids) != len(set(ids)):
            problems.append("a need id was given twice")
        return problems

    def report(self) -> dict[str, Any]:
        """The realized-distribution report and the counts the freeze carries: counts only."""
        needs = [r.need for r in self.ready.values()]
        cell_of = {r.need.need_id: r.cell_id for r in self.ready.values()}
        cells = {c.cell_id: c for c in self.world.grid.cells}
        rejected = [a for a in self.ledger.attempts if a.outcome not in (None, "accepted")]
        summary = self.store.summary()
        ready_attempts = {r.proposal_id for r in self.ready.values()}
        audited = [o for o in self.omission_log if o["proposal_id"] in ready_attempts]
        return {
            "plan_against_realized": hl.plan_against_realized(self.world.grid, needs, cell_of),
            "ledger": self.ledger.summary(self.world.grid),
            "rejections": {
                "by_reason": dict(Counter(a.outcome or "" for a in rejected)),
                "by_type": dict(Counter(cells[a.cell_id].type for a in rejected)),
                "by_act": dict(Counter("+".join(cells[a.cell_id].acts) for a in rejected)),
                # the invariants three executions all violated: codes, never text
                "semantic_inadmissible": dict(
                    Counter(a.note or "" for a in rejected if a.outcome == "semantic_inadmissible")
                ),
            },
            "gate": dict(Counter(f"{g['stage']}:{g['verdict']}" for g in self.gate_log)),
            # the audits of the READY attempts; every attempt audited, as provenance
            "omission_audit": {
                "cells": len(audited),
                "additional_nodes": sum(len(o["additional_nodes"]) for o in audited),
                "grade_0_additions": sum(
                    g == 0 for o in audited for g in o["additional_final_grades"].values()
                ),
                "attempts_audited": len(self.omission_log),
            },
            "evidence_discovery": self._discovery_report(),
            "diagnostic": {
                "needs": len(self.diagnostic_log),
                "with_discrepancies": sum(bool(d["discrepancies"]) for d in self.diagnostic_log),
                "judge_d": dict(
                    Counter(v for d in self.diagnostic_log for v in d["judge_d"].values())
                ),
                # unavailable is not agreement: those needs had judge D on every item instead
                "available": sum(d["status"] == "ok" for d in self.diagnostic_log),
                "unavailable": sum(d["status"] == "unavailable" for d in self.diagnostic_log),
                "unavailable_defects": dict(
                    Counter(x for d in self.diagnostic_log for x in d["defects"])
                ),
                "judge_d_in_place_of_the_diagnostic": dict(
                    Counter(
                        v
                        for d in self.diagnostic_log
                        if d["status"] == "unavailable"
                        for v in d["judge_d"].values()
                    )
                ),
            },
            "technical": {
                "runs": summary["runs"],
                "failures_by_reason": summary["technical_failures"],
            },
        }

    def _discovery_report(self) -> dict[str, Any]:
        """What the two searches found across the READY needs, as counts: how many candidates
        each listed, how many only search B listed, and how many of those the adjudication
        made relevant (grade >= 2, not ambiguous) or grade >= 1."""
        out: Counter[str] = Counter()
        for r in self.ready.values():
            found = r.provenance.get("evidence_discovery", {})
            grades = {f"{q.act}:{q.path}": q for q in r.qrels}
            for key in ("search_a", "search_b", "both", "only_b", "union"):
                out[key] += int(found.get(key, 0))
            only_b = [
                n for n, by in found.get("found_by", {}).items() if "B" in by and "A" not in by
            ]
            out["only_b_grade_1_or_more"] += sum(
                n in grades and grades[n].grade >= 1 for n in only_b
            )
            out["only_b_relevant"] += sum(
                n in grades and hj.relevant(grades[n].grade, grades[n].ambiguous) for n in only_b
            )
            out["answerability_a_b_agree"] += bool(found.get("answerability_a_b_agree"))
        return {"needs": len(self.ready), **dict(sorted(out.items()))}

    def freeze(
        self,
        export: Callable[[list[dict[str, Any]], Mapping[str, Any]], Mapping[str, str]],
        commit: Callable[[dict[str, Any]], None],
        required_minimum: int = hl.MIN_EVALUABLE,
    ) -> dict[str, Any]:
        """The freeze, after a replay that ran no model. Each step can run again if a freeze
        stopped half way:
        1. every invariant, or a refusal;
        2. export(): the benchmark files written from the READY records and sealed, the
           private freeze record and the private history, which returns their hashes;
        3. the frozen event, once: every ready cell frozen;
        4. a verified backup of the private history and the state: no commitment without it;
        5. commit(): the private freeze record and the public commitment."""
        problems = self.freeze_problems(required_minimum)
        if problems:
            raise ConstructionStop("the freeze refuses: " + "; ".join(problems[:8]))
        records = [r.record() for _, r in sorted(self.ready.items())]
        report = self.report()
        files = dict(export(records, report))
        record = {"report": report, "needs": sorted(self.ready), "files": files}
        self.store.record("frozen", "frozen", {"needs": record["needs"], "files": files})
        for cell_id, state in self.states.items():
            if state == "ready":
                self.states[cell_id] = "frozen"
        try:
            backup = dict(self.backup("freeze"))
        except Exception as error:
            raise ConstructionStop(
                f"the backup could not be verified: no commitment: {error}"
            ) from error
        commit({**record, "backup": backup, "state": self.store.summary()})
        return record


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def frozen_template_hashes(path: Path) -> dict[str, str]:
    files: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))["files"]
    return files


# The code that audits a run: a change to any of it changes the audit implementation
AUDIT_FILES = (
    "ingest/corpus_tools.py",
    "ingest/holdout_broker.py",
    "ingest/holdout_isolation.py",
    "ingest/holdout_roles.py",
)


def frozen_identities(
    contract_sha256: str,
    plan: Path,
    grid: ho.Grid,
    binary: Mapping[str, str],
    terms: Sequence[hi.Term],
    repo: Path,
    claude_runtime: Mapping[str, Any],
    mlx_runtime: Mapping[str, Any],
) -> dict[str, str]:
    """The hash of everything frozen that a construction is created with and every resume must
    find unchanged, read now from the files that define each: the pinned pre-authoring
    contract, the S4 plan, the systems, the role contracts, the templates, the order and the
    omission audit, the corpus and chunk layer, the Claude Code binary, the audit's code and
    terms, the construction's code, and the Claude and MLX runtimes (requested and effective
    model, policy; packages, model files, generation)."""
    from ingest import holdout_contract as hc

    fixed = ho.plan_record(grid)
    audit = {f: bm.sha256_file(repo / f) for f in AUDIT_FILES}
    audit_terms = [[t.category, t.label, t.pattern.pattern, t.normalized] for t in terms]
    return {
        "pre_authoring_contract": contract_sha256,
        "analysis_plan": bm.sha256_file(plan),
        "systems": hs.sha_value(hc.systems()),
        "role_contracts": hs.sha_value(hr.contracts_record()),
        "templates": hs.sha_value(hr.template_hashes()),
        "order": fixed["order"]["sha256"],
        "omission_audit": fixed["omission_audit"]["sha256"],
        "corpus": hs.sha_value(hc.corpus()),
        "binary": hs.sha_value({"version": binary["version"], "sha256": binary["sha256"]}),
        "audit": hs.sha_value({"files": audit, "terms": audit_terms}),
        "construction_code": hs.sha_value(
            {f: bm.sha256_file(repo / f) for f in hc.CONSTRUCTION_FILES}
        ),
        "claude_runtime": hs.sha_value(dict(claude_runtime)),
        "mlx_runtime": hs.sha_value(dict(mlx_runtime)),
    }
