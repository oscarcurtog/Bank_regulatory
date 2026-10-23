"""A stand-in for TEST's model runs, to test the orchestrator with no model and no TEST
content (Codex Gate 2, P1.6). Every reply is synthetic and depends only on the cell, the
attempt and the role, so a construction resumed in another process gets the same answers;
every evidence node comes from the pipeline fixtures (N001 and N003), and every session has
the shape of Gate 1's runtime. A scenario makes chosen stages reject, fail technically a
given number of times, carry contamination, or kill the process."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import signal
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ingest import benchmark as bm
from ingest import holdout as ho
from ingest import holdout_broker as hb
from ingest import holdout_exclusions as hxe
from ingest import holdout_isolation as hi
from ingest import holdout_packets as hp
from ingest import holdout_pipeline as hpl
from ingest import holdout_roles as hr
from ingest import holdout_runtime as hrt
from ingest import holdout_state as hs
from ingest import psd2_exclusions as px
from ingest.corpus_tools import CorpusTools

PIN = {"version": "2.1.301 (Claude Code)", "organization_sha256": hi.CONTROL_ORG_SHA256}
GENERIC = "/code-review ultra launches a multi-agent cloud review of the current branch."
# What the state is created with in the tests; a test changes one to see resume refuse
IDENTITIES = {
    name: hashlib.sha256(name.encode()).hexdigest()
    for name in (
        "pre_authoring_contract",
        "analysis_plan",
        "systems",
        "role_contracts",
        "templates",
        "order",
        "omission_audit",
        "corpus",
        "binary",
        "audit",
    )
}

COMPLETE = [
    ("MiCA:149.2", 3),
    ("MiCA:149.3", 2),
    ("MiCA:149.4", 2),
    ("MiCA:149.1", 1),
    ("DORA:64", 0),
]
PARTIAL = [
    ("DORA:19.4", 3),
    ("DORA:19.4.a", 2),
    ("DORA:20.a.ii", 2),
    ("DORA:19.1", 1),
    ("DORA:19.3", 0),
]
NONE = [("MiCA:149.2", 1), ("MiCA:149.1", 1), ("DORA:64", 0)]
EXTRA_NODE = "MiCA:143.3"  # what the omission audit's search adds in a scenario
B_EXTRA_NODE = "MiCA:143.2"  # what search_2 adds to search_1's candidates in a scenario
SEARCHERS = ("searcher_1", "searcher_2", "searcher_3", "audit_searcher")
POOL_JUDGES = ("pool_judge_1", "pool_judge_2")
OVER_LIMIT = 20000  # a token count over the diagnostic's frozen limit
CONTAMINATION = " The E5 baseline reached Recall@10 0.824."
# The model the stand-in reports, in the shape of the authenticated runtime's own report
FAKE_EFFECTIVE: dict[str, Any] = {
    "init": {
        "model": "m",
        "fast_mode_state": "off",
        "per_turn_effort_active": True,
        "permissionMode": "default",
        "output_style": "default",
        "apiKeySource": "none",
    },
    "assistant_model": "m",
    "model_usage": {
        "m": {
            "canonicalModel": "m",
            "provider": "firstParty",
            "contextWindow": 1000,
            "maxOutputTokens": 100,
        }
    },
    "model_attachment": {"modelId": "m", "marketingName": "M", "knowledgeCutoff": "x"},
}
# An MLX runtime in the shape ingest/holdout_runtime.py mlx_runtime reads
FAKE_MLX: dict[str, Any] = {
    "python": {"version": "3.12.0", "implementation": "CPython", "build": "3.12.0 (fake)"},
    "machine": "arm64",
    "host": {
        "macos_product_version": "27.0.0",
        "macos_build_version": "26A000",
        "darwin_release": "27.0.0",
        "hardware_model": "Mac0,0",
        "architecture": "arm64",
    },
    "packages": {name: "1.0.0" for name in hrt.MLX_PACKAGES},
    "model": {
        "config_file": "config/retrieval/qwen3-8b-mlx.json",
        "model_id": "fake/qwen",
        "revision": "0" * 40,
        "files": {"model.safetensors": "f" * 64},
    },
    "generation": {
        "max_new_tokens": 2048,
        "decoding": "greedy",
        "chat_template_kwargs": {"enable_thinking": False},
        "max_prompt_tokens": 16384,
    },
}
FAKE_RUNTIME: dict[str, Any] = {
    "claude_runtime": {
        "requested_model": "m",
        "binary": {"version": "2.1.301 (Claude Code)", "sha256": "b" * 64},
        "effective": FAKE_EFFECTIVE,
        "policy": {},
    },
    "mlx_runtime": FAKE_MLX,
}
# What a contract records of the Gate 1 certification it relies on, for the stand-in
FAKE_CERTIFICATION: dict[str, Any] = {
    "id": "gate1-stand-in",
    "schema": "gate1-certification.v1",
    "manifest_sha256": "c" * 64,
    "public_commit": "d" * 40,
    "private_commit": "e" * 40,
    "certified_code_sha256": {"ingest/holdout_isolation.py": "f" * 64},
    "gate1_run": "run-1",
    "broker_test_run": "run-1",
}
# Development need exclusions (ingest/holdout_exclusions.py) a contract names in the tests:
# synthetic statements, no TEST text; one the gate rejected before its lock, one that became
# READY in its run
FAKE_EXCLUSIONS = (
    hxe.Exclusion(
        "run-1/G01-a1", "A synthetic firm asks whether an old development rule binds it.", False
    ),
    hxe.Exclusion(
        "run-2/G02-a2",
        "A synthetic lender asks which development report is due first.",
        True,
        "a" * 64,
    ),
)


def fake_sources(
    entries: tuple[hxe.Exclusion, ...] = FAKE_EXCLUSIONS,
) -> tuple[hxe.SourceState, ...]:
    """A synthetic inventory: one state per aborted run the contract names (its contract, the
    public one), with the counts of these entries."""
    from ingest import holdout_contract as hc

    states = []
    for n, run in enumerate(hc.ABORTED_RUNS, start=1):
        label = f"run-{n}"
        own = [e for e in entries if e.key.split("/")[0] == label]
        states.append(
            hxe.SourceState(
                label,
                str(run["contract_sha256"]),
                10 + n,
                hashlib.sha256(label.encode()).hexdigest(),
                "stopped",
                len(own),
                sum(e.locked for e in own),
                sum(e.record_sha256 is not None for e in own),
            )
        )
    return tuple(states)


def write_exclusions(
    provenance: Path,
    entries: tuple[hxe.Exclusion, ...] = FAKE_EXCLUSIONS,
    sources: tuple[hxe.SourceState, ...] | None = None,
) -> dict[str, Any]:
    """The private exclusions file of a TEST provenance, written once as the CLI writes it, and
    the identity a contract records of it."""
    provenance.mkdir(parents=True, exist_ok=True)
    found = fake_sources(entries) if sources is None else sources
    return hxe.write(provenance / hxe.FILE, entries, found)


class Crash(BaseException):
    """The process dying at that instant: no handler of the construction catches it, and
    nothing it has not committed survives."""


def session(
    sent: list[str],
    replies: list[str],
    reminder: str = "",
    extra: tuple = (),
    workdir: str = "/tmp/tmp.x",
    identity: Mapping[str, Any] | None = None,
    records_extra: tuple = (),
) -> dict[str, Any]:
    """A session as the runtime records it; identity is the model it reports (FAKE_EFFECTIVE
    unless another is given); records_extra, more transcript records after the conversation."""
    workdir = str(Path(workdir).resolve(strict=False))
    clock = hi.runtime_clock_observation()
    model = dict(identity or FAKE_EFFECTIVE)
    attachments = [
        {"type": "environment", "snapshot": {**hi._SNAPSHOT, "workingDirectory": workdir}},
        {"type": "model", "identity": model["model_attachment"], "text": "x"},
        {"type": "total_tokens_reminder", "text": "<total_tokens>1</total_tokens>"},
        {"type": "session_context", "context": {"userEmail": hi.CONTROL_EMAIL}},
        {"type": "date", "date": clock["local_date"]},
        {"type": "credential_org", "organizationUuid": hi.CONTROL_ORG},
        {
            "type": "prompt_snapshot",
            "systemPrompt": [GENERIC],
            "reminderFold": False,
            "echoWireToolInputs": True,
            "contextRendering": "x",
        },
        *extra,
    ]
    records: list[dict[str, Any]] = [
        {"type": "attachment", "attachment": a, "cwd": workdir} for a in attachments
    ]
    for record in records:
        attachment = record["attachment"]
        if attachment["type"] == "date":
            rendered_content = (
                f"<system-reminder>\n{hi.rendered_date(attachment)}\n</system-reminder>"
            )
            record["rendered"] = [{"content": rendered_content}]
        elif attachment["type"] == "environment":
            rendered_content = (
                f"<system-reminder>\n - Primary working directory: {workdir}\n</system-reminder>"
            )
            record["rendered"] = [{"content": rendered_content}]
    reminder_text = f"<system-reminder>{hi.CONTROL_EMAIL}{reminder}</system-reminder>"
    records.append(
        {
            "type": "api-request-blob",
            "message": {"role": "user", "content": [{"type": "text", "text": reminder_text}]},
        }
    )
    for i, message in enumerate(sent):
        content = [{"type": "text", "text": message}]
        records.append(
            {"type": "user", "message": {"role": "user", "content": content}, "cwd": workdir}
        )
        if i < len(replies):
            text = [{"type": "text", "text": replies[i]}]
            answer = {"model": model["assistant_model"], "content": text}
            records.append({"type": "assistant", "message": answer, "cwd": workdir})
    records += list(records_extra)
    init = {
        "type": "system",
        "subtype": "init",
        "tools": [],
        "mcp_servers": [],
        "cwd": workdir,
        "original_cwd": workdir,
        "plugins": list(hi.BUILTIN_PLUGINS),
        "claude_code_version": "2.1.301",
        **model["init"],
    }
    result = {"type": "result", "is_error": False, "modelUsage": model["model_usage"]}
    return {
        "transcript": "\n".join(json.dumps(r) for r in records),
        "stream": json.dumps(init) + "\n" + json.dumps(result),
        "stderr": "",
        "_gate1_context": {
            "launcher_workdir": workdir,
            "clock_start": clock,
            "clock_end": clock,
        },
    }


def _candidates_in(packet: str) -> list[tuple[str, str]]:
    """(opaque id, node) of every candidate a judge's packet shows (judge_b.v2, judge_d.v2), in
    its order: what a judge reads, the id it must answer by."""
    found: list[tuple[str, str]] = re.findall(
        r"^CANDIDATE (C-[0-9a-f]{12})\n\[((?:DORA|MiCA|PSD2):[^\]]+)\]", packet, re.M
    )
    return found


IDENTITY_FAULTS = ("wrong", "missing", "unknown", "duplicate", "conflicting_target", "reversed")


def misbind(
    entries: list[dict[str, Any]], how: str, node_of: Mapping[str, str]
) -> list[dict[str, Any]]:
    """A judge's entries with their identity broken one way, the way a model could break it:
    another valid id, no id, an id nobody gave, an entry twice, a node of the same act with other
    numbers written beside the right id, or the right ids in another order; or left intact, for
    an execution before the one that breaks it."""
    entries = [dict(e) for e in entries]
    first = entries[0]
    if how == "intact":
        return entries
    if how == "wrong":
        others = [i for i in node_of if i != first["item_id"]]
        first["item_id"] = others[0] if others else "C-" + "f" * 12
    elif how == "missing":
        del first["item_id"]
    elif how == "unknown":
        first["item_id"] = "C-" + "0" * 12
    elif how == "duplicate":
        entries.append(dict(first))
    elif how == "conflicting_target":
        node = node_of.get(first["item_id"], "MiCA:1")
        act, _, path = node.partition(":")
        article = path.split(".")[0]
        number = int("".join(ch for ch in article if ch.isdigit()) or "1") + 1
        first["target"] = f"{act}:{number} (paragraph 2)"
    elif how == "reversed":
        entries.reverse()
    else:
        raise AssertionError(how)
    return entries


def eligible(base: Mapping[str, Any], node: str) -> bool:
    """A node that can be judged, computed here from the corpus alone: one with its own text,
    outside the excluded PSD2 articles."""
    act, _, path = node.partition(":")
    text = base["texts"].get((act, path))
    return bool(text and text.strip()) and not (
        act == "PSD2" and path.split(".")[0] in base["excluded"]
    )


def rule_a(
    base: Mapping[str, Any],
    listed: Iterable[str] = (),
    full: Iterable[str] = (),
    named: Iterable[str] = (),
) -> set[str]:
    """Exposure v2's rule A, computed here from the corpus alone, not by
    ingest/holdout_judging.py: the nodes listed, the nodes received in full or named by a snippet
    that can be judged, and every ancestor of any of them that can be judged."""
    found = set(listed) | {n for n in [*full, *named] if eligible(base, n)}
    for node in list(found):
        act, _, path = node.partition(":")
        parts = path.split(".")
        for i in range(1, len(parts)):
            ancestor = f"{act}:{'.'.join(parts[:i])}"
            if eligible(base, ancestor):
                found.add(ancestor)
    return found


def _refs_in(packet: str, marker: str) -> list[str]:
    part = packet.split(marker, 1)[1] if marker in packet else packet
    return re.findall(r"^\[((?:DORA|MiCA|PSD2):[^\]]+)\]", part, re.M)


def _ref_of(packet: str) -> str:
    found = re.search(r"\b(R-[0-9a-f]{12})\b", packet)
    return found.group(1) if found else "R-unknown"


@dataclass
class Scenario:
    gate: dict[str, list[str]] = field(default_factory=dict)  # cell -> verdict per attempt
    closest: str = "N004"  # the need a rejected proposal repeats (DEV by default)
    unfit: set[tuple[str, int]] = field(default_factory=set)  # (cell, attempt)
    # (cell, attempt) whose concept repeats a DEV statement word for word: the frozen
    # anti-copy guard fires on the statement writer's packet (CONSTRUCTION_AMENDMENT_1)
    copied_concept: set[tuple[str, int]] = field(default_factory=set)
    technical: dict[str, str] = field(default_factory=dict)  # cell -> role that fails
    # raise | contaminate | resource | protocol | malformed
    technical_kind: str = "raise"
    technical_times: int = 1000  # executions of that role that fail before one succeeds
    crash: tuple[str, str] | None = None  # (cell, role): a Crash inside that call, once
    kill: tuple[str, str] | None = None  # (cell, role): SIGKILL of the process in that call
    # (cell, role): the response reaches the spool, then the orchestrator dies, once
    die_after: tuple[str, str] | None = None
    extra_grade: dict[str, int] = field(default_factory=dict)  # cell -> grade of EXTRA_NODE
    b_extra: dict[str, int] = field(default_factory=dict)  # cell -> grade of B_EXTRA_NODE
    # (cell, role) -> nodes a search receives in full (a `nodes` request) and does not list
    expose: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    # (cell, role) -> a pattern a search sends to `search`: it sees snippets, never a full text
    snippet: dict[tuple[str, str], str] = field(default_factory=dict)
    # (cell, role) -> node -> what a pool judge gives it: a grade, or {"grade", "ambiguous"};
    # every other node gets its grade of the evidence (0 outside it)
    pool_grade: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    # (cell, role) -> a node of the evidence that search does not list
    drop: dict[tuple[str, str], str] = field(default_factory=dict)
    # (cell, role) -> the problem a search declares, with no candidates at all
    problem: dict[tuple[str, str], str] = field(default_factory=dict)
    qwen_overlength: set[str] = field(default_factory=set)  # cells whose prompt is too long
    # (cell, role) -> transcript records its sessions carry, while it fails technically
    records: dict[tuple[str, str], list[dict[str, Any]]] = field(default_factory=dict)
    # (cell, attempt) where judge D finds every item ambiguous: adjudication_uncertain
    d_uncertain: set[tuple[str, int]] = field(default_factory=set)
    # (cell, role) -> replies for its next executions, one per execution, in order: an object
    # (a search's final answer, or a headless role's JSON) or a raw text; then the usual ones
    per_execution: dict[tuple[str, str], list[Any]] = field(default_factory=dict)
    # (cell, role) -> how judge B, C or D breaks its entries' identity, one per execution, in
    # order (IDENTITY_FAULTS); then the usual entries
    identity: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    # cell -> nodes judge B grades one step off the evidence, so that judge C decides them
    b_shift: dict[str, list[str]] = field(default_factory=dict)
    # cell -> a reason judge C gives every item (to name another provision in its text)
    c_reason: dict[str, str] = field(default_factory=dict)
    qwen_crosses: set[str] = field(default_factory=set)  # cells where the diagnostic disagrees
    d_answer: dict[str, dict[str, Any]] = field(default_factory=dict)  # cell -> D's item fields
    # (cell, role) -> a reply that replaces the usual one, for a valid answer of another kind
    override: dict[tuple[str, str], Any] = field(default_factory=dict)
    # (cell, role) -> parts of the effective model the session reports instead of the frozen
    model: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    # cell -> how the diagnostic's answer is broken (empty, subset, duplicate, unknown,
    # invalid_grade, missing_answerability, invalid_answerability, missing_ambiguous,
    # outer_array, not_json), for the first qwen_bad_times executions
    qwen_bad: dict[str, str] = field(default_factory=dict)
    qwen_bad_times: int = 1
    # cell -> parts of the MLX runtime the diagnostic's output records instead of the frozen
    local_runtime: dict[str, dict[str, Any]] = field(default_factory=dict)


class FakeExecutor:
    """Replies by role, for the cell and attempt the pipeline is on. Like the real executor,
    it spools every response to a file named by its run before returning it, so that a
    resume can recover a response the orchestrator died before recording."""

    def __init__(
        self,
        scenario: Scenario | None = None,
        spool: Path | None = None,
        workdir: str | None = None,
        runtime: Mapping[str, Any] | None = None,
    ) -> None:
        self.scenario = scenario or Scenario()
        self.spool = spool
        self.workdir = workdir or str(Path(tempfile.gettempdir()) / "tmp.x")
        self.runtime = dict(runtime or FAKE_RUNTIME)  # what the sessions and outputs report
        self.pipeline: hpl.Pipeline | None = None
        self.calls: list[tuple[str, str, int]] = []  # (role, cell, attempt)
        self.packets: dict[tuple[str, str, int], str] = {}
        # every packet, in the order sent: a role with several batches sends more than one
        self.sent: list[tuple[str, str, int, str]] = []
        self.failed: Counter[tuple[str, str]] = Counter()
        self.recovered: list[str] = []

    def _spooled(self, role: str, run_id: str, kind: str, value: Any) -> None:
        """The response on disk; then, if the scenario says so, the orchestrator dies."""
        if self.spool is not None:
            self.spool.mkdir(parents=True, exist_ok=True)
            (self.spool / f"{run_id}.json").write_text(json.dumps({"kind": kind, "value": value}))
        cell, _, _ = self._where()
        if self.scenario.die_after == (cell, role):
            self.scenario.die_after = None
            raise Crash(f"killed after {role} of {cell} answered")

    def _recall(self, run_id: str) -> Any:
        path = None if self.spool is None else self.spool / f"{run_id}.json"
        if path is None or not path.exists():
            return None
        self.recovered.append(run_id)
        return json.loads(path.read_text())["value"]

    def recover_headless(self, role: str, run_id: str, packet: str) -> hpl.Session | None:
        found = self._recall(run_id)
        return None if found is None else hpl.Session(found["reply"], found["session"])

    def recover_broker(
        self, role: str, run_id: str, packet: str, tools: hb.Tools
    ) -> hpl.BrokerSession | None:
        found = self._recall(run_id)
        if found is None:
            return None
        run = hb.BrokerRun(**found["run"])
        return hpl.BrokerSession(run, found["replies"], found["session"])

    def recover_local(
        self, role: str, run_id: str, messages: list[dict[str, str]]
    ) -> dict[str, Any] | None:
        found: dict[str, Any] | None = self._recall(run_id)
        return found

    def _where(self) -> tuple[str, int, str]:
        assert self.pipeline is not None
        attempt = self.pipeline.ledger.attempts[-1]
        cell = self.pipeline.world.cell(attempt.cell_id)
        return attempt.cell_id, attempt.number, cell.answerability

    def _evidence(self) -> list[tuple[str, int]]:
        _, _, answerability = self._where()
        return {"complete": COMPLETE, "partial": PARTIAL, "none": NONE}[answerability]

    def _truth(self) -> dict[str, int]:
        """Every node's grade as the judges see it: the evidence and the scenario's additions."""
        cell, _, _ = self._where()
        grades = dict(self._evidence())
        if cell in self.scenario.extra_grade:
            grades[EXTRA_NODE] = self.scenario.extra_grade[cell]
        if cell in self.scenario.b_extra:
            grades[B_EXTRA_NODE] = self.scenario.b_extra[cell]
        return grades

    def _enter(self, role: str, packet: str) -> tuple[str, int, str, bool]:
        """Record the call, die if the scenario says so, and say whether it fails."""
        cell, number, answerability = self._where()
        self.calls.append((role, cell, number))
        self.packets[(role, cell, number)] = packet
        self.sent.append((role, cell, number, packet))
        s = self.scenario
        if s.kill == (cell, role):
            os.kill(os.getpid(), signal.SIGKILL)
        if s.crash == (cell, role):
            s.crash = None
            raise Crash(f"killed inside {role} of {cell}")
        failing = s.technical.get(cell) == role and self.failed[(cell, role)] < s.technical_times
        if failing:
            self.failed[(cell, role)] += 1
        return cell, number, answerability, failing

    def _session(
        self, role: str, sent: list[str], replies: list[str], failing: bool
    ) -> dict[str, str]:
        contaminate = failing and self.scenario.technical_kind == "contaminate"
        reminder = CONTAMINATION if contaminate else ""
        cell, _, _ = self._where()
        identity = self.runtime["claude_runtime"]["effective"]
        identity = {**identity, **self.scenario.model.get((cell, role), {})}
        extra = self.scenario.records.get((cell, role), []) if failing else []
        return session(
            sent,
            replies,
            reminder=reminder,
            workdir=self.workdir,
            identity=identity,
            records_extra=tuple(extra),
        )

    def headless(self, role: str, run_id: str, packet: str) -> hpl.Session:
        cell, number, answerability, failing = self._enter(role, packet)
        kind = self.scenario.technical_kind
        if failing and kind in ("raise", "resource", "protocol"):
            raise RuntimeError("the API answered 529 overloaded")
        n = int(cell[1:]) * 10 + number
        queued = self.scenario.per_execution.get((cell, role))
        if queued:
            given = queued.pop(0)
            reply = given if isinstance(given, str) else json.dumps(given)
            result = hpl.Session(reply, self._session(role, [packet], [reply], failing))
            self._spooled(role, run_id, "headless", {"reply": reply, "session": result.session})
            return result
        if role == "query_writer":
            reply = (
                f"en formal: What does the law require in situation {n}?\n"
                f"en natural: What do I have to do in case {n}?\n"
                f"es natural: Q{n}-es?"
            )
        else:
            answer = self.scenario.override.get((cell, role))
            if answer is None:
                answer = self._reply(role, _ref_of(packet), packet, cell, number, answerability)
            faults = self.scenario.identity.get((cell, role))
            if faults:
                key = "grades" if role in ("judge_b", *POOL_JUDGES) else "items"
                node_of = dict(_candidates_in(packet)) or {
                    m: m for m in re.findall(r"^ITEM (C-[0-9a-f]{12})", packet, re.M)
                }
                answer = {**answer, key: misbind(answer[key], faults.pop(0), node_of)}
            reply = json.dumps(answer)
        if failing and kind == "malformed":
            reply = "I would rather explain in prose."
        result = hpl.Session(reply, self._session(role, [packet], [reply], failing))
        self._spooled(role, run_id, "headless", {"reply": reply, "session": result.session})
        return result

    def _reply(
        self, role: str, ref: str, packet: str, cell: str, number: int, answerability: str
    ) -> Any:
        s = self.scenario
        n = int(cell[1:]) * 10 + number
        if role == "concept_author":
            fit = "the cell cannot hold it" if (cell, number) in s.unfit else "fits"
            concept = "A firm asks a plain question."
            if (cell, number) in s.copied_concept:
                assert self.pipeline is not None
                concept = copied(self.pipeline.world.dev.needs)
            return {
                "cell_id": ref,
                "concept": concept,
                "facts": ["a firm"],
                "atomicity": "one",
                "figure_or_date": "no",
                "fit": fit,
            }
        if role == "statement_writer":
            return {
                "proposal_id": ref,
                "primary": (
                    f"A small firm in situation {n} wonders whether a new rule {n} covers its "
                    "service. Does it?"
                ),
                "primary_atomicity": "one",
                "alternative": None,
                "alternative_reason": None,
            }
        if role == "gatekeeper":
            verdicts = s.gate.get(cell, [])
            verdict = verdicts[number - 1] if number <= len(verdicts) else "independent"
            return {
                "proposal_id": ref,
                "verdict": verdict,
                "closest": None if verdict in ("independent", "uncertain") else s.closest,
                "reason": "x",
            }
        if role == "gatekeeper_evidence":
            return {"proposal_id": ref, "verdict": "independent", "closest": None, "reason": "x"}
        if role == "judge_b":
            grades = self._truth()
            for node in s.b_shift.get(cell, []):
                grades[node] = grades[node] + 1 if grades[node] < 3 else grades[node] - 1
            return {
                "need_id": ref,
                "grades": [
                    {
                        "item_id": i,
                        "grade": grades.get(n, 0),
                        "ambiguous": False,
                        "scope_basis": [n] if grades.get(n, 0) == 0 else [],
                        "reason": "B",
                    }
                    for i, n in _candidates_in(packet)
                ],
                "answerability": answerability,
                "answerability_reason": "B",
                "missing": None if answerability == "complete" else "the corpus lacks it",
            }
        if role in POOL_JUDGES:
            grades = self._truth()
            given = s.pool_grade.get((cell, role), {})
            out = []
            for i, node in _candidates_in(packet):
                found = given.get(node, grades.get(node, 0))
                grade, ambiguous = (
                    (found["grade"], bool(found.get("ambiguous")))
                    if isinstance(found, dict)
                    else (found, False)
                )
                out.append(
                    {
                        "item_id": i,
                        "grade": grade,
                        "ambiguous": ambiguous,
                        "scope_basis": [node] if grade == 0 else [],
                        "reason": "P",
                    }
                )
            return {"need_id": ref, "grades": out}
        if role == "judge_d":
            # the evidence as the searches and judge B see it: D holds the consensus unless told
            # otherwise
            section = packet.split("ITEMS TO GRADE", 2)[-1]
            node_of = dict(_candidates_in(packet))
            items = [
                (i, bm.ANSWERABILITY_TARGET if text.startswith("the answerability") else node_of[i])
                for i, text in re.findall(r"^ITEM (C-[0-9a-f]{12}): (.*)$", section, re.M)
            ]
            grades = self._truth()
            base: dict[str, Any] = {"ambiguous": False, "scope_basis": [], "reason": "D"}

            def item(item_id: str, target: str) -> dict[str, Any]:
                found: dict[str, Any]
                if target == bm.ANSWERABILITY_TARGET:
                    found = {"grade": None, "answerability": answerability}
                else:
                    grade = grades.get(target, 0)
                    found = {"grade": grade, "answerability": None}
                    if grade == 0:
                        found["scope_basis"] = [target]
                uncertain = {"ambiguous": True} if (cell, number) in s.d_uncertain else {}
                answer = {**base, **found, **s.d_answer.get(cell, {}), **uncertain}
                return {**answer, "item_id": item_id}

            return {"need_id": ref, "items": [item(i, t) for i, t in items]}
        if role == "metadata_adjudicator":
            kind = "direct" if answerability == "complete" else "partial_out_of_scope"
            return {
                "need_id": ref,
                "type": kind,
                "difficulty": "medium",
                "drivers": {},
                "reason": "x",
            }
        if role == "judge_c":
            # each item decided by the evidence, as the other judges see it
            grades = self._truth()
            decided = []
            for m in re.finditer(r"^ITEM (C-[0-9a-f]{12})(.*)\n(.*)$", packet, re.M):
                iid = m.group(1)
                if m.group(2).startswith(": the answerability"):
                    target, grade, answer = bm.ANSWERABILITY_TARGET, None, answerability
                else:
                    found = re.match(r"\[((?:DORA|MiCA|PSD2):[^\]]+)\]", m.group(3))
                    target = found.group(1) if found else "?"
                    grade, answer = grades.get(target, 0), None
                decided.append(
                    {
                        "item_id": iid,
                        "grade": grade,
                        "ambiguous": False,
                        "answerability": answer,
                        "scope_basis": [target] if grade == 0 else [],
                        "reason": s.c_reason.get(cell, "C"),
                    }
                )
            return {"need_id": ref, "items": decided}
        raise AssertionError(role)

    def broker(self, role: str, run_id: str, packet: str, tools: hb.Tools) -> hpl.BrokerSession:
        cell, number, answerability, failing = self._enter(role, packet)
        assert self.pipeline is not None and self.pipeline.states[cell] == "judging"
        kind = self.scenario.technical_kind
        if failing and kind in ("raise", "malformed"):
            raise RuntimeError("the stream-json process died")
        s = self.scenario
        evidence = list(self._evidence())
        if role == "audit_searcher" and cell in s.extra_grade:
            evidence.append((EXTRA_NODE, s.extra_grade[cell]))
        if role in ("searcher_2", "audit_searcher") and cell in s.b_extra:
            # search_2's addition, which the audit search finds too
            evidence.append((B_EXTRA_NODE, s.b_extra[cell]))
        dropped = s.drop.get((cell, role))
        evidence = [(n, g) for n, g in evidence if n != dropped]
        ref = _ref_of(packet)
        final: dict[str, Any] = {
            "need_id": ref,
            "candidates": [
                {
                    "node": n,
                    "scope": [],
                    "grade": g,
                    "ambiguous": False,
                    "scope_basis": [n] if g == 0 else [],
                    "rationale": "A quotes the node",
                }
                for n, g in evidence
            ],
            "answerability": answerability,
            "answerability_reason": "A",
            "gap_note": None if answerability == "complete" else "the corpus lacks the element",
            "drivers": {},
            "problem": None,
        }
        if (cell, role) in s.problem:
            # a declared problem, and nothing else: its transition comes before any candidate
            final = {"need_id": ref, "problem": s.problem[(cell, role)]}
        queued = s.per_execution.get((cell, role))
        if queued:
            final = {"need_id": ref, **queued.pop(0)}
        ask = json.dumps({"request": {"op": "nodes", "args": {"refs": [evidence[0][0]]}}})
        more = []
        if (cell, role) in s.expose:
            refs = s.expose[(cell, role)]
            more.append(json.dumps({"request": {"op": "nodes", "args": {"refs": refs}}}))
        if (cell, role) in s.snippet:
            pattern = s.snippet[(cell, role)]
            more.append(
                json.dumps(
                    {"request": {"op": "search", "args": {"scope": "ALL", "pattern": pattern}}}
                )
            )
        if failing and kind == "resource":
            replies = [ask] * 33 + [json.dumps({"final": final})]
        elif failing and kind == "protocol":
            replies = ["not a request"] * 3
        else:
            replies = [ask, ask, *more, json.dumps({"final": final})]

        class Judge:
            received: list[str] = []

            def send(self, text: str) -> str:
                self.received.append(text)
                return replies[len(self.received) - 1]

        run = hb.run_judge_a(Judge(), packet, tools)
        used = replies[: len(run.sent)]
        result = hpl.BrokerSession(run, used, self._session(role, list(run.sent), used, failing))
        value = {"run": dataclasses.asdict(run), "replies": used, "session": result.session}
        self._spooled(role, run_id, "broker", value)
        return result

    def local(self, role: str, run_id: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        cell, number, answerability, failing = self._enter(role, messages[0]["content"])
        if failing:
            raise RuntimeError("MLX ran out of memory")
        grades = self._truth()
        if cell in self.scenario.qwen_crosses:
            grades = {n: (0 if g >= 2 else g) for n, g in grades.items()}
        answer: dict[str, Any] = {
            "need_id": "x",
            "grades": [
                {"item_id": i, "grade": grades.get(n, 0), "ambiguous": False}
                for i, n in _candidates_in(messages[0]["content"])
            ],
            "answerability": answerability,
        }
        output = json.dumps(answer)
        bad = self.scenario.qwen_bad.get(cell)
        if bad and self.failed[(cell, "qwen_bad")] < self.scenario.qwen_bad_times:
            self.failed[(cell, "qwen_bad")] += 1
            output = broken_output(answer, bad)
        digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()
        runtime = self.runtime["mlx_runtime"]
        changed = self.scenario.local_runtime.get(cell)
        if changed:
            runtime = {**runtime, **changed}
        limit = self.runtime["mlx_runtime"]["generation"]["max_prompt_tokens"]
        tokens = OVER_LIMIT if cell in self.scenario.qwen_overlength else 1000
        record = {
            "messages": messages,
            "prompt_sha256": digest,
            "preflight": {
                "prompt_tokens": tokens,
                "max_prompt_tokens": limit,
                "fits": tokens <= limit,
            },
            # a prompt over the limit is never generated
            "output": output if tokens <= limit else None,
            "versions": hrt.generator_versions(runtime),
            "runtime": runtime,
        }
        self._spooled(role, run_id, "local", record)
        return record


def copied(needs: Any) -> str:
    """A DEV statement long enough for the anti-copy guard's eight words."""
    return str(next(n.statement for n in needs if len(n.statement.split()) >= 12))


def broken_diagnostic(answer: dict[str, Any], how: str) -> dict[str, Any]:
    """A diagnostic answer broken one way, from a complete one."""
    grades = [dict(g) for g in answer["grades"]]
    broken = {**answer, "grades": grades}
    if how == "empty":
        broken["grades"] = []
    elif how == "subset":
        broken["grades"] = grades[:-1]
    elif how == "duplicate":
        broken["grades"] = [*grades, dict(grades[0])]
    elif how == "unknown":
        broken["grades"] = [*grades, {"item_id": "C-" + "0" * 12, "grade": 0, "ambiguous": False}]
    elif how == "invalid_grade":
        grades[0]["grade"] = 7
    elif how == "missing_answerability":
        del broken["answerability"]
    elif how == "invalid_answerability":
        broken["answerability"] = "mostly"
    elif how == "missing_ambiguous":
        del grades[0]["ambiguous"]
    else:
        raise AssertionError(how)
    return broken


def broken_output(answer: dict[str, Any], how: str) -> str:
    """The diagnostic's raw output broken one way: the shapes of broken_diagnostic, the whole
    answer inside a one-element array (the first START's), or no JSON at all."""
    if how == "outer_array":
        return json.dumps([answer])
    if how == "not_json":
        return "I would rather explain the grades in prose."
    return json.dumps(broken_diagnostic(answer, how))


class Backups:
    """A stand-in for the verified backup of a checkpoint or the freeze: it records the label,
    or fails when told to."""

    def __init__(self) -> None:
        self.labels: list[str] = []
        self.fail = False

    def __call__(self, label: str) -> dict[str, Any]:
        if self.fail:
            raise RuntimeError("the bundle did not verify")
        self.labels.append(label)
        return {"bundle": f"fake-{len(self.labels)}", "label": label}


def load_base() -> dict[str, Any]:
    """The real grid, corpus, DEV and fixtures the construction reads: no model, no TEST."""
    dev = bm.load_benchmark(bm.DEV_DIR)
    docs = bm.load_corpus()
    grid = ho.build_grid(dev, ho.load_areas())
    return {
        "dev": dev,
        "fixtures": bm.load_benchmark(bm.FIXTURES_DIR),
        "grid": grid,
        "corpus": hp.Corpus(docs),
        "tools": CorpusTools(docs),
        "texts": bm.node_texts(docs),
        "excluded": px.load_psd2_exclusions(px.PSD2_EXCLUSIONS_PATH).excluded(),
        "terms": hi.forbidden_terms(None, None, []),
    }


def world(base: dict[str, Any], identities: dict[str, str] | None = None) -> hpl.World:
    return hpl.World(
        grid=base["grid"],
        order=ho.cell_order(base["grid"]),
        omission=ho.omission_cells(base["grid"]),
        dev=base["dev"],
        fixtures=base["fixtures"],
        corpus=base["corpus"],
        tools=base["tools"],
        texts=base["texts"],
        excluded_psd2=base["excluded"],
        terms=base["terms"],
        frozen_templates=hr.template_hashes(),
        pin=PIN,
        tmp_root=str(Path(tempfile.gettempdir()).resolve()),
        salt=b"test-salt",
        identities=dict(identities or IDENTITIES),
        contract_check=lambda: None,
        runtime_check=lambda: None,
        runtime=FAKE_RUNTIME,
    )


def raw_dir(directory: Path) -> Path:
    return directory.with_name(directory.name + "-raw")


def make(
    base: dict[str, Any],
    directory: Path,
    scenario: Scenario | None = None,
    backup: Any = None,
) -> tuple[hpl.Pipeline, FakeExecutor]:
    """A new construction in directory, with its state created there."""
    w = world(base)
    store = hs.Store.create(
        directory, raw_dir(directory), w.identities, w.order, w.omission, runtime=w.runtime
    )
    return attach(w, store, scenario, backup)


def reopen(
    base: dict[str, Any],
    directory: Path,
    scenario: Scenario | None = None,
    backup: Any = None,
    **changes: Any,
) -> tuple[hpl.Pipeline, FakeExecutor]:
    """A new process on the same directory: nothing in memory, the state opened from disk."""
    w = dataclasses.replace(world(base), **changes)
    return attach(w, hs.Store.open(directory, raw_dir(directory)), scenario, backup)


def attach(
    w: hpl.World, store: hs.Store, scenario: Scenario | None, backup: Any
) -> tuple[hpl.Pipeline, FakeExecutor]:
    executor = FakeExecutor(scenario, store.raw_dir / "fake-spool")
    pipeline = hpl.Pipeline(w, executor, store, backup if backup is not None else Backups())
    executor.pipeline = pipeline
    return pipeline, executor
