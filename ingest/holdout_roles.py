"""One contract per role of TEST's construction, and the audit of a run against it (S3,
D-054, Codex Gate 2 P1.1).

A global forbidden-term scan cannot work: a judge's packet legitimately holds the protocol,
which names the metrics, and the gatekeeper's packet holds DEV by design. So each run is
checked in three separate parts:

1. the AUTHORIZED INPUT: the packet is an instance of the role's frozen template (every
   fixed word in place), its SHA-256 is the one recorded when it was rendered, and the
   template files have their frozen hashes; packets for authors are also scanned for legal
   wording and for any run of eight words from DEV or the fixtures;
2. the ORCHESTRATOR MESSAGES: every message after the packet has one of the role's schemas,
   and a broker result is exactly what the broker's own operation returns for the request it
   logged, re-executed on the corpus;
3. the RUNTIME CONTEXT: everything the run received besides those two (system prompt,
   reminders, attachments) passes the isolation policy (attachments with exact schemas,
   built-in plugins only, no tools, no MCP, a fresh directory) and the forbidden terms.

The authorized texts, and the role's own replies, are masked before the scan: the terms
look only at what the role was not supposed to receive.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ingest import benchmark as bm
from ingest import holdout_broker as hb
from ingest import holdout_isolation as hi
from ingest import holdout_packets as hp

Mechanism = Literal["headless", "headless_broker", "local_mlx"]
QUERY_TEMPLATE = "query_variants.v1"
# The person's authorization of 2026-10-19, for every isolated role that runs Claude through
# the pinned binary: the macOS keychain only to authenticate Claude, never to create, change,
# delete or read any other credential. The local diagnostic needs none.
KEYCHAIN = "claude_keychain_authentication_only"


@dataclass(frozen=True)
class RoleContract:
    role: str
    template: str  # the template the packet is an instance of
    templates: tuple[str, ...]  # every template file whose hash the role depends on
    mechanism: Mechanism
    messages: tuple[str, ...]  # schemas of the orchestrator messages after the packet
    evidence: str  # the corpus or evidence inputs the role receives
    forbidden: tuple[str, ...]  # what must never reach it
    author: bool = False  # packets scanned for legal wording and DEV runs
    # the placeholder that holds the protocol's rules, and the words of DEV examples those
    # rules must not hold; the corpus text elsewhere in the packet may cite any provision
    rules_field: str | None = None
    banned_in_packet: tuple[str, ...] = ()

    @property
    def authentication(self) -> str:
        return "none" if self.mechanism == "local_mlx" else KEYCHAIN


_SYSTEMS = ("the systems under evaluation (E, E5, BM25)", "their results and failure modes")
_PROJECT = ("CLAUDE.md", "the memory index", "the repository and its git status")
ROLE_CONTRACTS: dict[str, RoleContract] = {
    c.role: c
    for c in (
        RoleContract(
            "concept_author",
            "concept_author.v1",
            ("concept_author.v1", "attempt_frames.v2"),
            "headless",
            (),
            "none: the cell and the corpus description of the template",
            (
                "DEV",
                "the fixtures",
                "other TEST needs",
                "earlier attempts",
                "legal text",
                *_SYSTEMS,
                *_PROJECT,
            ),
            author=True,
        ),
        RoleContract(
            "statement_writer",
            "statement_writer.v1",
            ("statement_writer.v1",),
            "headless",
            (),
            "none: the cell, the concept and its facts",
            ("DEV", "the fixtures", "other TEST needs", "legal text", *_SYSTEMS, *_PROJECT),
            author=True,
        ),
        RoleContract(
            "query_writer",
            QUERY_TEMPLATE,
            (QUERY_TEMPLATE,),
            "headless",
            (),
            "none: the statement alone",
            ("DEV", "the fixtures", "other TEST needs", "legal text", *_SYSTEMS, *_PROJECT),
            author=True,
        ),
        RoleContract(
            "gatekeeper",
            "overlap_gate.v1",
            ("overlap_gate.v1",),
            "headless",
            (),
            "by design: the proposal, DEV's and the fixtures' statements, the ready TEST needs",
            ("legal text", *_SYSTEMS, *_PROJECT),
        ),
        RoleContract(
            "gatekeeper_evidence",
            "overlap_gate_evidence.v1",
            ("overlap_gate_evidence.v1",),
            "headless",
            (),
            "the two statements and the shared and other grade-3 nodes, from the corpus",
            ("grades other than those shown", *_SYSTEMS, *_PROJECT),
        ),
        RoleContract(
            "judge_a",
            "judge_a.v2",
            ("judge_a.v2",),
            "headless_broker",
            ("broker_result", "protocol_error"),
            "the corpus, only through the evidence broker",
            ("DEV", "other judges' labels", "files", "Bash", *_SYSTEMS, *_PROJECT),
            rules_field="rules",
            banned_in_packet=hp.BANNED_FOR_JUDGE_A,
        ),
        RoleContract(
            # search B of every need: search A's packet and rules, in a context of its own
            "second_searcher",
            "judge_a.v2",
            ("judge_a.v2",),
            "headless_broker",
            ("broker_result", "protocol_error"),
            "the corpus, only through the evidence broker",
            (
                "search A's candidates, grades, path, logs and output",
                "DEV",
                "qrels",
                *_SYSTEMS,
                *_PROJECT,
            ),
            rules_field="rules",
            banned_in_packet=hp.BANNED_FOR_JUDGE_A,
        ),
        RoleContract(
            # the omission audit's search C, in its 12 cells
            "omission_searcher",
            "judge_a.v2",
            ("judge_a.v2",),
            "headless_broker",
            ("broker_result", "protocol_error"),
            "the corpus, only through the evidence broker",
            (
                "search A's and search B's candidates, grades, paths, logs and outputs",
                "DEV",
                "qrels",
                *_SYSTEMS,
                *_PROJECT,
            ),
            rules_field="rules",
            banned_in_packet=hp.BANNED_FOR_JUDGE_A,
        ),
        RoleContract(
            "judge_b",
            "judge_b.v1",
            ("judge_b.v1",),
            "headless",
            (),
            "every candidate with its legal context and scope provisions, from the corpus",
            ("the searches' labels (A, B and C)", "DEV", *_SYSTEMS, *_PROJECT),
            rules_field="rules",
            banned_in_packet=hp.BANNED_FOR_B_AND_C,
        ),
        RoleContract(
            "judge_c",
            "judge_c.v1",
            ("judge_c.v1",),
            "headless",
            (),
            "the disputed candidates with their context, and the two positions in a drawn order",
            ("which judge was first", "DEV", *_SYSTEMS, *_PROJECT),
            rules_field="rules",
            banned_in_packet=hp.BANNED_FOR_B_AND_C,
        ),
        RoleContract(
            "judge_d",
            "judge_d.v1",
            ("judge_d.v1",),
            "headless",
            (),
            "every candidate with its context, and the disputed items",
            ("the labels of A, B, C and the diagnostic", "DEV", "rankings", *_SYSTEMS, *_PROJECT),
            rules_field="rules",
            banned_in_packet=hp.BANNED_FOR_B_AND_C,
        ),
        RoleContract(
            "metadata_adjudicator",
            "metadata_adjudicator.v1",
            ("metadata_adjudicator.v1",),
            "headless",
            (),
            "the locked need, its final grades with their text, the answerability, the rubric",
            ("the planned labels", "DEV", "rankings", *_SYSTEMS, *_PROJECT),
            rules_field="drivers",
            banned_in_packet=hp.BANNED_FOR_JUDGE_A,
        ),
        RoleContract(
            "qwen_diagnostic",
            "judge_b.v1",
            ("judge_b.v1",),
            "local_mlx",
            (),
            "judge B's packet",
            ("every label", "DEV", *_SYSTEMS),
            rules_field="rules",
            banned_in_packet=hp.BANNED_FOR_B_AND_C,
        ),
    )
}
MESSAGE_SCHEMAS = {
    "broker_result": re.compile(
        r"RESULT \d+: (titles|article|nodes|search) \{.*?\}, page \d+ of \d+, \d+ characters\n.*",
        re.S,
    ),
    "protocol_error": re.compile(r"PROTOCOL ERROR E[1-5]: .*", re.S),
}


def authorized_rules(role: str) -> str:
    """The exact rules text a judge role may receive: judge A's without DEV examples, judges
    B, C, D and the diagnostic with DEV batch 2's removals, the metadata rubric's drivers."""
    if role in ("judge_a", "second_searcher", "omission_searcher"):
        return hp.judge_a_rules()
    if role == "metadata_adjudicator":
        return hp.driver_rubric()
    return hp.judge_bc_rules()


def template_path(name: str) -> Path:
    if name == QUERY_TEMPLATE:
        return bm.TEMPLATES_DIR / f"{name}.md"
    suffix = ".json" if name.startswith("attempt_frames") else ".md"
    return hp.TEMPLATES / f"{name}{suffix}"


def template_hashes() -> dict[str, str]:
    names = sorted({t for c in ROLE_CONTRACTS.values() for t in c.templates})
    return {n: hashlib.sha256(template_path(n).read_bytes()).hexdigest() for n in names}


def contracts_record() -> dict[str, Any]:
    """The role contracts as data, with the hashes of their templates: frozen before authoring."""
    hashes = template_hashes()
    return {
        role: {
            "template": c.template,
            "templates_sha256": {t: hashes[t] for t in c.templates},
            "mechanism": c.mechanism,
            "authentication": c.authentication,
            "orchestrator_messages": list(c.messages),
            "evidence": c.evidence,
            "forbidden": list(c.forbidden),
            "author_scan": c.author,
            "banned_in_packet": list(c.banned_in_packet),
        }
        for role, c in sorted(ROLE_CONTRACTS.items())
    }


def _template_text(name: str) -> str:
    text = template_path(name).read_text(encoding="utf-8")
    if name == QUERY_TEMPLATE:
        return text.split(bm.PROMPT_MARKER + "\n", 1)[1]
    return text


def instance_pattern(name: str) -> re.Pattern[str]:
    """The template as a pattern: its fixed text literally, each placeholder ($name in the
    holdout templates, {name} in the query template) any text, the same text where it repeats."""
    text = _template_text(name)
    placeholder = re.compile(r"\{(\w+)\}" if name == QUERY_TEMPLATE else r"\$(\w+)")
    out, seen, pos = [], set(), 0
    for m in placeholder.finditer(text):
        out.append(re.escape(text[pos : m.start()]))
        key = m.group(1)
        out.append(f"(?P={key})" if key in seen else f"(?P<{key}>.*?)")
        seen.add(key)
        pos = m.end()
    out.append(re.escape(text[pos:]))
    return re.compile("".join(out), re.S)


def is_instance(name: str, packet: str) -> bool:
    if name == QUERY_TEMPLATE:
        return instance_pattern(name).fullmatch(packet.rstrip("\n") + "\n") is not None
    return instance_pattern(name).fullmatch(packet) is not None


@dataclass
class RoleRun:
    """What one run of a role received and returned. For a headless role, session holds its
    transcript, stream and stderr; for the local diagnostic, local holds its output record."""

    role: str
    run_id: str
    packet: str
    packet_sha256: str
    sent: list[str]
    replies: list[str]
    session: dict[str, str] = field(default_factory=dict)
    broker_log: list[dict[str, Any]] = field(default_factory=list)
    local: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoleAudit:
    role: str
    run_id: str
    ok: bool
    problems: list[str]
    findings: dict[str, list[str]]


def _masked(texts: Mapping[str, str], authorized: Sequence[str]) -> dict[str, str]:
    """Every source decoded, with each authorized text and each of the role's replies cut out:
    what is left is what the run received besides its packet and the orchestrator's messages."""
    out = {}
    for name, raw in texts.items():
        decoded = hi.decoded_text(raw)
        for text in sorted({a for a in authorized if a.strip()}, key=len, reverse=True):
            decoded = decoded.replace(text, "<AUTHORIZED>")
        out[name] = decoded
    return out


def _broker_messages_problems(run: RoleRun, tools: hb.Tools) -> list[str]:
    """Each broker message must be exactly what the broker sends for its logged request: the
    corpus operation re-executed, its page, and the ceiling notice on the last request."""
    problems = []
    results = [m for m in run.sent[1:] if m.startswith("RESULT ")]
    if len(results) != len(run.broker_log):
        return [f"{len(results)} broker results for {len(run.broker_log)} logged requests"]
    for entry, message in zip(run.broker_log, results, strict=True):
        request = hb.Request(entry["op"], dict(entry["args"]), entry["page"])
        pages = hb.paginate(hb.execute(tools, request))
        expected = hb.result_message(entry["n"], request, pages[entry["page"] - 1], len(pages))
        if entry["n"] == hb.MAX_REQUESTS:
            expected += "\n" + hb.BUDGET_NOTICE
        if message != expected:
            problems.append(f"broker result {entry['n']} is not what the corpus returns")
    errors = {hb.error_message(code) for code in hb.ERRORS}
    for message in run.sent[1:]:
        if message.startswith("PROTOCOL ERROR") and message not in errors:
            problems.append("a protocol error that is not one of the fixed messages")
    return problems


def audit_role_run(
    run: RoleRun,
    terms: Sequence[hi.Term],
    frozen_templates: Mapping[str, str],
    forbidden_statements: Sequence[str] = (),
    pin: Mapping[str, str] | None = None,
    tmp_root: str | None = None,
    tools: hb.Tools | None = None,
) -> RoleAudit:
    contract = ROLE_CONTRACTS.get(run.role)
    if contract is None:
        return RoleAudit(run.role, run.run_id, False, [f"no contract for role {run.role!r}"], {})
    problems: list[str] = []
    # 1. the authorized input
    current = template_hashes()
    for t in contract.templates:
        if current.get(t) != frozen_templates.get(t):
            problems.append(f"template {t} does not have its frozen hash")
    if hashlib.sha256(run.packet.encode("utf-8")).hexdigest() != run.packet_sha256:
        problems.append("the packet is not the one recorded when it was rendered")
    if not is_instance(contract.template, run.packet):
        problems.append(f"the packet is not an instance of {contract.template}")
    if contract.author:
        leaks = hp.author_leaks(run.packet, forbidden_statements)
        if leaks:
            problems.append(f"the author's packet leaks: {leaks[:3]}")
    if contract.rules_field is not None:
        match = instance_pattern(contract.template).fullmatch(run.packet)
        rules = match.group(contract.rules_field) if match else ""
        if rules != authorized_rules(run.role):
            problems.append("the packet's rules are not the authorized ones")
        named = [b for b in contract.banned_in_packet if b in rules]
        if named:
            problems.append(f"the packet's rules name DEV examples: {named}")
    # 2. the orchestrator messages
    if not run.sent or run.sent[0] != run.packet:
        problems.append("the first message is not the packet")
    for message in run.sent[1:]:
        if not any(MESSAGE_SCHEMAS[s].fullmatch(message) for s in contract.messages):
            problems.append(f"unexpected orchestrator message: {message[:60]!r}")
    if contract.mechanism == "headless_broker":
        if tools is None:
            problems.append("a broker role cannot be audited without the corpus tools")
        else:
            problems += _broker_messages_problems(run, tools)
    findings: dict[str, list[str]] = {}
    # 3. the runtime context
    if contract.mechanism == "local_mlx":
        messages = run.local.get("messages")
        digest = hashlib.sha256(
            json.dumps(messages, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if not messages or messages[0].get("content") != run.packet:
            problems.append("the local prompt is not the packet")
        if run.local.get("prompt_sha256") != digest:
            problems.append("the local output was not generated from this prompt")
    else:
        texts = {k: run.session.get(k, "") for k in ("transcript", "stream", "stderr")}
        if not texts["transcript"]:
            problems.append("no transcript: the run cannot be audited")
        structural = hi.audit_run(
            texts,
            [],
            allowed_cwd_prefix=tmp_root,
            expected_version=pin["version"] if pin else None,
            expected_org_sha256=pin["organization_sha256"] if pin else None,
        )
        problems += structural.problems
        foreign = hb.foreign_user_messages(texts["transcript"], run.sent)
        if foreign:
            problems.append(f"{len(foreign)} user messages nobody authorized")
        masked = _masked(texts, [*run.sent, *run.replies])
        for source, decoded in masked.items():
            flat = hi.normalize(decoded)
            for term in terms:
                if term.found_in(decoded, flat):
                    findings.setdefault(term.category, []).append(f"{source}: {term.label}")
            if contract.author:
                for leak in hp.author_leaks(decoded, forbidden_statements):
                    if leak.startswith("a run of"):
                        findings.setdefault("dev_statement", []).append(f"{source}: {leak[:60]}")
        if findings:
            problems.append(f"forbidden content outside the authorized input: {sorted(findings)}")
    return RoleAudit(run.role, run.run_id, not problems, problems, findings)
