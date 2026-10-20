"""Isolation of TEST's authors (S3, D-054): the headless run that gives a context nothing
but its packet, and the audit that proves it.

An author of TEST must not know the project: not its instructions (CLAUDE.md), not the
memory index, not the repository's git status, not the systems under evaluation, their
results on DEV, the external validation, their failures or the retrieval decisions. A
subagent of the working session receives the first three by itself, so authors run as
separate headless processes:

    claude -p --safe-mode --tools "" --model <model> --output-format stream-json --verbose

from a new empty directory in the system's temporary folder (a path that names nothing),
with an environment reduced to a whitelist (the working session exports the repository
path and its own session variables), and with the packet on standard input. No tool is
available, so the packet is the only thing the context can read.

The audit reads everything the run left: its transcript, its event stream and its error
stream. It fails on any attachment of project context, and on any forbidden string. The
strings are built from the real CLAUDE.md, the real memory index and the real git log, plus
the names, figures and terms of the systems and their evaluation, so a leak of any of them
is caught, and a positive control proves that each would be.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The only attachments a headless author may receive, each with the exact shape seen in the
# authenticated runtime of Claude Code 2.1.301 (isolation test 3): every key, and the type of
# every value. A key missing or added, or a type changed, fails the audit, and so does any
# other attachment (D-054). A type may also be a nested shape, or EMPTY for a list that must
# be empty. A kind with several entries accepts exactly one of those shapes.
EMPTY = "empty list"
Shape = Mapping[str, Any]
RUNTIME_ATTACHMENTS: dict[str, list[Shape]] = {
    "environment": [
        {
            "snapshot": {
                "workingDirectory": str,
                "isWorktree": bool,
                "isGitRepo": bool,
                "additionalWorkingDirectories": EMPTY,
                "platform": str,
                "shell": str,
                "osVersion": str,
            }
        }
    ],
    "model": [
        {"identity": {"modelId": str, "marketingName": str, "knowledgeCutoff": str}, "text": str}
    ],
    "total_tokens_reminder": [{"text": str}],
    "date": [{"date": str}],
    "prompt_snapshot": [
        {
            "systemPrompt": list,
            "reminderFold": bool,
            "echoWireToolInputs": bool,
            "contextRendering": str,
        },
        {
            "cliPrefix": str,
            "contextRendering": str,
            "echoWireToolInputs": bool,
            "inlineTools": bool,
            "keptReminders": bool,
            "reminderFold": bool,
            "systemPrompt": list,
            "systemTurns": bool,
            "toolChangeHeader": bool,
            "tools": EMPTY,
        },
    ],
}
# Account identity, accepted by the user only as it is: never project or system context.
IDENTITY_ATTACHMENTS: dict[str, list[Shape]] = {
    "session_context": [{"context": {"userEmail": str}}],
    "credential_org": [{"organizationUuid": str}],
}
# The plugins built into the pinned binary, which safe-mode keeps by design; accepted only
# with that binary (version and SHA-256) and with no attachment of their own.
BUILTIN_PLUGINS = (
    {"name": "cc-plugin-agents-md", "path": "builtin", "source": "cc-plugin-agents-md@builtin"},
    {"name": "cc-plugin-telemetry", "path": "builtin", "source": "cc-plugin-telemetry@builtin"},
    {
        "name": "cc-plugin-plugin-authoring",
        "path": "builtin",
        "source": "cc-plugin-plugin-authoring@builtin",
    },
)
REDACTED = "<account-identity-redacted>"
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Inherited from the working session, and only these: the home folder holds Claude Code's
# configuration and its login. Any other variable is added only when a run proves it is
# needed for authentication or execution (D-054), and is listed in the provenance.
ENV_INHERITED = ("HOME",)
SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
# Set, never inherited: they switch off two sources of project context.
ENV_SET = {"CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1", "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}


@dataclass(frozen=True)
class Term:
    category: str
    pattern: re.Pattern[str]
    label: str
    # a line of a file is matched on normalized text (markdown and spacing removed);
    # a name, a figure or a commit subject on the decoded text as it is
    normalized: bool = False

    def found_in(self, decoded: str, normalized: str) -> bool:
        return self.pattern.search(normalized if self.normalized else decoded) is not None


def _literal(category: str, text: str, normalized: bool = False) -> Term:
    return Term(category, re.compile(re.escape(text), re.I), text, normalized)


def _regex(category: str, pattern: str) -> Term:
    return Term(category, re.compile(pattern, re.I), pattern)


def _exact(category: str, text: str) -> Term:
    """Case-sensitive: the structure of an injected block, not a phrase of ordinary prose."""
    return Term(category, re.compile(re.escape(text)), text)


# The git status block as Claude Code injects it (the DEV transcripts): its own labels, with
# their colons and capitals. Claude Code's system prompt speaks of "the current branch" in
# ordinary prose, which is not a git status and must not fail the audit.
GIT_STATUS_STRUCTURE = (
    "This is the git status at the start of the conversation",
    "Current branch:",
    "Main branch (you will usually use this for PRs):",
    "Git user:",
    "Recent commits:",
)


STATIC_TERMS: dict[str, tuple[str, ...]] = {
    "project": (
        "Fundamento",
        "Bank_regulatory",
        "Bank-regulatory",
        "oscarcurtog",
        "CLAUDE.md",
        "MEMORY.md",
    ),
    "git_status": ("gitStatus",),
    "systems": (
        "pipeline E",
        "system E",
        "candidate E",
        "multilingual-e5",
        "Qwen3-Reranker",
        "Qwen3-4B",
        "reranker",
        "decomposition",
        "assemble rule",
    ),
    "dev_metrics": (
        "0.824",
        "0,824",
        "0.763",
        "0,763",
        "0.061",
        "0,061",
        "Recall@",
        "nDCG",
        "MRR@",
        "dev-0.10.0",
        "DEV 0.10.0",
        "kappa",
    ),
    "external_validation": (
        "JuriFindIT",
        "Antiriciclaggio",
        "d765a619",
        "validation-27",
        "pooled-108",
    ),
    "failure_modes": ("out of memory", "failure mode", "G1-G3"),
    "retrieval_decisions": (
        "BM25",
        "pgvector",
        "layer-500",
        "LAB-EMB",
        "retrieval round",
        "frozen as dev candidate",
        "embedding",
    ),
}
STATIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "systems": (r"\bE5\b",),
    "dev_metrics": (r"\bN0\d\d\b",),
    "external_validation": (r"\bT4\b",),
    "failure_modes": (r"\bOOM\b", r"\bMPS\b"),
    "retrieval_decisions": (r"\bD-0\d\d\b",),
}


def normalize(text: str) -> str:
    """Markdown links reduced to their text, markdown characters and dashes to spaces,
    spacing collapsed: a line reads the same wherever it was pasted."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*`#>|_\-\u2013\u2014]", " ", text)
    return " ".join(text.split())


def _lines(text: str, min_words: int) -> list[str]:
    """Lines of a file worth matching verbatim: normalized, long enough to be distinctive,
    and cut at 60 characters so that a line wrapped elsewhere still matches."""
    out = []
    for line in text.splitlines():
        line = normalize(line)
        if len(line.split()) >= min_words:
            out.append(line[:60].rstrip())
    return out


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def decoded_text(raw: str) -> str:
    """Every string a JSON Lines text holds, decoded (no escapes), one per line; a line
    that is not JSON is kept as it is."""
    out: list[str] = []
    for line in raw.splitlines():
        try:
            out.extend(_strings(json.loads(line)))
        except json.JSONDecodeError:
            out.append(line)
    return "\n".join(out)


def forbidden_terms(
    claude_md: Path | None,
    memory_index: Path | None,
    commit_subjects: Sequence[str],
    places: Sequence[str] = (),
) -> list[Term]:
    """Every string whose presence proves a leak: the static terms, the places of this
    machine that name the project (its path and folders), every distinctive line of
    CLAUDE.md and of the memory index, and the subjects of the recent commits."""
    terms = [_literal(c, t) for c, ts in STATIC_TERMS.items() for t in ts]
    terms += [_literal("project", place) for place in places if place]
    terms += [_regex(c, p) for c, ps in STATIC_PATTERNS.items() for p in ps]
    terms += [_exact("git_status", s) for s in GIT_STATUS_STRUCTURE]
    if claude_md is not None:
        terms += [_literal("claude_md", x, True) for x in _lines(claude_md.read_text(), 6)]
    if memory_index is not None:
        terms += [_literal("memory_index", x, True) for x in _lines(memory_index.read_text(), 4)]
    terms += [_literal("git_status", s) for s in commit_subjects if len(s.split()) >= 3]
    return terms


def recent_commit_subjects(repo: Path, count: int = 15) -> list[str]:
    done = subprocess.run(
        ["git", "-C", str(repo), "log", f"-{count}", "--format=%s"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [s for s in done.stdout.splitlines() if s.strip()]


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------


@dataclass
class IsolationReport:
    ok: bool
    attachments: list[str]
    attachment_problems: list[str]
    account_identity_metadata_present: bool
    tools_offered: list[str] | None
    cwd: str | None
    model: str | None
    findings: dict[str, list[str]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def _attachment_kinds(records: Iterable[Mapping[str, Any]]) -> list[str]:
    kinds = []
    for r in records:
        a = r.get("attachment")
        if isinstance(a, dict):
            kind = str(a.get("type"))
            files = a.get("files") or []
            kinds.append(kind + "".join(f":{f.get('type')}" for f in files if isinstance(f, dict)))
    return kinds


def _shape_problems(where: str, value: Any, shape: Any) -> list[str]:
    if shape == EMPTY:
        return [] if value == [] else [f"{where}: must be an empty list"]
    if isinstance(shape, Mapping):
        if not isinstance(value, dict):
            return [f"{where}: {type(value).__name__}, expected an object"]
        if set(value) != set(shape):
            return [f"{where}: keys {sorted(value)}, expected exactly {sorted(shape)}"]
        return [p for k in shape for p in _shape_problems(f"{where}.{k}", value[k], shape[k])]
    if type(value) is not shape:
        return [f"{where}: {type(value).__name__}, expected {shape.__name__}"]
    return []


def _matches(kind: str, body: Mapping[str, Any], shapes: Sequence[Shape]) -> list[str]:
    """No problem if the body has exactly one of the allowed shapes; otherwise the
    problems against the closest one (the one with the most keys in common)."""
    content = {k: v for k, v in body.items() if k != "type"}
    attempts = [_shape_problems(kind, content, s) for s in shapes]
    if any(not a for a in attempts):
        return []
    closest = max(range(len(shapes)), key=lambda i: len(set(content) & set(shapes[i])))
    return attempts[closest]


def _describe(shape: Any) -> Any:
    if shape == EMPTY or isinstance(shape, str):
        return shape
    if isinstance(shape, Mapping):
        return {k: _describe(v) for k, v in sorted(shape.items())}
    return shape.__name__


def schema_description() -> dict[str, list[Any]]:
    """The frozen attachment shapes, written out for the provenance."""
    return {
        kind: [_describe(s) for s in shapes]
        for kind, shapes in sorted({**RUNTIME_ATTACHMENTS, **IDENTITY_ATTACHMENTS}.items())
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def attachment_problems(
    records: Iterable[Mapping[str, Any]],
    tmp_root: str | None = None,
    expected_org_sha256: str | None = None,
) -> tuple[list[str], bool]:
    """Every attachment against the policy: the runtime ones and the two identity ones,
    each with its exact schema, and nothing else. Returns the problems and whether the
    account identity metadata was present."""
    problems: list[str] = []
    identity = False
    for r in records:
        body = r.get("attachment")
        if not isinstance(body, dict):
            continue
        kind = str(body.get("type"))
        shapes = RUNTIME_ATTACHMENTS.get(kind) or IDENTITY_ATTACHMENTS.get(kind)
        if shapes is None:
            problems.append(f"attachment {kind!r} is not allowed")
            continue
        found = _matches(kind, body, shapes)
        problems += found
        if found:
            continue
        if kind == "environment":
            snap = body["snapshot"]
            if snap["isGitRepo"] or snap["isWorktree"]:
                problems.append("environment: a repository or another directory is visible")
            if tmp_root is not None and not str(snap["workingDirectory"]).startswith(tmp_root):
                problems.append("environment: the working directory is not a fresh temporary one")
        if kind in IDENTITY_ATTACHMENTS:
            identity = True
        if kind == "credential_org" and expected_org_sha256 is not None:
            if sha256_text(str(body["organizationUuid"])) != expected_org_sha256:
                problems.append("credential_org: not the expected organization")
    return problems, identity


def identity_values(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """The personal values of the identity attachments: the email address and the
    organization id. They are redacted from anything that is kept in git."""
    values: list[str] = []
    for r in records:
        body = r.get("attachment")
        if not isinstance(body, dict):
            continue
        if body.get("type") == "session_context":
            values += _EMAIL.findall(str((body.get("context") or {}).get("userEmail", "")))
        if body.get("type") == "credential_org" and body.get("organizationUuid"):
            values.append(str(body["organizationUuid"]))
    return sorted(set(values), key=len, reverse=True)


def redact(text: str, values: Sequence[str]) -> str:
    """The text with every personal value replaced, in plain and JSON-escaped form."""
    for value in values:
        for form in {value, json.dumps(value)[1:-1]}:
            text = text.replace(form, REDACTED)
    leftover = [v for v in values if v in text]
    if leftover:
        raise RuntimeError("a personal value survived the redaction")
    return text


def parse_records(texts: Mapping[str, str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in ("transcript", "stream"):
        for line in texts.get(name, "").splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    records.append({"unparsed": line})
    return records


def audit_run(
    texts: Mapping[str, str],
    terms: Sequence[Term],
    allowed_cwd_prefix: str | None = None,
    expected_version: str | None = None,
    expected_org_sha256: str | None = None,
    allowed_plugins: Sequence[Mapping[str, str]] = BUILTIN_PLUGINS,
) -> IsolationReport:
    """texts maps a source ("transcript", "stream", "stderr") to its raw content. The
    transcript and the stream are JSON Lines; the stream's init event says which tools,
    directory and model the run had."""
    records = parse_records(texts)
    attachments = _attachment_kinds(records)
    attached, identity = attachment_problems(records, allowed_cwd_prefix, expected_org_sha256)
    init = next(
        (r for r in records if r.get("type") == "system" and r.get("subtype") == "init"), {}
    )
    tools = init.get("tools")
    cwd = init.get("cwd")
    model = init.get("model")
    findings: dict[str, list[str]] = {}
    for source, text in texts.items():
        decoded = decoded_text(text)
        flat = normalize(decoded)
        for term in terms:
            if term.found_in(decoded, flat) or term.found_in(text, normalize(text)):
                findings.setdefault(term.category, []).append(f"{source}: {term.label}")
    problems = list(attached)
    plugins = init.get("plugins") or []
    if sorted(map(json.dumps, plugins)) != sorted(map(json.dumps, allowed_plugins)):
        problems.append(f"plugins {[p.get('name') for p in plugins]}, not exactly the built-ins")
    if not init:
        problems.append("no init event: the run cannot be checked")
    if tools:
        problems.append(f"tools offered: {tools}")
    if init.get("mcp_servers"):
        problems.append(f"MCP servers: {init['mcp_servers']}")
    if allowed_cwd_prefix is not None and not str(cwd or "").startswith(allowed_cwd_prefix):
        problems.append(f"working directory {cwd!r} is not a fresh temporary one")
    if expected_version is not None and not expected_version.startswith(
        str(init.get("claude_code_version") or "?")
    ):
        problems.append(
            f"the run says Claude Code {init.get('claude_code_version')}, pinned {expected_version}"
        )
    if findings:
        problems.append(f"forbidden strings: {sum(len(v) for v in findings.values())}")
    return IsolationReport(
        not problems, attachments, attached, identity, tools, cwd, model, findings, problems
    )


def _example(term: Term) -> str:
    """A string the term must catch: its own text, or an instance of its pattern."""
    examples = {
        r"\bE5\b": "the E5 run",
        r"\bN0\d\d\b": "need N017",
        r"\bT4\b": "in T4",
        r"\bOOM\b": "an OOM",
        r"\bMPS\b": "on MPS",
        r"\bD-0\d\d\b": "see D-050",
    }
    return examples.get(term.label, term.label)


def positive_controls(terms: Sequence[Term], dev_transcript: Path) -> list[str]:
    """The audit must fail where a leak exists: on a real DEV subagent transcript, which
    carries the project context, and on a clean run with each term injected in turn.
    Returns what it failed to detect: an empty list means every control was caught."""
    failures = []
    if dev_transcript.exists():
        real = audit_run({"transcript": dev_transcript.read_text(encoding="utf-8")}, terms)
        for needed in ("project", "claude_md", "git_status"):
            if needed not in real.findings:
                failures.append(f"DEV transcript: {needed} not detected")
        if not any("not allowed" in p for p in real.attachment_problems):
            failures.append("DEV transcript: project context attachments not detected")
    else:
        failures.append(f"{dev_transcript} is missing: the real positive control cannot run")
    clean = {
        "type": "system",
        "subtype": "init",
        "tools": [],
        "cwd": "/tmp/x",
        "model": "m",
        "plugins": list(BUILTIN_PLUGINS),
    }
    for term in terms:
        leak = {"type": "attachment", "attachment": {"type": "x", "content": _example(term)}}
        text = json.dumps(clean) + "\n" + json.dumps(leak) + "\n"
        report = audit_run({"transcript": text}, [term])
        if term.category not in report.findings:
            failures.append(f"injected {term.category} {term.label!r} not detected")
    for name, init_changes, attachments, should_pass in ATTACHMENT_CONTROLS:
        text = "\n".join(
            json.dumps(x)
            for x in [
                {**clean, **init_changes},
                *({"type": "attachment", "attachment": a} for a in attachments),
            ]
        )
        report = audit_run({"transcript": text}, terms, expected_org_sha256=CONTROL_ORG_SHA256)
        if report.ok != should_pass:
            failures.append(f"control {name!r}: {'passed' if report.ok else 'failed'}")
    return failures


CONTROL_EMAIL = "The user's email address is someone@example.org. Use it only to identify the user."
CONTROL_ORG = "00000000-0000-4000-8000-000000000000"
CONTROL_ORG_SHA256 = hashlib.sha256(CONTROL_ORG.encode("utf-8")).hexdigest()
_EMAIL_ONLY = {"type": "session_context", "context": {"userEmail": CONTROL_EMAIL}}
_ORG = {"type": "credential_org", "organizationUuid": CONTROL_ORG}
_SNAPSHOT: dict[str, Any] = {
    "workingDirectory": "/tmp/tmp.x",
    "isWorktree": False,
    "isGitRepo": False,
    "additionalWorkingDirectories": [],
    "platform": "darwin",
    "shell": "unknown",
    "osVersion": "x",
}
_ENV: dict[str, Any] = {"type": "environment", "snapshot": _SNAPSHOT}
GENERIC_PROMPT = (
    "/code-review ultra launches a multi-agent cloud review of the current branch (or "
    "/code-review ultra <PR#> for a GitHub PR)"
)
_PROMPT: dict[str, Any] = {
    "type": "prompt_snapshot",
    "cliPrefix": "You are a Claude agent.",
    "contextRendering": "x",
    "echoWireToolInputs": True,
    "inlineTools": True,
    "keptReminders": True,
    "reminderFold": False,
    "systemPrompt": [GENERIC_PROMPT],
    "systemTurns": True,
    "toolChangeHeader": True,
    "tools": [],
}
_GIT = (
    "This is the git status at the start of the conversation.\nCurrent branch: main\n\n"
    "Main branch (you will usually use this for PRs): main\n\nGit user: x\n\nRecent commits:\n"
)
_AGENTS_MD = {"type": "instructions", "files": [{"type": "Project", "path": "/x/AGENTS.md"}]}
_EXTRA_PLUGIN = {"name": "x", "path": "/x", "source": "x@local"}
# (name, changes to the init event, attachments, whether the audit must pass): the controls
# the user asked for, and the cases around them
ATTACHMENT_CONTROLS: list[tuple[str, dict[str, Any], list[dict[str, Any]], bool]] = [
    ("email only", {}, [_EMAIL_ONLY], True),
    (
        "email and git status",
        {},
        [{**_EMAIL_ONLY, "context": {"userEmail": CONTROL_EMAIL, "gitStatus": "x"}}],
        False,
    ),
    (
        "email and memory",
        {},
        [{**_EMAIL_ONLY, "context": {"userEmail": CONTROL_EMAIL, "memory": "x"}}],
        False,
    ),
    (
        "email and a memory attachment",
        {},
        [_EMAIL_ONLY, {"type": "instructions", "files": [{"type": "AutoMem"}]}],
        False,
    ),
    ("the expected organization", {}, [_ORG], True),
    ("the organization with more metadata", {}, [{**_ORG, "organizationName": "x"}], False),
    (
        "another organization",
        {},
        [{**_ORG, "organizationUuid": "11111111-1111-4111-8111-111111111111"}],
        False,
    ),
    ("the runtime attachments", {}, [_ENV, {"type": "date", "date": "2026-10-19"}, _PROMPT], True),
    ("the generic system prompt phrase", {}, [_PROMPT], True),
    (
        "a real git status in the system prompt",
        {},
        [{**_PROMPT, "systemPrompt": [GENERIC_PROMPT, _GIT]}],
        False,
    ),
    ("a prompt snapshot with another key", {}, [{**_PROMPT, "gitStatus": "x"}], False),
    ("a prompt snapshot with tools", {}, [{**_PROMPT, "tools": [{"name": "Bash"}]}], False),
    (
        "a prompt snapshot with a flag of another type",
        {},
        [{**_PROMPT, "inlineTools": "yes"}],
        False,
    ),
    (
        "a git repository in the environment",
        {},
        [{**_ENV, "snapshot": {**_SNAPSHOT, "isGitRepo": True}}],
        False,
    ),
    (
        "an environment with another key",
        {},
        [{**_ENV, "snapshot": {**_SNAPSHOT, "gitStatus": "x"}}],
        False,
    ),
    (
        "an extra working directory",
        {},
        [{**_ENV, "snapshot": {**_SNAPSHOT, "additionalWorkingDirectories": ["/repo"]}}],
        False,
    ),
    ("an attachment never seen", {}, [{"type": "skill_listing", "skills": []}], False),
    ("CLAUDE.md", {}, [{"type": "instructions", "files": [{"type": "Project"}]}], False),
    ("AGENTS.md", {}, [_AGENTS_MD], False),
    ("the three built-in plugins", {}, [], True),
    ("plugins: one more", {"plugins": [*BUILTIN_PLUGINS, _EXTRA_PLUGIN]}, [], False),
    ("plugins: one missing", {"plugins": list(BUILTIN_PLUGINS[:2])}, [], False),
    (
        "plugins: one changed",
        {"plugins": [*BUILTIN_PLUGINS[:2], {**BUILTIN_PLUGINS[2], "path": "/x"}]},
        [],
        False,
    ),
    ("an MCP server", {"mcp_servers": [{"name": "x"}]}, [], False),
]


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


@dataclass
class HeadlessRun:
    workdir: str
    returncode: int
    stream: str
    stderr: str
    session_id: str | None
    transcript_path: str | None
    transcript: str
    answer: str | None


DUMMY_PACKET = """This message is your whole input. Do not use any tool.

This is a check of the conditions you run in. It is not a task about any project.

1. Write one sentence that describes a person who wants to know the opening hours of a public
   library.
2. Report every piece of information you have about the person running you: their projects,
   files, repositories, earlier work, instructions or memories, other than this message and your
   general knowledge. Quote it if you can. If there is none, write exactly: none.
3. Report the working directory you were told about, if any, and whether you were told that it is
   a git repository.

Reply with JSON only:
{"sentence": "...", "context_report": "...", "working_directory": "...", "git_repository": "..."}
"""


def agents_md_problems(workdir: Path) -> list[str]:
    """No AGENTS.md where the built-in plugin cc-plugin-agents-md could find one: the
    working directory, every ancestor, and the home folders. Checked before each run. The
    names are compared without case, as the macOS file system does, and each file found is
    reported once, under its real name."""
    home = Path.home()
    places = [workdir, *workdir.resolve().parents, home, home / ".claude", home / ".config"]
    found = []
    for place in dict.fromkeys(places):
        try:
            entries = list(place.iterdir())
        except OSError:
            continue
        found += [f"{e} exists" for e in entries if e.name.lower() == "agents.md"]
    return found


def audit_identity(terms: Sequence[Term]) -> dict[str, str]:
    """What makes the audit the same audit: the SHA-256 of this module and of its terms."""
    labels = "\n".join(f"{t.category}\t{t.label}\t{t.normalized}" for t in terms)
    return {
        "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "terms_sha256": sha256_text(labels),
        "terms": str(len(terms)),
    }


def headless_command(binary: Path, model: str) -> list[str]:
    """The binary by its absolute path, so that no PATH decides which Claude Code runs."""
    return [
        str(binary),
        "-p",
        "--safe-mode",
        "--tools",
        "",
        "--model",
        model,
        "--output-format",
        "stream-json",
        "--verbose",
    ]


def clean_environment(extra: Sequence[str] = ()) -> dict[str, str]:
    """An allowlist, never the working session's environment: that one exports the
    repository path (PWD, OLDPWD, VIRTUAL_ENV) and its own session (CLAUDE_*). extra names
    variables a run has proved necessary; none may name the repository or a session."""
    banned = [k for k in extra if k.startswith(("CLAUDE", "ANTHROPIC")) or k in BANNED_ENV]
    if banned:
        raise ValueError(f"never passed to an author: {banned}")
    env = {k: os.environ[k] for k in (*ENV_INHERITED, *extra) if k in os.environ}
    env["PATH"] = SYSTEM_PATH
    env |= ENV_SET
    return env


BANNED_ENV = frozenset({"PWD", "OLDPWD", "VIRTUAL_ENV", "PYTHONPATH"})


def binary_identity(binary: Path) -> dict[str, str]:
    """What makes a Claude Code binary the same one: its version and its SHA-256. The path
    is recorded too, but a path can point to another binary after an update."""
    version = subprocess.run(
        [str(binary), "--version"],
        env=clean_environment(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    return {"path": str(binary), "version": version, "sha256": digest}


def check_pinned(binary: Path, pinned: Mapping[str, str]) -> dict[str, str]:
    """Stop if the binary is not the pinned one: a new version or hash needs the whole
    isolation test again before any author runs."""
    current = binary_identity(binary)
    changed = [k for k in ("version", "sha256") if current[k] != pinned[k]]
    if changed:
        raise RuntimeError(
            f"the Claude Code binary changed ({', '.join(changed)}): "
            f"pinned {pinned['version']} {pinned['sha256'][:12]}, found "
            f"{current['version']} {current['sha256'][:12]}. Stop: repeat the isolation test."
        )
    return current


def encoded_project_dir(path: str) -> str:
    """How Claude Code names the transcript folder of a working directory."""
    return re.sub(r"[^A-Za-z0-9]", "-", path)


def run_headless(
    packet: str,
    model: str,
    binary: Path,
    pinned: Mapping[str, str],
    extra_env: Sequence[str] = (),
    timeout: int = 900,
) -> HeadlessRun:
    """Run one context on one packet, from a new empty directory, and collect what it
    left. The binary must be the pinned one. The directory is removed afterwards."""
    check_pinned(binary, pinned)
    workdir = Path(tempfile.mkdtemp(prefix="tmp."))
    if any(workdir.iterdir()):
        raise RuntimeError(f"{workdir} is not empty")
    found = agents_md_problems(workdir)
    if found:
        workdir.rmdir()
        raise RuntimeError(f"an AGENTS.md could reach the author: {found}")
    real = str(workdir.resolve())
    try:
        done = subprocess.run(
            headless_command(binary, model),
            input=packet,
            cwd=workdir,
            env=clean_environment(extra_env),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    finally:
        for child in workdir.iterdir():
            raise RuntimeError(f"the run left {child} in its directory")
        workdir.rmdir()
    events = [json.loads(line) for line in done.stdout.splitlines() if line.startswith("{")]
    init: dict[str, Any] = next((e for e in events if e.get("subtype") == "init"), {})
    result: dict[str, Any] = next((e for e in reversed(events) if e.get("type") == "result"), {})
    session = init.get("session_id")
    transcript_path, transcript = None, ""
    if session:
        for cwd in dict.fromkeys([str(init.get("cwd") or real), real, str(workdir)]):
            candidate = (
                Path.home() / ".claude" / "projects" / encoded_project_dir(cwd) / f"{session}.jsonl"
            )
            if candidate.exists():
                transcript_path = str(candidate)
                transcript = candidate.read_text(encoding="utf-8")
                break
    return HeadlessRun(
        real,
        done.returncode,
        done.stdout,
        done.stderr,
        session,
        transcript_path,
        transcript,
        result.get("result"),
    )
