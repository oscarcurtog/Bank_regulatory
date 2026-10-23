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
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
    # Claude Code 2.1.301 emits either the initial date or a date-rollover reminder.
    # The latter is accepted only after the launcher proves the observed local date change.
    "date": [{"date": str}, {"date": str, "changed": bool}],
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
            out.extend(_strings(_load_json(line)))
        except ValueError:
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
    date_validated: bool = False
    cwd_validated: bool = False


def _attachment_kinds(records: Iterable[Mapping[str, Any]]) -> list[str]:
    kinds = []
    for r in records:
        a = r.get("attachment")
        if isinstance(a, dict):
            kind = str(a.get("type"))
            raw_files = a.get("files")
            files = raw_files if isinstance(raw_files, list) else []
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


def _system_timezone() -> str:
    """The child inherits the host timezone, not the controller's possible TZ override."""
    localtime = Path("/etc/localtime").resolve()
    marker = "/zoneinfo/"
    resolved = localtime.as_posix()
    if marker not in resolved:
        raise RuntimeError("the host timezone cannot be identified from /etc/localtime")
    zone = resolved.split(marker, 1)[1]
    try:
        ZoneInfo(zone)
    except ZoneInfoNotFoundError as error:
        raise RuntimeError("the host timezone is not an installed IANA zone") from error
    return zone


def runtime_clock_observation() -> dict[str, str]:
    """A launcher-owned wall-clock observation, with a date reproducible from its UTC instant."""
    now = datetime.now(UTC)
    zone = _system_timezone()
    local = now.astimezone(ZoneInfo(zone))
    return {
        "utc": now.isoformat(timespec="microseconds"),
        "local_date": local.date().isoformat(),
        "timezone": zone,
    }


def _clock_observation_date(value: Any) -> tuple[date | None, datetime | None, str | None]:
    """Validate one persisted clock observation without trusting its claimed local date."""
    if not isinstance(value, dict) or set(value) != {"utc", "local_date", "timezone"}:
        return None, None, None
    try:
        instant = datetime.fromisoformat(value["utc"])
        claimed = date.fromisoformat(value["local_date"])
        zone_name = value["timezone"]
        if not isinstance(zone_name, str) or not zone_name:
            return None, None, None
        zone = ZoneInfo(zone_name)
    except (TypeError, ValueError, ZoneInfoNotFoundError):
        return None, None, None
    if instant.tzinfo is None or instant.utcoffset() != timedelta(0):
        return None, None, None
    if instant.astimezone(zone).date() != claimed or claimed.isoformat() != value["local_date"]:
        return None, None, None
    return claimed, instant, zone_name


def _runtime_context(
    value: Any,
) -> tuple[list[str], str | None, tuple[date, ...], bool]:
    """Check the launcher binding and the bounded local-date window used by Gate 1."""
    if not isinstance(value, dict) or set(value) != {
        "launcher_workdir",
        "clock_start",
        "clock_end",
    }:
        return ["launcher provenance is missing or has an unknown shape"], None, (), False
    raw_cwd = value.get("launcher_workdir")
    if not isinstance(raw_cwd, str) or not Path(raw_cwd).is_absolute():
        return ["launcher working-directory provenance is invalid"], None, (), False
    try:
        expected_cwd = str(Path(raw_cwd).resolve(strict=False))
    except (OSError, RuntimeError):
        return ["launcher working-directory provenance cannot be resolved"], None, (), False
    if raw_cwd != expected_cwd:
        return ["launcher working-directory provenance is not canonical"], None, (), False

    start_date, start_at, start_zone = _clock_observation_date(value.get("clock_start"))
    if start_date is None or start_at is None or start_zone is None:
        return ["launcher start-clock provenance is invalid"], expected_cwd, (), False
    raw_end = value.get("clock_end")
    if raw_end is None:
        return [], expected_cwd, (start_date,), False
    end_date, end_at, end_zone = _clock_observation_date(raw_end)
    if end_date is None or end_at is None or end_zone is None:
        return ["launcher end-clock provenance is invalid"], expected_cwd, (), False
    if end_zone != start_zone or end_at < start_at:
        return (
            ["launcher clock window changed timezone or moved backwards"],
            expected_cwd,
            (),
            False,
        )
    day_delta = (end_date - start_date).days
    if day_delta < 0 or day_delta > 1:
        return ["launcher clock window spans an unexplained date change"], expected_cwd, (), False
    dates = (start_date,) if day_delta == 0 else (start_date, end_date)
    return [], expected_cwd, dates, True


def attachment_problems(
    records: Iterable[Mapping[str, Any]],
    expected_cwd: str | None = None,
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
            if r.get("type") == "attachment":
                problems.append("attachment body is not an object")
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
            if expected_cwd is not None and snap["workingDirectory"] != expected_cwd:
                problems.append("environment: workingDirectory differs from the launcher path")
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


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _load_json(text: str) -> Any:
    # A normal json.loads silently discards earlier values of duplicate keys. None of
    # the audit's structural or content checks may lose those values.
    return json.loads(text, object_pairs_hook=_unique_keys)


def parse_records(texts: Mapping[str, str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in ("transcript", "stream"):
        for number, line in enumerate(texts.get(name, "").splitlines(), 1):
            if line.strip():
                try:
                    record = _load_json(line)
                    if not isinstance(record, dict):
                        raise ValueError("JSON record is not an object")
                    records.append(record)
                except ValueError as error:
                    records.append({"unparsed": line, "_parse_error": f"{name}:{number}: {error}"})
    return records


def _date_attachments(text: str) -> list[tuple[Any, Any]]:
    found: list[tuple[Any, Any]] = []
    for line in text.splitlines():
        try:
            record = _load_json(line)
        except ValueError:
            continue
        attachment = record.get("attachment") if isinstance(record, dict) else None
        if not isinstance(attachment, dict) or attachment.get("type") != "date":
            continue
        value = attachment.get("date")
        changed = attachment.get("changed") if "changed" in attachment else None
        found.append((value, changed))
    return found


def _date_problems(
    texts: Mapping[str, str], runtime_context: Any
) -> tuple[list[str], str | None, bool]:
    sequences = {
        source: _date_attachments(texts.get(source, "")) for source in ("transcript", "stream")
    }
    nonempty = [sequence for sequence in sequences.values() if sequence]
    if runtime_context is None:
        if nonempty:
            return ["date attachments have no launcher clock provenance"], None, False
        return [], None, False

    problems, expected_cwd, observed_dates, has_end = _runtime_context(runtime_context)
    if problems:
        return problems, expected_cwd, False
    if not nonempty:
        return ["the pinned Claude run has no date attachment"], expected_cwd, False
    allowed = set(observed_dates)
    end_date = observed_dates[-1]

    for record in parse_records(texts):
        attachment = record.get("attachment")
        if not isinstance(attachment, dict) or attachment.get("type") != "date":
            continue
        problems += _matches("date", attachment, RUNTIME_ATTACHMENTS["date"])
        if "rendered" in record:
            content = f"<system-reminder>\n{rendered_date(attachment)}\n</system-reminder>"
            expected = [{"content": content}]
            if record["rendered"] != expected:
                problems.append("date attachment has an unknown rendered template")

    for sequence in nonempty:
        previous: date | None = None
        for raw_date, changed in sequence:
            if not isinstance(raw_date, str):
                problems.append("date attachment value is not text")
                continue
            try:
                current = date.fromisoformat(raw_date)
            except ValueError:
                problems.append("date attachment is not a calendar date")
                continue
            if current.isoformat() != raw_date:
                problems.append("date attachment is not in canonical YYYY-MM-DD form")
                continue
            if current not in allowed:
                problems.append("date attachment is outside the launcher's observed date window")
            if changed is not None and type(changed) is not bool:
                problems.append("date attachment changed flag is not boolean")
            elif changed is False:
                problems.append("the pinned runtime does not emit changed=false")
            elif changed is True:
                if not has_end:
                    problems.append("a rollover attachment has no launcher end-clock proof")
                if previous is None or current == previous or current != end_date:
                    problems.append("date rollover is not supported by the observed clock sequence")
            elif previous is not None and current != previous:
                problems.append("date changed without the pinned rollover marker")
            previous = current

    return problems, expected_cwd, not problems


def rendered_date(attachment: Mapping[str, Any]) -> str:
    """The two exact strings verified in the SHA-pinned Claude Code 2.1.301 binary."""
    if attachment.get("changed") is True:
        return (
            f"The date has changed. Today's date is now {attachment.get('date')}. "
            "No need to announce the new date — the user's own clock shows it."
        )
    return f"Today's date is {attachment.get('date')}."


_ENV_CWD = re.compile(r"^ - Primary working directory: ([^\r\n]*)$", re.M)
_CWD_RECORD_TYPES = frozenset({"system", "attachment", "user", "assistant"})


def _environment_texts(record: Mapping[str, Any]) -> Iterable[str]:
    """Only runtime-owned environment renderings, never a user's or model's text."""
    attachment = record.get("attachment")
    if isinstance(attachment, dict):
        if attachment.get("type") == "environment":
            renderings = record.get("rendered")
            for rendered in renderings if isinstance(renderings, list) else []:
                if isinstance(rendered, dict) and isinstance(rendered.get("content"), str):
                    yield rendered["content"]
        elif attachment.get("type") == "prompt_snapshot":
            prompts = attachment.get("systemPrompt")
            for prompt in prompts if isinstance(prompts, list) else []:
                if isinstance(prompt, str):
                    yield prompt
    message = record.get("message")
    if (
        record.get("type") == "api-request-blob"
        and isinstance(message, dict)
        and message.get("role") == "system"
    ):
        content = message.get("content")
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    yield block["text"]


def _cwd_problems(records: Sequence[Mapping[str, Any]], expected_cwd: str | None) -> list[str]:
    if expected_cwd is None:
        return []
    problems = []
    for record in records:
        kind = record.get("type")
        for key in ("cwd", "original_cwd"):
            if key not in record:
                continue
            known = isinstance(kind, str) and kind in _CWD_RECORD_TYPES
            if key == "original_cwd":
                known = kind == "system" and record.get("subtype") == "init"
            if not known or record[key] != expected_cwd:
                problems.append(f"{kind}: {key} is not the exact launcher-created path")
        attachment = record.get("attachment")
        if (
            isinstance(attachment, dict)
            and attachment.get("type") == "environment"
            and "rendered" in record
        ):
            renderings = record["rendered"]
            if (
                not isinstance(renderings, list)
                or not renderings
                or any(
                    not isinstance(item, dict)
                    or set(item) != {"content"}
                    or not isinstance(item["content"], str)
                    or len(list(_ENV_CWD.finditer(item["content"]))) != 1
                    for item in renderings
                )
            ):
                problems.append("rendered environment has an unknown working-directory format")
        for text in _environment_texts(record):
            if any(match[1] != expected_cwd for match in _ENV_CWD.finditer(text)):
                problems.append("rendered environment: working directory differs from launcher")
    return problems


def _normalize_environment(value: Any, expected_cwd: str) -> Any:
    if isinstance(value, str):
        return _ENV_CWD.sub(
            lambda match: (
                " - Primary working directory: <verified-launcher-cwd>"
                if match[1] == expected_cwd
                else match[0]
            ),
            value,
        )
    if isinstance(value, list):
        return [_normalize_environment(item, expected_cwd) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_environment(item, expected_cwd) for key, item in value.items()}
    return value


def audit_content_copy(text: str, expected_cwd: str | None, normalize_date: bool) -> str:
    """Normalize only launcher-proven fields in an in-memory content-audit copy."""
    output: list[str] = []
    for line in text.splitlines():
        try:
            record = _load_json(line)
        except ValueError:
            output.append(line)
            continue
        if not isinstance(record, dict):
            output.append(line)
            continue
        if expected_cwd is not None:
            kind = record.get("type")
            if (
                isinstance(kind, str)
                and kind in _CWD_RECORD_TYPES
                and record.get("cwd") == expected_cwd
            ):
                record["cwd"] = "<verified-launcher-cwd>"
            if (
                record.get("type") == "system"
                and record.get("subtype") == "init"
                and record.get("original_cwd") == expected_cwd
            ):
                record["original_cwd"] = "<verified-launcher-cwd>"
        if expected_cwd is not None and record.get("type") == "api-request-blob":
            message = record.get("message")
            if isinstance(message, dict) and message.get("role") == "system":
                message["content"] = _normalize_environment(message.get("content"), expected_cwd)
        attachment = record.get("attachment")
        if isinstance(attachment, dict):
            kind = attachment.get("type")
            if expected_cwd is not None and kind == "environment":
                snapshot = attachment.get("snapshot")
                if isinstance(snapshot, dict) and snapshot.get("workingDirectory") == expected_cwd:
                    snapshot["workingDirectory"] = "<verified-launcher-cwd>"
                if "rendered" in record:
                    record["rendered"] = _normalize_environment(record["rendered"], expected_cwd)
            if expected_cwd is not None and kind == "prompt_snapshot":
                attachment["systemPrompt"] = _normalize_environment(
                    attachment.get("systemPrompt"), expected_cwd
                )
            if (
                normalize_date
                and kind == "date"
                and set(attachment) in ({"type", "date"}, {"type", "date", "changed"})
            ):
                # changed is a runtime control flag, retained in the raw artifact and checked
                # above; it is excluded only from this copy used to scan for leaked content.
                attachment.pop("changed", None)
        output.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return "\n".join(output)


def audit_run(
    texts: Mapping[str, str],
    terms: Sequence[Term],
    expected_cwd: str | None = None,
    expected_version: str | None = None,
    expected_org_sha256: str | None = None,
    allowed_plugins: Sequence[Mapping[str, str]] = BUILTIN_PLUGINS,
    runtime_context: Mapping[str, Any] | None = None,
) -> IsolationReport:
    """texts maps a source ("transcript", "stream", "stderr") to its raw content. The
    transcript and the stream are JSON Lines; the stream's init event says which tools,
    directory and model the run had."""
    records = parse_records(texts)
    parse_problems = [str(record["_parse_error"]) for record in records if "_parse_error" in record]
    attachments = _attachment_kinds(records)
    date_findings, context_cwd, dates_valid = _date_problems(texts, runtime_context)
    binding_valid = runtime_context is None or not _runtime_context(runtime_context)[0]
    if context_cwd is not None:
        if expected_cwd is not None and expected_cwd != context_cwd:
            date_findings.append(
                "launcher working-directory provenance differs from the expected path"
            )
            binding_valid = False
            dates_valid = False
        else:
            expected_cwd = context_cwd
    attached, identity = attachment_problems(records, expected_cwd, expected_org_sha256)
    cwd_findings = _cwd_problems(records, expected_cwd)
    cwd_valid = (
        expected_cwd is not None
        and binding_valid
        and not (parse_problems or cwd_findings or attached)
    )
    attached += date_findings
    attached += cwd_findings
    dates_valid = dates_valid and not parse_problems
    init = next(
        (r for r in records if r.get("type") == "system" and r.get("subtype") == "init"), {}
    )
    cwd_valid = cwd_valid and init.get("cwd") == expected_cwd
    tools = init.get("tools")
    cwd = init.get("cwd")
    model = init.get("model")
    findings: dict[str, list[str]] = {}
    for source in ("transcript", "stream", "stderr"):
        text = texts.get(source, "")
        if not isinstance(text, str):
            continue
        audit_text = audit_content_copy(text, expected_cwd if cwd_valid else None, dates_valid)
        decoded = decoded_text(audit_text)
        flat = normalize(decoded)
        for term in terms:
            if term.found_in(decoded, flat) or term.found_in(audit_text, normalize(audit_text)):
                findings.setdefault(term.category, []).append(f"{source}: {term.label}")
    problems = [*parse_problems, *attached]
    init_fields = ("tools", "mcp_servers", "cwd", "plugins", "claude_code_version")
    if any(
        any(record.get(key) != init.get(key) for key in init_fields)
        for record in records
        if record.get("type") == "system" and record.get("subtype") == "init"
    ):
        problems.append("the runtime artifacts contain conflicting init events")
    plugins = init.get("plugins") or []
    if (
        not isinstance(plugins, list)
        or any(not isinstance(plugin, dict) for plugin in plugins)
        or sorted(json.dumps(plugin, sort_keys=True) for plugin in plugins)
        != sorted(json.dumps(plugin, sort_keys=True) for plugin in allowed_plugins)
    ):
        problems.append("plugins are not exactly the built-ins")
    if not init:
        problems.append("no init event: the run cannot be checked")
    if tools:
        problems.append(f"tools offered: {tools}")
    if init.get("mcp_servers"):
        problems.append(f"MCP servers: {init['mcp_servers']}")
    if expected_cwd is not None and cwd != expected_cwd:
        problems.append("working directory differs from the exact launcher-created path")
    if expected_version is not None and init.get(
        "claude_code_version"
    ) != expected_version.removesuffix(" (Claude Code)"):
        problems.append(
            f"the run says Claude Code {init.get('claude_code_version')}, pinned {expected_version}"
        )
    if findings:
        problems.append(f"forbidden strings: {sum(len(v) for v in findings.values())}")
    return IsolationReport(
        not problems,
        attachments,
        attached,
        identity,
        tools,
        cwd,
        model,
        findings,
        problems,
        date_validated=dates_valid,
        cwd_validated=cwd_valid,
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
        "cwd": str(Path("/tmp/tmp.x").resolve(strict=False)),
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
        runtime_context = attachment_control_context(attachments)
        expected_cwd = runtime_context["launcher_workdir"] if runtime_context else None
        report = audit_run(
            {"transcript": text},
            terms,
            expected_cwd=expected_cwd,
            expected_org_sha256=CONTROL_ORG_SHA256,
            runtime_context=runtime_context,
        )
        if report.ok != should_pass:
            failures.append(f"control {name!r}: {'passed' if report.ok else 'failed'}")
    for name, transcript, runtime_context, should_pass in PROVENANCE_CONTROLS:
        report = audit_run(
            {"transcript": transcript},
            terms,
            expected_cwd=runtime_context["launcher_workdir"] if runtime_context else None,
            runtime_context=runtime_context,
        )
        if report.ok != should_pass:
            failures.append(f"control {name!r}: {'passed' if report.ok else 'failed'}")
    return failures


def attachment_control_context(attachments: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """A fixed, independently reproducible clock window for the recorded date controls."""
    if not any(a.get("type") == "date" for a in attachments):
        return None
    return {
        "launcher_workdir": str(Path("/tmp/tmp.x").resolve(strict=False)),
        "clock_start": {
            "utc": "2026-10-19T21:59:00+00:00",
            "local_date": "2026-10-19",
            "timezone": "Europe/Madrid",
        },
        "clock_end": {
            "utc": "2026-10-19T22:01:00+00:00",
            "local_date": "2026-10-20",
            "timezone": "Europe/Madrid",
        },
    }


CONTROL_EMAIL = "The user's email address is someone@example.org. Use it only to identify the user."
CONTROL_ORG = "00000000-0000-4000-8000-000000000000"
CONTROL_ORG_SHA256 = hashlib.sha256(CONTROL_ORG.encode("utf-8")).hexdigest()
_EMAIL_ONLY = {"type": "session_context", "context": {"userEmail": CONTROL_EMAIL}}
_ORG = {"type": "credential_org", "organizationUuid": CONTROL_ORG}
_SNAPSHOT: dict[str, Any] = {
    "workingDirectory": str(Path("/tmp/tmp.x").resolve(strict=False)),
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
    (
        "a clock-proven date rollover",
        {},
        [
            {"type": "date", "date": "2026-10-19"},
            {"type": "date", "date": "2026-10-20", "changed": True},
        ],
        True,
    ),
    (
        "a rollover without a previous date",
        {},
        [{"type": "date", "date": "2026-10-20", "changed": True}],
        False,
    ),
    (
        "a false rollover flag",
        {},
        [{"type": "date", "date": "2026-10-19", "changed": False}],
        False,
    ),
    ("a date outside the clock window", {}, [{"type": "date", "date": "2026-10-18"}], False),
    (
        "a date with an unknown key",
        {},
        [{"type": "date", "date": "2026-10-19", "retrieval_results": []}],
        False,
    ),
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

# The launcher's provenance (D-064), on transcripts in the shape the pinned runtime writes them
# (Gate 1 run 4): a cwd on every record, the init's original_cwd, the rendered environment and
# date. The directory's random suffix holds a static term, as in run 3: the verified cwd must
# pass, and the same path anywhere else, or any other deviation, must not.
CONTROL_CWD = str(Path("/tmp/tmp.k_ndcg4x").resolve(strict=False))
_LEAK = "The answer comes from pipeline E."
_DAY = {"type": "date", "date": "2026-10-19"}
_ROLLOVER = {"type": "date", "date": "2026-10-20", "changed": True}


def _clock(utc: str, local_date: str, zone: str = "Europe/Madrid") -> dict[str, str]:
    return {"utc": utc, "local_date": local_date, "timezone": zone}


_SAME_DAY = (
    _clock("2026-10-19T10:00:00+00:00", "2026-10-19"),
    _clock("2026-10-19T10:05:00+00:00", "2026-10-19"),
)
_MIDNIGHT = (
    _clock("2026-10-19T21:59:00+00:00", "2026-10-19"),
    _clock("2026-10-19T22:01:00+00:00", "2026-10-20"),
)


def _provenance(
    start: dict[str, str] | None, end: dict[str, str] | None, cwd: str = CONTROL_CWD
) -> dict[str, Any]:
    return {"launcher_workdir": cwd, "clock_start": start, "clock_end": end}


def _control_run(
    dates: Sequence[Mapping[str, Any]],
    reply: str = "{}",
    moved: Mapping[str, str] | None = None,
    extra: Sequence[str] = (),
) -> str:
    """One control's transcript: every runtime position of the cwd holds CONTROL_CWD unless
    moved gives another path for one of them ("init", "record" or "environment"); the extra
    lines are appended as they are."""
    at = {"init": CONTROL_CWD, "record": CONTROL_CWD, "environment": CONTROL_CWD, **(moved or {})}
    environment = (
        "# Environment\nYou have been invoked in the following environment: \n"
        f" - Primary working directory: {at['environment']}\n - Is a git repository: false"
    )
    records: list[dict[str, Any]] = [
        {
            "type": "system",
            "subtype": "init",
            "tools": [],
            "cwd": at["init"],
            "original_cwd": at["init"],
            "model": "m",
            "plugins": list(BUILTIN_PLUGINS),
        },
        {
            "type": "attachment",
            "cwd": at["record"],
            "attachment": {
                "type": "environment",
                "snapshot": {**_SNAPSHOT, "workingDirectory": at["environment"]},
            },
            "rendered": [{"content": f"<system-reminder>\n{environment}\n</system-reminder>"}],
        },
        *(
            {
                "type": "attachment",
                "cwd": at["record"],
                "attachment": dict(date),
                "rendered": [
                    {"content": f"<system-reminder>\n{rendered_date(date)}\n</system-reminder>"}
                ],
            }
            for date in dates
        ),
        {
            "type": "user",
            "cwd": at["record"],
            "message": {"role": "user", "content": "A synthetic packet."},
        },
        {
            "type": "assistant",
            "cwd": at["record"],
            "message": {"content": [{"type": "text", "text": reply}]},
        },
    ]
    return "\n".join([*map(json.dumps, records), *extra])


_CWD_JSON = json.dumps(CONTROL_CWD)
# (name, transcript, the launcher's provenance, whether the audit must pass)
PROVENANCE_CONTROLS: list[tuple[str, str, dict[str, Any] | None, bool]] = [
    (
        "a verified cwd whose random suffix holds a forbidden term",
        _control_run([_DAY]),
        _provenance(*_SAME_DAY),
        True,
    ),
    (
        "a verified midnight rollover",
        _control_run([_DAY, _ROLLOVER]),
        _provenance(*_MIDNIGHT),
        True,
    ),
    (
        "the verified cwd copied into a model reply",
        _control_run([_DAY], reply=f"I work in {CONTROL_CWD}."),
        _provenance(*_SAME_DAY),
        False,
    ),
    (
        "a leak beside a verified cwd",
        _control_run([_DAY], reply=_LEAK),
        _provenance(*_SAME_DAY),
        False,
    ),
    (
        "a leak beside a verified rollover",
        _control_run([_DAY, _ROLLOVER], reply=_LEAK),
        _provenance(*_MIDNIGHT),
        False,
    ),
    (
        "a sibling directory",
        _control_run([_DAY], moved={"init": CONTROL_CWD + "-sibling"}),
        _provenance(*_SAME_DAY),
        False,
    ),
    (
        "a directory under the launcher's",
        _control_run([_DAY], moved={"record": CONTROL_CWD + "/sub"}),
        _provenance(*_SAME_DAY),
        False,
    ),
    (
        "the launcher directory's parent",
        _control_run([_DAY], moved={"environment": str(Path(CONTROL_CWD).parent)}),
        _provenance(*_SAME_DAY),
        False,
    ),
    (
        "provenance for another directory",
        _control_run([_DAY]),
        _provenance(*_SAME_DAY, cwd=CONTROL_CWD + "-sibling"),
        False,
    ),
    (
        "a launcher directory that is not canonical",
        _control_run([_DAY]),
        _provenance(*_SAME_DAY, cwd=str(Path(CONTROL_CWD).parent / ".." / "tmp.k_ndcg4x")),
        False,
    ),
    (
        "a relative launcher directory",
        _control_run([_DAY]),
        _provenance(*_SAME_DAY, cwd="tmp/tmp.k_ndcg4x"),
        False,
    ),
    (
        "a cwd on a record kind never seen",
        _control_run([_DAY], extra=[json.dumps({"type": "progress", "cwd": CONTROL_CWD})]),
        _provenance(*_SAME_DAY),
        False,
    ),
    (
        "a rollover flag as text",
        _control_run([_DAY, {**_ROLLOVER, "changed": "true"}]),
        _provenance(*_MIDNIGHT),
        False,
    ),
    (
        "a rollover flag as a number",
        _control_run([_DAY, {**_ROLLOVER, "changed": 1}]),
        _provenance(*_MIDNIGHT),
        False,
    ),
    (
        "a null rollover flag",
        _control_run([_DAY, {**_ROLLOVER, "changed": None}]),
        _provenance(*_MIDNIGHT),
        False,
    ),
    (
        "an impossible calendar date",
        _control_run([{"type": "date", "date": "2026-02-30"}]),
        _provenance(*_SAME_DAY),
        False,
    ),
    (
        "a date in another form",
        _control_run([{"type": "date", "date": "20261019"}]),
        _provenance(*_SAME_DAY),
        False,
    ),
    (
        "a date going back",
        _control_run([{"type": "date", "date": "2026-10-20"}, {**_DAY, "changed": True}]),
        _provenance(*_MIDNIGHT),
        False,
    ),
    (
        "a date change without the rollover flag",
        _control_run([_DAY, {"type": "date", "date": "2026-10-20"}]),
        _provenance(*_MIDNIGHT),
        False,
    ),
    (
        "two rollovers",
        _control_run([_DAY, _ROLLOVER, _ROLLOVER]),
        _provenance(*_MIDNIGHT),
        False,
    ),
    (
        "a rollover inside one day",
        _control_run([_DAY, _ROLLOVER]),
        _provenance(*_SAME_DAY),
        False,
    ),
    (
        "a rollover without the launcher's end clock",
        _control_run([_DAY, _ROLLOVER]),
        _provenance(_MIDNIGHT[0], None),
        False,
    ),
    (
        "a rollover rendered with another text",
        _control_run(
            [_DAY],
            extra=[
                json.dumps(
                    {
                        "type": "attachment",
                        "cwd": CONTROL_CWD,
                        "attachment": _ROLLOVER,
                        "rendered": [
                            {"content": "<system-reminder>\nIt is late.\n</system-reminder>"}
                        ],
                    }
                )
            ],
        ),
        _provenance(*_MIDNIGHT),
        False,
    ),
    (
        "a duplicate key in a record",
        _control_run(
            [_DAY],
            extra=[
                '{"type": "attachment", "cwd": '
                + _CWD_JSON
                + ', "attachment": {"type": "date", "date": "2026-10-19", "date": "2026-10-18"}}'
            ],
        ),
        _provenance(*_SAME_DAY),
        False,
    ),
    (
        "a duplicate key nested in a message",
        _control_run(
            [_DAY],
            extra=[
                '{"type": "assistant", "cwd": '
                + _CWD_JSON
                + ', "message": {"content": '
                + json.dumps(_LEAK)
                + ', "content": "{}"}}'
            ],
        ),
        _provenance(*_SAME_DAY),
        False,
    ),
    (
        "a cut record",
        _control_run([_DAY], extra=['{"type": "assistant", "message": {"content": "{}"']),
        _provenance(*_SAME_DAY),
        False,
    ),
    ("no launcher provenance", _control_run([_DAY]), None, False),
    (
        "a launcher clock that changes timezone",
        _control_run([_DAY]),
        _provenance(_SAME_DAY[0], _clock("2026-10-19T10:05:00+00:00", "2026-10-19", "UTC")),
        False,
    ),
    (
        "a launcher clock that moves backwards",
        _control_run([_DAY]),
        _provenance(_SAME_DAY[1], _SAME_DAY[0]),
        False,
    ),
    (
        "a launcher window longer than a day",
        _control_run([_DAY]),
        _provenance(_SAME_DAY[0], _clock("2026-10-21T10:00:00+00:00", "2026-10-21")),
        False,
    ),
    (
        "a local date its instant contradicts",
        _control_run([_DAY]),
        _provenance(_clock("2026-10-19T10:00:00+00:00", "2026-10-20"), _SAME_DAY[1]),
        False,
    ),
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
    runtime_context: dict[str, Any] = field(default_factory=dict)


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
    clock_start = runtime_clock_observation()
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
        clock_end = runtime_clock_observation()
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
        {
            "launcher_workdir": real,
            "clock_start": clock_start,
            "clock_end": clock_end,
        },
    )
