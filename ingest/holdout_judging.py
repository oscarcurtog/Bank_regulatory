"""From the judges' answers to TEST's qrels and adjudication record (S3, D-054), and the
audit of every isolated context, read from its own transcript.

The logic is DEV batch 2's: judge A's file is checked before judge B starts; every
difference between A and B goes to judge C; the final label of each candidate is the
agreed one or C's; the adjudication record keeps the three. One rule is new: the
gap_note of a partial or none need is the one written by the judge whose answerability
became final, never a text written afterwards by whoever orchestrates.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

from ingest import benchmark as bm

# ---------------------------------------------------------------------------
# judge A's file
# ---------------------------------------------------------------------------


def judge_a_problems(
    data: Mapping[str, Any], texts: Mapping[bm.NodeKey, str], excluded: frozenset[str]
) -> list[str]:
    """What makes judge A's file unusable before judge B ever sees its candidates."""
    nid = data.get("need_id", "?")
    problems: list[str] = []
    seen: set[str] = set()
    graded: dict[str, Mapping[str, Any]] = {}
    for c in data.get("candidates", []):
        ref = c["node"]
        act, _, path = ref.partition(":")
        if (act, path) not in texts:
            problems.append(f"{nid} {ref}: no such node with own text")
        if act == "PSD2" and path.split(".")[0] in excluded:
            problems.append(f"{nid} {ref}: excluded PSD2 article")
        for s in [*c.get("scope", []), *c.get("scope_basis", [])]:
            if tuple(s.partition(":")[::2]) not in texts:
                problems.append(f"{nid} {ref}: scope node {s} does not exist")
        if ref in seen:
            problems.append(f"{nid} {ref}: listed twice")
        seen.add(ref)
        if c.get("grade") == 0 and not c.get("scope_basis"):
            problems.append(f"{nid} {ref}: grade 0 without scope_basis")
        if c.get("ambiguous") and c.get("grade") is None:
            problems.append(f"{nid} {ref}: ambiguous without a best grade")
        if not str(c.get("rationale", "")).strip():
            problems.append(f"{nid} {ref}: empty rationale")
        graded[ref] = c
    answerability = data.get("answerability")
    relevant = [
        r for r, c in graded.items() if (c.get("grade") or 0) >= 2 and not c.get("ambiguous")
    ]
    top = [r for r, c in graded.items() if c.get("grade") == 3 and not c.get("ambiguous")]
    if answerability == "complete" and not top:
        problems.append(f"{nid}: complete without a non-ambiguous grade 3")
    if answerability == "none" and relevant:
        problems.append(f"{nid}: none, but relevant nodes {relevant}")
    if answerability == "partial" and not relevant:
        problems.append(f"{nid}: partial without any relevant node")
    if answerability != "complete" and not str(data.get("gap_note") or "").strip():
        problems.append(f"{nid}: {answerability} without gap_note")
    return problems


def a_judgments(data: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Judge A's labels in the shape judge C's packet shows them."""
    out: dict[str, dict[str, Any]] = {
        c["node"]: {
            "grade": c.get("grade"),
            "ambiguous": bool(c.get("ambiguous")),
            "scope_basis": c.get("scope_basis", []),
            "rationale": c["rationale"],
        }
        for c in data["candidates"]
    }
    out[bm.ANSWERABILITY_TARGET] = {
        "answerability": data["answerability"],
        "rationale": data.get("answerability_reason", ""),
    }
    return out


# The broker operations whose pages hold a node's whole own text (ingest/holdout_broker.py): a
# `search` page shows a snippet of each node it names, and `titles` names no node
FULL_TEXT_OPS = ("article", "nodes")


def exposure(log: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every node one search received in full: the ids on the pages the broker returned for its
    `article` and `nodes` requests (each log entry's own record of its page), sorted. Mechanical:
    read from the broker's log, never from what the model wrote, and the same on every replay."""
    return sorted({n for e in log if e["op"] in FULL_TEXT_OPS for n in e.get("nodes", [])})


def poolable(node: str, texts: Mapping[bm.NodeKey, str], excluded: frozenset[str]) -> bool:
    """A node received in full that can be judged: one of the corpus with its own, non-empty
    text (a structural parent whose text is all in its children has nothing to judge), outside
    the PSD2 articles excluded from TEST."""
    act, _, path = node.partition(":")
    own = texts.get((act, path))
    return bool(own and own.strip()) and not (act == "PSD2" and path.split(".")[0] in excluded)


@dataclass(frozen=True)
class Discovery:
    """The evidence of one need after its three blind searches (config/holdout/
    evidence-discovery.md).

    listed: every node a search listed, once (by its exact ACT:path), in the order first listed;
    the caller puts them in document order.
    first: each listed node's first judgment, the one the adjudication compares with judge B:
    the label of the first search, in the order search_1, search_2, search_3, that listed it;
    and the answerability, its reason and the gap note, search_1's. A later search's label on
    a node an earlier one listed, and its answerability, are provenance only: no vote.
    scope: the provisions that fix each listed node's scope, from the search whose label is
    first.
    listed_by: which searches listed each node.
    exposed_by: which searches received each node in full (an `article` or `nodes` page).
    pool: the exposure pool: every node a search received in full or listed, that can be
    judged (poolable). No model decides that a node enters it.
    unlisted: the pool's nodes no search listed: each has no first judgment (an abstention),
    and is judged explicitly by judge B's pool variant.
    answerabilities: each search's answerability, as provenance."""

    listed: list[str]
    first: dict[str, dict[str, Any]]
    scope: dict[str, list[str]]
    listed_by: dict[str, list[str]]
    exposed_by: dict[str, list[str]]
    pool: list[str]
    unlisted: list[str]
    answerabilities: dict[str, str]

    def record(self) -> dict[str, Any]:
        """Counts, and which searches listed and received each node."""
        searches = list(self.answerabilities)
        return {
            "searches": searches,
            "listed": {s: sum(s in v for v in self.listed_by.values()) for s in searches},
            "exposed": {s: sum(s in v for v in self.exposed_by.values()) for s in searches},
            "listed_union": len(self.listed),
            "listed_by_one": sum(len(v) == 1 for v in self.listed_by.values()),
            "pool": len(self.pool),
            "unlisted": len(self.unlisted),
            "answerabilities_agree": len(set(self.answerabilities.values())) == 1,
            "listed_by": {n: list(v) for n, v in sorted(self.listed_by.items())},
            "exposed_by": {n: list(v) for n, v in sorted(self.exposed_by.items())},
        }


def exposure_pool(
    files: Mapping[str, Mapping[str, Any]],
    exposures: Mapping[str, Sequence[str]],
    texts: Mapping[bm.NodeKey, str],
    excluded: frozenset[str],
) -> Discovery:
    """The three searches' candidates joined, with the first judgment of each, and the exposure
    pool. files and exposures map each search to its admitted answer and to the nodes it
    received in full, in the search order. Each file is admitted already, with no declared
    problem (ingest/holdout_admission.py search_problems)."""
    order = list(files)
    first: dict[str, dict[str, Any]] = {}
    scope: dict[str, list[str]] = {}
    listed_by: dict[str, list[str]] = {}
    for stage in order:
        data = files[stage]
        labels = a_judgments(data)
        for c in data["candidates"]:
            node = c["node"]
            if node in first:
                if stage not in listed_by[node]:
                    listed_by[node].append(stage)
                continue
            first[node] = labels[node]
            scope[node] = list(c.get("scope", []))
            listed_by[node] = [stage]
    first[bm.ANSWERABILITY_TARGET] = a_judgments(files[order[0]])[bm.ANSWERABILITY_TARGET]
    listed = [n for n in first if n != bm.ANSWERABILITY_TARGET]
    exposed_by: dict[str, list[str]] = {}
    for stage in order:
        for node in exposures.get(stage, []):
            if poolable(node, texts, excluded) and stage not in exposed_by.get(node, []):
                exposed_by.setdefault(node, []).append(stage)
    pool = sorted({*listed, *exposed_by})
    unlisted = [n for n in pool if n not in listed_by]
    answerabilities = {stage: str(files[stage]["answerability"]) for stage in order}
    return Discovery(listed, first, scope, listed_by, exposed_by, pool, unlisted, answerabilities)


@dataclass(frozen=True)
class AuditNodes:
    """What the omission audit's search brings outside the normal exposure pool.

    listed: the new nodes it listed, with its own label as their first judgment.
    scope: the provisions that fix each of those nodes' scope, from its answer.
    exposed: the new nodes it received in full and did not list (no first judgment).
    on_pool: its labels on nodes already in the pool: provenance only, never a label of the
    need, a stop or a rejection (an audit judgment discrepancy, counted)."""

    listed: dict[str, dict[str, Any]]
    scope: dict[str, list[str]]
    exposed: list[str]
    on_pool: dict[str, dict[str, Any]]


def audit_nodes(
    audit_file: Mapping[str, Any],
    audit_exposure: Sequence[str],
    pool: Sequence[str],
    texts: Mapping[bm.NodeKey, str],
    excluded: frozenset[str],
) -> AuditNodes:
    """The audit search's nodes, split by the normal exposure pool: a node is new for the
    omission audit when the audit search listed it or received it in full, and it was in no
    normal search's exposure pool."""
    labels = a_judgments(audit_file)
    in_pool = set(pool)
    listed: dict[str, dict[str, Any]] = {}
    scope: dict[str, list[str]] = {}
    on_pool: dict[str, dict[str, Any]] = {}
    for c in audit_file["candidates"]:
        node = c["node"]
        if node in in_pool:
            on_pool[node] = labels[node]
        else:
            listed[node] = labels[node]
            scope[node] = list(c.get("scope", []))
    exposed = sorted(
        {
            n
            for n in audit_exposure
            if poolable(n, texts, excluded) and n not in in_pool and n not in listed
        }
    )
    return AuditNodes(listed, scope, exposed, on_pool)


def pool_judgments(data: Mapping[str, Any], ids: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """Judge B's pool variant's labels by node (judge_b_pool.v2): each entry's opaque id resolved
    by the orchestrator's own mapping (ids: id -> node), never by what the judge wrote."""
    return {
        ids[g["item_id"]]: {
            "grade": g.get("grade"),
            "ambiguous": bool(g.get("ambiguous")),
            "scope_basis": g.get("scope_basis", []),
            "rationale": g.get("reason", ""),
        }
        for g in data["grades"]
    }


def positive_or_ambiguous(j: Mapping[str, Any]) -> bool:
    """A label with a positive or an uncertain signal: a grade of 1 or more, or an ambiguity.
    No grade (a judge C that leaves a node unjudged) is neither."""
    grade, ambiguous = _label(j)
    return ambiguous or (grade is not None and grade >= 1)


def pool_route(j: Mapping[str, Any]) -> str:
    """Where one blind pool judgment sends its node: an explicit grade 0 that is not ambiguous
    is the final negative judgment ("final_negative"); a grade of 1 or more, or an ambiguity,
    needs a second blind judgment ("second_judgment"). An unjudged label is no judgment: the
    admission refuses it (holdout_admission.pool_problems), and here it is an error, never a 0."""
    grade, ambiguous = _label(j)
    if grade is None:
        raise ValueError("an unjudged pool label is no judgment, and never a grade 0")
    return "final_negative" if grade == 0 and not ambiguous else "second_judgment"


def diagnostic_set(
    listed: Sequence[str],
    pool_labels: Mapping[str, Sequence[Mapping[str, Any]]],
    final_grades: Mapping[str, int | None],
) -> list[str]:
    """The candidates judge D and the diagnostic receive: every node a search listed; every
    pool node whose final grade is 1 or more; every pool node any blind judgment graded 1 or
    more or found ambiguous (pool_labels: each unlisted pool node's labels, in the order they
    were given). A pool node nobody listed, graded 0 or left unjudged by judge B, with no
    positive or ambiguous label, is not among them. Unique and sorted; the caller puts them in
    document order."""
    out = set(listed)
    for node, labels in pool_labels.items():
        if (final_grades.get(node) or 0) >= 1 or any(positive_or_ambiguous(j) for j in labels):
            out.add(node)
    return sorted(out)


def b_judgments(data: Mapping[str, Any], ids: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """Judge B's labels by node: each entry's opaque id resolved by the orchestrator's own
    mapping (ids: id -> node), never by what the judge wrote about the node."""
    out: dict[str, dict[str, Any]] = {
        ids[g["item_id"]]: {
            "grade": g.get("grade"),
            "ambiguous": bool(g.get("ambiguous")),
            "scope_basis": g.get("scope_basis", []),
            "rationale": g.get("reason", ""),
        }
        for g in data["grades"]
    }
    out[bm.ANSWERABILITY_TARGET] = {
        "answerability": data["answerability"],
        "rationale": data.get("answerability_reason", ""),
        "missing": data.get("missing"),
    }
    return out


def _label(j: Mapping[str, Any]) -> tuple[int | None, bool]:
    grade = j.get("grade")
    return grade, bool(j.get("ambiguous")) if grade is not None else False


def label(j: Mapping[str, Any]) -> tuple[int | None, bool]:
    """A label as agreement reads it: the grade, and the ambiguity of a grade."""
    return _label(j)


@dataclass(frozen=True)
class Disagreements:
    nodes: list[str]
    answerability: bool


def disagreements(
    candidates: Sequence[str], a: Mapping[str, Any], b: Mapping[str, Any]
) -> Disagreements:
    """Every difference between A and B, of grade, ambiguity or answerability, goes to C."""
    missing = [r for r in candidates if r not in b]
    if missing:
        raise ValueError(f"judge B did not grade {missing}")
    nodes = [r for r in candidates if _label(a[r]) != _label(b[r])]
    target = bm.ANSWERABILITY_TARGET
    return Disagreements(nodes, a[target]["answerability"] != b[target]["answerability"])


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def as_judgment(j: Mapping[str, Any]) -> bm.Judgment:
    """A judge's label as the adjudication record holds it: what admission builds before an
    output is admitted (ingest/holdout_admission.py), and assembly again after."""
    return _judgment(j)


def _judgment(j: Mapping[str, Any]) -> bm.Judgment:
    data: dict[str, Any] = {"rationale": str(j.get("rationale") or j.get("reason") or "").strip()}
    if j.get("answerability"):
        data["answerability"] = j["answerability"]
        return bm.Judgment.model_validate(data)
    if j.get("grade") is not None:
        data["grade"] = j["grade"]
        if j.get("ambiguous"):
            data["ambiguous"] = True
    if j.get("scope_basis"):
        data["scope_basis"] = list(dict.fromkeys(j["scope_basis"]))
    return bm.Judgment.model_validate(data)


@dataclass
class Assembled:
    qrels: list[bm.Qrel] = field(default_factory=list)
    adjudications: list[bm.Adjudication] = field(default_factory=list)
    answerability: str = "complete"
    gap_note: str | None = None


def assemble_nodes(
    need_id: str,
    candidates: Sequence[str],
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    c: Mapping[str, Any],
    texts: Mapping[bm.NodeKey, str],
) -> Assembled:
    """The adjudication record of each candidate node and a qrel for every one whose final
    label is a grade: two judgments, a and b, and judge C's where they differ."""
    out = Assembled()
    for ref in candidates:
        record = bm.Adjudication(
            need_id=need_id,
            target=ref,
            a=_judgment(a[ref]),
            b=_judgment(b[ref]),
            c=_judgment(c[ref]) if ref in c else None,
        )
        out.adjudications.append(record)
        final = record.final
        if final.grade is not None:
            act, _, path = ref.partition(":")
            out.qrels.append(
                bm.Qrel(
                    need_id=need_id,
                    act=act,  # type: ignore[arg-type]
                    path=path,
                    grade=final.grade,
                    rationale=final.rationale,
                    text_sha256=bm.sha256_text(texts[(act, path)]),
                    ambiguous=final.ambiguous,
                )
            )
    return out


def assemble_audit(
    need_id: str,
    candidates: Sequence[str],
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    c: Mapping[str, Any],
    texts: Mapping[bm.NodeKey, str],
) -> Assembled:
    """The omission audit's new nodes, adjudicated like the pool's: no answerability. c maps
    each disputed node to judge C's answer; a node without a dispute must not appear in it."""
    disputed = {r for r in candidates if _label(a[r]) != _label(b[r])}
    if set(c) != disputed:
        raise ValueError(
            f"{need_id}: judge C answered {sorted(c)}, the disputes are {sorted(disputed)}"
        )
    return assemble_nodes(need_id, candidates, a, b, c, texts)


def assemble(
    need_id: str,
    candidates: Sequence[str],
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    c: Mapping[str, Any],
    a_gap_note: str | None,
    texts: Mapping[bm.NodeKey, str],
) -> Assembled:
    """One need: an adjudication record for every candidate and for the answerability, and
    a qrel for every candidate whose final label is a grade. c maps each disputed target to
    judge C's answer; a target without a dispute must not appear in it."""
    found = disagreements(candidates, a, b)
    disputed = set(found.nodes) | ({bm.ANSWERABILITY_TARGET} if found.answerability else set())
    if set(c) != disputed:
        raise ValueError(
            f"{need_id}: judge C answered {sorted(c)}, the disputes are {sorted(disputed)}"
        )
    out = assemble_nodes(need_id, candidates, a, b, c, texts)
    target = bm.ANSWERABILITY_TARGET
    record = bm.Adjudication(
        need_id=need_id,
        target=target,
        a=_judgment(a[target]),
        b=_judgment(b[target]),
        c=_judgment(c[target]) if target in c else None,
    )
    out.adjudications.append(record)
    final_answerability = record.final.answerability
    assert final_answerability is not None
    out.answerability = final_answerability
    if final_answerability != "complete":
        if final_answerability == a[target]["answerability"]:
            out.gap_note = a_gap_note
        elif final_answerability == b[target]["answerability"]:
            out.gap_note = b[target].get("missing")
        else:
            out.gap_note = record.final.rationale
        if not (out.gap_note or "").strip():
            raise ValueError(f"{need_id}: {final_answerability}, but its judge wrote no gap")
    return out


# ---------------------------------------------------------------------------
# the audit of an isolated context, from its transcript
# ---------------------------------------------------------------------------

# Context that Claude Code attaches to a subagent by itself (seen in DEV's transcripts).
PROJECT_CONTEXT = frozenset({"Project", "AutoMem", "session_context"})


@dataclass(frozen=True)
class Audit:
    ok: bool
    tools: list[str]
    reads: list[str]
    writes: list[str]
    commands: list[str]
    context: list[str]  # the project context Claude Code attached
    answer: str
    problems: list[str]


def audit_transcript(
    transcript: Path,
    packet: Path,
    allowed_tools: frozenset[str] = frozenset({"Read", "SubagentHandback"}),
    output: Path | None = None,
    command_prefix: str | None = None,
    forbidden_context: frozenset[str] = frozenset(),
) -> Audit:
    """What one context did and received: it may read only its packet, use only the
    allowed tools, write only its output, run only commands that start with the prefix,
    and must not have received the forbidden project context."""
    records = [
        json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines() if line
    ]
    calls: list[Mapping[str, Any]] = []
    context: list[str] = []
    for r in records:
        content = (r.get("message") or {}).get("content") or []
        if isinstance(content, list):
            calls += [c for c in content if isinstance(c, dict) and c.get("type") == "tool_use"]
        attachment = r.get("attachment") or {}
        if attachment.get("type") == "instructions":
            context += [f.get("type", "?") for f in attachment.get("files", [])]
        elif attachment.get("type") == "session_context":
            context.append("session_context")
    tools = [c["name"] for c in calls]
    inputs = [c.get("input", {}) for c in calls]
    reads = [i["file_path"] for t, i in zip(tools, inputs, strict=True) if t == "Read"]
    writes = [i["file_path"] for t, i in zip(tools, inputs, strict=True) if t == "Write"]
    commands = [i["command"] for t, i in zip(tools, inputs, strict=True) if t == "Bash"]
    answers = [
        i.get("message", "") for t, i in zip(tools, inputs, strict=True) if t == "SubagentHandback"
    ]
    problems = []
    if set(tools) - allowed_tools:
        problems.append(
            f"tools beyond {sorted(allowed_tools)}: {sorted(set(tools) - allowed_tools)}"
        )
    if set(reads) != {str(packet)}:
        problems.append(f"reads {sorted(set(reads))}, not only {packet}")
    if output is not None and set(writes) != {str(output)}:
        problems.append(f"writes {sorted(set(writes))}, not only {output}")
    if output is None and writes:
        problems.append(f"writes {writes}")
    if command_prefix is None and commands:
        problems.append(f"runs commands: {commands[:2]}")
    if command_prefix is not None:
        stray = [c for c in commands if not c.startswith(command_prefix)]
        if stray:
            problems.append(f"commands outside the corpus tools: {stray[:2]}")
    if not tools or tools[-1] != "SubagentHandback":
        problems.append("does not end with its answer")
    received = sorted(set(context) & forbidden_context)
    if received:
        problems.append(f"received project context {received}")
    return Audit(
        not problems,
        tools,
        reads,
        writes,
        commands,
        sorted(set(context)),
        answers[-1] if answers else "",
        problems,
    )


def as_json(text: str) -> Any:
    """An answer may come inside a fenced block; its content is the JSON."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(stripped)


# ---------------------------------------------------------------------------
# the diagnostic judge and judge D (D-054, decision 7)
# ---------------------------------------------------------------------------


def relevant(grade: int | None, ambiguous: bool) -> bool:
    """The material line: relevant is a grade >= 2 that is not ambiguous; 0, 1, unjudged
    and ambiguous are on the other side."""
    return grade is not None and grade >= bm.RELEVANT_GRADE and not ambiguous


@dataclass(frozen=True)
class Diagnostic:
    discrepancies: list[str]  # targets where the diagnostic crosses a material line
    missing: list[str]  # candidates the diagnostic did not grade: recorded, never a discrepancy


GRADES = (0, 1, 2, 3)
ANSWERABILITIES = ("complete", "partial", "none")


def diagnostic_problems(ids: Mapping[str, str], diagnostic: Any) -> list[str]:
    """What keeps the diagnostic's answer from being complete, by the rules of judge B's
    packet, which it receives (judge_b.v2): one entry per candidate, by its opaque id, and only
    those, each with a grade 0 to 3 or null and a true or false "ambiguous"; and an
    answerability of the protocol. The candidates and their ids are the stage's own input
    (ids: id -> node), never read from the answer."""
    if not isinstance(diagnostic, Mapping):
        return ["the answer is not a JSON object"]
    problems = []
    grades = diagnostic.get("grades")
    if not isinstance(grades, list):
        return ["the answer has no list of grades"]
    nodes = []
    for entry in grades:
        node = entry.get("item_id") if isinstance(entry, Mapping) else None
        if not isinstance(node, str):
            problems.append("an entry names no candidate")
            continue
        nodes.append(node)
        grade = entry.get("grade")
        if isinstance(grade, bool) or (grade is not None and grade not in GRADES):
            problems.append(f"{node}: grade {grade!r} is not 0 to 3 or null")
        if not isinstance(entry.get("ambiguous"), bool):
            problems.append(f"{node}: no true or false ambiguous")
    expected = list(ids)
    duplicated = sorted({n for n in nodes if nodes.count(n) > 1})
    unknown = sorted(set(nodes) - set(expected))
    missing = [n for n in expected if n not in nodes]
    problems += [f"{n}: graded more than once" for n in duplicated]
    problems += [f"{n}: not a candidate" for n in unknown]
    problems += [f"{n}: not graded" for n in missing]
    if diagnostic.get("answerability") not in ANSWERABILITIES:
        problems.append(f"answerability {diagnostic.get('answerability')!r} is not of the protocol")
    return problems


# What keeps the diagnostic's answer from being admitted, by class, never by content. An answer
# of a run that ended normally with any of them makes the diagnostic unavailable: the first
# START, aborted on 2026-10-20, showed that the greedy local model gives the same answer to the
# same prompt, so another execution of it repairs nothing.
DIAGNOSTIC_DEFECTS = (
    # the prompt, exactly as frozen, does not fit the model's frozen limit: found by the
    # preflight before any generation (P1 #6), never truncated, never run three times
    "input_exceeds_context",
    "not_json",
    "not_an_object",
    "no_grades",
    "entry_without_candidate",
    "invalid_grade",
    "invalid_ambiguous",
    "duplicate_candidate",
    "unknown_candidate",
    "missing_candidate",
    "invalid_answerability",
)
DIAGNOSTIC_UNAVAILABLE = "qwen_diagnostic_unavailable"


def _defect(problem: str) -> str:
    whole = {
        "the answer is not a JSON object": "not_an_object",
        "the answer has no list of grades": "no_grades",
        "an entry names no candidate": "entry_without_candidate",
    }
    if problem in whole:
        return whole[problem]
    if problem.startswith("answerability ") and problem.endswith(" is not of the protocol"):
        return "invalid_answerability"
    for suffix, defect in (
        (" is not 0 to 3 or null", "invalid_grade"),
        (": no true or false ambiguous", "invalid_ambiguous"),
        (": graded more than once", "duplicate_candidate"),
        (": not a candidate", "unknown_candidate"),
        (": not graded", "missing_candidate"),
    ):
        if problem.endswith(suffix):
            return defect
    raise ValueError(f"a diagnostic problem of no known class: {problem!r}")


def diagnostic_defects(ids: Mapping[str, str], output: str) -> list[str]:
    """The classes of defect of the diagnostic's raw answer, in DIAGNOSTIC_DEFECTS order: none
    for an answer diagnostic_problems finds complete. Never the answer's content, which names
    TEST's candidates."""
    try:
        answer = as_json(output)
    except (json.JSONDecodeError, IndexError):
        return ["not_json"]
    found = {_defect(p) for p in diagnostic_problems(ids, answer)}
    return [d for d in DIAGNOSTIC_DEFECTS if d in found]


def unavailable_diagnostic(defects: Sequence[str]) -> dict[str, Any]:
    """The diagnostic stage's value when the answer is not admitted: its classes of defect only."""
    if not defects or any(d not in DIAGNOSTIC_DEFECTS for d in defects):
        raise ValueError(f"not classes of defect: {list(defects)}")
    return {DIAGNOSTIC_UNAVAILABLE: True, "defects": list(defects)}


def diagnostic_unavailable(value: Any) -> bool:
    """Whether the diagnostic stage recorded no diagnostic. Unavailable is not agreement."""
    return (
        isinstance(value, Mapping)
        and set(value) == {DIAGNOSTIC_UNAVAILABLE, "defects"}
        and value[DIAGNOSTIC_UNAVAILABLE] is True
    )


def material_discrepancies(
    ids: Mapping[str, str], final: Assembled, diagnostic: Mapping[str, Any]
) -> Diagnostic:
    """Where the diagnostic judge (Qwen3-8B, judge B's packet) and the final labels of
    A, B and C fall on different sides: relevance >= 2 against < 2 for a node, or another
    answerability. Each entry's opaque id is resolved by the orchestrator (ids: id -> node). It
    never changes a label; each target found goes to judge D."""
    candidates = list(ids.values())
    finals = {f"{q.act}:{q.path}": relevant(q.grade, q.ambiguous) for q in final.qrels}
    graded = {
        ids[g["item_id"]]: g
        for g in diagnostic.get("grades", [])
        if isinstance(g, dict) and g.get("item_id") in ids
    }
    missing = [r for r in candidates if r not in graded]
    found = [
        r
        for r in candidates
        if r in graded
        and finals.get(r, False)
        != relevant(graded[r].get("grade"), bool(graded[r].get("ambiguous")))
    ]
    if diagnostic.get("answerability") != final.answerability:
        found.append(bm.ANSWERABILITY_TARGET)
    return Diagnostic(found, missing)


def judge_d_outcome(
    final: Assembled, d_items: Sequence[Mapping[str, Any]], ids: Mapping[str, str]
) -> dict[str, str]:
    """For each target judge D graded: "consensus_held" when D, unambiguous, falls on the side
    of the final label; "adjudication_uncertain" otherwise. Ambiguity takes precedence (Codex
    Gate 2): an ambiguous D never holds the consensus, even when its best grade and the final
    one both sit below the relevance line. For the answerability, anything but the final label
    is uncertain: a third label does not hold the consensus either."""
    finals = {f"{q.act}:{q.path}": relevant(q.grade, q.ambiguous) for q in final.qrels}
    out: dict[str, str] = {}
    for item in d_items:
        target = ids[item["item_id"]]  # the orchestrator's mapping, never the judge's word
        if item.get("ambiguous"):
            held = False
        elif target == bm.ANSWERABILITY_TARGET:
            held = item.get("answerability") == final.answerability
        else:
            held = finals.get(target, False) == relevant(item.get("grade"), False)
        out[target] = "consensus_held" if held else "adjudication_uncertain"
    return out


# ---------------------------------------------------------------------------
# final metadata (Codex Gate 2, P1.6): deterministic where the protocol decides it, the
# blind metadata adjudicator for the rest, and nothing inferred by the orchestrator
# ---------------------------------------------------------------------------

GAP_DRIVERS = frozenset({"numeric", "date", "level2_gap"})


@dataclass(frozen=True)
class Metadata:
    type: str
    acts: list[str]
    difficulty: str
    drivers: list[str]
    driver_nodes: dict[str, list[str]]
    source: dict[str, str]  # field -> "deterministic: ..." or "metadata adjudicator"


def deterministic_metadata(
    qrels: Sequence[bm.Qrel], answerability: str, planned_acts: Sequence[str]
) -> dict[str, Any]:
    """What the protocol fixes from the final qrels alone:
    - acts: the acts of the relevant evidence (grade >= 2, not ambiguous); a none need has no
      relevant evidence, so it keeps the acts its question was written for (the cell's);
    - type partial_out_of_scope exactly when the answerability is partial or none (the
      dataset schema ties them);
    - cross_act: the relevant evidence lies in more than one act (section 13)."""
    relevant_acts = sorted(
        {r.act for r in qrels if relevant(r.grade, r.ambiguous)}, key=bm.ACTS.index
    )
    order: dict[str, int] = {a: i for i, a in enumerate(bm.ACTS)}
    planned: list[str] = sorted(planned_acts, key=order.__getitem__)
    acts: list[str] = list(relevant_acts) or planned
    return {
        "acts": acts,
        "acts_source": (
            "deterministic: the acts of the relevant evidence"
            if relevant_acts
            else "deterministic: no relevant evidence, the acts the question was written for"
        ),
        "type": None if answerability == "complete" else "partial_out_of_scope",
        "cross_act": len(relevant_acts) > 1,
    }


def metadata_problems(
    answer: Mapping[str, Any], qrels: Sequence[bm.Qrel], answerability: str
) -> list[str]:
    """What makes the adjudicator's answer unusable: a label outside the rubric, or a driver
    the final grades do not support."""
    problems = []
    allowed = (
        {"direct", "multi_provision", "exception_confusion"}
        if answerability == "complete"
        else {"partial_out_of_scope"}
    )
    if answer.get("type") not in allowed:
        problems.append(f"type {answer.get('type')!r} is not one of {sorted(allowed)}")
    if answer.get("difficulty") not in {"easy", "medium", "hard"}:
        problems.append(f"difficulty {answer.get('difficulty')!r}")
    graded = {f"{r.act}:{r.path}": r for r in qrels}
    drivers = answer.get("drivers")
    if not isinstance(drivers, dict):
        return [*problems, "drivers is not an object"]
    known = set(get_args(bm.Driver)) - {"cross_act"}
    for driver, nodes in drivers.items():
        if driver not in known:
            problems.append(f"driver {driver!r} is not one the adjudicator may give")
            continue
        if not isinstance(nodes, list) or not nodes:
            problems.append(f"{driver}: no node named")
            continue
        for node in nodes:
            if node == "gap_note":
                if driver not in GAP_DRIVERS or answerability == "complete":
                    problems.append(f"{driver}: the gap holds it only for a partial or none need")
                continue
            r = graded.get(node)
            if r is None:
                problems.append(f"{driver}: {node} is not a judged node")
            elif driver == "negation" and not (r.grade == 3 and not r.ambiguous):
                problems.append(f"negation: {node} is not a grade 3")
            elif driver != "lexical_twin" and not relevant(r.grade, r.ambiguous):
                problems.append(f"{driver}: {node} is not relevant")
    return problems


def final_metadata(
    answer: Mapping[str, Any],
    qrels: Sequence[bm.Qrel],
    answerability: str,
    planned_acts: Sequence[str],
) -> Metadata:
    """The realized type, acts, difficulty and drivers of a need. Planned labels enter only
    for the acts of a need without relevant evidence; the comparison with the plan is made
    afterwards, from this."""
    problems = metadata_problems(answer, qrels, answerability)
    if problems:
        raise ValueError(f"the metadata answer is unusable: {problems}")
    fixed = deterministic_metadata(qrels, answerability, planned_acts)
    nodes = {d: list(n) for d, n in answer["drivers"].items()}
    if fixed["cross_act"]:
        nodes["cross_act"] = sorted(
            f"{r.act}:{r.path}" for r in qrels if relevant(r.grade, r.ambiguous)
        )
    order = list(get_args(bm.Driver))
    drivers = sorted(nodes, key=order.index)
    return Metadata(
        type=fixed["type"] or answer["type"],
        acts=fixed["acts"],
        difficulty=answer["difficulty"],
        drivers=drivers,
        driver_nodes=nodes,
        source={
            "type": "deterministic: the answerability" if fixed["type"] else "metadata adjudicator",
            "acts": fixed["acts_source"],
            "difficulty": "metadata adjudicator",
            "drivers": "metadata adjudicator; cross_act deterministic from the relevant evidence",
        },
    )
