"""What a model's answer must be before it is admitted (S3, P1 #1 of the review before the
clean restart).

Every stage's answer is checked against its COMPLETE contract before the output is admitted:
its shape, its required fields with their types, non-empty required text, valid grades and
answerabilities, the exact candidate set (none missing, none twice, none unknown), the
rationale each judgment needs, and every invariant its consumers rely on downstream. One
source of truth: each validator builds, from the answer, the very records its consumers will
build (a bm.Judgment for every label, the gap note the final answerability takes), so that an
admitted output can never fail later in a consumer for a reason of shape. A problem is a failed
technical execution of the stage (ingest/holdout_pipeline.py), never a judgment of the need.

The problems name a field, a position and a class of defect, never the answer's text: they
are recorded in the state and read by the controller.
"""

from __future__ import annotations

import dataclasses
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, get_args

from ingest import benchmark as bm
from ingest import holdout_judging as hj

GRADES = hj.GRADES
ANSWERABILITIES = hj.ANSWERABILITIES
GATE_VERDICTS = (
    "independent",
    "paraphrase",
    "same_legal_question",
    "same_rule_superficial_change",
    "uncertain",
)
NO_PROBLEM = (None, "", "null")
METADATA_TYPES = frozenset(
    {"direct", "multi_provision", "exception_confusion", "partial_out_of_scope"}
)
DIFFICULTIES = frozenset({"easy", "medium", "hard"})
# The frozen semantic invariants an answer of the right form can still violate: each ties the
# answer's labels to the need's evidence or answerability. Three independent executions of a
# stage violating one of them is a property of the need or its adjudication, not of the run.
SEARCH_INVARIANTS = (
    "complete_without_grade_3",
    "none_with_relevant_evidence",
    "partial_without_relevant_evidence",
)
METADATA_INVARIANTS = (
    "type_against_answerability",
    "gap_driver_on_complete_need",
    "negation_without_grade_3",
    "driver_without_relevant_support",
)


@dataclass(frozen=True)
class Verdict:
    """What admission finds: problems of the role's contract (form, fields, types, vocabulary,
    references), and, only when there is none, the frozen semantic invariants the answer
    violates (codes, never text)."""

    problems: list[str] = dataclasses.field(default_factory=list)
    violations: list[str] = dataclasses.field(default_factory=list)

    @property
    def admitted(self) -> bool:
        return not self.problems and not self.violations


_NODE_REF = re.compile(bm.NODE_REF_PATTERN)
_NODE_IN_TEXT = re.compile(r"\b(?:DORA|MiCA|PSD2):[\w.]+")


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _grade(value: Any, nullable: bool) -> bool:
    if value is None:
        return nullable
    return isinstance(value, int) and not isinstance(value, bool) and value in GRADES


def _refs(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(r, str) and _NODE_REF.match(r) for r in value)


def _judgment(entry: Mapping[str, Any], where: str) -> list[str]:
    """The label as the adjudication record will hold it (bm.Judgment), built now."""
    try:
        hj.as_judgment(entry)
    except (ValueError, TypeError):
        return [f"{where}: not a judgment the adjudication record can hold"]
    return []


def _set_problems(nodes: Sequence[Any], expected: Sequence[str], where: str) -> list[str]:
    counts = Counter(n for n in nodes if isinstance(n, str))
    problems = [f"{where}: a candidate named twice"] * sum(c > 1 for c in counts.values())
    if set(counts) - set(expected):
        problems.append(f"{where}: {len(set(counts) - set(expected))} unknown candidates")
    if set(expected) - set(counts):
        problems.append(f"{where}: {len(set(expected) - set(counts))} candidates missing")
    return problems


# ---------------------------------------------------------------------------
# authors and gatekeepers
# ---------------------------------------------------------------------------


def concept_problems(value: Any) -> list[str]:
    """The concept author's answer (concept_author.v1): the concept, its facts and the fit,
    which the statement writer's packet and the fit check read."""
    if not isinstance(value, Mapping):
        return ["the answer is not an object"]
    problems = []
    if not _text(value.get("concept")):
        problems.append("concept: no text")
    facts = value.get("facts")
    if not isinstance(facts, list):
        problems.append("facts: not a list")
    elif not all(_text(f) for f in facts):
        problems.append("facts: an entry that is not text")
    if not _text(value.get("fit")):
        problems.append("fit: no text")
    return problems


def statement_problems(value: Any) -> list[str]:
    """The statement writer's answer (statement_writer.v1): a primary statement."""
    if not isinstance(value, Mapping):
        return ["the answer is not an object"]
    return [] if _text(value.get("primary")) else ["primary: no text"]


def gate_problems(value: Any, shown: Sequence[str]) -> list[str]:
    """A gatekeeper's verdict (overlap_gate.v2), naming only a need it was shown."""
    if not isinstance(value, Mapping):
        return ["the answer is not an object"]
    verdict = value.get("verdict")
    if verdict not in GATE_VERDICTS:
        return ["verdict: not one of the gatekeeper's"]
    closest = value.get("closest")
    if closest is not None and not isinstance(closest, str):
        return ["closest: neither an id nor null"]
    if verdict not in ("independent", "uncertain") and closest not in shown:
        return ["closest: a need it was not shown"]
    return []


def evidence_gate_problems(value: Any) -> list[str]:
    """The evidence gatekeeper's verdict (overlap_gate_evidence.v1)."""
    if not isinstance(value, Mapping):
        return ["the answer is not an object"]
    return [] if value.get("verdict") in GATE_VERDICTS else ["verdict: not one of the gatekeeper's"]


# ---------------------------------------------------------------------------
# the evidence searches (judge_a.v3: search_1, search_2, search_3 and the audit search)
# ---------------------------------------------------------------------------


def declared_problem(found: Mapping[str, Any]) -> str | None:
    """The problem a search declares, or None. Only text declares one."""
    problem = found.get("problem")
    return None if problem in NO_PROBLEM else str(problem)


def search_verdict(
    found: Any, texts: Mapping[bm.NodeKey, str], excluded: frozenset[str]
) -> Verdict:
    """A search's final answer (judge_a.v3). A declared problem is an answer of its own: its
    text, and nothing else, is required (its transition is the pipeline's, before any
    candidate is read). Otherwise every candidate and the answerability, complete (the role's
    contract), and then the protocol's rules tying the answerability to the search's own
    grades (DEV batch 2's checks: the semantic invariants)."""
    if not isinstance(found, Mapping):
        return Verdict(["the answer is not an object"])
    problem = found.get("problem")
    if problem not in NO_PROBLEM:
        return Verdict([] if _text(problem) else ["problem: neither null nor text"])
    candidates = found.get("candidates")
    if not isinstance(candidates, list):
        return Verdict(["candidates: not a list"])
    problems = []
    nodes = []
    for i, c in enumerate(candidates, start=1):
        found_here = _candidate_problems(c, f"candidate {i}", texts, excluded)
        if isinstance(c, Mapping) and isinstance(c.get("node"), str):
            nodes.append(c["node"])
        problems += found_here
    if len(set(nodes)) != len(nodes):
        problems.append("candidates: a node named twice")
    if found.get("answerability") not in ANSWERABILITIES:
        return Verdict([*problems, "answerability: not complete, partial or none"])
    if not _text(found.get("answerability_reason")):
        problems.append("answerability_reason: no text")
    gap = found.get("gap_note")
    if found["answerability"] != "complete" and not _text(gap):
        problems.append("gap_note: none for a need that is not complete")
    elif gap is not None and not isinstance(gap, str):
        problems.append("gap_note: neither text nor null")
    if not problems:
        # every label exactly as the adjudication record will build it (hj.a_judgments)
        labels = hj.a_judgments(found)
        problems += [p for n, j in labels.items() for p in _judgment(j, _place(n, nodes))]
    if problems:
        return Verdict(problems)
    relevant = [c for c in candidates if c["grade"] >= 2 and not c.get("ambiguous")]
    top = [c for c in candidates if c["grade"] == 3 and not c.get("ambiguous")]
    answerability = found["answerability"]
    violations = []
    if answerability == "complete" and not top:
        violations.append("complete_without_grade_3")
    if answerability == "none" and relevant:
        violations.append("none_with_relevant_evidence")
    if answerability == "partial" and not relevant:
        violations.append("partial_without_relevant_evidence")
    if not violations:
        # nothing else of the protocol's rules may remain: if it does, the role's contract
        problems += sorted(
            {
                "rule: " + _NODE_IN_TEXT.sub("<node>", p.split(": ", 1)[-1])
                for p in hj.judge_a_problems(found, texts, excluded)
            }
        )
    return Verdict(problems, violations)


def search_problems(
    found: Any, texts: Mapping[bm.NodeKey, str], excluded: frozenset[str]
) -> list[str]:
    """Everything that keeps a search's answer from admission, contract and invariants alike."""
    verdict = search_verdict(found, texts, excluded)
    return [*verdict.problems, *(f"invariant: {v}" for v in verdict.violations)]


def _candidate_problems(
    c: Any, where: str, texts: Mapping[bm.NodeKey, str], excluded: frozenset[str]
) -> list[str]:
    if not isinstance(c, Mapping):
        return [f"{where}: not an object"]
    node = c.get("node")
    if not isinstance(node, str) or not _NODE_REF.match(node):
        return [f"{where}: no node of the corpus's form"]
    problems = []
    act, _, path = node.partition(":")
    if (act, path) not in texts:
        problems.append(f"{where}: no such node with own text")
    if act == "PSD2" and path.split(".")[0] in excluded:
        problems.append(f"{where}: an excluded PSD2 article")
    if not _grade(c.get("grade"), nullable=False):
        problems.append(f"{where}: grade not 0 to 3")
    if "ambiguous" in c and not isinstance(c["ambiguous"], bool):
        problems.append(f"{where}: ambiguous not true or false")
    for field in ("scope", "scope_basis"):
        if field in c and not _refs(c[field]):
            problems.append(f"{where}: {field} not a list of nodes")
        elif any(tuple(s.partition(":")[::2]) not in texts for s in c.get(field) or []):
            problems.append(f"{where}: {field} names a node that does not exist")
    if not _text(c.get("rationale")):
        problems.append(f"{where}: no rationale")
    return problems or _judgment(_first_label(c), where)


def _first_label(c: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "grade": c.get("grade"),
        "ambiguous": bool(c.get("ambiguous")),
        "scope_basis": c.get("scope_basis", []),
        "rationale": c.get("rationale"),
    }


# ---------------------------------------------------------------------------
# judges B, C and D
# ---------------------------------------------------------------------------


def _identity_conflicts(entry: Mapping[str, Any], target: str | None, where: str) -> list[str]:
    """An identity the answer writes beside the opaque id: the orchestrator owns it, so one
    that names anything else than the id's own node is a conflict, never a correction."""
    problems = []
    for name in ("target", "node"):
        if name in entry and entry[name] is not None and entry[name] != target:
            problems.append(f"{where}: an identity other than its id's")
    return problems


def judge_b_problems(answer: Any, ids: Mapping[str, str]) -> list[str]:
    """Judge B's answer (judge_b.v2): one entry for every candidate shown, by its exact opaque
    id (ids: id -> node), and only those, each a label the adjudication can hold, and the
    answerability with what the corpus lacks."""
    if not isinstance(answer, Mapping):
        return ["the answer is not an object"]
    grades = answer.get("grades")
    if not isinstance(grades, list):
        return ["grades: not a list"]
    problems = []
    named = []
    for i, g in enumerate(grades, start=1):
        where = f"entry {i}"
        if not isinstance(g, Mapping) or not isinstance(g.get("item_id"), str):
            problems.append(f"{where}: names no candidate id")
            continue
        named.append(g["item_id"])
        problems += _identity_conflicts(g, ids.get(g["item_id"]), where)
        problems += _label_problems(g, where, "reason", grade_nullable=True)
    problems += _set_problems(named, list(ids), "grades")
    problems += _answerability_problems(answer, "answerability_reason")
    if answer.get("answerability") in ("partial", "none") and not _text(answer.get("missing")):
        problems.append("missing: none for a need that is not complete")
    if not problems:
        # every label exactly as the adjudication record will build it (hj.b_judgments)
        nodes = [ids[n] for n in named]
        labels = hj.b_judgments(answer, ids)
        problems += [p for n, j in labels.items() for p in _judgment(j, _place(n, nodes))]
    return problems


def pool_problems(answer: Any, ids: Mapping[str, str]) -> list[str]:
    """Judge B's pool variant's answer (judge_b_pool.v2), the exact-coverage contract of the
    person's decision of 2026-10-22: the set of answered ids is exactly the set shown (ids: id
    -> node), each once, each with an explicit grade 0 to 3 and an explicit true or false
    ambiguity, each a label the adjudication can hold. A node left out, ungraded (null),
    unknown, repeated or replaced is a failure of the role's contract: never a grade 0. It
    gives no answerability: none is asked."""
    if not isinstance(answer, Mapping):
        return ["the answer is not an object"]
    grades = answer.get("grades")
    if not isinstance(grades, list):
        return ["grades: not a list"]
    problems = []
    named = []
    for i, g in enumerate(grades, start=1):
        where = f"entry {i}"
        if not isinstance(g, Mapping) or not isinstance(g.get("item_id"), str):
            problems.append(f"{where}: names no candidate id")
            continue
        named.append(g["item_id"])
        problems += _identity_conflicts(g, ids.get(g["item_id"]), where)
        if not isinstance(g.get("ambiguous"), bool):
            problems.append(f"{where}: ambiguous not given as true or false")
        problems += _label_problems(g, where, "reason", grade_nullable=False)
    problems += _set_problems(named, list(ids), "grades")
    if not problems:
        # every label exactly as the adjudication record will build it (hj.pool_judgments)
        nodes = [ids[n] for n in named]
        labels = hj.pool_judgments(answer, ids)
        problems += [p for n, j in labels.items() for p in _judgment(j, _place(n, nodes))]
    return problems


def judge_c_problems(reply: Any, items: Sequence[Any]) -> list[str]:
    """Judge C's answer (judge_c.v2): every item it was given, once, by its exact opaque id,
    and no other, each with a label the adjudication can hold. The id alone says which node or
    answerability an entry decides: never its position, never what it writes about the node."""
    if not isinstance(reply, Mapping) or not isinstance(reply.get("items"), list):
        return ["items: not a list"]
    expected = {i.item_id: i.target for i in items}
    problems = []
    seen = []
    for k, entry in enumerate(reply["items"], start=1):
        where = f"entry {k}"
        if not isinstance(entry, Mapping) or not isinstance(entry.get("item_id"), str):
            problems.append(f"{where}: names no item id")
            continue
        number = entry["item_id"]
        seen.append(number)
        if number not in expected:
            continue  # counted as unknown below
        problems += _identity_conflicts(entry, expected[number], where)
        found = _item_problems(entry, expected[number], where)
        # judge C's answer is the label itself: built exactly as the assembly will build it
        problems += found or _judgment(entry, where)
    problems += _set_problems(seen, list(expected), "items")
    return problems


def judge_d_problems(reply: Any, ids: Mapping[str, str]) -> list[str]:
    """Judge D's answer (judge_d.v2): every item it was asked, once, by its exact opaque id
    (ids: id -> node or "answerability"), and only those."""
    if not isinstance(reply, Mapping) or not isinstance(reply.get("items"), list):
        return ["items: not a list"]
    problems = []
    seen = []
    for k, entry in enumerate(reply["items"], start=1):
        where = f"entry {k}"
        if not isinstance(entry, Mapping) or not isinstance(entry.get("item_id"), str):
            problems.append(f"{where}: names no item id")
            continue
        seen.append(entry["item_id"])
        if entry["item_id"] in ids:
            target = ids[entry["item_id"]]
            problems += _identity_conflicts(entry, target, where)
            problems += _item_problems(entry, target, where)
    problems += _set_problems(seen, list(ids), "items")
    return problems


def _place(node: str, nodes: Sequence[str]) -> str:
    if node == bm.ANSWERABILITY_TARGET:
        return "the answerability"
    return f"candidate {nodes.index(node) + 1}" if node in nodes else "a candidate"


def _item_problems(entry: Mapping[str, Any], target: str, where: str) -> list[str]:
    if target == bm.ANSWERABILITY_TARGET:
        problems = []
        if entry.get("answerability") not in ANSWERABILITIES:
            problems.append(f"{where}: answerability not complete, partial or none")
        if entry.get("grade") is not None:
            problems.append(f"{where}: a grade on the answerability")
        if "ambiguous" in entry and not isinstance(entry["ambiguous"], bool):
            problems.append(f"{where}: ambiguous not true or false")
        if not _text(entry.get("reason")):
            problems.append(f"{where}: no reason")
        return problems
    if entry.get("answerability") is not None:
        return [f"{where}: an answerability on a node"]
    return _label_problems(entry, where, "reason", grade_nullable=True)


def _label_problems(
    entry: Mapping[str, Any], where: str, reason: str, grade_nullable: bool
) -> list[str]:
    problems = []
    if not _grade(entry.get("grade"), grade_nullable):
        problems.append(f"{where}: grade not 0 to 3" + (" or null" if grade_nullable else ""))
    if not isinstance(entry.get("ambiguous", False), bool):
        problems.append(f"{where}: ambiguous not true or false")
    # null is an absent scope basis, as the adjudication record reads it (bm.Judgment's default);
    # a grade 0 still needs one, which building the judgment below checks
    if entry.get("scope_basis") is not None and not _refs(entry["scope_basis"]):
        problems.append(f"{where}: scope_basis not a list of nodes")
    if not _text(entry.get(reason)):
        problems.append(f"{where}: no {reason}")
    if not problems:
        label = {
            "grade": entry.get("grade"),
            "ambiguous": bool(entry.get("ambiguous")),
            "scope_basis": entry.get("scope_basis") or [],
            "rationale": entry.get(reason),
        }
        problems += _judgment(label, where)
    return problems


def _answerability_problems(answer: Mapping[str, Any], reason: str) -> list[str]:
    if answer.get("answerability") not in ANSWERABILITIES:
        return ["answerability: not complete, partial or none"]
    if not _text(answer.get(reason)):
        return [f"{reason}: no text"]
    return []


# ---------------------------------------------------------------------------
# the metadata adjudicator
# ---------------------------------------------------------------------------


def metadata_verdict(reply: Any, qrels: Sequence[bm.Qrel], answerability: str) -> Verdict:
    """The metadata adjudicator's answer (metadata_adjudicator.v1), as holdout_judging's
    metadata_problems judges it, split in two: the role's contract (form, the rubric's
    vocabulary, a node it was shown, the gap note only for a driver a gap can hold) and the
    semantic invariants that tie its labels to the final grades and answerability."""
    problems = metadata_shape_problems(reply)
    if problems:
        return Verdict(problems)
    violations = []
    kind = reply["type"]
    if kind not in METADATA_TYPES:
        problems.append("type: not a type of the rubric")
    elif (answerability == "complete") == (kind == "partial_out_of_scope"):
        violations.append("type_against_answerability")
    if reply["difficulty"] not in DIFFICULTIES:
        problems.append("difficulty: not one of the rubric")
    graded = {f"{r.act}:{r.path}": r for r in qrels}
    known = set(get_args(bm.Driver)) - {"cross_act"}
    for driver, nodes in reply["drivers"].items():
        if driver not in known:
            problems.append("drivers: a driver the adjudicator may not give")
            continue
        if not nodes:
            problems.append("drivers: a driver with no node named")
            continue
        for node in nodes:
            if node == "gap_note":
                if driver not in hj.GAP_DRIVERS:
                    problems.append("drivers: the gap named for a driver no gap can hold")
                elif answerability == "complete":
                    violations.append("gap_driver_on_complete_need")
                continue
            r = graded.get(node)
            if r is None:
                problems.append("drivers: a node that is not among the final grades")
            elif driver == "negation" and not (r.grade == 3 and not r.ambiguous):
                violations.append("negation_without_grade_3")
            elif driver != "lexical_twin" and not hj.relevant(r.grade, r.ambiguous):
                violations.append("driver_without_relevant_support")
    if problems:
        return Verdict(problems)
    return Verdict([], sorted(set(violations)))


def metadata_shape_problems(reply: Any) -> list[str]:
    """The metadata adjudicator's answer as data (metadata_adjudicator.v1): its labels are
    then checked against the final grades by holdout_judging.final_metadata."""
    if not isinstance(reply, Mapping):
        return ["the answer is not an object"]
    problems = []
    for field in ("type", "difficulty"):
        if not isinstance(reply.get(field), str):
            problems.append(f"{field}: not text")
    drivers = reply.get("drivers")
    if not isinstance(drivers, Mapping):
        problems.append("drivers: not an object")
    elif not all(
        isinstance(k, str) and isinstance(v, list) and all(isinstance(n, str) for n in v)
        for k, v in drivers.items()
    ):
        problems.append("drivers: not lists of nodes")
    return problems
