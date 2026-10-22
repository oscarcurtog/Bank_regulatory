"""The complete contract of every answer, checked before it is admitted (P1 #1 of the review
before the clean restart): nothing admitted can fail later in a consumer for a reason of shape.
Synthetic answers on a stand-in corpus; no model, no TEST content."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import pytest

from ingest import benchmark as bm
from ingest import holdout_admission as ha
from ingest import holdout_judging as hj

TEXTS = {
    ("MiCA", "149.2"): "text",
    ("MiCA", "149.3"): "text",
    ("DORA", "64"): "text",
    ("PSD2", "97.1"): "text",
    ("PSD2", "10.1"): "text",
}
EXCLUDED = frozenset({"10"})
CANDIDATES = ["MiCA:149.2", "MiCA:149.3", "DORA:64"]
UNKNOWN = "C-" + "0" * 12  # an id the orchestrator never gave


def ids_of(*targets: str) -> dict[str, str]:
    """The orchestrator's mapping of one stage: opaque id -> node (or the answerability)."""
    return {f"C-{i:012x}": t for i, t in enumerate(targets, start=1)}


def id_of(ids: dict[str, str], target: str) -> str:
    return next(i for i, t in ids.items() if t == target)


B_IDS = ids_of(*CANDIDATES)
C_IDS = ids_of("MiCA:149.3", bm.ANSWERABILITY_TARGET)
D_IDS = ids_of("DORA:64", bm.ANSWERABILITY_TARGET)


def search() -> dict[str, Any]:
    return {
        "need_id": "N101",
        "candidates": [
            {
                "node": "MiCA:149.2",
                "scope": [],
                "grade": 3,
                "ambiguous": False,
                "scope_basis": [],
                "rationale": "quotes the node",
            },
            {
                "node": "MiCA:149.3",
                "scope": ["MiCA:149.2"],
                "grade": 2,
                "ambiguous": False,
                "scope_basis": [],
                "rationale": "quotes the node",
            },
            {
                "node": "DORA:64",
                "scope": [],
                "grade": 0,
                "ambiguous": False,
                "scope_basis": ["DORA:64"],
                "rationale": "out of scope",
            },
        ],
        "answerability": "complete",
        "answerability_reason": "149.2 answers",
        "gap_note": None,
        "drivers": {},
        "problem": None,
    }


def judge_b() -> dict[str, Any]:
    return {
        "need_id": "x",
        "grades": [
            {
                "item_id": id_of(B_IDS, "MiCA:149.2"),
                "grade": 3,
                "ambiguous": False,
                "scope_basis": [],
                "reason": "B",
            },
            {
                "item_id": id_of(B_IDS, "MiCA:149.3"),
                "grade": 2,
                "ambiguous": False,
                "scope_basis": [],
                "reason": "B",
            },
            {
                "item_id": id_of(B_IDS, "DORA:64"),
                "grade": 0,
                "ambiguous": False,
                "scope_basis": ["DORA:64"],
                "reason": "B",
            },
        ],
        "answerability": "complete",
        "answerability_reason": "B",
        "missing": None,
    }


@dataclass(frozen=True)
class Item:
    item_id: str
    target: str


ITEMS = [Item(i, t) for i, t in C_IDS.items()]


def judge_c() -> dict[str, Any]:
    return {
        "need_id": "x",
        "items": [
            {
                "item_id": id_of(C_IDS, "MiCA:149.3"),
                "grade": 1,
                "ambiguous": False,
                "answerability": None,
                "scope_basis": [],
                "reason": "C",
            },
            {
                "item_id": id_of(C_IDS, "answerability"),
                "grade": None,
                "ambiguous": False,
                "answerability": "partial",
                "scope_basis": [],
                "reason": "C",
            },
        ],
    }


def judge_d() -> dict[str, Any]:
    return {
        "need_id": "x",
        "items": [
            {
                "item_id": id_of(D_IDS, "DORA:64"),
                "grade": 0,
                "ambiguous": False,
                "answerability": None,
                "scope_basis": ["DORA:64"],
                "reason": "D",
            },
            {
                "item_id": id_of(D_IDS, "answerability"),
                "grade": None,
                "ambiguous": False,
                "answerability": "complete",
                "scope_basis": [],
                "reason": "D",
            },
        ],
    }


def problems(value: Any) -> list[str]:
    return ha.search_problems(value, TEXTS, EXCLUDED)


# ---------------------------------------------------------------------------
# the authors' answers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"facts": None}, "facts: not a list"),
        ({"facts": ["a fact", 3]}, "facts: an entry that is not text"),
        ({"facts": ["  "]}, "facts: an entry that is not text"),
        ({"concept": ""}, "concept: no text"),
        ({"fit": None}, "fit: no text"),
    ],
)
def test_a_concept_is_admitted_only_whole(change, expected):
    concept = {"concept": "A firm asks.", "facts": ["a firm"], "fit": "fits"}
    assert ha.concept_problems(concept) == []
    assert expected in ha.concept_problems({**concept, **change})
    assert ha.concept_problems(["not an object"]) == ["the answer is not an object"]


def test_a_statement_and_a_verdict_are_admitted_only_whole():
    assert ha.statement_problems({"primary": "  "}) == ["primary: no text"]
    assert ha.statement_problems({"primary": "Does it?"}) == []
    shown = ["N004"]
    assert ha.gate_problems({"verdict": "independent", "closest": None}, shown) == []
    assert ha.gate_problems({"verdict": "maybe"}, shown) == ["verdict: not one of the gatekeeper's"]
    assert ha.gate_problems({"verdict": "paraphrase", "closest": "N999"}, shown) == [
        "closest: a need it was not shown"
    ]
    assert ha.gate_problems({"verdict": "uncertain", "closest": 4}, shown) == [
        "closest: neither an id nor null"
    ]
    assert ha.evidence_gate_problems({"verdict": "independent"}) == []


# ---------------------------------------------------------------------------
# a search's answer
# ---------------------------------------------------------------------------


def test_a_complete_search_is_admitted():
    assert problems(search()) == []


@pytest.mark.parametrize(
    ("break_it", "expected"),
    [
        (lambda f: f["candidates"][0].update(rationale=""), "candidate 1: no rationale"),
        (lambda f: f["candidates"][0].update(rationale=None), "candidate 1: no rationale"),
        (lambda f: f["candidates"][1].update(grade=7), "candidate 2: grade not 0 to 3"),
        (lambda f: f["candidates"][1].update(grade=True), "candidate 2: grade not 0 to 3"),
        (lambda f: f["candidates"][1].update(grade="2"), "candidate 2: grade not 0 to 3"),
        (lambda f: f["candidates"][1].update(grade=None), "candidate 2: grade not 0 to 3"),
        (lambda f: f["candidates"][1].update(ambiguous="no"), "ambiguous not true or false"),
        (lambda f: f["candidates"][2].update(scope_basis=[]), "not a judgment the adjudication"),
        (lambda f: f["candidates"][2].update(scope_basis="DORA:64"), "scope_basis not a list"),
        (lambda f: f["candidates"][1].update(scope=["MiCA:999"]), "scope names a node that does"),
        (lambda f: f["candidates"].append(copy.deepcopy(f["candidates"][0])), "a node named twice"),
        (lambda f: f["candidates"][0].update(node="MiCA:999.9"), "no such node with own text"),
        (lambda f: f["candidates"][0].update(node="PSD2:10.1"), "an excluded PSD2 article"),
        (lambda f: f["candidates"][0].update(node="nonsense"), "no node of the corpus's form"),
        (lambda f: f["candidates"].append("MiCA:149.2"), "candidate 4: not an object"),
        (lambda f: f.update(candidates=None), "candidates: not a list"),
        (lambda f: f.update(answerability="mostly"), "answerability: not complete, partial"),
        (lambda f: f.update(answerability_reason=""), "answerability_reason: no text"),
        (lambda f: f.update(answerability="partial"), "gap_note: none for a need"),
        (lambda f: f.update(gap_note=3), "gap_note: neither text nor null"),
        (lambda f: f.update(problem=5), "problem: neither null nor text"),
    ],
)
def test_a_broken_search_is_never_admitted(break_it, expected):
    found = search()
    break_it(found)
    assert any(expected in p for p in problems(found)), problems(found)


def test_the_protocol_rules_on_a_search_file_are_invariants_named_by_code():
    """A search's answerability against its own grades: semantic invariants, as codes, never a
    node or a text; and never mixed with the role's contract, which comes first."""
    found = search()
    found["candidates"][0]["grade"] = 2  # complete, with no grade 3 left
    assert ha.search_verdict(found, TEXTS, EXCLUDED) == ha.Verdict([], ["complete_without_grade_3"])
    assert problems(found) == ["invariant: complete_without_grade_3"]
    found = search()
    found["answerability"], found["gap_note"] = "none", "the corpus lacks it"
    assert ha.search_verdict(found, TEXTS, EXCLUDED).violations == ["none_with_relevant_evidence"]
    found = search()
    found["answerability"], found["gap_note"] = "partial", "lacks"
    for c in found["candidates"]:
        c["grade"], c["scope_basis"] = 0, [c["node"]]
    assert ha.search_verdict(found, TEXTS, EXCLUDED).violations == [
        "partial_without_relevant_evidence"
    ]
    # a contract problem as well: the contract alone, no invariant judged
    found = search()
    found["candidates"][0]["grade"] = 2
    found["candidates"][1]["rationale"] = ""
    verdict = ha.search_verdict(found, TEXTS, EXCLUDED)
    assert verdict.problems and verdict.violations == []


def test_a_declared_problem_is_an_answer_of_its_own():
    """P1 #2: a search that declares a problem needs its text and nothing else; its transition
    is the pipeline's, before any candidate is read."""
    assert problems({"need_id": "x", "problem": "the need cannot be judged"}) == []
    assert ha.declared_problem({"problem": "the need cannot be judged"}) is not None
    for none in (None, "", "null"):
        assert ha.declared_problem({**search(), "problem": none}) is None
    # a problem with garbage candidates still admits: nothing of them is ever read
    assert problems({"problem": "x", "candidates": [{"node": 1}]}) == []


# ---------------------------------------------------------------------------
# judges B, C and D
# ---------------------------------------------------------------------------


def test_a_complete_judge_b_is_admitted():
    assert ha.judge_b_problems(judge_b(), B_IDS) == []


@pytest.mark.parametrize(
    ("break_it", "expected"),
    [
        (lambda b: b["grades"].pop(), "grades: 1 candidates missing"),
        (lambda b: b["grades"].append(copy.deepcopy(b["grades"][0])), "a candidate named twice"),
        (lambda b: b["grades"].append({**b["grades"][0], "item_id": UNKNOWN}), "unknown"),
        (lambda b: b["grades"][0].update(grade=4), "entry 1: grade not 0 to 3 or null"),
        (lambda b: b["grades"][0].update(reason=""), "entry 1: no reason"),
        (lambda b: b["grades"][2].update(scope_basis=[]), "not a judgment the adjudication"),
        (lambda b: b["grades"][0].update(item_id=None), "entry 1: names no candidate id"),
        (lambda b: b["grades"][0].pop("item_id"), "entry 1: names no candidate id"),
        (lambda b: b["grades"][0].update(node="DORA:64"), "an identity other than its id's"),
        (lambda b: b.update(answerability=None), "answerability: not complete"),
        (lambda b: b.update(answerability_reason=" "), "answerability_reason: no text"),
        (lambda b: b.update(answerability="partial"), "missing: none for a need"),
        (lambda b: b.update(grades={}), "grades: not a list"),
    ],
)
def test_a_broken_judge_b_is_never_admitted(break_it, expected):
    answer = judge_b()
    break_it(answer)
    found = ha.judge_b_problems(answer, B_IDS)
    assert any(expected in p for p in found), found


def test_a_judges_null_scope_basis_is_an_absent_one_but_not_for_a_grade_0():
    """As the adjudication record reads it: no stricter than the consumer, no looser."""
    answer = judge_b()
    answer["grades"][0]["scope_basis"] = None  # grade 3
    assert ha.judge_b_problems(answer, B_IDS) == []
    answer["grades"][2]["scope_basis"] = None  # grade 0
    assert any("not a judgment" in p for p in ha.judge_b_problems(answer, B_IDS))
    reply = judge_d()
    reply["items"][0]["grade"], reply["items"][0]["scope_basis"] = 2, None
    assert ha.judge_d_problems(reply, D_IDS) == []


def test_judge_b_may_leave_a_candidate_unjudged():
    answer = judge_b()
    answer["grades"][1]["grade"] = None
    assert ha.judge_b_problems(answer, B_IDS) == []


@pytest.mark.parametrize(
    ("break_it", "expected"),
    [
        (lambda c: c["items"].pop(), "items: 1 candidates missing"),
        (lambda c: c["items"].append(copy.deepcopy(c["items"][0])), "a candidate named twice"),
        (lambda c: c["items"][0].update(item_id=UNKNOWN), "items: 1 unknown candidates"),
        (lambda c: c["items"][0].pop("item_id"), "entry 1: names no item id"),
        (lambda c: c["items"][0].update(target="DORA:64"), "an identity other than its id's"),
        (lambda c: c["items"][0].update(answerability="complete"), "an answerability on a node"),
        (lambda c: c["items"][1].update(answerability="maybe"), "answerability not complete"),
        (lambda c: c["items"][1].update(grade=2), "a grade on the answerability"),
        (lambda c: c["items"][1].update(reason=""), "entry 2: no reason"),
        (lambda c: c.update(items=None), "items: not a list"),
    ],
)
def test_a_broken_judge_c_is_never_admitted(break_it, expected):
    assert ha.judge_c_problems(judge_c(), ITEMS) == []
    reply = judge_c()
    break_it(reply)
    found = ha.judge_c_problems(reply, ITEMS)
    assert any(expected in p for p in found), found


@pytest.mark.parametrize(
    ("break_it", "expected"),
    [
        (lambda d: d["items"].pop(), "items: 1 candidates missing"),
        (lambda d: d["items"].append(copy.deepcopy(d["items"][0])), "a candidate named twice"),
        (lambda d: d["items"].append({**d["items"][0], "item_id": UNKNOWN}), "unknown"),
        (lambda d: d["items"][0].update(grade=-1), "entry 1: grade not 0 to 3"),
        (lambda d: d["items"][0].update(ambiguous=None), "ambiguous not true or false"),
        (lambda d: d["items"][0].update(item_id=None), "entry 1: names no item id"),
        (lambda d: d["items"][0].update(target="MiCA:149.2"), "an identity other than its id's"),
        (lambda d: d["items"][1].update(answerability=None), "answerability not complete"),
    ],
)
def test_a_broken_judge_d_is_never_admitted(break_it, expected):
    assert ha.judge_d_problems(judge_d(), D_IDS) == []
    reply = judge_d()
    break_it(reply)
    found = ha.judge_d_problems(reply, D_IDS)
    assert any(expected in p for p in found), found


def test_the_metadata_answer_is_checked_as_data_before_its_labels():
    assert ha.metadata_shape_problems({"type": "direct", "difficulty": "easy", "drivers": {}}) == []
    assert "drivers: not an object" in ha.metadata_shape_problems(
        {"type": "direct", "difficulty": "easy", "drivers": []}
    )
    assert "drivers: not lists of nodes" in ha.metadata_shape_problems(
        {"type": "direct", "difficulty": "easy", "drivers": {"negation": [{"x": 1}]}}
    )


def test_what_admission_accepts_the_assembly_can_always_build():
    """One source of truth: the labels admitted are the very judgments the adjudication record
    holds, and the gap note a partial answerability takes is there."""
    a_file = search()
    b = judge_b()
    b["answerability"], b["missing"] = "partial", "the corpus lacks the period"
    b["grades"][1]["grade"] = 1  # a disagreement: judge C decides it
    assert problems(a_file) == [] and ha.judge_b_problems(b, B_IDS) == []
    a = hj.a_judgments(a_file)
    found = hj.disagreements(CANDIDATES, a, hj.b_judgments(b, B_IDS))
    assert found.nodes == ["MiCA:149.3"] and found.answerability
    c = judge_c()
    assert ha.judge_c_problems(c, ITEMS) == []
    answers = {i.target: entry for i, entry in zip(ITEMS, c["items"], strict=True)}
    texts = {k: v for k, v in TEXTS.items()}
    assembled = hj.assemble("N101", CANDIDATES, a, hj.b_judgments(b, B_IDS), answers, None, texts)
    # the final answerability is judge B's, so its gap note is judge B's: there to be taken
    assert assembled.answerability == "partial"
    assert assembled.gap_note == "the corpus lacks the period"


def test_a_judge_c_label_is_checked_as_the_assembly_reads_it():
    """The assembly prefers a "rationale" over a "reason" (hj.as_judgment): a blank rationale
    beside a good reason is refused before admission, never found at assembly."""
    reply = judge_c()
    reply["items"][0]["rationale"] = "   "
    assert ha.judge_c_problems(reply, ITEMS) == [
        "entry 1: not a judgment the adjudication record can hold"
    ]
    with pytest.raises(ValueError):
        hj.as_judgment(reply["items"][0])  # what the assembly would have met
    for entry in judge_c()["items"]:
        hj.as_judgment(entry)  # every admitted label builds
