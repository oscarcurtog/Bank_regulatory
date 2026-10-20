"""From judges A, B and C to qrels and adjudications, and the audit of a context from its
transcript (S3, D-054). Synthetic judgments and transcripts; no corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingest import benchmark as bm
from ingest import holdout_judging as hj

TEXTS = {("DORA", "19.1"): "report", ("DORA", "19.2"): "notify", ("PSD2", "96.1"): "old"}


def _a_file(**overrides) -> dict:
    data = {
        "need_id": "N101",
        "candidates": [
            {"node": "DORA:19.1", "grade": 3, "rationale": "it says report", "scope": []},
            {"node": "DORA:19.2", "grade": 0, "scope_basis": ["DORA:19.2"], "rationale": "no"},
        ],
        "answerability": "complete",
        "answerability_reason": "19.1 answers",
        "gap_note": None,
    }
    return data | overrides


def test_a_clean_judge_a_file_has_no_problem():
    assert hj.judge_a_problems(_a_file(), TEXTS, frozenset({"96"})) == []


@pytest.mark.parametrize(
    ("candidates", "answerability", "expected"),
    [
        ([{"node": "DORA:99.9", "grade": 1, "rationale": "r"}], "partial", "no such node"),
        ([{"node": "PSD2:96.1", "grade": 1, "rationale": "r"}], "partial", "excluded PSD2"),
        ([{"node": "DORA:19.1", "grade": 0, "rationale": "r"}], "none", "without scope_basis"),
        ([{"node": "DORA:19.1", "grade": 2, "rationale": "r"}], "complete", "without a non-amb"),
        ([{"node": "DORA:19.1", "grade": 2, "rationale": "r"}] * 2, "partial", "listed twice"),
    ],
)
def test_judge_a_problems(candidates, answerability, expected):
    data = _a_file(candidates=candidates, answerability=answerability, gap_note="the RTS")
    assert any(expected in p for p in hj.judge_a_problems(data, TEXTS, frozenset({"96"})))


def _b_file(grade_191=3, answerability="complete", missing=None) -> dict:
    return {
        "need_id": "N101",
        "grades": [
            {"node": "DORA:19.1", "grade": grade_191, "reason": "B on 19.1"},
            {"node": "DORA:19.2", "grade": 0, "scope_basis": ["DORA:19.2"], "reason": "B no"},
        ],
        "answerability": answerability,
        "answerability_reason": "B",
        "missing": missing,
    }


def test_agreement_needs_no_judge_c():
    a, b = hj.a_judgments(_a_file()), hj.b_judgments(_b_file())
    found = hj.disagreements(["DORA:19.1", "DORA:19.2"], a, b)
    assert found == hj.Disagreements([], False)
    out = hj.assemble("N101", ["DORA:19.1", "DORA:19.2"], a, b, {}, None, TEXTS)
    assert [(q.path, q.grade) for q in out.qrels] == [("19.1", 3), ("19.2", 0)]
    assert len(out.adjudications) == 3 and all(r.c is None for r in out.adjudications)
    assert out.answerability == "complete" and out.gap_note is None


def test_every_disagreement_goes_to_judge_c_and_c_decides():
    a, b = hj.a_judgments(_a_file()), hj.b_judgments(_b_file(grade_191=2))
    found = hj.disagreements(["DORA:19.1", "DORA:19.2"], a, b)
    assert found.nodes == ["DORA:19.1"] and not found.answerability
    with pytest.raises(ValueError, match="judge C answered"):
        hj.assemble("N101", ["DORA:19.1", "DORA:19.2"], a, b, {}, None, TEXTS)
    c = {"DORA:19.1": {"grade": 3, "reason": "C: it states the duty"}}
    out = hj.assemble("N101", ["DORA:19.1", "DORA:19.2"], a, b, c, None, TEXTS)
    assert out.qrels[0].grade == 3 and out.qrels[0].rationale == "C: it states the duty"
    assert out.qrels[0].text_sha256 == bm.sha256_text("report")


def test_the_gap_note_comes_from_the_judge_whose_answerability_is_final():
    a = hj.a_judgments(
        _a_file(answerability="partial", gap_note="A: the fee is in a delegated act")
    )
    b = hj.b_judgments(_b_file(answerability="none", missing="B: nothing"))
    c_takes_b = {"answerability": {"answerability": "none", "reason": "C"}}
    # with 19.1 at grade 3 a "none" would contradict the dataset rules later; only the
    # provenance of the gap_note is under test here
    out = hj.assemble("N101", ["DORA:19.1", "DORA:19.2"], a, b, c_takes_b, "A gap", TEXTS)
    assert out.answerability == "none" and out.gap_note == "B: nothing"
    c_takes_a = {"answerability": {"answerability": "partial", "reason": "C"}}
    out = hj.assemble("N101", ["DORA:19.1", "DORA:19.2"], a, b, c_takes_a, "A gap", TEXTS)
    assert out.gap_note == "A gap"


def _transcript(tmp_path, calls, attachments=()) -> Path:
    lines = [{"type": "attachment", "attachment": a} for a in attachments]
    lines += [
        {"message": {"content": [{"type": "tool_use", "name": n, "input": i}]}} for n, i in calls
    ]
    path: Path = tmp_path / "agent.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    return path


def test_the_audit_accepts_a_context_that_read_its_packet_and_answered(tmp_path):
    packet = tmp_path / "packet.txt"
    path = _transcript(
        tmp_path,
        [("Read", {"file_path": str(packet)}), ("SubagentHandback", {"message": '{"ok": 1}'})],
    )
    audit = hj.audit_transcript(path, packet)
    assert audit.ok and audit.answer == '{"ok": 1}'
    assert hj.as_json("```json\n" + audit.answer + "\n```") == {"ok": 1}


def test_the_audit_finds_other_reads_tools_and_project_context(tmp_path):
    packet = tmp_path / "packet.txt"
    path = _transcript(
        tmp_path,
        [
            ("Read", {"file_path": str(packet)}),
            ("Read", {"file_path": "/repo/data/benchmark/dev/needs.jsonl"}),
            ("Bash", {"command": "ls"}),
            ("SubagentHandback", {"message": "{}"}),
        ],
        attachments=[
            {"type": "instructions", "files": [{"type": "Project"}, {"type": "AutoMem"}]},
            {"type": "session_context", "context": {}},
        ],
    )
    audit = hj.audit_transcript(path, packet, forbidden_context=hj.PROJECT_CONTEXT)
    assert not audit.ok
    assert audit.context == ["AutoMem", "Project", "session_context"]
    assert len(audit.problems) == 4


def test_judge_a_may_run_only_the_corpus_tools_and_write_only_its_file(tmp_path):
    packet, out = tmp_path / "packet.txt", tmp_path / "N101.json"
    tool = "cd repo && PYTHONPATH=. .venv/bin/python scripts/evidence_tools.py "
    calls = [
        ("Read", {"file_path": str(packet)}),
        ("Bash", {"command": tool + "article DORA 19"}),
        ("Write", {"file_path": str(out)}),
        ("SubagentHandback", {"message": "N101 written"}),
    ]
    allowed = frozenset({"Read", "Bash", "Write", "SubagentHandback"})
    good = hj.audit_transcript(_transcript(tmp_path, calls), packet, allowed, out, tool)
    assert good.ok
    calls.insert(2, ("Bash", {"command": "cat data/benchmark/dev/qrels.jsonl"}))
    bad = hj.audit_transcript(_transcript(tmp_path, calls), packet, allowed, out, tool)
    assert bad.problems == [
        "commands outside the corpus tools: ['cat data/benchmark/dev/qrels.jsonl']"
    ]


def _final() -> hj.Assembled:
    a, b = hj.a_judgments(_a_file()), hj.b_judgments(_b_file())
    return hj.assemble("N101", ["DORA:19.1", "DORA:19.2"], a, b, {}, None, TEXTS)


def test_the_diagnostic_crosses_a_material_line_only_across_relevance_or_answerability():
    final = _final()  # 19.1 grade 3, 19.2 grade 0, complete
    same_side = {
        "grades": [{"node": "DORA:19.1", "grade": 2}, {"node": "DORA:19.2", "grade": 1}],
        "answerability": "complete",
    }
    assert (
        hj.material_discrepancies(["DORA:19.1", "DORA:19.2"], final, same_side).discrepancies == []
    )
    crossed = {
        "grades": [
            {"node": "DORA:19.1", "grade": 3, "ambiguous": True},
            {"node": "DORA:19.2", "grade": 2},
        ],
        "answerability": "partial",
    }
    found = hj.material_discrepancies(["DORA:19.1", "DORA:19.2"], final, crossed)
    assert found.discrepancies == ["DORA:19.1", "DORA:19.2", "answerability"]
    partial = {"grades": [{"node": "DORA:19.1", "grade": 3}], "answerability": "complete"}
    found = hj.material_discrepancies(["DORA:19.1", "DORA:19.2"], final, partial)
    assert found.discrepancies == [] and found.missing == ["DORA:19.2"]


def test_judge_d_holds_the_consensus_or_leaves_it_uncertain():
    final = _final()
    outcome = hj.judge_d_outcome(
        final,
        [
            {"target": "DORA:19.1", "grade": 2},
            {"target": "DORA:19.2", "grade": 2},
            {"target": "answerability", "answerability": "none"},
        ],
    )
    assert outcome == {
        "DORA:19.1": "consensus_held",
        "DORA:19.2": "adjudication_uncertain",
        "answerability": "adjudication_uncertain",
    }


# judge D: ambiguity takes precedence (Codex Gate 2, P1.6). The final label of DORA:19.1 is
# grade 3 (relevant) and of DORA:19.2 grade 0 (not relevant).
@pytest.mark.parametrize(
    ("target", "grade", "ambiguous", "expected"),
    [
        ("DORA:19.1", 3, False, "consensus_held"),
        ("DORA:19.1", 2, False, "consensus_held"),
        ("DORA:19.1", 1, False, "adjudication_uncertain"),
        ("DORA:19.1", 0, False, "adjudication_uncertain"),
        ("DORA:19.1", 3, True, "adjudication_uncertain"),
        ("DORA:19.2", 0, False, "consensus_held"),
        ("DORA:19.2", 1, False, "consensus_held"),
        ("DORA:19.2", 2, False, "adjudication_uncertain"),
        ("DORA:19.2", 3, False, "adjudication_uncertain"),
        ("DORA:19.2", 0, True, "adjudication_uncertain"),
        ("DORA:19.2", 1, True, "adjudication_uncertain"),
        ("DORA:19.2", None, True, "adjudication_uncertain"),
    ],
)
def test_judge_d_regressions(target, grade, ambiguous, expected):
    item = {"target": target, "grade": grade, "ambiguous": ambiguous}
    assert hj.judge_d_outcome(_final(), [item]) == {target: expected}


def _q(path: str, grade: int, act: str = "DORA", ambiguous: bool = False) -> bm.Qrel:
    return bm.Qrel(
        need_id="N101",
        act=act,  # type: ignore[arg-type]
        path=path,
        grade=grade,
        rationale="r",
        text_sha256="a" * 64,
        ambiguous=ambiguous,
    )


def test_the_protocol_fixes_acts_type_and_cross_act_from_the_qrels():
    qrels = [_q("19.1", 3), _q("4.16", 2, act="PSD2"), _q("19.2", 0)]
    fixed = hj.deterministic_metadata(qrels, "complete", ["DORA"])
    assert fixed["acts"] == ["DORA", "PSD2"] and fixed["cross_act"] and fixed["type"] is None
    none = hj.deterministic_metadata([_q("19.2", 1)], "none", ["MiCA"])
    assert none["acts"] == ["MiCA"] and none["type"] == "partial_out_of_scope"
    assert "the acts the question was written for" in none["acts_source"]


def test_the_metadata_answer_must_keep_to_the_rubric_and_the_grades():
    qrels = [_q("19.1", 3), _q("19.4", 2), _q("19.2", 0)]
    good = {
        "type": "direct",
        "difficulty": "easy",
        "drivers": {"negation": ["DORA:19.1"], "lexical_twin": ["DORA:19.2"]},
    }
    assert hj.metadata_problems(good, qrels, "complete") == []
    bad = {
        "type": "partial_out_of_scope",
        "difficulty": "trivial",
        "drivers": {
            "cross_act": ["DORA:19.1"],
            "negation": ["DORA:19.4"],
            "exception": ["DORA:19.2"],
            "numeric": ["gap_note"],
        },
    }
    problems = hj.metadata_problems(bad, qrels, "complete")
    assert len(problems) == 6


def test_final_metadata_adds_the_deterministic_fields():
    qrels = [_q("19.1", 3), _q("4.16", 2, act="PSD2")]
    meta = hj.final_metadata(
        {"type": "multi_provision", "difficulty": "hard", "drivers": {}},
        qrels,
        "complete",
        ["DORA"],
    )
    assert meta.type == "multi_provision" and meta.acts == ["DORA", "PSD2"]
    assert meta.drivers == ["cross_act"] and meta.source["type"] == "metadata adjudicator"
    partial = hj.final_metadata(
        {
            "type": "partial_out_of_scope",
            "difficulty": "medium",
            "drivers": {"numeric": ["gap_note"]},
        },
        [_q("19.1", 2)],
        "partial",
        ["DORA"],
    )
    assert partial.type == "partial_out_of_scope" and partial.source["type"].startswith(
        "deterministic"
    )
    with pytest.raises(ValueError, match="unusable"):
        hj.final_metadata(
            {"type": "direct", "difficulty": "x", "drivers": {}}, qrels, "complete", ["DORA"]
        )


CANDIDATES = ["MiCA:149.1", "MiCA:149.2", "DORA:64"]


def _diagnostic(**changes):
    grades = [
        {"node": n, "grade": g, "ambiguous": False}
        for n, g in zip(CANDIDATES, (1, 3, 0), strict=True)
    ]
    return {"need_id": "x", "grades": grades, "answerability": "complete"} | changes


@pytest.mark.parametrize(
    ("name", "answer", "expected"),
    [
        ("empty list", _diagnostic(grades=[]), "not graded"),
        ("strict subset", _diagnostic(grades=_diagnostic()["grades"][:2]), "DORA:64: not graded"),
        (
            "duplicate candidate",
            _diagnostic(grades=[*_diagnostic()["grades"], _diagnostic()["grades"][0]]),
            "graded more than once",
        ),
        (
            "unknown candidate",
            _diagnostic(
                grades=[
                    *_diagnostic()["grades"],
                    {"node": "MiCA:1", "grade": 0, "ambiguous": False},
                ]
            ),
            "MiCA:1: not a candidate",
        ),
        (
            "invalid grade",
            _diagnostic(
                grades=[{**_diagnostic()["grades"][0], "grade": 4}, *_diagnostic()["grades"][1:]]
            ),
            "is not 0 to 3 or null",
        ),
        (
            "a boolean grade",
            _diagnostic(
                grades=[{**_diagnostic()["grades"][0], "grade": True}, *_diagnostic()["grades"][1:]]
            ),
            "is not 0 to 3 or null",
        ),
        (
            "missing ambiguous",
            _diagnostic(grades=[{"node": CANDIDATES[0], "grade": 1}, *_diagnostic()["grades"][1:]]),
            "no true or false ambiguous",
        ),
        (
            "missing answerability",
            {k: v for k, v in _diagnostic().items() if k != "answerability"},
            "answerability None",
        ),
        ("invalid answerability", _diagnostic(answerability="mostly"), "answerability 'mostly'"),
        ("no grades at all", {"answerability": "complete"}, "no list of grades"),
        ("not an object", ["MiCA:149.1"], "not a JSON object"),
    ],
)
def test_an_incomplete_diagnostic_is_never_complete(name, answer, expected):
    problems = hj.diagnostic_problems(CANDIDATES, answer)
    assert any(expected in p for p in problems), (name, problems)


def test_a_complete_diagnostic_passes_with_null_and_ambiguous_labels():
    grades = _diagnostic()["grades"]
    grades[0] = {**grades[0], "grade": None}  # unjudged, as judge B's packet allows
    grades[1] = {**grades[1], "ambiguous": True}
    assert hj.diagnostic_problems(CANDIDATES, _diagnostic(grades=grades)) == []
    assert hj.diagnostic_problems(CANDIDATES, _diagnostic()) == []
    assert hj.diagnostic_problems([], _diagnostic(grades=[], answerability="none")) == []
