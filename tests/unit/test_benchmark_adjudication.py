"""Adjudication records (PROTOCOL.md, section 12): judge A, then judge B without
seeing A, then judge C for every disagreement.

These tests pin what the record enforces. What a judge decides (whether a
definition is necessary, which nodes of a referred article matter, whether a scope
rules a node out) stays annotation policy and is not automated here."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from ingest import benchmark as bm


def judgment(
    grade: int | None = None,
    *,
    ambiguous: bool = False,
    answerability: str | None = None,
    scope: list[str] | None = None,
    why: str = "Quotes the text.",
) -> dict[str, Any]:
    data: dict[str, Any] = {"rationale": why}
    if grade is not None:
        data["grade"] = grade
        if grade == 0 and scope is None:
            scope = ["MiCA:149.2"]
    if ambiguous:
        data["ambiguous"] = True
    if answerability is not None:
        data["answerability"] = answerability
    if scope is not None:
        data["scope_basis"] = scope
    return data


def record(a, b, c=None, target: str = "MiCA:149.2", **extra) -> bm.Adjudication:
    data = {"need_id": "N001", "target": target, "a": a, "b": b} | extra
    if c is not None:
        data["c"] = c
    return bm.Adjudication.model_validate(data)


def need(answerability: str = "complete", status: str = "ready") -> bm.Need:
    partial = answerability != "complete"
    return bm.Need.model_validate(
        {
            "need_id": "N001",
            "split": "dev",
            "statement": "From what date does MiCA apply?",
            "type": "partial_out_of_scope" if partial else "direct",
            "drivers": [],
            "difficulty": "easy",
            "acts": ["MiCA"],
            "answerability": answerability,
            "gap_note": "Elsewhere." if partial else None,
            "status": status,
        }
    )


def qrel(path: str, grade: int, ambiguous: bool = False) -> bm.Qrel:
    return bm.Qrel.model_validate(
        {
            "need_id": "N001",
            "act": "MiCA",
            "path": path,
            "grade": grade,
            "rationale": "Quotes the text.",
            "text_sha256": "0" * 64,
            "ambiguous": ambiguous,
        }
    )


ANSWERABILITY = record(
    judgment(answerability="complete"), judgment(answerability="complete"), target="answerability"
)


# ---------------------------------------------------------------------------
# one record
# ---------------------------------------------------------------------------


def test_an_agreement_is_final_without_judge_c():
    agreed = record(judgment(3), judgment(3, why="Other words, the same grade."))
    assert agreed.agree and agreed.final.grade == 3 and not agreed.crosses_threshold


@pytest.mark.parametrize(
    "a,b",
    [
        (judgment(2), judgment(3)),  # 2 <-> 3
        (judgment(1), judgment(2)),  # 1 <-> 2, across the relevance line
        (judgment(0), judgment(1)),  # a hard negative against context
        (judgment(2, ambiguous=True), judgment(2)),  # ambiguous against judged
        (judgment(None), judgment(1)),  # unjudged against judged
    ],
)
def test_a_disagreement_is_never_final_without_judge_c(a, b):
    with pytest.raises(ValidationError, match="never final without judge C"):
        record(a, b)


@pytest.mark.parametrize("a,b,crosses", [(1, 2, True), (3, 0, True), (2, 3, False), (0, 1, False)])
def test_a_disagreement_records_whether_it_crosses_the_relevance_line(a, b, crosses):
    assert record(judgment(a), judgment(b), c=judgment(b)).crosses_threshold is crosses


def test_judge_c_may_take_a_label_neither_judge_gave():
    assert record(judgment(1), judgment(3), c=judgment(2)).final.grade == 2


def test_judge_c_may_find_the_node_ambiguous():
    final = record(judgment(1), judgment(2), c=judgment(2, ambiguous=True)).final
    assert final.ambiguous and final.grade == 2 and not final.relevant


def test_judge_c_may_leave_the_node_unjudged():
    assert record(judgment(0), judgment(1), c=judgment(None)).final.grade is None


def test_judge_c_on_an_agreement_needs_the_reason():
    with pytest.raises(ValidationError, match="escalated"):
        record(judgment(2), judgment(2), c=judgment(3))
    with pytest.raises(ValidationError, match="needs judge C"):
        record(judgment(2), judgment(2), escalated="Both break a dataset rule.")
    reason = "Both give grade 2, and the complete need would have no grade 3."
    assert record(judgment(2), judgment(2), c=judgment(3), escalated=reason).final.grade == 3


def test_a_grade_0_names_the_provisions_that_fix_the_scope():
    with pytest.raises(ValidationError, match="scope_basis"):
        bm.Judgment.model_validate({"grade": 0, "rationale": "Another regime."})
    scoped = bm.Judgment.model_validate(judgment(0, scope=["PSD2:77.1", "PSD2:64.1"]))
    assert scoped.scope_basis == ["PSD2:77.1", "PSD2:64.1"]
    with pytest.raises(ValidationError):
        bm.Judgment.model_validate(judgment(0, scope=["PSD2 77(1)"]))  # not a node reference


def test_an_ambiguous_judgment_still_gives_its_best_grade():
    with pytest.raises(ValidationError, match="best grade"):
        bm.Judgment.model_validate({"ambiguous": True, "rationale": "Unclear."})


def test_answerability_is_adjudicated_like_a_grade():
    decided = record(
        judgment(answerability="partial"),
        judgment(answerability="none"),
        c=judgment(answerability="partial"),
        target="answerability",
    )
    assert decided.final.answerability == "partial"
    with pytest.raises(ValidationError, match="never final without judge C"):
        record(
            judgment(answerability="complete"),
            judgment(answerability="partial"),
            target="answerability",
        )
    with pytest.raises(ValidationError, match="answerability target"):
        record(judgment(2), judgment(2), target="answerability")
    with pytest.raises(ValidationError, match="answerability target"):
        record(judgment(answerability="none"), judgment(answerability="none"))


# ---------------------------------------------------------------------------
# the record against the data
# ---------------------------------------------------------------------------


def test_the_final_labels_are_the_qrels():
    records = [record(judgment(2), judgment(3), c=judgment(3)), ANSWERABILITY]
    assert bm.adjudication_problems(records, [need()], [qrel("149.2", 3)], require_all=True) == []


@pytest.mark.parametrize(
    "records,qrels,message",
    [
        ([record(judgment(3), judgment(3))], [qrel("149.2", 2)], "the qrel has 2"),
        (
            [record(judgment(1), judgment(2), c=judgment(2, ambiguous=True))],
            [qrel("149.2", 2)],
            "(ambiguous True), the qrel has 2 (ambiguous False)",
        ),
        ([record(judgment(None), judgment(None))], [qrel("149.2", 1)], "but a qrel exists"),
        ([record(judgment(1), judgment(1))], [], "but there is no qrel"),
        (
            [record(judgment(1), judgment(1)), record(judgment(1), judgment(1))],
            [qrel("149.2", 1)],
            "appears 2 times",
        ),
    ],
)
def test_a_record_that_disagrees_with_the_qrels_is_refused(records, qrels, message):
    problems = bm.adjudication_problems(records, [need()], qrels, require_all=False)
    assert any(message in p for p in problems), problems


def test_the_final_answerability_is_the_needs():
    problems = bm.adjudication_problems(
        [
            record(
                judgment(answerability="none"),
                judgment(answerability="none"),
                target="answerability",
            )
        ],
        [need("partial")],
        [],
        require_all=False,
    )
    assert problems == ["adjudication N001 answerability: final 'none', the need says 'partial'"]


def test_with_a_record_every_ready_need_is_fully_adjudicated():
    qrels = [qrel("149.2", 3), qrel("149.1", 1)]
    problems = bm.adjudication_problems(
        [record(judgment(3), judgment(3))], [need()], qrels, require_all=True
    )
    assert problems == [
        "need N001: no adjudication of its answerability",
        "qrel N001 MiCA:149.1: no adjudication record",
    ]
    draft = bm.adjudication_problems([], [need(status="draft")], [], require_all=True)
    assert draft == []  # judged later: the record comes with the qrels


def test_definitions_graded_2_or_3_are_listed_for_the_necessity_check():
    qrels = [
        bm.Qrel.model_validate(
            {
                "need_id": "N001",
                "act": act,
                "path": path,
                "grade": grade,
                "rationale": "Quotes the text.",
                "text_sha256": "0" * 64,
            }
        )
        for act, path, grade in [
            ("DORA", "3.22", 2),  # a definition at 2: listed
            ("DORA", "3.19", 1),  # a definition at 1: not listed
            ("PSD2", "4.23", 3),  # PSD2 defines in Article 4
            ("PSD2", "3.1", 2),  # PSD2 Article 3 is not its definitions article
            ("MiCA", "30.1", 2),  # not a definition
        ]
    ]
    assert [(r.act, r.path) for r in bm.definition_qrels(qrels)] == [
        ("DORA", "3.22"),
        ("PSD2", "4.23"),
    ]
