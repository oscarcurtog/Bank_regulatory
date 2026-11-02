"""The deterministic split, contexts and negatives of S7-S8, on a synthetic world of needs."""

from __future__ import annotations

import random

import pytest

from ingest import answer_cases as ac

ACTS = ("DORA", "MiCA", "PSD2")


def _needs() -> list[ac.NeedInfo]:
    """24 complete needs (7 DORA, 10 MiCA, 5 PSD2, 2 cross-act), 3 partial, 3 none: DEV's shape."""
    out = []
    n = 4
    for act, count in (("DORA", 7), ("MiCA", 10), ("PSD2", 5)):
        for _ in range(count):
            out.append(ac.NeedInfo(f"N{n:03d}", "complete", (act,)))
            n += 1
    for _ in range(2):
        out.append(ac.NeedInfo(f"N{n:03d}", "complete", ("DORA", "PSD2")))
        n += 1
    out += [ac.NeedInfo(f"N{n + i:03d}", "partial", ("MiCA",)) for i in range(3)]
    out += [ac.NeedInfo(f"N{n + 3 + i:03d}", "none", ("PSD2",)) for i in range(3)]
    return out


def test_the_split_has_two_single_act_needs_per_act_and_the_rest_for_evaluation() -> None:
    needs = _needs()
    split = ac.make_split(needs)
    acts = {n.need_id: n.acts for n in needs}
    assert len(split.prompt_dev) == 6 and len(split.evaluation) == 18
    assert sorted(acts[i][0] for i in split.prompt_dev) == [
        "DORA",
        "DORA",
        "MiCA",
        "MiCA",
        "PSD2",
        "PSD2",
    ]
    assert all(
        len(acts[i]) == 1 for i in split.prompt_dev
    )  # cross-act needs never go to prompt dev
    assert not set(split.prompt_dev) & set(split.evaluation)
    assert len(split.partial) == 3 and len(split.none) == 3
    complete = {n.need_id for n in needs if n.answerability == "complete"}
    assert set(split.prompt_dev) | set(split.evaluation) == complete


def test_the_split_does_not_depend_on_the_order_of_the_input() -> None:
    needs = _needs()
    shuffled = needs[:]
    random.Random(1).shuffle(shuffled)
    assert ac.make_split(needs) == ac.make_split(shuffled)


def test_the_split_changes_with_the_seed_and_fails_without_two_needs_per_act() -> None:
    needs = _needs()
    assert ac.make_split(needs, "another seed").prompt_dev != ac.make_split(needs).prompt_dev
    few = [n for n in needs if not (n.acts == ("PSD2",) and n.answerability == "complete")][:]
    with pytest.raises(ac.CaseError):
        ac.make_split(few)


def test_hash_order_is_fixed_by_sha256_and_not_by_the_input_order() -> None:
    items = ["N004", "N005", "N006", "N007"]
    assert ac.hash_order(items, "k") == ac.hash_order(items[::-1], "k")
    # golden: sha256 of "s7-s8-v1|k|<id>" is 67ca…, b330…, e7a8…, eb60… for N004, N007, N005, N006
    assert ac.hash_order(items, "k") == ["N004", "N007", "N005", "N006"]


GRADES = {"R3": 3, "R2": 2, "R2b": 2, "G1": 1, "G1b": 1, "G0": 0, "G0b": 0}


def test_the_gold_context_is_the_known_relevant_evidence_by_grade_then_id() -> None:
    assert ac.gold_known_context(GRADES) == ("R3", "R2", "R2b")


def test_the_insufficient_context_holds_only_chunks_judged_grade_0_or_1() -> None:
    ranking = ["U9", "G1b", "R3", "G0", "U8", "G1", "R2", "G0b"]
    context = ac.insufficient_context(GRADES, ranking)
    assert context == ("G1b", "G0", "G1", "G0b")  # E5's order; no unjudged, no grade 2 or 3
    assert ac.insufficient_context(GRADES, ranking, minimum=5) is None
    assert ac.insufficient_context(GRADES, ranking, maximum=2, minimum=1) == ("G1b", "G0")


def _world() -> tuple[ac.Split, dict[str, tuple[str, ...]], dict[str, dict[str, int]]]:
    needs = _needs()
    split = ac.make_split(needs)
    acts = {n.need_id: n.acts for n in needs}
    grades = {
        n.need_id: {
            f"{n.need_id}:rel": 3,
            f"{n.need_id}:a": 1,
            f"{n.need_id}:b": 0,
            f"{n.need_id}:c": 0,
        }
        for n in needs
    }
    return split, acts, grades


def test_the_negative_selection_has_six_and_six_disjoint_valid_needs() -> None:
    split, acts, grades = _world()
    selection = ac.select_negatives(
        split, acts, grades, contexts_of=lambda b: {f"{b}:x"}, has_insufficient=lambda n: True
    )
    assert len(selection.insufficient) == 6 and len(selection.mismatched) == 6
    assert set(selection.insufficient) <= set(split.evaluation)
    questions = [a for a, _ in selection.mismatched]
    assert not set(questions) & set(selection.insufficient) and len(set(questions)) == 6
    for a, b in selection.mismatched:
        assert a != b and not set(acts[a]) & set(acts[b])  # a different act
    again = ac.select_negatives(
        split, acts, grades, contexts_of=lambda b: {f"{b}:x"}, has_insufficient=lambda n: True
    )
    assert again == selection  # deterministic


def test_a_partner_whose_context_holds_known_relevant_evidence_of_the_question_is_refused() -> None:
    split, acts, grades = _world()
    first = ac.select_negatives(split, acts, grades, lambda b: set(), lambda n: True)
    a, b = first.mismatched[0]
    blocked = ac.select_negatives(
        split, acts, grades, lambda x: {f"{a}:rel"} if x == b else set(), lambda n: True
    )
    assert (a, b) not in blocked.mismatched


def test_the_selection_fails_if_six_cases_cannot_be_found() -> None:
    split, acts, grades = _world()
    with pytest.raises(ac.CaseError):
        ac.select_negatives(
            split, acts, grades, lambda b: set(), lambda n: n == split.evaluation[0]
        )
    same_act = {n: ("MiCA",) for n in acts}
    with pytest.raises(ac.CaseError):
        ac.select_negatives(split, same_act, grades, lambda b: set(), lambda n: True)


def _rankings(split: ac.Split, acts: dict[str, tuple[str, ...]]):
    questions = {n: ac.Question(f"{n}-en-1", f"Question {n}?") for n in acts}
    e5 = {
        q.query_id: [f"{n}:rel"] + [f"{n}:e5{i}" for i in range(20)] for n, q in questions.items()
    }
    e = {q.query_id: [f"{n}:rel"] + [f"{n}:e{i}" for i in range(20)] for n, q in questions.items()}
    return questions, e5, e


def test_the_positive_cases_have_the_four_contexts_in_the_documented_shapes() -> None:
    split, acts, grades = _world()
    questions, e5, e = _rankings(split, acts)
    cases = ac.positive_cases(split.evaluation, questions, grades, e5, e)
    assert len(cases) == 18 * 4
    by = {(c.need_id, c.context_name): c for c in cases}
    need = split.evaluation[0]
    assert by[(need, ac.GOLD_KNOWN)].chunk_ids == (f"{need}:rel",)
    assert by[(need, ac.E5_10)].chunk_ids == tuple(e5[f"{need}-en-1"][:10])
    assert by[(need, ac.E5_5)].chunk_ids == tuple(e5[f"{need}-en-1"][:5])
    assert by[(need, ac.E_10)].chunk_ids == tuple(e[f"{need}-en-1"][:10])
    assert len({c.case_id for c in cases}) == len(cases)


def test_the_negative_cases_are_the_fifteen_as_twenty_four_generations() -> None:
    split, acts, grades = _world()
    questions, e5, e = _rankings(split, acts)
    for n in grades:  # give each need judged non-relevant chunks inside E5's ranking
        e5[f"{n}-en-1"][1:4] = [f"{n}:a", f"{n}:b", f"{n}:c"]
    selection = ac.select_negatives(
        split,
        acts,
        grades,
        lambda b: set(e5[f"{b}-en-1"][:10]) | set(e[f"{b}-en-1"][:10]),
        lambda n: ac.insufficient_context(grades[n], e5[f"{n}-en-1"]) is not None,
    )
    cases = ac.negative_cases(split, selection, questions, grades, e5, e)
    kinds = [c.kind for c in cases]
    assert kinds.count(ac.NEG_NATURAL) == 6 and kinds.count(ac.NEG_INSUFFICIENT) == 6
    assert kinds.count(ac.NEG_MISMATCHED) == 12 and len(cases) == 24
    for c in cases:
        if c.kind == ac.NEG_INSUFFICIENT:
            assert c.context_name == ac.INSUFFICIENT
            assert all(grades[c.need_id].get(x, -1) in (0, 1) for x in c.chunk_ids)
        if c.kind == ac.NEG_MISMATCHED:
            assert c.source_need_id and all(x.startswith(c.source_need_id) for x in c.chunk_ids)
            assert not any(x.startswith(c.need_id) for x in c.chunk_ids)
