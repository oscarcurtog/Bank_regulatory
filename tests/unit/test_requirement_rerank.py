"""E's requirement reranking (ingest/requirement_rerank.py): the reorder within E5's 50, the best
rank over a query's requirements, and the requirement-level recall and MRR, macro by need."""

from __future__ import annotations

from ingest import benchmark as bm
from ingest import requirement_rerank as rq


def _query(qid: str) -> bm.Query:
    return bm.Query.model_validate(
        {
            "query_id": qid,
            "need_id": qid[:4],
            "lang": "en",
            "style": "natural",
            "origin": "human",
            "text": "x",
        }
    )


def test_the_instruction_is_relevance_and_names_no_label():
    assert "answer" in rq.INSTRUCTION
    for banned in ("necessary", "qrel", "grade", "gold", "anchor"):
        assert banned not in rq.INSTRUCTION.lower()


def test_reorder_sorts_by_score_and_ties_keep_e5s_order():
    out = rq.reorder(["a", "b", "c", "d"], {"a": 0.1, "b": 2.0, "c": 0.1, "d": -1.0})
    assert out == ["b", "a", "c", "d"]


def test_best_rank_is_the_smallest_over_requirements():
    assert rq.best_rank([["a", "b", "c"], ["c", "a"]]) == {"a": 1, "b": 2, "c": 1}
    assert rq.best_rank([]) == {}


def test_requirement_recall_and_mrr_are_macro_by_need():
    queries = [_query("N001-en-1"), _query("N001-en-2"), _query("N002-en-1")]
    hits = {
        "N001-en-1": [["x", "r1"], ["r2"]],  # r1 at 2, r2 at 1
        "N001-en-2": [["x", "y", "r1"]],  # r1 at 3, r2 absent
        "N002-en-1": [],  # no requirement
    }
    relevant = {q.query_id: frozenset({"r1", "r2"}) for q in queries}
    # k = 1: N001-en-1 finds r2 only (0.5), N001-en-2 nothing; N001 0.25, N002 0.0
    assert rq.requirement_recall(hits, relevant, queries, 1)[0] == 0.125
    # N001: (1.0 + 0.5) / 2 at k = 3
    assert rq.requirement_recall(hits, relevant, queries, 3)[0] == 0.375
    # N001: (1/1 + 1/3) / 2; N002: 0
    assert abs(rq.requirement_mrr(hits, relevant, queries) - (1 + 1 / 3) / 4) < 1e-12
