"""What every retriever of the iteration shares (ingest/retrieval.py): the ranking rule,
the run file, weighted Reciprocal Rank Fusion and the diagnostics."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from ingest import benchmark as bm
from ingest import retrieval as rt


def test_rank_by_scores_orders_by_score_then_id():
    r = rt.rank_by_scores("q", ["X:3", "X:1", "X:2"], np.array([1.0, 2.0, 1.0]))
    assert r.chunk_ids == ("X:1", "X:2", "X:3") and r.scores == (2.0, 1.0, 1.0)
    with pytest.raises(rt.RetrievalError, match="twice"):
        rt.rank_by_scores("q", ["X:1", "X:1"], np.array([1.0, 2.0]))
    with pytest.raises(rt.RetrievalError, match="shape"):
        rt.rank_by_scores("q", ["X:1"], np.array([1.0, 2.0]))


def test_rrf_is_the_weighted_sum_of_reciprocal_ranks():
    dense, lexical = ["A", "B", "C"], ["C", "A", "B"]
    fused = rt.rrf("q", [dense, lexical], [0.75, 0.25], k=60)
    expected = {
        "A": 0.75 / 61 + 0.25 / 62,
        "B": 0.75 / 62 + 0.25 / 63,
        "C": 0.75 / 63 + 0.25 / 61,
    }
    assert fused.chunk_ids == ("A", "B", "C")
    assert fused.scores == pytest.approx([expected[c] for c in fused.chunk_ids])
    # all the weight on one ranking gives that ranking back
    assert rt.rrf("q", [dense, lexical], [0.0, 1.0], k=20).chunk_ids == ("C", "A", "B")


def test_rrf_breaks_an_exact_tie_by_chunk_id():
    fused = rt.rrf("q", [["B", "A"], ["A", "B"]], [0.5, 0.5], k=60)
    assert fused.chunk_ids == ("A", "B") and fused.scores[0] == fused.scores[1]


def test_rrf_refuses_rankings_of_different_documents_and_bad_parameters():
    with pytest.raises(rt.RetrievalError, match="same documents"):
        rt.rrf("q", [["A", "B"], ["A", "C"]], [0.5, 0.5], k=60)
    with pytest.raises(rt.RetrievalError, match="same documents"):
        rt.rrf("q", [["A", "B"], ["A", "A"]], [0.5, 0.5], k=60)
    with pytest.raises(rt.RetrievalError, match="k must be positive"):
        rt.rrf("q", [["A"], ["A"]], [0.5, 0.5], k=0)
    with pytest.raises(rt.RetrievalError, match="one weight per ranking"):
        rt.rrf("q", [["A"], ["A"]], [1.0], k=60)


def test_a_run_file_round_trips_and_a_changed_file_is_refused(tmp_path):
    ranked = [
        rt.Ranked("N001-en-1", ("X:1", "X:2"), (2.0, 1.0)),
        rt.Ranked("N010-en-3", ("X:2", "X:1"), (0.5, 0.25)),
    ]
    path = tmp_path / "run.jsonl"
    sha = rt.write_run(path, ranked, ["N001-en-1"])
    back, evaluated = rt.read_run(path, sha)
    assert back == ranked and evaluated == {"N001-en-1"}
    path.write_text(path.read_text().replace("0.5", "0.6"))
    with pytest.raises(rt.RetrievalError, match="SHA-256 differs"):
        rt.read_run(path, sha)
    with pytest.raises(rt.RetrievalError, match="no run file"):
        rt.read_run(tmp_path / "absent.jsonl")


def _target(need_id: str, grades: dict[str, int]) -> Any:
    """Only what dependent_evidence reads of an ev.Target."""
    return cast(Any, SimpleNamespace(need=SimpleNamespace(need_id=need_id), chunk_grades=grades))


def _query(query_id: str) -> bm.Query:
    return bm.Query.model_validate(
        {
            "query_id": query_id,
            "need_id": query_id.split("-")[0],
            "lang": query_id.split("-")[1],
            "style": "natural",
            "origin": "human",
            "text": "x",
        }
    )


def test_dependent_evidence_needs_an_anchor_and_a_grade_2_chunk():
    targets = [
        _target("N001", {"X:1": 3, "X:2": 2, "X:3": 1, "X:4": 0}),
        _target("N002", {"X:5": 3, "X:6": 3}),  # no grade 2: not in the diagnostic
        _target("N003", {"X:7": 2}),  # no anchor
    ]
    dep = rt.dependent_evidence(targets)
    assert list(dep) == ["N001"]
    assert dep["N001"].anchors == {"X:1"} and dep["N001"].dependents == {"X:2"}


def test_recall_and_the_two_aggregations():
    assert rt.recall_of(["X:1", "X:9"], frozenset({"X:1", "X:2"})) == 0.5
    with pytest.raises(rt.RetrievalError):
        rt.recall_of(["X:1"], frozenset())
    queries = [_query(q) for q in ("N001-en-1", "N001-en-2", "N001-es-1", "N002-en-1")]
    values = {"N001-en-1": 1.0, "N001-en-2": 0.0, "N001-es-1": 0.5, "N002-en-1": 1.0}
    # need N001: mean of its three queries 0.5; N002: 1.0; then over needs
    assert rt.macro_by_need(values, queries) == pytest.approx(0.75)
    # English: N001 0.5 and N002 1.0, then over needs; Spanish: N001 only
    assert rt.per_language(values, queries) == pytest.approx({"en": 0.75, "es": 0.5})
