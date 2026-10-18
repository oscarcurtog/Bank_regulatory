"""Tests of OUR integration with ir_measures (D-020), not of the library. Each one
pins a trap measured on ir_measures 0.4.3 with pytrec-eval-terrier 0.5.10, with a
case where the right and the wrong integration give different numbers."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from ingest import benchmark as bm
from ingest import evaluation as ev
from ingest import ir_metrics as im
from tests.conftest import N002_RANKING, n002_excerpt_layer

pytestmark = pytest.mark.evaluator

LOCK = Path(__file__).parents[2] / "requirements-dev.lock.txt"


def _pin(package: str) -> str:
    found = re.search(rf"^{re.escape(package)}==(\S+)", LOCK.read_text(encoding="utf-8"), re.M)
    assert found, package
    return found.group(1)


def test_the_metrics_are_computed_by_the_locked_backend():
    backend = im.backend()
    assert (backend.ir_measures, backend.pytrec_eval_terrier) == (
        _pin("ir-measures"),
        _pin("pytrec-eval-terrier"),
    )


def test_scores_are_strictly_decreasing():
    scores = im.strict_scores(["c", "a", "b"])
    assert list(scores) == ["c", "a", "b"]
    assert scores["c"] > scores["a"] > scores["b"]


def test_our_order_survives_trec_evals_reverse_docid_tie_break():
    # with equal scores trec_eval puts "d" before "b" and MRR would be 0.5
    assert im.standard_scores({"q": {"b": 2, "d": 0}}, {"q": ["b", "d"]})["q"]["MRR@10"] == 1.0


def test_mrr_at_10_ignores_a_first_hit_at_rank_11():
    """The pytrec_eval provider computes RR@10 over the whole ranking: 1/11 here."""
    ranking = [f"d{i:02d}" for i in range(1, 11)] + ["hit"]
    qrels = {"q": {"hit": 2}}
    assert im.standard_scores(qrels, {"q": ranking})["q"]["MRR@10"] == 0.0
    assert im.standard_scores(qrels, {"q": ranking[1:]})["q"]["MRR@10"] == 0.1


def test_binary_measures_count_grade_two_and_above_only():
    qrels = {"q": {"a": 3, "c": 1}}
    assert (
        im.standard_scores(qrels, {"q": ["c"]})["q"]["Recall@5"] == 0.0
    )  # grade 1 is not relevant
    assert im.standard_scores(qrels, {"q": ["a"]})["q"]["Recall@5"] == 1.0  # one relevant, not two
    assert im.standard_scores(qrels, {"q": ["c", "a"]})["q"]["MRR@10"] == 0.5


def test_ndcg_uses_the_grades_as_linear_gains():
    got = im.standard_scores({"q": {"a": 3, "c": 1}}, {"q": ["c", "a"]})["q"]["nDCG@10"]
    linear = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    exponential = (1 + 7 / math.log2(3)) / (7 + 1 / math.log2(3))
    assert got == pytest.approx(linear, abs=1e-9)
    assert abs(got - exponential) > 0.05


def test_judged_is_the_judged_share_of_what_was_returned():
    scores = im.standard_scores({"q": {"a": 2, "b": 0}}, {"q": ["a", "x", "b", "y", "z"]})
    assert scores["q"]["Judged@10"] == pytest.approx(2 / 5, abs=1e-12)


def test_an_empty_or_absent_ranking_scores_zero_and_still_counts():
    """The judged provider divides by zero on an empty ranking; it never sees one."""
    qrels = {"q1": {"a": 2}, "q2": {"a": 2}, "q3": {"a": 2}}
    scores = im.standard_scores(qrels, {"q1": ["a"], "q2": []})
    assert set(scores) == {"q1", "q2", "q3"}
    assert set(scores["q2"].values()) == set(scores["q3"].values()) == {0.0}
    assert scores["q1"]["Recall@5"] == 1.0


def test_rankings_for_queries_without_qrels_are_refused():
    with pytest.raises(im.MetricError, match="without qrels"):
        im.standard_scores({"q1": {"a": 2}}, {"q9": ["a"]})


def test_a_measure_the_provider_cannot_compute_is_refused():
    import ir_measures

    with pytest.raises(im.MetricError, match="does not support"):
        im._require_support(ir_measures.pytrec_eval, [ir_measures.RR @ 10])


def test_the_known_ranking_of_fixture_n002():
    """The values computed by hand in the design study, on the real chunk ids."""
    judged = ev.judged_nodes(bm.load_benchmark(bm.FIXTURES_DIR).qrels_of("N002"))
    grades = ev.project(judged, n002_excerpt_layer())
    scores = im.standard_scores({"q": grades}, {"q": N002_RANKING}, recall_cutoffs=(1, 2, 5))["q"]
    assert (scores["Recall@1"], scores["Recall@2"], scores["Recall@5"]) == (0.0, 0.5, 1.0)
    assert scores["MRR@10"] == 0.5
    # gains in rank order: 0, 3, 1, 2, unjudged, unjudged; ideal: 3, 2, 1, 1
    dcg = 3 / math.log2(3) + 1 / math.log2(4) + 2 / math.log2(5)
    idcg = 3 + 2 / math.log2(3) + 1 / math.log2(4) + 1 / math.log2(5)
    assert scores["nDCG@10"] == pytest.approx(dcg / idcg, abs=1e-9)
    assert round(scores["nDCG@10"], 4) == 0.6267
    assert scores["Judged@10"] == pytest.approx(4 / 6, abs=1e-12)


# ---------------------------------------------------------------------------
# binary, positive-only judgments of the external benchmark (T4, D-053)
# ---------------------------------------------------------------------------


def test_binary_scores_follow_the_published_definitions():
    """q1 has three relevant documents, one ("z") not in any ranking: it still counts in the
    denominators of Recall and AP and in the ideal DCG, as trec_eval and JuriFindIT count it."""
    relevant = {"q1": {"a", "c", "z"}, "q2": {"d"}}
    rankings = {"q1": ["a", "b", "c", "d"], "q2": ["a", "b", "c", "d"]}
    v = im.binary_scores(relevant, rankings, recall=(1, 3), ndcg=(3,), rr=(1, 3), ap=(3,))
    q1 = v["q1"]
    assert q1["Recall@1"] == pytest.approx(1 / 3) and q1["Recall@3"] == pytest.approx(2 / 3)
    assert q1["mAP@3"] == pytest.approx((1 + 2 / 3) / 3)
    ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert q1["nDCG@3"] == pytest.approx((1 + 1 / math.log2(4)) / ideal)
    assert q1["MRR@1"] == 1.0
    # q2's only relevant document is at rank 4: everything at cutoff 3 or less is 0
    assert all(value == 0.0 for value in v["q2"].values())


def test_binary_mrr_is_cut_at_its_cutoff():
    v = im.binary_scores({"q": {"c"}}, {"q": ["a", "b", "c"]}, recall=(), ndcg=(), rr=(2, 3), ap=())
    assert v["q"]["MRR@2"] == 0.0 and v["q"]["MRR@3"] == pytest.approx(1 / 3)


def test_binary_scores_refuse_a_query_without_relevant_documents():
    with pytest.raises(im.MetricError, match="without any relevant"):
        im.binary_scores({"q": set()}, {"q": ["a"]}, recall=(1,), ndcg=(), rr=(), ap=())
