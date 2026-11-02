"""The glue of the selection read, on the synthetic benchmark of the evaluation tests: the
project's evaluator gives the numbers, the rules module checks them against its own oracle."""

from __future__ import annotations

import json

import pytest

from ingest import evaluation as ev
from ingest import selection_read as rd
from ingest import selection_rules as sr
from tests.integration.test_evaluation_end_to_end import _benchmark, _layer

pytestmark = pytest.mark.evaluator

FILLERS = tuple(f"x{i}" for i in range(1, 11))


def _top(*head: str) -> tuple[str, ...]:
    return (*head, *FILLERS)[: sr.TOP]


def _lists(needs: list[sr.NeedMeta]) -> sr.Lists:
    heads: dict[str, dict[str, tuple[str, ...]]] = {
        "E5": {"N001": ("MiCA:3", "MiCA:4"), "N002": ("MiCA:6",)},
        "RRF": {"N001": ("MiCA:3", "MiCA:1"), "N002": ("MiCA:6", "MiCA:5")},
        "N20": {"N001": ("MiCA:1", "MiCA:2"), "N002": ("MiCA:5",)},
        "N50": {"N001": ("MiCA:1", "MiCA:2"), "N002": ("MiCA:5", "MiCA:6")},
        "D-4B": {"N001": ("MiCA:1",), "N002": ("MiCA:5",)},
        "E": {"N001": ("MiCA:1", "MiCA:2"), "N002": ("MiCA:5", "MiCA:6")},
    }
    return {
        system: {q: _top(*heads[system][n.need_id]) for n in needs for q in n.query_ids}
        for system in sr.SYSTEMS
    }


def _needs() -> list[sr.NeedMeta]:
    return rd.needs_from_targets(ev.targets(_benchmark(), _layer()))


def test_the_needs_are_the_evaluable_ones_in_order_with_chunk_grades() -> None:
    needs = _needs()
    assert [n.need_id for n in needs] == ["N001", "N002"]  # N003 is a `none` need: never enters
    assert needs[0].query_ids == ("N001-en-1", "N001-es-1")
    assert needs[0].grades == {"MiCA:1": 3, "MiCA:2": 2, "MiCA:3": 0}  # max grade of owned nodes
    assert "MiCA:4" not in needs[0].grades  # a chunk with no judged node is unjudged


def test_the_vectors_are_the_evaluator_numbers_and_the_oracle_agrees() -> None:
    needs = _needs()
    lists = _lists(needs)
    per_query, per_need = rd.all_vectors(needs, lists)
    assert set(per_query) == set(sr.SYSTEMS)
    assert per_need["N20"]["N001"]["Recall@10"] == pytest.approx(1.0)  # MiCA:1 and MiCA:2
    assert per_need["E5"]["N001"]["Recall@10"] == pytest.approx(0.0)
    # Judged@10 counts the judged chunks of the ten: MiCA:3 only (MiCA:4 is unjudged)
    assert per_need["E5"]["N001"]["Judged@10"] == pytest.approx(0.1)
    # nothing but the three measures survives
    assert set(per_query["E5"]["N001-en-1"]) == set(rd.METRICS)
    analysis = sr.analyse(
        needs,
        lists,
        rd.metric_of(per_need, "Recall@10"),
        rd.metric_of(per_need, "nDCG@10"),
        {
            "E5": sr.CostRow(0.154, 0.66, *sr.STRUCTURE["E5"]),
            "RRF": sr.CostRow(0.156, 0.65, *sr.STRUCTURE["RRF"]),
            "N20": sr.CostRow(9.594, 5.95, *sr.STRUCTURE["N20"]),
            "D-4B": sr.CostRow(18.29, 4.14, *sr.STRUCTURE["D-4B"]),
            "N50": sr.CostRow(19.689, 6.05, *sr.STRUCTURE["N50"]),
            "E": sr.CostRow(85.741, 8.99, *sr.STRUCTURE["E"]),
        },
    )
    # with two needs and unjudged chunks everywhere, the labels exist and are well formed
    assert all(label in sr.LABELS for label in analysis.labels.values())
    assert analysis.paths.quality_path == "E"
    json.dumps(rd.analysis_to_dict(analysis))  # serializable
    coverage = rd.coverage_figures(needs, lists, list(analysis.comparisons))
    assert coverage["N20|E5"]["mean_chunks_only_in_candidate_top10"] > 0
    report = rd.render_report(
        analysis, rd.system_summary(per_need), coverage, {"commit": "x"}, looks=6
    )
    assert "FAST PATH" in report and "not independently confirmed" in report


def test_a_list_that_does_not_meet_the_contract_is_refused() -> None:
    needs = _needs()
    lists = {s: dict(q) for s, q in _lists(needs).items()}
    lists["E5"]["N001-en-1"] = ("MiCA:1",) * 10  # ten times the same chunk
    with pytest.raises(rd.SelectionReadError, match="distinct"):
        rd.check_lists(needs, lists)
    del lists["E"]["N002-es-1"]
    lists["E5"]["N001-en-1"] = _top("MiCA:1")
    with pytest.raises(rd.SelectionReadError, match="no list for"):
        rd.check_lists(needs, lists)
