"""The pure rules of the S5-S6 selection: no qrel, no model, no clock, only synthetic numbers."""

from __future__ import annotations

import itertools
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import pytest

from ingest import benchmark as bm
from ingest import ir_metrics
from ingest import run_report as rr
from ingest import selection_rules as sr

# ---------------------------------------------------------------------------
# H1, bootstrap, H3, guardrail
# ---------------------------------------------------------------------------


def test_latency_classes_follow_the_limits_and_the_measured_p95() -> None:
    assert sr.latency_class(10.0) == sr.FAST_LATENCY  # the limit belongs to the lower class
    assert sr.latency_class(10.0000000005) == sr.FAST_LATENCY  # tolerance 1e-9
    assert sr.latency_class(10.01) == sr.MID_LATENCY
    assert sr.latency_class(30.0) == sr.MID_LATENCY
    assert sr.latency_class(30.01) == sr.SLOW_LATENCY
    measured = {
        "E5": 0.154,
        "RRF": 0.156,
        "N20": 9.594,
        "D-4B": 18.290,
        "N50": 19.689,
        "E": 85.741,
    }  # p95 of S5.3, in seconds
    assert {s: sr.latency_class(v) for s, v in measured.items()} == {
        "E5": sr.FAST_LATENCY,
        "RRF": sr.FAST_LATENCY,
        "N20": sr.FAST_LATENCY,
        "D-4B": sr.MID_LATENCY,
        "N50": sr.MID_LATENCY,
        "E": sr.SLOW_LATENCY,
    }


@dataclass(frozen=True)
class _Query:
    need_id: str
    query_id: str


def test_the_interval_is_the_paired_bootstrap_of_d067() -> None:
    """Same draws, same percentiles: the mean and the two-sided limits equal paired_bootstrap."""
    assert rr.SEED == sr.BOOTSTRAP_SEED
    rng = random.Random(7)
    needs = [f"N{i:03d}" for i in range(4, 31)]
    queries = [_Query(n, f"{n}-{j}") for n in needs for j in range(3)]
    deltas = {q.query_id: rng.choice([-0.5, 0.0, 0.0, 0.25, 0.5, 1.0]) for q in queries}
    reference = rr.paired_bootstrap(deltas, cast("Sequence[bm.Query]", queries))
    per_need = [
        statistics.fmean(deltas[q.query_id] for q in queries if q.need_id == n) for n in needs
    ]
    ours = sr.paired_interval(per_need, sr.resample_indices(len(needs)))
    assert ours.mean == reference["mean"]
    assert ours.low2 == reference["ci95_low"]
    assert ours.high2 == reference["ci95_high"]
    assert ours.low2 <= ours.low1 <= ours.high1 <= ours.high2


def test_the_one_sided_limits_use_the_fifth_and_ninety_fifth_percentile() -> None:
    values = [float(i) for i in range(10)]
    indices = sr.resample_indices(len(values), reps=1000, seed=1)
    means = sorted(statistics.fmean([values[i] for i in draw]) for draw in indices)
    interval = sr.paired_interval(values, indices)
    assert interval.low1 == means[50] and interval.high1 == means[949]
    assert interval.low2 == means[25] and interval.high2 == means[974]


def test_the_most_influential_need_is_the_largest_difference_first_on_a_tie() -> None:
    assert sr.drop_most_influential([0.0, 0.5, 0.1, 0.5]) == statistics.fmean([0.0, 0.1, 0.5])
    assert sr.drop_most_influential([0.9, 0.0]) == 0.0
    with pytest.raises(ValueError):
        sr.drop_most_influential([0.3])


def _interval(mean: float, low2: float, high2: float, low1: float, high1: float) -> sr.Interval:
    return sr.Interval(mean, low2, high2, low1, high1)


def test_the_guardrail_states() -> None:
    assert sr.ndcg_guardrail(_interval(-0.02, -0.1, 0.1, -0.08, 0.05)) == sr.VIOLATED  # inclusive
    assert sr.ndcg_guardrail(_interval(0.01, -0.01, 0.1, -0.0, 0.05)) == sr.PASSED
    assert sr.ndcg_guardrail(_interval(-0.01, -0.1, 0.1, -0.05, 0.05)) == sr.GUARD_UNRESOLVED
    # violated wins over passed when both could be read: the point estimate decides first
    assert sr.ndcg_guardrail(_interval(-0.03, -0.1, 0.1, -0.019, 0.05)) == sr.VIOLATED


# ---------------------------------------------------------------------------
# the nine labels: one rule per comparison, exhaustive and exclusive
# ---------------------------------------------------------------------------


def test_every_label_is_reachable_with_the_expected_rule() -> None:
    ok = sr.PASSED
    cases = {
        sr.FRAGILE: (_interval(0.05, 0.01, 0.1, 0.02, 0.09), 0.01, ok),
        sr.STRONG: (_interval(0.09, 0.04, 0.14, 0.05, 0.13), 0.08, ok),
        sr.SUPPORTED: (_interval(0.06, 0.01, 0.11, 0.02, 0.1), 0.05, ok),
        sr.MATERIAL: (_interval(0.04, -0.03, 0.1, -0.02, 0.09), 0.04, ok),
        sr.BELOW_MEPI: (_interval(0.02, 0.005, 0.04, 0.007, 0.036), 0.01, ok),
        sr.INFERIOR: (_interval(-0.10, -0.16, -0.04, -0.15, -0.05), -0.1, ok),
        sr.NON_INFERIOR: (_interval(0.0, -0.04, 0.04, -0.02, 0.03), 0.0, ok),
        sr.UNRESOLVED: (_interval(-0.02, -0.09, 0.05, -0.08, 0.04), -0.02, ok),
    }
    for expected, (interval, drop, guardrail) in cases.items():
        assert sr.recall_label(interval, drop, guardrail) == expected
    # a violated guardrail blocks `non-inferior` and falls to the last rule
    violated = sr.recall_label(_interval(0.0, -0.04, 0.04, -0.02, 0.03), 0.0, sr.VIOLATED)
    assert violated == sr.UNRESOLVED
    assert set(cases) | {sr.NOT_EVALUABLE_LABEL} == set(sr.LABELS) and len(sr.LABELS) == 9


def test_the_label_list_is_exhaustive_and_exclusive_on_a_grid() -> None:
    rng = random.Random(3)
    values = [-0.2, -0.08, -0.04, -0.03, -0.02, 0.0, 0.01, 0.02, 0.03, 0.05, 0.09, 0.15]
    reached = set()
    for _ in range(20000):
        d = rng.choice(values)
        low1 = d - rng.choice([0, 0.01, 0.04, 0.08])
        low2 = low1 - rng.choice([0, 0.005, 0.02])
        high1 = d + rng.choice([0, 0.01, 0.04, 0.08])
        interval = _interval(d, low2, high1 + 0.02, low1, high1)
        guardrail = rng.choice([sr.PASSED, sr.GUARD_UNRESOLVED, sr.VIOLATED])
        label = sr.recall_label(interval, d - rng.choice([0, 0.01, 0.03, 0.06, -0.01]), guardrail)
        reached.add(label)
        assert label in sr.LABELS and label != sr.NOT_EVALUABLE_LABEL
        # the two rules that look at the lower side of -delta cannot both hold
        assert not (sr.below(high1, -sr.DELTA_RECALL) and sr.above(low1, -sr.DELTA_RECALL))
    assert reached == set(sr.LABELS) - {sr.NOT_EVALUABLE_LABEL}


# ---------------------------------------------------------------------------
# the coverage sensitivity envelope
# ---------------------------------------------------------------------------


def _random_need(rng: random.Random, n_unjudged: int) -> sr.NeedData:
    judged = [f"j{i}" for i in range(12)]
    unjudged = [f"u{i}" for i in range(n_unjudged)]
    grades = {c: rng.choice([0, 0, 1, 2, 3]) for c in judged}
    grades[judged[0]] = rng.choice([2, 3])  # at least one relevant chunk
    pool = judged + unjudged
    queries = tuple(
        sr.QueryLists(tuple(rng.sample(pool, 10)), tuple(rng.sample(pool, 10)))
        for _ in range(rng.randint(1, 3))
    )
    return sr.NeedData("N", grades, queries)


def _brute_force(need: sr.NeedData) -> tuple[float, float, float, float]:
    unjudged = sr.unjudged_chunks(need)
    recall, ndcg = [], []
    for combo in itertools.product(sr.GRADES, repeat=len(unjudged)):
        extra = dict(zip(unjudged, combo, strict=True))
        recall.append(sr.recall_delta(need, extra))
        ndcg.append(sr.ndcg_delta(need, extra))
    return min(recall), max(recall), min(ndcg), max(ndcg)


def test_the_envelope_is_exact_against_brute_force() -> None:
    rng = random.Random(11)
    checked = 0
    for _ in range(60):
        need = _random_need(rng, rng.randint(0, 5))
        r_low, r_high, n_low, n_high = _brute_force(need)
        recall, ndcg = sr.recall_range(need), sr.ndcg_range(need)
        assert recall.low == pytest.approx(r_low, abs=1e-12)
        assert recall.high == pytest.approx(r_high, abs=1e-12)
        assert ndcg.low == pytest.approx(n_low, abs=1e-12)
        assert ndcg.high == pytest.approx(n_high, abs=1e-12)
        # the assignments given for the audit really reach the bounds, and are legal grades
        for rng_, oracle in ((recall, sr.recall_delta), (ndcg, sr.ndcg_delta)):
            assert oracle(need, rng_.low_assignment) == pytest.approx(rng_.low, abs=1e-12)
            assert oracle(need, rng_.high_assignment) == pytest.approx(rng_.high, abs=1e-12)
            assert set(rng_.low_assignment.values()) | set(rng_.high_assignment.values()) <= set(
                sr.GRADES
            )
        checked += 1
    assert checked == 60


def test_without_unjudged_chunks_the_envelope_collapses_to_the_primary_value() -> None:
    need = _random_need(random.Random(5), 0)
    recall, ndcg = sr.recall_range(need), sr.ndcg_range(need)
    assert recall.low == recall.high == pytest.approx(sr.recall_delta(need), abs=1e-12)
    assert ndcg.low == ndcg.high == pytest.approx(sr.ndcg_delta(need), abs=1e-12)


def test_the_primary_value_always_lies_inside_the_envelope() -> None:
    rng = random.Random(21)
    for _ in range(40):
        need = _random_need(rng, rng.randint(0, 7))
        recall, ndcg = sr.recall_range(need), sr.ndcg_range(need)
        assert recall.low - 1e-12 <= sr.recall_delta(need) <= recall.high + 1e-12
        assert ndcg.low - 1e-12 <= sr.ndcg_delta(need) <= ndcg.high + 1e-12


def test_a_chunk_in_both_lists_gets_one_grade_and_only_moves_the_denominator() -> None:
    # c is unjudged and in both top-10s of the only query: for Recall it cannot favour either
    # side, it can only add to the relevant set. One judged relevant chunk, hit by the candidate.
    base = [f"f{i}" for i in range(8)]
    grades = {"r": 3, **{c: 0 for c in base}}
    candidate = tuple(["r", "c", *base[:8]])
    reference = tuple(["c", *base])
    need = sr.NeedData("N", grades, (sr.QueryLists(candidate, reference),))
    recall = sr.recall_range(need)
    assert recall.high == pytest.approx(1.0)  # primary: candidate hits r, reference none
    # worst case: c is relevant too (R = 2), the candidate hits r and c, the reference hits c
    assert recall.low == pytest.approx(1.0 - 0.5)
    assert recall.low_assignment["c"] == 2 and recall.high_assignment["c"] == 0


@pytest.mark.evaluator
def test_the_oracle_is_the_evaluator_of_the_project() -> None:
    """recall_delta and ndcg_delta equal Recall@10 and nDCG@10 of ir_measures on the same qrels,
    also with hypothetical grades added to the qrels."""
    rng = random.Random(8)
    for _ in range(12):
        need = _random_need(rng, rng.randint(0, 5))
        unjudged = sr.unjudged_chunks(need)
        extra = {c: rng.choice(sr.GRADES) for c in unjudged}
        grades = {**need.grades, **extra}
        qrels = {f"q{i}": dict(grades) for i in range(len(need.queries))}
        cand = {f"q{i}": list(q.candidate) for i, q in enumerate(need.queries)}
        ref = {f"q{i}": list(q.reference) for i, q in enumerate(need.queries)}
        scores_c = ir_metrics.standard_scores(qrels, cand, recall_cutoffs=(10,))
        scores_r = ir_metrics.standard_scores(qrels, ref, recall_cutoffs=(10,))
        qids = sorted(qrels)
        recall = statistics.fmean(scores_c[q]["Recall@10"] - scores_r[q]["Recall@10"] for q in qids)
        ndcg = statistics.fmean(scores_c[q]["nDCG@10"] - scores_r[q]["nDCG@10"] for q in qids)
        assert sr.recall_delta(need, extra) == pytest.approx(recall, abs=1e-9)
        assert sr.ndcg_delta(need, extra) == pytest.approx(ndcg, abs=1e-9)


# ---------------------------------------------------------------------------
# coverage states, actions, dominance
# ---------------------------------------------------------------------------


def _scenario(label: str, guardrail: str = sr.PASSED) -> sr.Scenario:
    zero = _interval(0.0, 0.0, 0.0, 0.0, 0.0)
    return sr.Scenario(zero, 0.0, zero, guardrail, label)


def _comparison(a: sr.Scenario, b: sr.Scenario, c: sr.Scenario) -> sr.Comparison:
    return sr.Comparison("X", "R", a, b, c)


def test_the_three_coverage_states() -> None:
    def action(s: sr.Scenario) -> str:
        return sr.fast_action(sr.FAST_LATENCY, s)

    same = _scenario(sr.SUPPORTED)
    assert sr.coverage_state(_comparison(same, same, same), action) == sr.COVERAGE_ROBUST
    # the label changes (supported, strong) but both displace: affected
    strong = _scenario(sr.STRONG)
    assert sr.coverage_state(_comparison(same, strong, same), action) == sr.COVERAGE_AFFECTED
    # displaces under the favourable bound and not under the unfavourable one: not evaluable
    unresolved = _scenario(sr.UNRESOLVED)
    assert sr.coverage_state(_comparison(same, same, unresolved), action) == sr.NOT_EVALUABLE
    # a guardrail change that flips the action counts as well
    violated = _scenario(sr.SUPPORTED, sr.VIOLATED)
    assert sr.coverage_state(_comparison(same, same, violated), action) == sr.NOT_EVALUABLE
    # discard against frontier is also a change of action
    inferior = _scenario(sr.INFERIOR)
    assert sr.coverage_state(_comparison(unresolved, unresolved, inferior), action) == (
        sr.NOT_EVALUABLE
    )

    # without an action to change, the state can only be robust or affected
    def constant(_: sr.Scenario) -> str:
        return "descriptive"

    assert (
        sr.coverage_state(_comparison(same, strong, unresolved), constant) == sr.COVERAGE_AFFECTED
    )


def test_final_label_is_not_evaluable_only_when_the_state_says_so() -> None:
    s = _scenario(sr.SUPPORTED)
    comparison = _comparison(s, s, s)
    assert sr.final_label(comparison, sr.COVERAGE_ROBUST) == sr.SUPPORTED
    assert sr.final_label(comparison, sr.COVERAGE_AFFECTED) == sr.SUPPORTED
    assert sr.final_label(comparison, sr.NOT_EVALUABLE) == sr.NOT_EVALUABLE_LABEL


def test_h2b_actions() -> None:
    assert sr.fast_action(sr.FAST_LATENCY, _scenario(sr.SUPPORTED)) == sr.DISPLACE
    assert sr.fast_action(sr.FAST_LATENCY, _scenario(sr.STRONG)) == sr.DISPLACE
    # a material signal on its own never pays for more
    for label in (sr.MATERIAL, sr.FRAGILE, sr.BELOW_MEPI, sr.NON_INFERIOR, sr.UNRESOLVED):
        assert sr.fast_action(sr.FAST_LATENCY, _scenario(label)) == sr.FRONTIER
    assert sr.fast_action(sr.FAST_LATENCY, _scenario(sr.SUPPORTED, sr.VIOLATED)) == sr.FRONTIER
    assert sr.fast_action(sr.MID_LATENCY, _scenario(sr.STRONG)) == sr.FRONTIER  # class
    assert sr.fast_action(sr.SLOW_LATENCY, _scenario(sr.STRONG)) == sr.FRONTIER
    assert sr.fast_action(sr.FAST_LATENCY, _scenario(sr.INFERIOR)) == sr.DISCARD
    assert sr.fast_action(sr.MID_LATENCY, _scenario(sr.INFERIOR)) == sr.DISCARD
    assert sr.prefers_candidate(_scenario(sr.SUPPORTED))
    assert not sr.prefers_candidate(_scenario(sr.MATERIAL))


S53 = {
    "E5": sr.CostRow(0.154, 0.66, *sr.STRUCTURE["E5"]),
    "RRF": sr.CostRow(0.156, 0.65, *sr.STRUCTURE["RRF"]),
    "N20": sr.CostRow(9.594, 5.95, *sr.STRUCTURE["N20"]),
    "D-4B": sr.CostRow(18.290, 4.14, *sr.STRUCTURE["D-4B"]),
    "N50": sr.CostRow(19.689, 6.05, *sr.STRUCTURE["N50"]),
    "E": sr.CostRow(85.741, 8.99, *sr.STRUCTURE["E"]),
}


def test_cost_dominance_of_s53_is_the_closed_list_of_nine_pairs() -> None:
    assert sr.dominance_pairs(S53) == [
        ("D-4B", "E"),
        ("E5", "D-4B"),
        ("E5", "E"),
        ("E5", "N20"),
        ("E5", "N50"),
        ("E5", "RRF"),
        ("N20", "E"),
        ("N20", "N50"),
        ("N50", "E"),
    ]


def test_cost_axes_use_a_ten_percent_band_and_complexity_is_component_wise() -> None:
    base = sr.CostRow(10.0, 1.0, (1, 1, 1, 1, 1), 1)
    near = sr.CostRow(9.2, 0.95, (1, 1, 1, 1, 1), 1)  # within 10 %: a tie, no dominance
    assert not sr.cost_dominates(near, base) and not sr.cost_dominates(base, near)
    clear = sr.CostRow(9.0, 1.0, (1, 1, 1, 1, 1), 1)  # exactly 10 % lower: clearly better
    assert sr.cost_dominates(clear, base)
    incomparable = sr.CostRow(1.0, 1.0, (0, 1, 1, 1, 2), 1)  # fewer models, one more index
    assert not sr.cost_dominates(incomparable, base)
    worse_port = sr.CostRow(1.0, 1.0, (1, 1, 1, 1, 1), 2)
    assert not sr.cost_dominates(worse_port, base)


def test_dominance_needs_non_inferiority_and_a_guardrail_that_is_not_violated() -> None:
    def scenario(low1: float, guardrail: str) -> sr.Scenario:
        zero = _interval(0.0, 0.0, 0.0, 0.0, 0.0)
        return sr.Scenario(_interval(0.0, 0.0, 0.0, low1, 0.0), 0.0, zero, guardrail, sr.UNRESOLVED)

    assert sr.dominance_quality(scenario(-0.02, sr.PASSED)) == "certified"
    assert sr.dominance_quality(scenario(-0.02, sr.GUARD_UNRESOLVED)) == "certified"
    assert sr.dominance_quality(scenario(-0.02, sr.VIOLATED)) == "unresolved"
    assert sr.dominance_quality(scenario(-0.03, sr.PASSED)) == "unresolved"  # not strictly above


# ---------------------------------------------------------------------------
# the whole analysis on a synthetic world
# ---------------------------------------------------------------------------

NEEDS = [f"N{i:03d}" for i in range(4, 12)]


def _world(
    n20_known_hit: bool, reference_unjudged: bool
) -> tuple[list[sr.NeedMeta], sr.Lists, dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Eight needs of one query. The only relevant chunk of each need is r. E5 never finds it. In
    the good world N20 finds it at rank 1 and everything in the lists is judged. With
    `reference_unjudged`, E5's list holds unjudged chunks that could all be relevant."""
    fillers = [f"f{i}" for i in range(20)]
    needs: list[sr.NeedMeta] = []
    lists: dict[str, dict[str, tuple[str, ...]]] = {s: {} for s in sr.SYSTEMS}
    for need in NEEDS:
        grades = {"r": 3, **{c: 0 for c in fillers}}
        if reference_unjudged:
            e5 = tuple([f"u{i}" for i in range(5)] + fillers[:5])
        else:
            e5 = tuple(fillers[:10])
        good = tuple(["r", *fillers[:9]]) if n20_known_hit else tuple(fillers[10:20])
        for system in sr.SYSTEMS:
            lists[system][f"{need}-q"] = e5 if system == "E5" else good
        needs.append(sr.NeedMeta(need, grades, (f"{need}-q",)))
    recall: dict[str, dict[str, float]] = {}
    ndcg: dict[str, dict[str, float]] = {}
    for system in sr.SYSTEMS:
        recall[system], ndcg[system] = {}, {}
        for n in needs:
            judged = n.grades
            relevant = sum(g >= sr.RELEVANT_GRADE for g in judged.values())
            top = lists[system][n.query_ids[0]]
            recall[system][n.need_id] = sum(judged.get(c, -1) >= 2 for c in top) / relevant
            ideal = sr._ideal_dcg(list(judged.values()))
            ndcg[system][n.need_id] = sr._dcg(top, judged) / ideal
    return needs, lists, recall, ndcg


def test_a_clear_and_fully_judged_gain_displaces_e5_robustly() -> None:
    needs, lists, recall, ndcg = _world(n20_known_hit=True, reference_unjudged=False)
    result = sr.analyse(needs, lists, recall, ndcg, S53)
    assert result.classes["N20"] == sr.FAST_LATENCY
    assert result.states[("N20", "E5")] == sr.COVERAGE_ROBUST
    assert result.labels[("N20", "E5")] == sr.STRONG
    assert result.actions["N20"] == sr.DISPLACE and result.actions["N50"] == sr.FRONTIER
    # RRF returns the same list as N20 in this world, so it displaces E5 too, and the tie-break
    # between two equal candidates (N20 against RRF) does not prefer N20
    assert result.actions["RRF"] == sr.DISPLACE and not result.n20_over_rrf
    assert result.paths.fast_path == "RRF" and result.paths.quality_path == "E"
    # E5 is clearly worse than N20 here: it is inferior beyond delta, but E5 is never discarded
    assert "E5" not in result.paths.discarded
    # N20 is a clear gain over N50? they are equal: no discard by dominance of N50 (not certified
    # against an equal system is fine: non-inferior holds), so N50 is dominated by N20
    assert result.dominance[("N20", "N50")] == "certified"
    assert "N50" in result.paths.discarded


def test_unjudged_chunks_in_the_reference_make_the_gain_not_evaluable_and_e5_stays() -> None:
    needs, lists, recall, ndcg = _world(n20_known_hit=True, reference_unjudged=True)
    result = sr.analyse(needs, lists, recall, ndcg, S53)
    assert result.states[("N20", "E5")] == sr.NOT_EVALUABLE
    assert result.labels[("N20", "E5")] == sr.NOT_EVALUABLE_LABEL
    assert result.actions["N20"] == sr.FRONTIER  # the system does not displace the reference
    assert result.paths.fast_path == "E5"
    assert "N20" not in result.paths.discarded


def test_the_evaluator_must_agree_with_the_oracle() -> None:
    needs, lists, recall, ndcg = _world(n20_known_hit=True, reference_unjudged=False)
    recall["N20"][needs[0].need_id] += 0.1
    with pytest.raises(ValueError, match="the evaluator"):
        sr.analyse(needs, lists, recall, ndcg, S53)


def test_needs_must_be_ordered_by_id() -> None:
    needs, lists, recall, ndcg = _world(n20_known_hit=True, reference_unjudged=False)
    with pytest.raises(ValueError, match="ordered"):
        sr.analyse(list(reversed(needs)), lists, recall, ndcg, S53)


def test_paths_never_discard_e_and_a_dominated_e_is_only_descriptive() -> None:
    actions = {"RRF": sr.FRONTIER, "N20": sr.FRONTIER, "N50": sr.FRONTIER, "D-4B": sr.FRONTIER}
    paths = sr.assign_paths(actions, False, {("N20", "E"): True, ("N20", "N50"): True})
    assert paths.quality_path == "E" and "E" not in paths.discarded
    assert paths.discarded == {"N50": "cost-dominated by N20 with non-inferiority certified (H7)"}
    assert paths.fast_path == "E5" and "E5" not in paths.frontier  # an occupant is not frontier
    assert set(paths.frontier) == {"RRF", "N20", "D-4B"}
    displaced = sr.assign_paths({**actions, "N20": sr.DISPLACE}, False, {})
    assert displaced.fast_path == "N20" and set(displaced.frontier) == {"E5", "RRF", "N50", "D-4B"}


def test_two_challengers_are_settled_by_the_direct_comparison() -> None:
    both = {"RRF": sr.DISPLACE, "N20": sr.DISPLACE, "N50": sr.FRONTIER, "D-4B": sr.FRONTIER}
    assert sr.assign_paths(both, True, {}).fast_path == "N20"
    assert sr.assign_paths(both, False, {}).fast_path == "RRF"
    assert sr.assign_paths({**both, "RRF": sr.FRONTIER}, False, {}).fast_path == "N20"
    discarded = {**both, "RRF": sr.DISCARD}
    assert sr.assign_paths(discarded, False, {}).fast_path == "N20"
