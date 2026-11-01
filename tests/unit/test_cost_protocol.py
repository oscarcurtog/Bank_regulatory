"""The pure rules of the S5.3 cost protocol: no model, no label, no clock."""

import math

import pytest

from ingest import cost_protocol as cp


def test_nearest_rank_percentiles_follow_the_rule_of_m054() -> None:
    values = [float(v) for v in range(1, 91)]  # 90 samples
    assert cp.nearest_rank_percentile(values, 50) == 45.0  # rank ceil(45.0) = 45
    assert cp.nearest_rank_percentile(values, 95) == 86.0  # rank ceil(85.5) = 86
    assert cp.nearest_rank_percentile(values, 100) == 90.0
    assert cp.nearest_rank_percentile([7.0], 95) == 7.0
    assert cp.nearest_rank_percentile([3.0, 1.0, 2.0], 50) == 2.0  # unsorted input


def test_percentile_rejects_empty_input_and_bad_percent() -> None:
    with pytest.raises(ValueError):
        cp.nearest_rank_percentile([], 50)
    with pytest.raises(ValueError):
        cp.nearest_rank_percentile([1.0], 0)


def test_summary_holds_the_protocol_statistics() -> None:
    summary = cp.summarize([4.0, 1.0, 3.0, 2.0])
    assert (summary.n, summary.p50, summary.p95, summary.maximum, summary.mean) == (
        4,
        2.0,
        4.0,
        4.0,
        2.5,
    )


def test_seeds_and_orders() -> None:
    assert [cp.seed_for_pass(n) for n in range(1, 6)] == [
        20261102,
        20261103,
        20261104,
        20261105,
        20261106,
    ]
    assert cp.repeat_seed(cp.seed_for_pass(2)) == 20261203
    ids = [f"N{i:03d}-en-{j}" for i in range(4, 34) for j in (1, 2, 3)]
    first = cp.pass_order(ids, cp.seed_for_pass(1))
    reversed_ids = list(reversed(ids))  # the order of the input is irrelevant
    assert first == cp.pass_order(reversed_ids, cp.seed_for_pass(1))
    assert sorted(first) == sorted(ids)
    orders = {tuple(cp.pass_order(ids, cp.seed_for_pass(n))) for n in range(1, 6)}
    assert len(orders) == 5  # no two passes share an order
    with pytest.raises(ValueError):
        cp.seed_for_pass(0)


def test_rrf_uses_the_first_fifty_and_gives_nothing_for_an_absent_chunk() -> None:
    first = [f"a{i}" for i in range(60)]
    second = [f"b{i}" for i in range(60)]
    fused = cp.rrf_fuse(first, second)
    assert len(fused) == 10
    assert "a55" not in cp.rrf_fuse(first, second, top=200)  # beyond depth 50: not in the fusion
    # a chunk present in both lists beats one present in a single list at the same best rank
    both = cp.rrf_fuse(["x", "y"], ["z", "x"], top=3)
    assert both[0] == "x"
    # an absent chunk contributes exactly zero: x alone in the first list
    assert cp.rrf_fuse(["x"], [], top=1) == ["x"]


def test_rrf_scores_and_tie_break_are_deterministic() -> None:
    k = cp.RRF_K
    # b and c tie on score (rank 1 in one list each): the best rank is 1 for both, then chunk id
    assert cp.rrf_fuse(["c"], ["b"], top=2) == ["b", "c"]
    # score of a chunk in both lists at ranks 1 and 2
    fused = cp.rrf_fuse(["p", "q"], ["q", "p"], top=2)
    assert fused == ["p", "q"]  # equal scores, equal best rank 1, tie broken by id
    assert math.isclose(1 / (k + 1) + 1 / (k + 2), 1 / (k + 2) + 1 / (k + 1))


def test_heavy_systems_are_valid_when_two_passes_agree() -> None:
    verdict = cp.judge_heavy(10.0, 10.9)
    assert verdict.status == "valid" and verdict.reported == ("1", "2")
    assert cp.judge_heavy(10.0, 10.0).drift == 0.0


def test_heavy_systems_need_one_repeat_then_are_valid_or_unstable() -> None:
    assert cp.judge_heavy(10.0, 12.0).status == "needs_repeat"
    good = cp.judge_heavy(10.0, 12.0, 10.5)
    assert good.status == "valid_after_repeat" and good.reported == ("1", "2r")
    bad = cp.judge_heavy(10.0, 12.0, 13.0)
    assert bad.status == "measurement_unstable" and bad.reported == ("1",)


def test_the_drift_limit_is_inclusive() -> None:
    assert cp.judge_heavy(100.0, 110.0).status == "valid"  # exactly 10 %
    assert cp.judge_heavy(100.0, 110.0001).status == "needs_repeat"


def test_fast_systems_compare_each_pass_with_the_median_of_the_five() -> None:
    assert cp.judge_fast([1.0, 1.02, 0.98, 1.05, 0.95]).status == "valid"
    assert cp.judge_fast([1.0, 1.0, 1.0, 1.0, 1.5]).status == "needs_repeat"
    assert cp.judge_fast([1.0, 1.0, 1.0, 1.0, 1.5], [1.0, 1.01, 1.0, 0.99, 1.0]).status == (
        "valid_after_repeat"
    )
    assert cp.judge_fast([1.0, 1.0, 1.0, 1.0, 1.5], [1.0, 1.0, 1.0, 1.0, 1.5]).status == (
        "measurement_unstable"
    )


def test_ranking_identity_looks_at_the_first_ten_in_order() -> None:
    a = [f"c{i}" for i in range(20)]
    assert cp.same_top(a, a)
    assert cp.same_top(a, a[:10] + ["other"])  # only the first ten count
    assert not cp.same_top(a, [a[1], a[0], *a[2:]])
    assert not cp.same_top(a[:9], a[:10])


def test_score_comparison_reports_exact_within_tolerance_or_differs() -> None:
    base = {("q1", "c1"): 1.5, ("q1", "c2"): -0.25}
    assert cp.compare_scores(base, dict(base)).outcome == "exact"
    near = {("q1", "c1"): 1.5 + 5e-7, ("q1", "c2"): -0.25}
    result = cp.compare_scores(base, near)
    assert result.outcome == "within tolerance" and result.max_abs_diff < 1e-6
    assert cp.compare_scores(base, {("q1", "c1"): 1.5 + 2e-6, ("q1", "c2"): -0.25}).outcome == (
        "differs"
    )
    assert cp.compare_scores(base, {("q1", "c1"): 1.5}).outcome == "differs"  # other keys


BATTERY = (
    "Now drawing from 'Battery Power'\n"
    " -InternalBattery-0 (id=1)\t76%; discharging; 11:48 remaining present: true\n"
)
AC = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t80%; charging; present: true\n"


def test_power_and_low_power_mode() -> None:
    assert not cp.on_ac_power(BATTERY)
    assert cp.on_ac_power(AC)
    assert not cp.on_ac_power("")
    assert cp.low_power_mode_off("System-wide power settings:\n lowpowermode         0\n")
    assert not cp.low_power_mode_off(" lowpowermode         1\n")
    assert not cp.low_power_mode_off("nothing about it")


QUIET = (
    "Note: No thermal warning level has been recorded\n"
    "Note: No performance warning level has been recorded\n"
    "Note: No CPU power status has been recorded\n"
)


def test_thermal_state() -> None:
    assert cp.thermal_ok(QUIET)
    assert not cp.thermal_ok(QUIET + "CPU_Speed_Limit \t= 80\n")
    assert cp.thermal_ok(QUIET + "CPU_Speed_Limit \t= 100\n")
    assert not cp.thermal_ok("Thermal warning level: 2\n")


def test_load_average() -> None:
    line = "18:56  up 13 days, 11:15, 1 user, load averages: 2.44 2.27 2.16"
    assert cp.one_minute_load(line) == 2.44
    assert cp.load_ok(line)
    assert not cp.load_ok(line.replace("2.44", "3.00"))
    with pytest.raises(ValueError):
        cp.one_minute_load("nothing")


def test_time_report_parsing() -> None:
    report = (
        "        0.00 real         0.00 user         0.00 sys\n"
        "              999424  maximum resident set size\n"
        "             1048816  peak memory footprint\n"
    )
    assert cp.parse_time_l(report) == {
        "maximum_resident_set_size": 999424,
        "peak_memory_footprint": 1048816,
    }
    assert cp.parse_time_l("nothing") == {}


def test_resident_memory_of_a_process_tree() -> None:
    ps = "10 1 100\n11 10 200\n12 11 300\n20 1 5000\n"
    assert cp.descendants_rss(ps, 10) == 600  # 10, its child 11 and grandchild 12
    assert cp.descendants_rss(ps, 20) == 5000
    assert cp.descendants_rss(ps, 99) == 0


def test_the_warmup_queries_are_three_fixed_strings() -> None:
    assert len(cp.WARMUP_QUERIES) == 3 and len(set(cp.WARMUP_QUERIES)) == 3


def test_power_source_level_and_profile() -> None:
    assert cp.power_source(BATTERY) == "Battery" and cp.power_source(AC) == "AC"
    assert cp.power_source("") == "unknown" and cp.power_source("nothing") == "unknown"
    assert cp.battery_percent(BATTERY) == 76 and cp.battery_percent(AC) == 80
    assert cp.battery_percent("no level here") is None
    assert (cp.B_START, cp.B_CRITICAL) == (40, 15)
    assert cp.power_profile(["AC", "AC"]) == "ac_throughout"
    assert cp.power_profile(["Battery", "Battery", "Battery"]) == "battery_throughout"
    assert cp.power_profile(["Battery", "AC"]) == "mixed"
    assert cp.power_profile(["AC", "unknown"]) == "mixed"
    assert cp.power_profile([]) == "mixed"
