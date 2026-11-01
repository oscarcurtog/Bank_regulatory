"""The pure rules of the S5.3 operational-cost protocol (stage A of the S5-S6 preregistration).

The private protocol document is pinned by its SHA-256 in
config/retrieval/s5-s6-preregistration-a.json. This module holds only what can be decided
without a model, a label or a clock: nearest-rank percentiles, the order and seed of a pass, the
fusion of two top-50 rankings, the validity of a system's passes, the comparison of rankings and
scores, and the parsing of what macOS reports about power, heat, load and memory. Nothing here
reads a qrel, a metric summary or TEST.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

BASE_SEED = 20261101  # the seed of pass n is BASE_SEED + n
REPEAT_OFFSET = 100  # a permitted repetition reuses the seed of its pass plus this
DRIFT_LIMIT = 0.10
FLOAT_SLACK = 1e-9  # so that exactly 10 % is not lost to the rounding of a division
LOAD_LIMIT = 2.5
SCORE_TOLERANCE = 1e-6
RRF_K = 60
RRF_DEPTH = 50
TOP = 10
WARMUP_QUERIES = (
    "Which obligations apply to a payment service provider that outsources part of its operations?",
    "What must a crypto-asset service provider disclose to its clients before offering a service?",
    "How should a financial entity report a major incident to its supervisor?",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def nearest_rank_percentile(values: Sequence[float], percent: float) -> float:
    """The nearest-rank percentile: the value at rank ceil(percent / 100 * n) of the sorted
    values (M-054 used the same rule)."""
    if not values:
        raise ValueError("no values")
    if not 0 < percent <= 100:
        raise ValueError(f"percent must be in (0, 100], got {percent}")
    ordered = sorted(values)
    rank = max(1, math.ceil(percent / 100 * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True)
class Summary:
    n: int
    p50: float
    p95: float
    maximum: float
    mean: float


def summarize(values: Sequence[float]) -> Summary:
    return Summary(
        len(values),
        nearest_rank_percentile(values, 50),
        nearest_rank_percentile(values, 95),
        max(values),
        sum(values) / len(values),
    )


# ---------------------------------------------------------------------------
# order and seeds
# ---------------------------------------------------------------------------


def seed_for_pass(number: int) -> int:
    if number < 1:
        raise ValueError("passes are numbered from 1")
    return BASE_SEED + number


def repeat_seed(seed: int) -> int:
    return seed + REPEAT_OFFSET


def pass_order(query_ids: Sequence[str], seed: int) -> list[str]:
    """The ids sorted, then shuffled with random.Random(seed): the same seed always gives the
    same order, and different seeds give different ones."""
    ordered = sorted(query_ids)
    random.Random(seed).shuffle(ordered)
    return ordered


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------


def rrf_fuse(
    first: Sequence[str],
    second: Sequence[str],
    k: int = RRF_K,
    depth: int = RRF_DEPTH,
    top: int = TOP,
) -> list[str]:
    """Reciprocal rank fusion of the first `depth` of two rankings, equal weights: the score of a
    chunk is the sum over the lists that hold it of 1 / (k + rank), ranks from 1. A chunk absent
    from a list's first `depth` contributes nothing from that list; no rank is invented for it.
    Order: score descending, then the best rank it has in either list, then the chunk id."""
    lists = (list(first[:depth]), list(second[:depth]))
    score: dict[str, float] = {}
    best: dict[str, int] = {}
    for ranking in lists:
        for position, chunk in enumerate(ranking, start=1):
            score[chunk] = score.get(chunk, 0.0) + 1.0 / (k + position)
            best[chunk] = min(best.get(chunk, position), position)
    ordered = sorted(score, key=lambda c: (-score[c], best[c], c))
    return ordered[:top]


# ---------------------------------------------------------------------------
# validity of a system's passes
# ---------------------------------------------------------------------------


def within_limit(value: float) -> bool:
    """The drift limit is inclusive: exactly 10 % passes."""
    return value <= DRIFT_LIMIT + FLOAT_SLACK


def drift(first: float, second: float) -> float:
    """max / min - 1 of two medians."""
    low, high = min(first, second), max(first, second)
    if low <= 0:
        raise ValueError("a median must be positive")
    return high / low - 1.0


@dataclass(frozen=True)
class Verdict:
    status: str  # valid, valid_after_repeat, measurement_unstable, needs_repeat
    drift: float
    reported: tuple[str, ...]  # the passes the headline figures come from


def judge_heavy(median_1: float, median_2: float, median_2_repeat: float | None = None) -> Verdict:
    """Two passes agree when their medians differ by at most DRIFT_LIMIT. If not, pass 2 may be
    repeated once; if the repetition agrees with pass 1 the headline figures use passes 1 and its
    repetition, otherwise the system is unstable and pass 1 is reported with that mark. Every pass
    is kept either way."""
    first = drift(median_1, median_2)
    if within_limit(first):
        return Verdict("valid", first, ("1", "2"))
    if median_2_repeat is None:
        return Verdict("needs_repeat", first, ())
    second = drift(median_1, median_2_repeat)
    if within_limit(second):
        return Verdict("valid_after_repeat", second, ("1", "2r"))
    return Verdict("measurement_unstable", second, ("1",))


def judge_fast(medians: Sequence[float], repeat_medians: Sequence[float] | None = None) -> Verdict:
    """Five passes are valid when every pass median is within DRIFT_LIMIT of the median of the
    five. If not, the whole block of five may be repeated once and is judged the same way."""

    def worst(values: Sequence[float]) -> float:
        reference = nearest_rank_percentile(values, 50)
        return max(abs(v / reference - 1.0) for v in values)

    first = worst(medians)
    if within_limit(first):
        return Verdict("valid", first, tuple(str(i) for i in range(1, len(medians) + 1)))
    if repeat_medians is None:
        return Verdict("needs_repeat", first, ())
    second = worst(repeat_medians)
    if within_limit(second):
        return Verdict(
            "valid_after_repeat", second, tuple(f"{i}r" for i in range(1, len(repeat_medians) + 1))
        )
    return Verdict(
        "measurement_unstable", second, tuple(str(i) for i in range(1, len(medians) + 1))
    )


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------


def same_top(ours: Sequence[str], stored: Sequence[str], k: int = TOP) -> bool:
    """The first k ids, in order, are identical."""
    return list(ours[:k]) == list(stored[:k])


@dataclass(frozen=True)
class ScoreComparison:
    outcome: str  # exact, within tolerance, differs
    max_abs_diff: float
    compared: int


def compare_scores[K: Hashable](
    first: Mapping[K, float],
    second: Mapping[K, float],
    tolerance: float = SCORE_TOLERANCE,
) -> ScoreComparison:
    """Scores of the same keys in two passes: `exact` when every pair is identical,
    `within tolerance` when the largest absolute difference is at most `tolerance`, else
    `differs`. Different key sets always differ."""
    if set(first) != set(second):
        return ScoreComparison("differs", math.inf, min(len(first), len(second)))
    largest = max((abs(first[key] - second[key]) for key in first), default=0.0)
    if largest == 0.0:
        return ScoreComparison("exact", 0.0, len(first))
    if largest <= tolerance:
        return ScoreComparison("within tolerance", largest, len(first))
    return ScoreComparison("differs", largest, len(first))


# ---------------------------------------------------------------------------
# what macOS reports
# ---------------------------------------------------------------------------


def on_ac_power(pmset_batt: str) -> bool:
    return "'AC Power'" in pmset_batt.splitlines()[0] if pmset_batt.strip() else False


def low_power_mode_off(pmset_all: str) -> bool:
    match = re.search(r"^\s*lowpowermode\s+(\d+)", pmset_all, re.MULTILINE)
    return match is not None and match.group(1) == "0"


def thermal_ok(pmset_therm: str) -> bool:
    """No recorded warning and no CPU limit below its maximum. The notes `No ... has been
    recorded` are the quiet state; a warning level without `No`, or a limit under 100 (or fewer
    available CPUs than the machine has), is not."""
    for line in pmset_therm.splitlines():
        text = line.strip()
        if not text:
            continue
        limit = re.search(r"(CPU_Speed_Limit|CPU_Scheduler_Limit)\s*=\s*(\d+)", text)
        if limit and int(limit.group(2)) < 100:
            return False
        if "warning level" in text and "No " not in text and "no " not in text:
            return False
    return True


def one_minute_load(uptime: str) -> float:
    match = re.search(r"load averages?:\s*([\d.,]+)", uptime)
    if not match:
        raise ValueError(f"no load average in {uptime!r}")
    return float(match.group(1).replace(",", "."))


def load_ok(uptime: str, limit: float = LOAD_LIMIT) -> bool:
    return one_minute_load(uptime) <= limit


def parse_time_l(report: str) -> dict[str, int]:
    """Bytes of `peak memory footprint` and `maximum resident set size` from /usr/bin/time -l."""
    out: dict[str, int] = {}
    for label, key in (
        ("peak memory footprint", "peak_memory_footprint"),
        ("maximum resident set size", "maximum_resident_set_size"),
    ):
        match = re.search(rf"^\s*(\d+)\s+{label}\s*$", report, re.MULTILINE)
        if match:
            out[key] = int(match.group(1))
    return out


def descendants_rss(ps_output: str, root: int) -> int:
    """Kilobytes of resident memory summed over `root` and its descendants, from
    `ps -axo pid=,ppid=,rss=`. An approximate, coarse observation, never a peak."""
    rows: dict[int, tuple[int, int]] = {}
    for line in ps_output.splitlines():
        parts = line.split()
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            rows[int(parts[0])] = (int(parts[1]), int(parts[2]))
    tree = {root}
    changed = True
    while changed:
        changed = False
        for pid, (parent, _) in rows.items():
            if parent in tree and pid not in tree:
                tree.add(pid)
                changed = True
    return sum(rows[pid][1] for pid in tree if pid in rows)
