"""The pure rules of the S5-S6 retrieval selection (stage B of the preregistration).

The private document of stage B is pinned by its SHA-256 in
config/retrieval/s5-s6-preregistration-b.json. This module holds only what can be decided from
numbers: the latency classes, the paired bootstrap of a difference between two systems, the
label of a comparison, the nDCG guardrail, the influence of the single most influential need,
the coverage sensitivity envelope, the three coverage states, cost dominance, and the assignment
of the two paths. It reads no qrel and no file: the caller passes the numbers in.

Coverage sensitivity envelope. A chunk that no qrel judges for a need is "unjudged". For a
comparison of a candidate against a reference, the unjudged chunks that appear in the top-10 of
either system, for any query of the need, may be given a hypothetical grade of the DEV scale
(0, 1, 2 or 3). One chunk gets ONE grade per need, whichever system or query it appears in, so
the same piece of evidence is graded the same everywhere. The candidate-favourable bound is the
assignment that maximizes the candidate's advantage, the candidate-unfavourable bound the one
that minimizes it, separately for Recall@10 (relevant = grade >= 2, the relevant set and its
size grow with the assignment) and for nDCG@10 (linear gain, the ideal DCG grows with the
assignment). Both optima are exact:

- Recall: with A the set of unjudged chunks graded relevant, the need's difference is
  (base + sum of w_c over A) / (R + |A|), where R is the number of judged relevant chunks, base
  the judged difference in relevant hits summed over the queries, and w_c the number of queries
  whose candidate top-10 holds c minus the number whose reference top-10 holds it, all divided by
  the number of queries. For a fixed |A| = k the best A takes the k largest w_c (the worst, the k
  smallest); scanning k gives the exact maximum and minimum.
- nDCG: the numerator is linear in the grades (sum over chunks of grade times v_c, the net
  discounted presence) and the ideal DCG depends only on how many unjudged chunks get each grade.
  For each count profile (n3, n2, n1) the best numerator pairs the highest grades with the
  largest v_c (the worst, with the smallest); enumerating every profile gives the exact optimum.

The two metrics are bounded separately, so the favourable (or unfavourable) scenario need not be
attainable by one grade assignment for both at once. The envelope is therefore conservative: it
can be wider than the set of consistent assignments, never narrower. No assignment is stored as
a label.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

# thresholds fixed by D-069 and carried by the preregistration
MEPI_RECALL = 0.03
MEPI_NDCG = 0.02
DELTA_RECALL = 0.03
DELTA_NDCG = 0.02
BOOTSTRAP_SEED = 20261013
BOOTSTRAP_REPS = 10_000
TOLERANCE = 1e-9
TOP = 10
RELEVANT_GRADE = 2
GRADES = (0, 1, 2, 3)
COST_BAND = 0.10  # engineering tolerance on latency and memory, not a statistical interval
FAST_LATENCY_LIMIT = 10.0
MID_LATENCY_LIMIT = 30.0

FAST_LATENCY = "FAST_LATENCY"
MID_LATENCY = "MID_LATENCY"
SLOW_LATENCY = "SLOW_LATENCY"

# the nine labels
FRAGILE = "fragile material signal"
STRONG = "strong practical improvement evidence"
SUPPORTED = "supported improvement"
MATERIAL = "material improvement signal"
BELOW_MEPI = "improvement below MEPI"
INFERIOR = "inferior beyond δ"
NON_INFERIOR = "non-inferior within δ"
UNRESOLVED = "quality unresolved"
NOT_EVALUABLE_LABEL = "not evaluable"
LABELS = (
    NOT_EVALUABLE_LABEL,
    FRAGILE,
    STRONG,
    SUPPORTED,
    MATERIAL,
    BELOW_MEPI,
    INFERIOR,
    NON_INFERIOR,
    UNRESOLVED,
)

# nDCG guardrail states
PASSED = "passed"
GUARD_UNRESOLVED = "unresolved"
VIOLATED = "violated"

# coverage states
COVERAGE_ROBUST = "coverage_robust"
COVERAGE_AFFECTED = "coverage_affected"
NOT_EVALUABLE = "not_evaluable"

# actions of H2b
DISPLACE = "displace"
FRONTIER = "frontier"
DISCARD = "discard"


def at_least(a: float, b: float) -> bool:
    return a >= b - TOLERANCE


def at_most(a: float, b: float) -> bool:
    return a <= b + TOLERANCE


def above(a: float, b: float) -> bool:
    return a > b + TOLERANCE


def below(a: float, b: float) -> bool:
    return a < b - TOLERANCE


# ---------------------------------------------------------------------------
# H1: latency classes
# ---------------------------------------------------------------------------


def latency_class(p95_seconds: float) -> str:
    """FAST_LATENCY up to 10 s, MID_LATENCY up to 30 s, SLOW_LATENCY above; the limits belong to
    the lower class. A product assumption measured on one machine, never a production SLA."""
    if at_most(p95_seconds, FAST_LATENCY_LIMIT):
        return FAST_LATENCY
    if at_most(p95_seconds, MID_LATENCY_LIMIT):
        return MID_LATENCY
    return SLOW_LATENCY


# ---------------------------------------------------------------------------
# paired bootstrap (the procedure of M-035 and D-067, ingest/run_report.paired_bootstrap)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    mean: float
    low2: float  # two-sided 95 %, lower limit
    high2: float
    low1: float  # one-sided 95 %, lower limit
    high1: float  # one-sided 95 %, upper limit


def resample_indices(
    n: int, reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED
) -> list[list[int]]:
    """The needs drawn in every replicate, as positions. The generator is created here, from the
    seed, so every comparison sees the same draws, exactly as paired_bootstrap does."""
    rng = random.Random(seed)
    return [rng.choices(range(n), k=n) for _ in range(reps)]


def paired_interval(deltas: Sequence[float], indices: Sequence[Sequence[int]]) -> Interval:
    """The mean of the per-need differences (ordered by need id) and its percentile intervals."""
    reps = len(indices)
    means = sorted(statistics.fmean([deltas[i] for i in draw]) for draw in indices)
    return Interval(
        statistics.fmean(deltas),
        means[int(0.025 * reps)],
        means[int(0.975 * reps) - 1],
        means[int(0.05 * reps)],
        means[int(0.95 * reps) - 1],
    )


def drop_most_influential(deltas: Sequence[float]) -> float:
    """H3: the mean of the differences without the need with the largest one (ties: the first
    need by id, so the first position). The signal is robust if this still reaches MEPI."""
    if len(deltas) < 2:
        raise ValueError("at least two needs are needed")
    top = max(range(len(deltas)), key=lambda i: (deltas[i], -i))
    return statistics.fmean(d for i, d in enumerate(deltas) if i != top)


# ---------------------------------------------------------------------------
# labels, guardrail
# ---------------------------------------------------------------------------


def ndcg_guardrail(interval: Interval) -> str:
    """Against the architecture the candidate aims to displace: violated if the point estimate is
    at or below -delta, passed if the one-sided lower limit is above -delta, else unresolved."""
    if at_most(interval.mean, -DELTA_NDCG):
        return VIOLATED
    if above(interval.low1, -DELTA_NDCG):
        return PASSED
    return GUARD_UNRESOLVED


def recall_label(interval: Interval, drop_one: float, guardrail: str) -> str:
    """Rules R1 to R8 of the preregistration, first match wins. R0 (`not evaluable`) is decided
    from the coverage states, not from these numbers. Exhaustive: R8 closes the list."""
    d = interval.mean
    if at_least(d, MEPI_RECALL):
        if not at_least(drop_one, MEPI_RECALL):
            return FRAGILE
        if above(interval.low2, MEPI_RECALL):
            return STRONG
        if above(interval.low2, 0.0):
            return SUPPORTED
        return MATERIAL
    if above(interval.low2, 0.0):
        return BELOW_MEPI
    if below(interval.high1, -DELTA_RECALL):
        return INFERIOR
    if above(interval.low1, -DELTA_RECALL) and guardrail != VIOLATED:
        return NON_INFERIOR
    return UNRESOLVED


@dataclass(frozen=True)
class Scenario:
    recall: Interval
    drop_one: float
    ndcg: Interval
    guardrail: str
    label: str  # R1 to R8


def evaluate_scenario(
    delta_recall: Sequence[float],
    delta_ndcg: Sequence[float],
    indices: Sequence[Sequence[int]],
) -> Scenario:
    recall = paired_interval(delta_recall, indices)
    ndcg = paired_interval(delta_ndcg, indices)
    guardrail = ndcg_guardrail(ndcg)
    drop = drop_most_influential(delta_recall)
    return Scenario(recall, drop, ndcg, guardrail, recall_label(recall, drop, guardrail))


# ---------------------------------------------------------------------------
# coverage sensitivity envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryLists:
    candidate: tuple[str, ...]
    reference: tuple[str, ...]


@dataclass(frozen=True)
class NeedData:
    need_id: str
    grades: Mapping[str, int]  # judged chunks (grade 0 included); a chunk not here is unjudged
    queries: tuple[QueryLists, ...]


@dataclass(frozen=True)
class Range:
    low: float
    high: float
    low_assignment: Mapping[str, int]  # audit only, never stored as labels
    high_assignment: Mapping[str, int]


def _discount(rank: int) -> float:
    return 1.0 / math.log2(rank + 1)


def _top(items: Sequence[str]) -> Sequence[str]:
    return items[:TOP]


def unjudged_chunks(need: NeedData) -> list[str]:
    """The unjudged chunks in the top-10 of either system, for any query of the need."""
    seen: set[str] = set()
    for q in need.queries:
        seen.update(c for c in _top(q.candidate) if c not in need.grades)
        seen.update(c for c in _top(q.reference) if c not in need.grades)
    return sorted(seen)


def recall_delta(need: NeedData, extra: Mapping[str, int] | None = None) -> float:
    """Recall@10 of the candidate minus that of the reference for the need, mean over its queries,
    with the optional hypothetical grades of unjudged chunks. The oracle of the tests."""
    grades = {**need.grades, **(extra or {})}
    relevant = {c for c, g in grades.items() if g >= RELEVANT_GRADE}
    if not relevant:
        raise ValueError(f"need {need.need_id}: no relevant chunk")
    diffs = []
    for q in need.queries:
        hit_c = sum(c in relevant for c in _top(q.candidate))
        hit_r = sum(c in relevant for c in _top(q.reference))
        diffs.append((hit_c - hit_r) / len(relevant))
    return statistics.fmean(diffs)


def _ideal_dcg(positive_grades: Sequence[int]) -> float:
    ordered = sorted((g for g in positive_grades if g > 0), reverse=True)[:TOP]
    return sum(g * _discount(i) for i, g in enumerate(ordered, start=1))


def _dcg(items: Sequence[str], grades: Mapping[str, int]) -> float:
    return sum(grades.get(c, 0) * _discount(i) for i, c in enumerate(_top(items), start=1))


def ndcg_delta(need: NeedData, extra: Mapping[str, int] | None = None) -> float:
    """nDCG@10 (linear gain) of the candidate minus that of the reference for the need, mean over
    its queries, with the optional hypothetical grades. The oracle of the tests."""
    grades = {**need.grades, **(extra or {})}
    ideal = _ideal_dcg(list(grades.values()))
    if ideal <= 0:
        raise ValueError(f"need {need.need_id}: no positive grade")
    diffs = [(_dcg(q.candidate, grades) - _dcg(q.reference, grades)) / ideal for q in need.queries]
    return statistics.fmean(diffs)


def recall_range(need: NeedData) -> Range:
    """The exact minimum and maximum of recall_delta over every assignment of the unjudged
    chunks."""
    unjudged = unjudged_chunks(need)
    relevant = sum(g >= RELEVANT_GRADE for g in need.grades.values())
    if relevant == 0:
        raise ValueError(f"need {need.need_id}: no relevant chunk")
    n_queries = len(need.queries)
    base = 0.0
    for q in need.queries:
        base += sum(need.grades.get(c, -1) >= RELEVANT_GRADE for c in _top(q.candidate))
        base -= sum(need.grades.get(c, -1) >= RELEVANT_GRADE for c in _top(q.reference))
    weight = {
        c: sum(
            (c in _top(q.candidate)) - (c in _top(q.reference))  # bool arithmetic: -1, 0 or 1
            for q in need.queries
        )
        for c in unjudged
    }

    def best(descending: bool) -> tuple[float, Mapping[str, int]]:
        order = sorted(unjudged, key=lambda c: (-weight[c] if descending else weight[c], c))
        chosen_value = base / (relevant * n_queries)
        chosen_k = 0
        running = 0.0
        for k, chunk in enumerate(order, start=1):
            running += weight[chunk]
            value = (base + running) / ((relevant + k) * n_queries)
            if (value > chosen_value) if descending else (value < chosen_value):
                chosen_value, chosen_k = value, k
        return chosen_value, {c: (2 if c in order[:chosen_k] else 0) for c in unjudged}

    high, high_assignment = best(True)
    low, low_assignment = best(False)
    return Range(low, high, low_assignment, high_assignment)


def ndcg_range(need: NeedData) -> Range:
    """The exact minimum and maximum of ndcg_delta over every assignment of the unjudged chunks."""
    unjudged = unjudged_chunks(need)
    judged_positive = [g for g in need.grades.values() if g > 0]
    if not judged_positive:
        raise ValueError(f"need {need.need_id}: no positive grade")
    n_queries = len(need.queries)
    base = 0.0
    for q in need.queries:
        base += _dcg(q.candidate, need.grades) - _dcg(q.reference, need.grades)
    base /= n_queries
    presence = {}
    for c in unjudged:
        total = 0.0
        for q in need.queries:
            cand, ref = _top(q.candidate), _top(q.reference)
            total += (_discount(cand.index(c) + 1) if c in cand else 0.0) - (
                _discount(ref.index(c) + 1) if c in ref else 0.0
            )
        presence[c] = total / n_queries
    ideal_cache: dict[tuple[int, int, int], float] = {}

    def ideal(n3: int, n2: int, n1: int) -> float:
        key = (n3, n2, n1)
        if key not in ideal_cache:
            ideal_cache[key] = _ideal_dcg([*judged_positive, *([3] * n3), *([2] * n2), *([1] * n1)])
        return ideal_cache[key]

    def best(descending: bool) -> tuple[float, Mapping[str, int]]:
        order = sorted(unjudged, key=lambda c: (-presence[c] if descending else presence[c], c))
        prefix = [0.0]
        for chunk in order:
            prefix.append(prefix[-1] + presence[chunk])
        n = len(order)
        chosen_value = base / ideal(0, 0, 0)
        chosen = (0, 0, 0)
        for n3 in range(n + 1):
            for n2 in range(n - n3 + 1):
                for n1 in range(n - n3 - n2 + 1):
                    numerator = (
                        base
                        + 3 * prefix[n3]
                        + 2 * (prefix[n3 + n2] - prefix[n3])
                        + (prefix[n3 + n2 + n1] - prefix[n3 + n2])
                    )
                    value = numerator / ideal(n3, n2, n1)
                    if (value > chosen_value) if descending else (value < chosen_value):
                        chosen_value, chosen = value, (n3, n2, n1)
        n3, n2, n1 = chosen
        assignment = dict.fromkeys(unjudged, 0)
        for position, chunk in enumerate(order):
            if position < n3:
                assignment[chunk] = 3
            elif position < n3 + n2:
                assignment[chunk] = 2
            elif position < n3 + n2 + n1:
                assignment[chunk] = 1
        return chosen_value, assignment

    high, high_assignment = best(True)
    low, low_assignment = best(False)
    return Range(low, high, low_assignment, high_assignment)


# ---------------------------------------------------------------------------
# a comparison under the three scenarios, and the coverage state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    candidate: str
    reference: str
    primary: Scenario  # the frozen qrels, exactly
    favourable: Scenario  # candidate-favourable bound
    unfavourable: Scenario  # candidate-unfavourable bound

    def scenarios(self) -> tuple[Scenario, Scenario, Scenario]:
        return (self.primary, self.favourable, self.unfavourable)


def build_comparison(
    candidate: str,
    reference: str,
    primary_recall: Sequence[float],
    primary_ndcg: Sequence[float],
    recall_ranges: Sequence[Range],
    ndcg_ranges: Sequence[Range],
    indices: Sequence[Sequence[int]],
) -> Comparison:
    """All sequences are ordered by need id. The primary differences come from the evaluator."""
    n = len(primary_recall)
    if not (len(primary_ndcg) == len(recall_ranges) == len(ndcg_ranges) == n):
        raise ValueError("one value per need is needed everywhere")
    return Comparison(
        candidate,
        reference,
        evaluate_scenario(primary_recall, primary_ndcg, indices),
        evaluate_scenario([r.high for r in recall_ranges], [r.high for r in ndcg_ranges], indices),
        evaluate_scenario([r.low for r in recall_ranges], [r.low for r in ndcg_ranges], indices),
    )


def coverage_state(
    comparison: Comparison,
    action: Callable[[Scenario], str],
) -> str:
    """coverage_robust: the label and the guardrail are the same in the three scenarios.
    coverage_affected: they change, the action of H2b does not.
    not_evaluable: the action changes."""
    scenarios = comparison.scenarios()
    if len({(s.label, s.guardrail) for s in scenarios}) == 1:
        return COVERAGE_ROBUST
    if len({action(s) for s in scenarios}) == 1:
        return COVERAGE_AFFECTED
    return NOT_EVALUABLE


def final_label(comparison: Comparison, state: str) -> str:
    """R0 first: `not evaluable` if the coverage state says so, else the label of the frozen
    qrels."""
    return NOT_EVALUABLE_LABEL if state == NOT_EVALUABLE else comparison.primary.label


# ---------------------------------------------------------------------------
# H2b actions and H7 dominance
# ---------------------------------------------------------------------------


def fast_action(latency: str, scenario: Scenario) -> str:
    """Candidate against E5. `discard` if inferior beyond delta; `displace` if it is in
    FAST_LATENCY and the label is supported improvement or strong practical improvement evidence
    (which already means a point estimate >= MEPI, a two-sided lower limit above 0 and a signal
    that survives without the most influential need) and the nDCG guardrail is not violated;
    else `frontier`. The coverage state and the reproducibility gate are applied by the caller."""
    if scenario.label == INFERIOR:
        return DISCARD
    if (
        latency == FAST_LATENCY
        and scenario.label in (SUPPORTED, STRONG)
        and scenario.guardrail != VIOLATED
    ):
        return DISPLACE
    return FRONTIER


def prefers_candidate(scenario: Scenario) -> bool:
    """For the tie-break of N=20 against RRF: the same two conditions as a displacement."""
    return scenario.label in (SUPPORTED, STRONG) and scenario.guardrail != VIOLATED


def dominance_quality(scenario: Scenario) -> str:
    """W against V: `certified` if W is non-inferior within delta in Recall (one-sided lower
    limit above -delta) and its nDCG guardrail is not violated, else `unresolved`."""
    if above(scenario.recall.low1, -DELTA_RECALL) and scenario.guardrail != VIOLATED:
        return "certified"
    return "unresolved"


@dataclass(frozen=True)
class CostRow:
    p95_seconds: float
    memory_gib: float
    complexity: tuple[int, int, int, int, int]  # models, stages, runtimes, processes, indices
    portability: int  # 0 best


def _axis(w: float, v: float) -> int:
    """-1: w is better (at least 10 % lower), 0: tie within the band, 1: w is worse."""
    if w * (1 + COST_BAND) <= v:
        return -1
    if v * (1 + COST_BAND) <= w:
        return 1
    return 0


def cost_dominates(w: CostRow, v: CostRow) -> bool:
    """W is not worse than V on any cost axis (outside the 10 % band) and clearly better on one."""
    latency, memory = _axis(w.p95_seconds, v.p95_seconds), _axis(w.memory_gib, v.memory_gib)
    complexity_ok = all(a <= b for a, b in zip(w.complexity, v.complexity, strict=True))
    complexity_better = any(a < b for a, b in zip(w.complexity, v.complexity, strict=True))
    not_worse = latency <= 0 and memory <= 0 and complexity_ok and w.portability <= v.portability
    better = latency < 0 or memory < 0 or complexity_better or w.portability < v.portability
    return not_worse and better


def dominance_pairs(costs: Mapping[str, CostRow]) -> list[tuple[str, str]]:
    """Every ordered pair (W, V) such that W dominates V on cost, in a fixed order."""
    names = sorted(costs)
    return [(w, v) for w in names for v in names if w != v and cost_dominates(costs[w], costs[v])]


# structure of each system (A section 3 and the preregistration): complexity vector, portability
STRUCTURE: dict[str, tuple[tuple[int, int, int, int, int], int]] = {
    "E5": ((1, 2, 1, 1, 0), 0),
    "RRF": ((1, 3, 1, 1, 1), 0),
    "N20": ((2, 4, 1, 1, 0), 1),
    "N50": ((2, 4, 1, 1, 0), 1),
    "D-4B": ((2, 7, 2, 2, 0), 2),
    "E": ((3, 10, 2, 2, 0), 2),
}


# ---------------------------------------------------------------------------
# the two paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Paths:
    fast_path: str
    quality_path: str
    discarded: Mapping[str, str]  # system -> the reason
    frontier: tuple[str, ...]


def assign_paths(
    actions: Mapping[str, str],
    n20_over_rrf: bool,
    certified: Mapping[tuple[str, str], bool],
) -> Paths:
    """actions: the H2b action of RRF, N20, N50 and D-4B against E5, already `frontier` where the
    coverage state is not_evaluable. certified: dominance W over V, already False where the
    coverage state is not_evaluable or the quality rule fails. E is the QUALITY PATH by
    construction (TEST v2) and is never discarded; a dominance of E is descriptive."""
    discarded: dict[str, str] = {}
    for system in ("RRF", "N20", "N50", "D-4B"):
        if actions.get(system) == DISCARD:
            discarded[system] = "inferior beyond δ against E5"
    for (w, v), holds in sorted(certified.items()):
        if holds and v != "E" and v not in discarded:
            discarded[v] = f"cost-dominated by {w} with non-inferiority certified (H7)"
    challengers = [s for s in ("RRF", "N20") if actions.get(s) == DISPLACE and s not in discarded]
    if len(challengers) == 2:
        fast = "N20" if n20_over_rrf else "RRF"
    elif challengers:
        fast = challengers[0]
    else:
        fast = "E5"
    members = ("E5", "RRF", "N20", "N50", "D-4B", "E")
    frontier = tuple(s for s in members if s not in discarded and s not in (fast, "E"))
    return Paths(fast, "E", discarded, frontier)


# ---------------------------------------------------------------------------
# the whole analysis, from numbers
# ---------------------------------------------------------------------------

SYSTEMS = ("E5", "RRF", "N20", "N50", "D-4B", "E")
CANDIDATES = ("RRF", "N20", "N50", "D-4B")  # compared with E5 for an H2b action
VS_E5 = ("RRF", "N20", "N50", "D-4B", "E")
VS_E = ("E5", "RRF", "N20", "N50", "D-4B")
TIE_BREAK = ("N20", "RRF")
UNTESTED = ("E5", "E")  # TEST v2 already puts E5 below E by more than delta: not tested on DEV
PRIMARY_CHECK = 1e-9

Pair = tuple[str, str]
Lists = Mapping[str, Mapping[str, tuple[str, ...]]]  # system -> query id -> top-10 chunk ids
PerNeed = Mapping[str, Mapping[str, float]]  # system -> need id -> mean over the need's queries


@dataclass(frozen=True)
class NeedMeta:
    need_id: str
    grades: Mapping[str, int]
    query_ids: tuple[str, ...]


@dataclass(frozen=True)
class Analysis:
    classes: Mapping[str, str]
    comparisons: Mapping[Pair, Comparison]
    states: Mapping[Pair, str]  # coverage state of the labelled comparisons
    labels: Mapping[Pair, str]  # final label of the labelled comparisons
    actions: Mapping[str, str]  # H2b action against E5, after the coverage state
    n20_over_rrf: bool
    tie_break_state: str
    dominance_states: Mapping[Pair, str]
    dominance: Mapping[Pair, str]  # certified | unresolved | not_evaluable
    paths: Paths


def need_data(
    needs: Sequence[NeedMeta], lists: Lists, candidate: str, reference: str
) -> list[NeedData]:
    return [
        NeedData(
            n.need_id,
            n.grades,
            tuple(QueryLists(lists[candidate][q], lists[reference][q]) for q in n.query_ids),
        )
        for n in needs
    ]


def compare(
    candidate: str,
    reference: str,
    needs: Sequence[NeedMeta],
    lists: Lists,
    recall: PerNeed,
    ndcg: PerNeed,
    indices: Sequence[Sequence[int]],
) -> Comparison:
    """The comparison of a candidate against a reference. The primary differences come from the
    evaluator; they are checked against the oracle of this module, so that the envelope bounds
    the number that was reported."""
    data = need_data(needs, lists, candidate, reference)
    primary_recall = [recall[candidate][n.need_id] - recall[reference][n.need_id] for n in needs]
    primary_ndcg = [ndcg[candidate][n.need_id] - ndcg[reference][n.need_id] for n in needs]
    for need, r, g in zip(data, primary_recall, primary_ndcg, strict=True):
        if abs(r - recall_delta(need)) > PRIMARY_CHECK or abs(g - ndcg_delta(need)) > PRIMARY_CHECK:
            raise ValueError(f"{candidate} against {reference}, need {need.need_id}: the evaluator")
    return build_comparison(
        candidate,
        reference,
        primary_recall,
        primary_ndcg,
        [recall_range(d) for d in data],
        [ndcg_range(d) for d in data],
        indices,
    )


def analyse(
    needs: Sequence[NeedMeta],
    lists: Lists,
    recall: PerNeed,
    ndcg: PerNeed,
    costs: Mapping[str, CostRow],
) -> Analysis:
    """Every comparison of the preregistration, the three coverage scenarios of each, the labels,
    the H2b actions, the dominance tests and the two paths. needs ordered by need id."""
    if [n.need_id for n in needs] != sorted(n.need_id for n in needs):
        raise ValueError("needs must be ordered by need id")
    indices = resample_indices(len(needs))
    classes = {s: latency_class(costs[s].p95_seconds) for s in SYSTEMS}
    dominance_pairs_ = [p for p in dominance_pairs(costs) if p not in {("E5", "E")}]
    wanted: list[Pair] = []
    for pair in [
        *((s, "E5") for s in VS_E5),
        *((s, "E") for s in VS_E),
        *dominance_pairs_,
        TIE_BREAK,
    ]:
        if pair not in wanted:
            wanted.append(pair)
    comparisons = {(c, r): compare(c, r, needs, lists, recall, ndcg, indices) for c, r in wanted}

    def h2b(system: str) -> Callable[[Scenario], str]:
        return lambda sc: fast_action(classes[system], sc)

    def constant(_: Scenario) -> str:
        return "descriptive"

    states: dict[Pair, str] = {}
    labels: dict[Pair, str] = {}
    for pair in [*((s, "E5") for s in VS_E5), *((s, "E") for s in VS_E)]:
        candidate, reference = pair
        action = h2b(candidate) if reference == "E5" and candidate in CANDIDATES else constant
        states[pair] = coverage_state(comparisons[pair], action)
        labels[pair] = final_label(comparisons[pair], states[pair])
    actions = {
        s: FRONTIER
        if states[(s, "E5")] == NOT_EVALUABLE
        else fast_action(classes[s], comparisons[(s, "E5")].primary)
        for s in CANDIDATES
    }
    tie_state = coverage_state(
        comparisons[TIE_BREAK], lambda sc: "yes" if prefers_candidate(sc) else "no"
    )
    n20_over_rrf = tie_state != NOT_EVALUABLE and prefers_candidate(comparisons[TIE_BREAK].primary)
    dominance_states: dict[Pair, str] = {}
    dominance: dict[Pair, str] = {}
    for pair in dominance_pairs_:
        dominance_states[pair] = coverage_state(comparisons[pair], dominance_quality)
        if dominance_states[pair] == NOT_EVALUABLE:
            dominance[pair] = NOT_EVALUABLE
        else:
            dominance[pair] = dominance_quality(comparisons[pair].primary)
    paths = assign_paths(
        actions, n20_over_rrf, {pair: outcome == "certified" for pair, outcome in dominance.items()}
    )
    return Analysis(
        classes,
        comparisons,
        states,
        labels,
        actions,
        n20_over_rrf,
        tie_state,
        dominance_states,
        dominance,
        paths,
    )
