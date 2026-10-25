"""TEST v2's evaluation (D-067, config/holdout/s4-analysis-plan.v2.md): the frozen qrels against
the frozen runs, once the qrels are frozen and never before.

The primary comparison is S4's, unchanged: Recall@10 (relevant at grade >= 2) and nDCG@10 (the
grades as linear gains), co-primary, per query by the project's evaluator (ingest/ir_metrics.py)
on the chunk qrels projected from the final node labels (ingest/evaluation.py project: a chunk's
grade is the highest grade it owns; an ambiguous or unresolved label counts as unjudged, D-019);
a need's value is the mean of its three formulations; the paired difference E minus E5 has S4's
interval (ingest/run_report.py paired_bootstrap: needs resampled, 10,000 times, seed 20261013,
the 95% percentile interval); a query missing from a run scores 0. The verdict of each metric
is holds, inconclusive or reverses, and the claim is the one S4's table fixes.

Everything else is fixed here before any TEST result and none of it can become the claim:
- secondary comparisons and strata (S4 section 4), exploratory;
- the sensitivity analyses SA1 to SA8;
- the pool diagnostics 1 to 10, Judged@10 and Judged@20 among them;
- the construction record (attempted, duplicates, technical failures, not evaluable, N).

Nothing here changes a qrel, a need or the sample: every analysis reads the frozen artifacts and
writes only its own numbers.
"""

from __future__ import annotations

import dataclasses
import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ingest import benchmark as bm
from ingest import evaluation as ev
from ingest import holdout_v2 as v2
from ingest import ir_metrics as irm
from ingest import retrieval as rt
from ingest import run_report as rr

SCHEMA = "holdout-v2-evaluation.v1"
SEED = rr.SEED  # 20261013, S4's
REPS = 10_000
PRIMARY = ("Recall@10", "nDCG@10")
SECONDARY = ("Recall@5", "Recall@20", "MRR@10")
EVALUATED = ("E", "E5", "BM25")  # BGE-M3 and Qwen3-Embedding-0.6B contribute to the pool only
SECONDARY_PAIRS = (("BM25", "E5"), ("E", "BM25"))
CUTOFF = 10
JUDGED_DEPTHS = (10, 20)
DISCOVERY_DEPTHS = (1, 2, 3, 5, 10, 15, 20)
SMALL_STRATUM = 5  # fewer needs than this: the stratum is descriptive only
# SA5: the alternative gains of nDCG, applied to the projected chunk grades
GAINS = {"grade1_zero": {0: 0, 1: 0, 2: 2, 3: 3}, "exponential": {0: 0, 1: 1, 2: 3, 3: 7}}
# SA3: Astra's fixed-scale discounted gain: DCG@10 with the grades as linear gains, divided by
# the DCG@10 of ten chunks of the highest grade, so that nothing outside a top 10 moves it
G_MAX_GRADE = 3
# SA4 (b): relevance outside the pool, fixed in advance (the completeness diagnostic's own
# positives enter SA2 only): (name, relevant chunks each need lacks, their grade)
RESIDUAL_SCENARIOS = (("one_grade_2", 1, 2), ("three_grade_3", 3, 3))

CLAIMS = {
    ("holds", "holds"): "E improves retrieval over E5.",
    ("holds", "inconclusive"): (
        "E improves evidence coverage, but we do not have evidence of a general ranking "
        "improvement."
    ),
    ("inconclusive", "holds"): (
        "E improves ranking quality, but we do not have evidence that it covers more of the "
        "evidence."
    ),
    ("inconclusive", "inconclusive"): "We do not have evidence that E improves on E5 on TEST.",
}


class EvaluationRefused(Exception):
    """The evaluation cannot run on what it was given: nothing was computed."""


# ---------------------------------------------------------------------------
# the frozen inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Frozen:
    """What the evaluation reads, every piece of it frozen (holdout_v2_runner evaluation)."""

    primary: tuple[str, ...]  # the evaluable needs of the qrel freeze, in priority order
    queries: tuple[bm.Query, ...]  # their formulations
    qrels: Mapping[str, Mapping[str, Mapping[str, Any]]]  # need -> node -> final label
    rankings: Mapping[str, Mapping[str, Sequence[str]]]  # system -> query id -> chunk ids
    pools: Mapping[str, Mapping[str, Any]]  # need -> its frozen pool record
    layer: ev.ChunkLayer
    topics: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)  # need -> topic
    completeness: Mapping[str, Any] = field(default_factory=dict)
    completeness_finals: Mapping[str, Mapping[str, Mapping[str, Any]]] = field(default_factory=dict)
    answerability: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    construction: Mapping[str, Any] = field(default_factory=dict)
    human: Mapping[str, Any] | None = None  # SA7: the optional human audit's labels

    def queries_of(self, need: str) -> list[bm.Query]:
        return [q for q in self.queries if q.need_id == need]


def frozen_inputs(inputs: Mapping[str, Any], layer: ev.ChunkLayer) -> Frozen:
    """The runner's frozen artifacts as the evaluation reads them. Refused unless the qrels are
    the frozen manifest (holdout_v2_runner QRELS_SCHEMA): no metric exists before the freeze."""
    qrels = inputs.get("qrels")
    if not isinstance(qrels, Mapping) or qrels.get("schema") != "holdout-v2-qrels.v1":
        raise EvaluationRefused("the qrels are not frozen: no E-vs-E5 metric exists before")
    primary = tuple(qrels["primary"])
    if len(set(primary)) != len(primary) or set(primary) != set(qrels["qrels"]):
        raise EvaluationRefused("the frozen qrels do not hold exactly the primary needs")
    topics = {t["need_id"]: t for t in inputs["topics"]["topics"]}
    queries = tuple(bm.Query.model_validate(q) for n in primary for q in topics[n]["queries"])
    completeness = inputs.get("completeness") or {}
    return Frozen(
        primary,
        queries,
        qrels["qrels"],
        inputs["rankings"],
        inputs["pools"]["pools"],
        layer,
        topics,
        completeness,
        inputs.get("completeness_finals") or {},
        inputs.get("answerability") or {},
        inputs.get("metadata") or {},
        {k: qrels[k] for k in ("judged_unused", "not_evaluable", "technical", "not_judged")}
        | {
            "topics": len(topics),
            "dropped": list(inputs["topics"].get("dropped", [])),
            "deduplication": dict(inputs["topics"].get("deduplication", {})),
            "generation": inputs["topics"].get("generation"),
            "not_completed": list(inputs.get("not_completed", [])),
        },
        inputs.get("human"),
    )


# ---------------------------------------------------------------------------
# qrels, scores and the paired comparison
# ---------------------------------------------------------------------------


def _key(node: str) -> bm.NodeKey:
    act, _, path = node.partition(":")
    return act, path


def node_grades(
    record: Mapping[str, Mapping[str, Any]], ambiguous_graded: bool = False
) -> dict[bm.NodeKey, int]:
    """node -> grade the evaluation reads: every final label with a grade, explicit 0s
    included; an ambiguous one only when ambiguous_graded (SA1), otherwise it is unjudged."""
    out = {}
    for node, f in record.items():
        if f.get("grade") is None or (f.get("ambiguous") and not ambiguous_graded):
            continue
        out[_key(node)] = int(f["grade"])
    return out


def chunk_qrels(
    grades: Mapping[bm.NodeKey, int],
    layer: ev.ChunkLayer,
    gains: Mapping[int, int] | None = None,
    drop: Iterable[str] = (),
) -> dict[str, int]:
    """The node grades projected onto the layer (ingest/evaluation.py project), with the chunks
    in drop unjudged and the grades mapped by gains (SA5)."""
    projected = ev.project(grades, layer)
    dropped = set(drop)
    return {c: (gains[g] if gains else g) for c, g in projected.items() if c not in dropped}


def relevant_chunks(chunks: Mapping[str, int]) -> set[str]:
    return {c for c, g in chunks.items() if g >= bm.RELEVANT_GRADE}


def scores(
    chunk_qrels_of: Mapping[str, Mapping[str, int]],
    f: Frozen,
    systems: Sequence[str] = EVALUATED,
) -> dict[str, dict[str, dict[str, float]]]:
    """system -> query id -> metric, for every query of the needs given, by the project's
    evaluator: a query a system did not rank scores 0 and still counts."""
    qrels = {
        q.query_id: dict(chunk_qrels_of[q.need_id])
        for q in f.queries
        if q.need_id in chunk_qrels_of
    }
    out = {}
    for s in systems:
        run = {qid: list(f.rankings.get(s, {}).get(qid, ())) for qid in qrels}
        out[s] = irm.standard_scores(qrels, {q: r for q, r in run.items() if r})
    return out


def verdict(low: float, high: float) -> str:
    if low > 0:
        return "holds"
    if high < 0:
        return "reverses"
    return "inconclusive"


def need_means(values: Mapping[str, float], queries: Sequence[bm.Query]) -> dict[str, float]:
    by_need: dict[str, list[float]] = defaultdict(list)
    for q in queries:
        by_need[q.need_id].append(values[q.query_id])
    return {n: statistics.fmean(v) for n, v in sorted(by_need.items())}


def compare(
    a: Mapping[str, Mapping[str, float]],
    b: Mapping[str, Mapping[str, float]],
    metric: str,
    queries: Sequence[bm.Query],
) -> dict[str, Any]:
    """S4's paired comparison of one metric, a minus b, over the needs of the queries given."""
    qs = [q for q in queries if q.query_id in a]
    if not qs:
        return {"needs": 0}
    deltas = {q.query_id: a[q.query_id][metric] - b[q.query_id][metric] for q in qs}
    boot = rr.paired_bootstrap(deltas, qs, REPS)
    per_need = need_means(deltas, qs)
    sd = statistics.stdev(per_need.values()) if len(per_need) > 1 else 0.0
    return {
        "needs": len(per_need),
        "mean_a": rt.macro_by_need({q: a[q][metric] for q in deltas}, qs),
        "mean_b": rt.macro_by_need({q: b[q][metric] for q in deltas}, qs),
        "delta": boot["mean"],
        "ci95_low": boot["ci95_low"],
        "ci95_high": boot["ci95_high"],
        "verdict": verdict(boot["ci95_low"], boot["ci95_high"]),
        "d_z": boot["mean"] / sd if sd > 0 else None,
        "sd_delta": sd,
        "needs_better": sum(d > 1e-9 for d in per_need.values()),
        "needs_equal": sum(abs(d) <= 1e-9 for d in per_need.values()),
        "needs_worse": sum(d < -1e-9 for d in per_need.values()),
    }


def claim(recall: str, ndcg: str) -> str:
    """S4's table: what may be said, fixed before any result."""
    reversed_ = [m for m, v in zip(PRIMARY, (recall, ndcg), strict=True) if v == "reverses"]
    if reversed_:
        other = {PRIMARY[0]: ndcg, PRIMARY[1]: recall}
        said = "; ".join(f"{m}: the other metric is {other[m]}" for m in reversed_)
        return f"E is worse than E5 on {' and '.join(reversed_)} ({said})."
    return CLAIMS[(recall, ndcg)]


def primary_block(
    s: Mapping[str, Mapping[str, Mapping[str, float]]], queries: Sequence[bm.Query]
) -> dict[str, Any]:
    out: dict[str, Any] = {m: compare(s["E"], s["E5"], m, queries) for m in PRIMARY}
    if all(out[m].get("needs") for m in PRIMARY):
        out["claim"] = claim(out[PRIMARY[0]]["verdict"], out[PRIMARY[1]]["verdict"])
    return out


def bootstrap_mean(values: Mapping[str, float], reps: int = REPS) -> dict[str, float]:
    """S4's interval for a mean over needs of one value each (rr.paired_bootstrap with one
    query per need): needs sorted, resampled by random.Random(SEED).choices."""
    needs = sorted(values)
    if not needs:
        return {"mean": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
    rng = random.Random(SEED)
    means = sorted(
        statistics.fmean(values[n] for n in rng.choices(needs, k=len(needs))) for _ in range(reps)
    )
    return {
        "mean": statistics.fmean(values[n] for n in needs),
        "ci95_low": means[int(0.025 * reps)],
        "ci95_high": means[int(0.975 * reps) - 1],
    }


# ---------------------------------------------------------------------------
# the pieces every analysis reads
# ---------------------------------------------------------------------------


def primary_qrels(f: Frozen, **kwargs: Any) -> dict[str, dict[str, int]]:
    return {n: chunk_qrels(node_grades(f.qrels[n]), f.layer, **kwargs) for n in f.primary}


def top(f: Frozen, system: str, qid: str, k: int) -> list[str]:
    return list(f.rankings.get(system, {}).get(qid, ()))[:k]


def need_top(f: Frozen, system: str, need: str, k: int) -> set[str]:
    return {c for q in f.queries_of(need) for c in top(f, system, q.query_id, k)}


def contributors(pool: Mapping[str, Any]) -> dict[str, set[str]]:
    """chunk -> the contributors that brought it into the pool (over every formulation)."""
    return {c: {s[0] for s in sources} for c, sources in pool["sources"].items()}


# ---------------------------------------------------------------------------
# the analyses
# ---------------------------------------------------------------------------


def secondary(f: Frozen, s: Mapping[str, Any]) -> dict[str, Any]:
    """S4 section 4, exploratory: the other pairs, the other cutoffs and the strata."""
    out: dict[str, Any] = {
        "pairs": {
            f"{a}-{b}": {m: compare(s[a], s[b], m, f.queries) for m in PRIMARY}
            for a, b in SECONDARY_PAIRS
            if a in s and b in s
        },
        "cutoffs": {m: compare(s["E"], s["E5"], m, f.queries) for m in SECONDARY},
    }
    strata: dict[str, dict[str, list[bm.Query]]] = defaultdict(lambda: defaultdict(list))
    for q in f.queries:
        strata["formulation"][f"{q.lang}-{q.style}"].append(q)
    for n in f.primary:
        qs = f.queries_of(n)
        labels = {
            "type": [str(f.metadata.get(n, {}).get("type", "unknown"))],
            "answerability": [str(f.answerability.get(n, {}).get("answerability", "unknown"))],
            "difficulty": [str(f.metadata.get(n, {}).get("difficulty", "unknown"))],
            "act": sorted({c.split(":")[0] for c in relevant_nodes(f.qrels[n])}),
            "profile_acts": ["+".join(f.topics.get(n, {}).get("slot", {}).get("acts", ()))],
        }
        for name, values in labels.items():
            for value in values:
                strata[name][value].extend(qs)
    out["strata"] = {
        name: {
            value: {
                "descriptive_only": len({q.need_id for q in qs}) < SMALL_STRATUM,
                **{m: compare(s["E"], s["E5"], m, qs) for m in PRIMARY},
            }
            for value, qs in sorted(groups.items())
        }
        for name, groups in sorted(strata.items())
    }
    return out


def relevant_nodes(record: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        n
        for n, f in record.items()
        if f.get("grade") is not None and f["grade"] >= bm.RELEVANT_GRADE and not f.get("ambiguous")
    ]


def sa1_ambiguity(f: Frozen) -> dict[str, Any]:
    """SA1: the final ambiguous labels counted at their grade instead of as unjudged, and how
    many there are, how many sit in a compared system's top 10, and in the decisive set."""
    counts: Counter[str] = Counter()
    for n in f.primary:
        pool = f.pools[n]
        decisive = set(pool["decisive"])
        e_top = need_top(f, "E", n, CUTOFF)
        e5_top = need_top(f, "E5", n, CUTOFF)
        for node, label in f.qrels[n].items():
            if not label.get("ambiguous"):
                continue
            chunk = f.layer.owner[_key(node)]
            counts["ambiguous_nodes"] += 1
            counts["ambiguous_with_grade_2_or_more"] += (label.get("grade") or 0) >= 2
            counts["in_E_top10"] += chunk in e_top
            counts["in_E5_top10"] += chunk in e5_top
            counts["in_decisive_set"] += chunk in decisive
        counts["needs_with_ambiguity"] += any(x.get("ambiguous") for x in f.qrels[n].values())
    graded = {
        n: chunk_qrels(node_grades(f.qrels[n], ambiguous_graded=True), f.layer) for n in f.primary
    }
    return {
        "treatment": "every final ambiguous label with a grade counted at that grade",
        "counts": dict(sorted(counts.items())),
        "primary": primary_block(scores(graded, f), f.queries),
    }


def sa2_augmented(
    f: Frozen, base: Mapping[str, Mapping[str, int]], s: Mapping[str, Any]
) -> dict[str, Any]:
    """SA2 and the completeness report: the 15 completeness needs with the verified residual
    positives of the chunk screen added (a final node label of grade >= 2, not ambiguous), the
    co-primary metrics recomputed on them and on the whole sample; and the diagnostic's own
    numbers. The primary qrels are never touched."""
    needs = f.completeness.get("needs") or {}
    if not needs:
        return {"status": "not_run"}
    selection = f.completeness.get("selection") or {}
    augmented = dict(base)
    report: dict[str, Any] = {}
    recall_of_pool: dict[str, float] = {}
    for n, record in sorted(needs.items()):
        finals = f.completeness_finals.get(n, {})
        extra = {
            _key(node): int(x["grade"])
            for node, x in finals.items()
            if x.get("grade") is not None
            and x["grade"] >= bm.RELEVANT_GRADE
            and not x.get("ambiguous")
        }
        grades = node_grades(f.qrels[n]) | {
            k: max(g, node_grades(f.qrels[n]).get(k, g)) for k, g in extra.items()
        }
        augmented[n] = chunk_qrels(grades, f.layer)
        inside = len(relevant_chunks(base[n]))
        residual = len(relevant_chunks(augmented[n]) - relevant_chunks(base[n]))
        recall_of_pool[n] = inside / (inside + residual) if inside + residual else math.nan
        stratum = next((a for a, ns in selection.items() if n in ns), None)
        report[n] = {
            "stratum": stratum,
            "screened_chunks": record.get("screened"),
            "flagged_chunks": record.get("flagged"),
            "verified_residual_chunks": residual,
            "relevant_chunks_in_pool": inside,
            "pool_recall": recall_of_pool[n],
            "incomplete": record.get("incomplete"),
            "residual_by_act": dict(
                sorted(
                    Counter(
                        f.layer.chunks[c].nodes[0][0]
                        for c in relevant_chunks(augmented[n]) - relevant_chunks(base[n])
                    ).items()
                )
            ),
        }
    chosen = sorted(needs)
    sample = [q for q in f.queries if q.need_id in set(chosen)]
    s_aug = scores({n: augmented[n] for n in chosen}, f)
    by_act: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    for r in report.values():
        by_act[str(r["stratum"])]["needs"] += 1
        by_act[str(r["stratum"])]["verified_residual_chunks"] += r["verified_residual_chunks"]
        by_act[str(r["stratum"])]["relevant_chunks_in_pool"] += r["relevant_chunks_in_pool"]
    valid = {n: v for n, v in recall_of_pool.items() if not math.isnan(v)}
    residuals = {n: float(r["verified_residual_chunks"]) for n, r in report.items()}
    return {
        "needs": report,
        "pool_recall": bootstrap_mean(valid),
        "residual_chunks_per_need": bootstrap_mean(residuals),
        "by_stratum": {a: dict(c) for a, c in sorted(by_act.items())},
        "on_the_completeness_needs": {
            "primary_qrels": {m: compare(s["E"], s["E5"], m, sample) for m in PRIMARY},
            "augmented_qrels": {m: compare(s_aug["E"], s_aug["E5"], m, sample) for m in PRIMARY},
        },
        "whole_sample_augmented": primary_block(scores(augmented, f), f.queries),
    }


def g_at_10(ranking: Sequence[str], chunks: Mapping[str, int]) -> float:
    """SA3: the fixed-scale discounted gain at 10."""
    dcg = sum(chunks.get(c, 0) / math.log2(i + 1) for i, c in enumerate(ranking[:CUTOFF], 1))
    ceiling = sum(G_MAX_GRADE / math.log2(i + 1) for i in range(1, CUTOFF + 1))
    return dcg / ceiling


def sa3_fixed_scale(f: Frozen, base: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    g = {
        s: {
            q.query_id: {"G@10": g_at_10(top(f, s, q.query_id, CUTOFF), base[q.need_id])}
            for q in f.queries
        }
        for s in ("E", "E5")
    }
    return {
        "definition": "DCG@10 with the grades as linear gains / the DCG@10 of ten grade-3 chunks",
        "G@10": compare(g["E"], g["E5"], "G@10", f.queries),
    }


def ndcg_at_10(
    ranking: Sequence[str], chunks: Mapping[str, int], extra: Sequence[float] = ()
) -> float:
    """trec_eval's nDCG@10 (gain = grade, discount log2(rank + 1)), with extra relevant grades
    the ideal ranking may hold beyond the judged ones (SA4 b)."""
    dcg = sum(chunks.get(c, 0) / math.log2(i + 1) for i, c in enumerate(ranking[:CUTOFF], 1))
    ideal = sorted([g for g in chunks.values() if g > 0] + list(extra), reverse=True)[:CUTOFF]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg if idcg > 0 else 0.0


def sa4_residual(
    f: Frozen, base: Mapping[str, Mapping[str, int]], s: Mapping[str, Any]
) -> dict[str, Any]:
    """SA4: how missing relevance could move the difference.
    (a) the chunks of E's or E5's top 10 that are unjudged (ambiguous, unresolved, or owning
    no node to judge): the lower bound makes E5's alone relevant (grade 3) and E's 0, the
    upper bound the reverse;
    (b) relevance outside the pool, in scenarios fixed in advance (RESIDUAL_SCENARIOS): such a
    chunk is in neither top 10 (the pool holds both top 20s), so it only enlarges the
    denominators of both systems: Recall@10 = hits / (R + r), nDCG@10 with r more relevant
    chunks of grade g in the ideal ranking. SA2 reports the residual the diagnostic measured."""
    unjudged: dict[str, dict[str, set[str]]] = {}
    positions: Counter[str] = Counter()
    for n in f.primary:
        unjudged[n] = {}
        for system in ("E", "E5"):
            found = set()
            for q in f.queries_of(n):
                for c in top(f, system, q.query_id, CUTOFF):
                    if c not in base[n]:
                        found.add(c)
                        positions[system] += 1
            unjudged[n][system] = found
    bounds = {}
    for name, favoured in (("lower", "E5"), ("upper", "E")):
        other = "E" if favoured == "E5" else "E5"
        variant = {
            n: dict(base[n])
            | {c: 3 for c in unjudged[n][favoured] - unjudged[n][other]}
            | {c: 0 for c in unjudged[n][other] | (unjudged[n][favoured] & unjudged[n][other])}
            for n in f.primary
        }
        bounds[name] = primary_block(scores(variant, f), f.queries)
    out: dict[str, Any] = {
        "unjudged_top10": {
            "positions": dict(positions),
            "needs_with_any": sum(bool(u["E"] | u["E5"]) for u in unjudged.values()),
            "bounds": bounds,
        }
    }
    scenarios: dict[str, Any] = {}
    for name, r, grade in RESIDUAL_SCENARIOS:
        values: dict[str, dict[str, dict[str, float]]] = {"E": {}, "E5": {}}
        for q in f.queries:
            chunks = base[q.need_id]
            rel = relevant_chunks(chunks)
            for system in ("E", "E5"):
                ranking = top(f, system, q.query_id, CUTOFF)
                hits = len(rel & set(ranking))
                values[system][q.query_id] = {
                    "Recall@10": hits / (len(rel) + r) if rel else 0.0,
                    "nDCG@10": ndcg_at_10(ranking, chunks, [grade] * r),
                }
        scenarios[name] = {
            "residual_chunks_per_need": r,
            "residual_grade": grade,
            **primary_block(values, f.queries),
        }
    out["residual_outside_pool"] = {"scenarios": scenarios}
    # the own nDCG must be the evaluator's where nothing is added
    for q in f.queries[:50]:
        own = ndcg_at_10(top(f, "E", q.query_id, CUTOFF), base[q.need_id])
        if abs(own - s["E"][q.query_id]["nDCG@10"]) > 1e-9:
            raise EvaluationRefused("SA4's nDCG is not the evaluator's")
    return out


def sa5_gains(f: Frozen) -> dict[str, Any]:
    """SA5: nDCG@10 with grade 1 as 0, and with exponential gains (2^grade - 1), through the
    same evaluator on the mapped chunk grades. Recall@10 does not change: it reads grade >= 2."""
    out = {}
    for name, gains in GAINS.items():
        mapped = primary_qrels(f, gains=gains)
        s = scores(mapped, f, ("E", "E5"))
        out[name] = {"gains": gains, "nDCG@10": compare(s["E"], s["E5"], "nDCG@10", f.queries)}
    return out


def sa6_inference(f: Frozen, s: Mapping[str, Any]) -> dict[str, Any]:
    """SA6: the paired t interval and the paired sign-flip permutation test of each co-primary
    metric, on the per-need deltas (the same unit as the bootstrap)."""
    from scipy import stats  # type: ignore[import-untyped]

    out = {}
    for m in PRIMARY:
        deltas = need_means(
            {q.query_id: s["E"][q.query_id][m] - s["E5"][q.query_id][m] for q in f.queries},
            f.queries,
        )
        d = list(deltas.values())
        n = len(d)
        mean = statistics.fmean(d)
        sd = statistics.stdev(d) if n > 1 else 0.0
        t = float(stats.t.ppf(0.975, n - 1)) if n > 1 else math.nan
        rng = random.Random(SEED)
        extreme = 0
        for _ in range(REPS):
            flipped = statistics.fmean(x if rng.random() < 0.5 else -x for x in d)
            extreme += abs(flipped) >= abs(mean) - 1e-12
        out[m] = {
            "needs": n,
            "mean": mean,
            "t_ci95_low": mean - t * sd / math.sqrt(n) if n > 1 else math.nan,
            "t_ci95_high": mean + t * sd / math.sqrt(n) if n > 1 else math.nan,
            "permutation_p_two_sided": (1 + extreme) / (1 + REPS),
            "permutations": REPS,
        }
    return out


def sa7_human(f: Frozen, base: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    """SA7: the optional, blinded human audit (ingest/holdout_v2_audit.py). Its labels never
    replace a primary label: they are compared with the final ones, and the co-primary
    differences are recomputed on the audited needs with them, as a sensitivity only."""
    if not f.human:
        return {"status": "not_run"}
    # (need, chunk) -> the person's grade of the passage, or None where unsure; the chunk is
    # the audit's unit, compared with the chunk grade the projection gives
    labels: Sequence[Mapping[str, Any]] = f.human["labels"]
    agree: Counter[str] = Counter()
    by_need: dict[str, dict[str, int]] = defaultdict(dict)
    for h in labels:
        need, chunk = h["need_id"], h["chunk_id"]
        final = base.get(need, {}).get(chunk)
        agree["items"] += 1
        agree["decoys"] += bool(h.get("decoy"))
        if h.get("grade") is None:
            agree["human_unsure"] += 1
            continue
        if final is None:
            agree["final_unjudged"] += 1
        else:
            agree["exact"] += final == int(h["grade"])
            agree["same_side_of_grade_2"] += (final >= 2) == (int(h["grade"]) >= 2)
            agree["compared"] += 1
        by_need[need][chunk] = int(h["grade"])
    needs = sorted(n for n in by_need if n in base)
    replaced = {n: dict(base[n]) | by_need[n] for n in needs}
    sample = [q for q in f.queries if q.need_id in set(needs)]
    s_h = scores(replaced, f, ("E", "E5"))
    s_b = scores({n: base[n] for n in needs}, f, ("E", "E5"))
    return {
        "status": "run",
        "agreement": dict(sorted(agree.items())),
        "on_the_audited_needs": {
            "primary_qrels": {m: compare(s_b["E"], s_b["E5"], m, sample) for m in PRIMARY},
            "human_labels": {m: compare(s_h["E"], s_h["E5"], m, sample) for m in PRIMARY},
        },
    }


def sa8_leave_one_out(f: Frozen, base: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    """SA8 (and pool diagnostic 8): for each contributor, the chunks only it brought into a
    pool made unjudged, the needs that stay evaluable kept, and the co-primary differences
    recomputed; with the relevant chunks each contributor alone brought."""
    out = {}
    names = [c for c, _ in v2.POOL] + [v2.ANCHOR_CONTRIBUTOR]
    for name in names:
        reduced, removed = {}, 0
        for n in f.primary:
            by = contributors(f.pools[n])
            only = {c for c, who in by.items() if who == {name}}
            removed += len(relevant_chunks(base[n]) & only)
            q = {c: g for c, g in base[n].items() if c not in only}
            if relevant_chunks(q):
                reduced[n] = q
        sample = [q for q in f.queries if q.need_id in reduced]
        s = scores(reduced, f, ("E", "E5"))
        out[name] = {
            "relevant_chunks_unique_to_it": removed,
            "needs_still_evaluable": len(reduced),
            **{m: compare(s["E"], s["E5"], m, sample) for m in PRIMARY},
        }
    return out


def judged_at(
    f: Frozen, base: Mapping[str, Mapping[str, int]], system: str, k: int
) -> dict[str, Any]:
    values = {}
    for q in f.queries:
        ranking = top(f, system, q.query_id, k)
        values[q.query_id] = sum(c in base[q.need_id] for c in ranking) / k if ranking else 0.0
    per_need = need_means(values, f.queries)
    return {
        "mean": statistics.fmean(per_need.values()) if per_need else math.nan,
        "needs_below_1": sum(v < 1 - 1e-12 for v in per_need.values()),
    }


def distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    xs = sorted(values)
    q = statistics.quantiles(xs, n=4) if len(xs) > 1 else [xs[0]] * 3
    return {
        "n": len(xs),
        "min": xs[0],
        "q1": q[0],
        "median": statistics.median(xs),
        "q3": q[2],
        "max": xs[-1],
        "mean": statistics.fmean(xs),
    }


def pool_diagnostics(f: Frozen, base: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    """The pool diagnostics 1 to 10 of the protocol. Diagnostics: none replaces a topic."""
    out: dict[str, Any] = {}
    for k in JUDGED_DEPTHS:
        for system in v2.COMPARED:
            out[f"Judged@{k} {system}"] = judged_at(f, base, system, k)
    unique: Counter[str] = Counter()
    discovery: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    chunks, nodes, decisive, decisive_nodes = [], [], [], []
    decisive_judged: Counter[str] = Counter()
    only_compared: Counter[str] = Counter()
    for n in f.primary:
        pool = f.pools[n]
        by = contributors(pool)
        rel = relevant_chunks(base[n])
        for c in rel:
            who = by.get(c, set())
            if len(who) == 1:
                unique[next(iter(who))] += 1
        sources = pool["sources"]
        for name, depth in v2.POOL:
            for d in DISCOVERY_DEPTHS:
                if d > depth:
                    continue
                found = {
                    c
                    for c, srcs in sources.items()
                    if any(s[0] == name and 0 < s[2] <= d for s in srcs)
                }
                discovery[name][d].append(len(found & rel) / len(rel))
        for d in DISCOVERY_DEPTHS:
            found = {c for c, srcs in sources.items() if any(0 < s[2] <= d for s in srcs)}
            discovery["union"][d].append(len(found & rel) / len(rel))
        chunks.append(len(pool["chunks"]))
        nodes.append(sum(len(v) for v in pool["judged"].values()))
        decisive.append(len(pool["decisive"]))
        dn = [x for c in pool["decisive"] for x in pool["judged"].get(c, ())]
        decisive_nodes.append(len(dn))
        for x in dn:
            label = f.qrels[n].get(x, {})
            decisive_judged["nodes"] += 1
            decisive_judged["with_b2"] += label.get("b2") is not None
            decisive_judged["final_resolved_unambiguous"] += label.get(
                "grade"
            ) is not None and not label.get("ambiguous")
        for system in v2.COMPARED:
            if rel and all(by.get(c, set()) == {system} for c in rel):
                only_compared[system] += 1
    out["relevant_chunks_unique_to"] = dict(sorted(unique.items()))
    out["cumulative_relevant_discovery"] = {
        name: {str(d): statistics.fmean(v) for d, v in sorted(by_depth.items())}
        for name, by_depth in sorted(discovery.items())
    }
    out["pool_size"] = {"chunks": distribution(chunks), "nodes": distribution(nodes)}
    out["leave_one_contributor_out"] = "sensitivity SA8"
    out["decisive_set"] = {
        "chunks": distribution(decisive),
        "nodes": distribution(decisive_nodes),
        **dict(decisive_judged),
    }
    out["needs_whose_relevant_evidence_only_one_compared_system_brought"] = {
        s: {"needs": only_compared[s], "share": only_compared[s] / len(f.primary)}
        for s in v2.COMPARED
    }
    return out


def descriptive(f: Frozen) -> dict[str, Any]:
    """The construction record (plan, section 6) and the descriptive labels: no metric."""
    c = f.construction
    dedup = c.get("deduplication", {})
    return {
        "construction": {
            "n": len(f.primary),
            "attempted_candidates": len(dedup) + len(c.get("dropped", [])),
            "topics": c.get("topics"),
            "dropped_before_deduplication": dict(
                sorted(Counter(f"{d['stage']}:{d['reason']}" for d in c.get("dropped", [])).items())
            ),
            "duplicates_by_kind": dict(
                sorted(
                    Counter(
                        "technical_exhausted"
                        if "technical_exhausted" in d.get("verdicts", [])
                        else str(d.get("closest_kind"))
                        for d in dedup.values()
                        if not d.get("kept")
                    ).items()
                )
            ),
            "judged_unused": len(c.get("judged_unused", [])),
            "not_evaluable": len(c.get("not_evaluable", [])),
            "technical_failures": len(c.get("technical", [])),
            "not_judged": len(c.get("not_judged", [])),
            "systems_not_completed": list(c.get("not_completed", [])),
            "anchor_leakage": c.get("generation"),
        },
        "answerability": dict(
            sorted(Counter(str(a.get("answerability")) for a in f.answerability.values()).items())
        ),
        "answerability_paths": dict(
            sorted(Counter(str(a.get("path")) for a in f.answerability.values()).items())
        ),
        "type": dict(sorted(Counter(str(m.get("type")) for m in f.metadata.values()).items())),
        "difficulty": dict(
            sorted(Counter(str(m.get("difficulty")) for m in f.metadata.values()).items())
        ),
    }


def evaluate(f: Frozen) -> dict[str, Any]:
    """The whole evaluation of the plan: primary first, then everything that cannot change it.
    If E or E5 did not complete its run, the primary comparison is reported as not made (S4),
    and nothing stands in for it."""
    base = primary_qrels(f)
    for n in f.primary:
        if not relevant_chunks(base[n]):
            raise EvaluationRefused(f"{n}: a primary need with no relevant chunk")
    missing = [system for system in EVALUATED if system not in f.rankings]
    if any(system in missing for system in v2.COMPARED):
        return {
            "schema": SCHEMA,
            "n": len(f.primary),
            "primary": {"status": "not_made", "systems_not_completed": missing},
            "descriptive": descriptive(f),
        }
    s = scores(base, f, [system for system in EVALUATED if system not in missing])
    completeness = sa2_augmented(f, base, s)
    return {
        "schema": SCHEMA,
        "evaluator": dataclasses.asdict(irm.backend()),
        "n": len(f.primary),
        "primary": primary_block(s, f.queries),
        "secondary": secondary(f, s),
        "sensitivity": {
            "SA1_ambiguity": sa1_ambiguity(f),
            "SA2_augmented_qrels": completeness,
            "SA3_fixed_scale_gain": sa3_fixed_scale(f, base),
            "SA4_residual_relevance": sa4_residual(f, base, s),
            "SA5_gains": sa5_gains(f),
            "SA6_inference": sa6_inference(f, s),
            "SA7_human_audit": sa7_human(f, base),
            "SA8_leave_one_contributor_out": sa8_leave_one_out(f, base),
        },
        "pool_diagnostics": pool_diagnostics(f, base),
        "descriptive": descriptive(f),
    }
