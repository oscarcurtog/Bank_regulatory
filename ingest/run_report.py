"""The breakdowns of one run on the benchmark, the same for every retriever.

scripts/embed_baseline.py computed them inside its own step 5 for the dense models; they
live here so that BM25, the fused runs and the expanded runs are measured by exactly the
same code. Every number comes from the official evaluator (ingest/evaluation.py): the
official aggregation per language (D-020), then descriptive breakdowns by query style,
need type, difficulty, answerability and driver (a group under 5 needs is descriptive
only), the Spanish-against-English effect, a table per need, the scores per query with
the rank of every relevant chunk, and the queries with the lowest nDCG@10.

`diagnostics` adds what the retrieval iteration compares (config/retrieval/README.md):
Recall@50 and Recall@100 from the same evaluator, the group recalls, and the recall of
the dependent evidence.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from ingest import benchmark as bm
from ingest import evaluation as ev
from ingest import ir_metrics
from ingest import retrieval as rt

METRICS = [
    "nDCG@10",
    "Recall@10",
    "MRR@10",
    "Recall@5",
    "Recall@20",
    "Recall@T=1000",
    "Recall@T=2000",
]
STYLES = [("en", "formal"), ("en", "natural"), ("es", "natural")]
SMALL_GROUP = 5
LOW_NDCG = 0.5
SEED = 20261013
DEEP_CUTOFFS = (50, 100)
GROUPS = {
    "multi_provision": ("type", "multi_provision"),
    "cross_reference": ("driver", "cross_reference"),
    "cross_act": ("driver", "cross_act"),
    "lexical_twin": ("driver", "lexical_twin"),
}


def mean_of(rows: Sequence[Mapping[str, float]], metric: str) -> float:
    return statistics.fmean(r[metric] for r in rows)


def group_means(
    per_need: Mapping[str, Mapping[str, float]], keys: Mapping[str, Sequence[str]]
) -> dict[str, dict[str, Any]]:
    """Macro means over needs for each group a need belongs to (a need may be in several)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for need_id, labels in keys.items():
        for label in labels:
            groups[label].append(need_id)
    out = {}
    for label, ids in sorted(groups.items()):
        rows = [per_need[i] for i in ids]
        out[label] = {"needs": len(ids), "descriptive_only": len(ids) < SMALL_GROUP} | {
            m: mean_of(rows, m) for m in METRICS
        }
    return out


def paired(diffs: Sequence[float]) -> dict[str, float]:
    rng = random.Random(SEED)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(10_000))
    return {
        "mean": statistics.fmean(diffs),
        "ci95_low": means[249],
        "ci95_high": means[9749],
        "es_better": sum(d > 0 for d in diffs),
        "en_better": sum(d < 0 for d in diffs),
        "equal": sum(d == 0 for d in diffs),
    }


def evaluate_ranked(
    dev: bm.Benchmark,
    layer: ev.ChunkLayer,
    ranked: Sequence[rt.Ranked],
    bodies: Mapping[str, str],
    truncated: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """metrics, per_need, per_query and failures of a run that ranks every query.

    ranked holds every query of the benchmark; only the queries of evaluable needs are
    scored. bodies is each chunk's own text, for the previews of the failures. truncated
    names the chunks a dense model saw cut, to flag them (empty for any other retriever)."""
    evaluable = {q.query_id for t in ev.targets(dev, layer) for q in t.queries}
    run = rt.to_run([r for r in ranked if r.query_id in evaluable])
    scores, missing = ev.score_queries(dev, layer, run)
    if missing:
        raise rt.RetrievalError(f"queries absent from the run: {missing}")
    report = ev.aggregate(scores, [s.query.query_id for s in scores])
    by_query = {s.query.query_id: s for s in scores}
    targets = {t.need.need_id: t for t in ev.targets(dev, layer)}
    rank_of = {r.query_id: {c: i + 1 for i, c in enumerate(r.chunk_ids)} for r in ranked}
    score_of = {r.query_id: dict(zip(r.chunk_ids, r.scores, strict=True)) for r in ranked}
    ranked_of = {r.query_id: r for r in ranked}

    def first_relevant(query_id: str, need_id: str) -> int:
        rel = [c for c, g in targets[need_id].chunk_grades.items() if g >= bm.RELEVANT_GRADE]
        return min(rank_of[query_id][c] for c in rel)

    per_need: dict[str, dict[str, float]] = {}
    per_need_style: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for need_id, t in targets.items():
        rows = [by_query[q.query_id].values for q in t.queries]
        per_need[need_id] = {m: mean_of(rows, m) for m in METRICS}
        for q in t.queries:
            per_need_style[need_id][f"{q.lang} {q.style}"] = dict(by_query[q.query_id].values) | {
                "first_relevant_rank": first_relevant(q.query_id, need_id),
                "query_id": q.query_id,
            }
    styles = {}
    for lang, style in STYLES:
        rows = [per_need_style[n][f"{lang} {style}"] for n in per_need]
        styles[f"{lang} {style}"] = {"needs": len(rows)} | {m: mean_of(rows, m) for m in METRICS}
    official = {
        lang: {"needs": s.needs, "queries": s.queries} | {m: s.means[m] for m in METRICS}
        for lang, s in report.by_language.items()
    }
    macro = {"needs": len(per_need), "queries": len(scores)} | {
        m: mean_of(list(per_need.values()), m) for m in METRICS
    }

    needs = {n.need_id: n for n in dev.needs}
    by_type = group_means(per_need, {n: [needs[n].type] for n in per_need})
    by_difficulty = group_means(per_need, {n: [needs[n].difficulty] for n in per_need})
    by_answerability = group_means(per_need, {n: [needs[n].answerability] for n in per_need})
    by_driver = group_means(per_need, {n: list(needs[n].drivers) or ["(none)"] for n in per_need})
    language = {
        metric: {
            "es_natural_minus_en_natural": paired(
                [
                    per_need_style[n]["es natural"][metric]
                    - per_need_style[n]["en natural"][metric]
                    for n in per_need
                ]
            ),
            "es_natural_minus_en_formal": paired(
                [
                    per_need_style[n]["es natural"][metric] - per_need_style[n]["en formal"][metric]
                    for n in per_need
                ]
            ),
        }
        for metric in ("nDCG@10", "Recall@10")
    }

    rel_chunks = {
        n: sum(g >= bm.RELEVANT_GRADE for g in t.chunk_grades.values()) for n, t in targets.items()
    }
    need_table = []
    for need_id in sorted(per_need):
        n = needs[need_id]
        firsts = {k: v["first_relevant_rank"] for k, v in per_need_style[need_id].items()}
        ceiling = min(1.0, 5 / rel_chunks[need_id])
        flags = []
        if any(per_need_style[need_id][k]["Recall@10"] < 1 for k in per_need_style[need_id]):
            flags.append("Recall@10<1")
        if any(r > 10 for r in firsts.values()):
            flags.append("first relevant >10")
        if per_need[need_id]["nDCG@10"] < LOW_NDCG:
            flags.append("low nDCG@10")
        es, en = (
            per_need_style[need_id]["es natural"]["nDCG@10"],
            per_need_style[need_id]["en natural"]["nDCG@10"],
        )
        if es < en - 0.2:
            flags.append("ES much worse")
        rel = {c for c, g in targets[need_id].chunk_grades.items() if g >= bm.RELEVANT_GRADE}
        if rel & truncated:
            flags.append("truncation-exposed")
        need_table.append(
            {
                "need": need_id,
                "type": n.type,
                "difficulty": n.difficulty,
                "answerability": n.answerability,
                "drivers": list(n.drivers),
                "relevant_chunks": rel_chunks[need_id],
                "recall5_ceiling": ceiling,
                "recall5_over_ceiling": per_need[need_id]["Recall@5"] / ceiling,
            }
            | {m: per_need[need_id][m] for m in METRICS}
            | {"first_relevant_rank": firsts, "flags": flags}
        )

    # failures: the queries with the lowest nDCG@10, then the latest first relevant chunk
    ordered = sorted(
        scores,
        key=lambda s: (
            s.values["nDCG@10"],
            -first_relevant(s.query.query_id, s.query.need_id),
            s.query.query_id,
        ),
    )
    failures = []
    for s in ordered[:8]:
        q, t = s.query, targets[s.query.need_id]
        top = []
        for c in ranked_of[q.query_id].chunk_ids[:10]:
            grade = t.chunk_grades.get(c)
            text = bodies[c].replace("\n", " ")
            top.append(
                {
                    "rank": rank_of[q.query_id][c],
                    "chunk": c,
                    "score": score_of[q.query_id][c],
                    "grade": grade,
                    "relevant": grade is not None and grade >= bm.RELEVANT_GRADE,
                    "truncated": c in truncated,
                    "preview": text[:160],
                }
            )
        rel_ranked = sorted(
            (c for c, g in t.chunk_grades.items() if g >= bm.RELEVANT_GRADE),
            key=lambda c: rank_of[q.query_id][c],
        )
        failures.append(
            {
                "query_id": q.query_id,
                "need": q.need_id,
                "lang": q.lang,
                "style": q.style,
                "text": q.text if q.lang == "en" else None,
                "values": dict(s.values),
                "first_relevant_rank": first_relevant(q.query_id, q.need_id),
                "top10": top,
                "relevant_chunks": [
                    {
                        "chunk": c,
                        "rank": rank_of[q.query_id][c],
                        "score": score_of[q.query_id][c],
                        "grade": t.chunk_grades[c],
                        "truncated": c in truncated,
                    }
                    for c in rel_ranked
                ],
                "truncation_exposed": bool(set(rel_ranked) & truncated)
                or any(x["truncated"] for x in top),
            }
        )

    return {
        "metrics": {
            "official_by_language": official,
            "by_style": styles,
            "macro_by_need": macro,
            "by_type": by_type,
            "by_difficulty": by_difficulty,
            "by_answerability": by_answerability,
            "by_driver": by_driver,
            "language_effect": language,
        },
        "per_need": need_table,
        "per_query": {
            s.query.query_id: dict(s.values)
            | {
                "first_relevant_rank": first_relevant(s.query.query_id, s.query.need_id),
                "relevant_ranks": {
                    c: rank_of[s.query.query_id][c]
                    for c, g in sorted(targets[s.query.need_id].chunk_grades.items())
                    if g >= bm.RELEVANT_GRADE
                },
            }
            for s in scores
        },
        "failures": failures,
    }


def diagnostics(
    dev: bm.Benchmark, layer: ev.ChunkLayer, ranked: Sequence[rt.Ranked]
) -> dict[str, Any]:
    """What the retrieval iteration compares, beyond the official block: global (macro by
    need) and per-language values of every official metric and of Recall@50 and Recall@100,
    the group recalls, and the dependent evidence, with every per-query value kept."""
    targets = ev.targets(dev, layer)
    evaluable = {q.query_id for t in targets for q in t.queries}
    run = rt.to_run([r for r in ranked if r.query_id in evaluable])
    scores, missing = ev.score_queries(
        dev, layer, run, recall_cutoffs=(*ir_metrics.DEFAULT_RECALL_CUTOFFS, *DEEP_CUTOFFS)
    )
    if missing:
        raise rt.RetrievalError(f"queries absent from the run: {missing}")
    queries = [q for t in targets for q in t.queries]
    names = [*METRICS, *(f"Recall@{k}" for k in DEEP_CUTOFFS)]
    values = {m: {s.query.query_id: s.values[m] for s in scores} for m in names}
    needs = {t.need.need_id: t.need for t in targets}

    def members(kind: str, label: str) -> list[str]:
        if kind == "type":
            return [n for n, need in needs.items() if need.type == label]
        return [n for n, need in needs.items() if label in need.drivers]

    groups = {}
    for name, (kind, label) in GROUPS.items():
        ids = set(members(kind, label))
        mine = [q for q in queries if q.need_id in ids]
        groups[name] = {"needs": len(ids)} | {
            m: rt.macro_by_need(values[m], mine) for m in ("Recall@10", "nDCG@10", "Recall@20")
        }

    dep = rt.dependent_evidence(targets)
    top = {r.query_id: r.chunk_ids for r in ranked}
    dep_queries = [q for q in queries if q.need_id in dep]
    dependent: dict[str, Any] = {"needs": len(dep), "queries": len(dep_queries)}
    per_query: dict[str, dict[str, float]] = {}
    for k in (10, 20):
        vals = {
            q.query_id: rt.recall_of(top[q.query_id][:k], dep[q.need_id].dependents)
            for q in dep_queries
        }
        dependent[f"Recall@{k}"] = rt.macro_by_need(vals, dep_queries)
        for qid, v in vals.items():
            per_query.setdefault(qid, {})[f"Recall@{k}"] = v
    anchor_hits = {
        q.query_id: float(bool(set(top[q.query_id][:10]) & dep[q.need_id].anchors))
        for q in dep_queries
    }
    dependent["anchor_in_top10"] = rt.macro_by_need(anchor_hits, dep_queries)
    dependent["per_query"] = per_query

    return {
        "global": {m: rt.macro_by_need(values[m], queries) for m in names},
        "by_language": {m: rt.per_language(values[m], queries) for m in names},
        "groups": groups,
        "dependent_evidence": dependent,
        "per_query": {m: values[m] for m in ("Recall@10", "nDCG@10", "Recall@50", "Recall@100")},
    }


def candidate_recall(
    dev: bm.Benchmark,
    layer: ev.ChunkLayer,
    ranked: Sequence[rt.Ranked],
    cutoffs: Sequence[int] = (20, 50, 100),
) -> dict[str, Any]:
    """Recall of the first k chunks of a candidate list (binary relevance, grade >= 2): global
    (macro by need), per language, per group, and for the dependent evidence. The same
    quantity as the evaluator's Recall@k, read on a list that is not a final ranking."""
    targets = ev.targets(dev, layer)
    queries = [q for t in targets for q in t.queries]
    relevant = {q.query_id: rt.relevant_chunks(t) for t in targets for q in t.queries}
    needs = {t.need.need_id: t.need for t in targets}
    dep = rt.dependent_evidence(targets)
    lists = {r.query_id: r.chunk_ids for r in ranked}
    out: dict[str, Any] = {}
    for k in cutoffs:
        values = {
            q.query_id: rt.recall_of(lists[q.query_id][:k], relevant[q.query_id]) for q in queries
        }
        groups = {}
        for name, (kind, label) in GROUPS.items():
            ids = {
                n
                for n, need in needs.items()
                if (need.type == label if kind == "type" else label in need.drivers)
            }
            groups[name] = rt.macro_by_need(values, [q for q in queries if q.need_id in ids])
        dep_queries = [q for q in queries if q.need_id in dep]
        dep_values = {
            q.query_id: rt.recall_of(lists[q.query_id][:k], dep[q.need_id].dependents)
            for q in dep_queries
        }
        out[f"@{k}"] = {
            "global": rt.macro_by_need(values, queries),
            "by_language": rt.per_language(values, queries),
            "groups": groups,
            "dependent_evidence": rt.macro_by_need(dep_values, dep_queries),
            "per_query": values,
        }
    return out


def paired_bootstrap(
    deltas: Mapping[str, float], queries: Sequence[bm.Query], reps: int = 10_000
) -> dict[str, float]:
    """Mean per-query delta and a 95% interval from resampling needs (a need's queries are
    not independent), with how many queries got better, stayed equal or got worse."""
    by_need: dict[str, list[float]] = {}
    for q in queries:
        by_need.setdefault(q.need_id, []).append(deltas[q.query_id])
    needs = sorted(by_need)
    rng = random.Random(SEED)
    means = sorted(
        statistics.fmean(v for n in rng.choices(needs, k=len(needs)) for v in by_need[n])
        for _ in range(reps)
    )
    return {
        "mean": statistics.fmean(v for n in needs for v in by_need[n]),
        "ci95_low": means[int(0.025 * reps)],
        "ci95_high": means[int(0.975 * reps) - 1],
        "better": sum(d > 1e-9 for d in deltas.values()),
        "equal": sum(abs(d) <= 1e-9 for d in deltas.values()),
        "worse": sum(d < -1e-9 for d in deltas.values()),
    }
