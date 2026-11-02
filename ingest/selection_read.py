"""The glue of the single labelled read of the S5-S6 selection (stage B of the preregistration).

Where ingest/selection_rules.py holds the rules as pure functions of numbers, this module turns
the frozen DEV targets and the top-10 lists of the six systems into those numbers with the
project's own evaluator (ingest/ir_metrics.py), and renders what the analysis found. Only three
measures are kept, Recall@10, nDCG@10 and Judged@10; the evaluator computes MRR@10 on the way and
that value is discarded here, never stored and never shown. Nothing in this module reads a file.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from ingest import evaluation as ev
from ingest import ir_metrics
from ingest import selection_rules as sr

METRICS = ("Recall@10", "nDCG@10", "Judged@10")

Vectors = dict[str, dict[str, dict[str, float]]]  # system -> query id -> metric -> value
PerNeedTable = dict[str, dict[str, dict[str, float]]]  # system -> need id -> metric -> value


class SelectionReadError(Exception):
    """An input does not meet the contract of the read: an integrity stop, never worked around."""


def needs_from_targets(targets: Sequence[ev.Target]) -> list[sr.NeedMeta]:
    """The evaluable needs, ordered by need id, with their chunk-level grades (a chunk absent from
    the mapping is unjudged) and their queries."""
    return [
        sr.NeedMeta(
            t.need.need_id,
            dict(t.chunk_grades),
            tuple(sorted(q.query_id for q in t.queries)),
        )
        for t in sorted(targets, key=lambda t: t.need.need_id)
    ]


def check_lists(needs: Sequence[sr.NeedMeta], lists: sr.Lists) -> None:
    """Every system has a top-10 for every evaluable query, with ten distinct chunks."""
    expected = {q for n in needs for q in n.query_ids}
    problems: list[str] = []
    for system in sr.SYSTEMS:
        if system not in lists:
            problems.append(f"{system}: no lists")
            continue
        missing = sorted(expected - set(lists[system]))
        if missing:
            problems.append(f"{system}: no list for {missing[:3]}")
        for query_id in sorted(expected & set(lists[system])):
            top = lists[system][query_id]
            if len(top) != sr.TOP or len(set(top)) != sr.TOP:
                problems.append(f"{system} {query_id}: {len(top)} chunks, {len(set(top))} distinct")
    if problems:
        raise SelectionReadError("the lists do not meet the contract:\n" + "\n".join(problems[:20]))


def system_vectors(
    needs: Sequence[sr.NeedMeta], top10: Mapping[str, tuple[str, ...]]
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """The three measures per query, and their mean per need (each query counts once, each need
    once), for one system. The evaluator is the project's, on the first ten chunks."""
    qrels = {q: n.grades for n in needs for q in n.query_ids}
    rankings = {q: list(top10[q][: sr.TOP]) for q in qrels}
    scores = ir_metrics.standard_scores(qrels, rankings, recall_cutoffs=(10,))
    per_query = {q: {m: scores[q][m] for m in METRICS} for q in qrels}
    per_need = {
        n.need_id: {m: statistics.fmean(per_query[q][m] for q in n.query_ids) for m in METRICS}
        for n in needs
    }
    return per_query, per_need


def all_vectors(needs: Sequence[sr.NeedMeta], lists: sr.Lists) -> tuple[Vectors, PerNeedTable]:
    check_lists(needs, lists)
    per_query: Vectors = {}
    per_need: PerNeedTable = {}
    for system in sr.SYSTEMS:
        per_query[system], per_need[system] = system_vectors(needs, lists[system])
    return per_query, per_need


def metric_of(table: PerNeedTable, metric: str) -> dict[str, dict[str, float]]:
    return {s: {n: v[metric] for n, v in per_need.items()} for s, per_need in table.items()}


def system_summary(table: PerNeedTable) -> dict[str, dict[str, float]]:
    """Macro mean over the evaluable needs of each measure, per system."""
    return {
        s: {m: statistics.fmean(v[m] for v in per_need.values()) for m in METRICS}
        for s, per_need in table.items()
    }


def coverage_figures(
    needs: Sequence[sr.NeedMeta], lists: sr.Lists, pairs: Sequence[sr.Pair]
) -> dict[str, dict[str, float]]:
    """Descriptive, for each comparison: Judged@10 of both sides, their ratio, the mean number of
    chunks of the candidate's top-10 that the reference's does not hold, and how many of those
    (and of the reference's own) are unjudged. Macro over needs, mean over queries."""
    out: dict[str, dict[str, float]] = {}
    for candidate, reference in pairs:
        exclusive: list[float] = []
        cand_unjudged: list[float] = []
        ref_unjudged: list[float] = []
        cand_judged: list[float] = []
        ref_judged: list[float] = []
        for n in needs:
            ex, cu, ru, cj, rj = [], [], [], [], []
            for q in n.query_ids:
                c, r = lists[candidate][q][: sr.TOP], lists[reference][q][: sr.TOP]
                ex.append(len(set(c) - set(r)))
                cu.append(len([x for x in set(c) - set(r) if x not in n.grades]))
                ru.append(len([x for x in set(r) - set(c) if x not in n.grades]))
                cj.append(sum(x in n.grades for x in c) / sr.TOP)
                rj.append(sum(x in n.grades for x in r) / sr.TOP)
            exclusive.append(statistics.fmean(ex))
            cand_unjudged.append(statistics.fmean(cu))
            ref_unjudged.append(statistics.fmean(ru))
            cand_judged.append(statistics.fmean(cj))
            ref_judged.append(statistics.fmean(rj))
        out[f"{candidate}|{reference}"] = {
            "judged_at_10_candidate": statistics.fmean(cand_judged),
            "judged_at_10_reference": statistics.fmean(ref_judged),
            "judged_ratio": statistics.fmean(cand_judged) / statistics.fmean(ref_judged),
            "mean_chunks_only_in_candidate_top10": statistics.fmean(exclusive),
            "mean_unjudged_among_them": statistics.fmean(cand_unjudged),
            "mean_unjudged_among_reference_only": statistics.fmean(ref_unjudged),
        }
    return out


def _scenario_dict(s: sr.Scenario) -> dict[str, Any]:
    return {
        "label": s.label,
        "guardrail": s.guardrail,
        "recall": vars(s.recall),
        "recall_without_most_influential_need": s.drop_one,
        "ndcg": vars(s.ndcg),
    }


def analysis_to_dict(analysis: sr.Analysis) -> dict[str, Any]:
    def key(pair: sr.Pair) -> str:
        return f"{pair[0]}|{pair[1]}"

    return {
        "latency_classes": dict(analysis.classes),
        "comparisons": {
            key(pair): {
                "primary": _scenario_dict(c.primary),
                "candidate_favourable": _scenario_dict(c.favourable),
                "candidate_unfavourable": _scenario_dict(c.unfavourable),
            }
            for pair, c in analysis.comparisons.items()
        },
        "coverage_state": {key(p): s for p, s in analysis.states.items()},
        "final_label": {key(p): s for p, s in analysis.labels.items()},
        "h2b_action_against_e5": dict(analysis.actions),
        "n20_over_rrf": analysis.n20_over_rrf,
        "n20_over_rrf_coverage_state": analysis.tie_break_state,
        "dominance_coverage_state": {key(p): s for p, s in analysis.dominance_states.items()},
        "dominance": {key(p): s for p, s in analysis.dominance.items()},
        "paths": {
            "fast_path": analysis.paths.fast_path,
            "quality_path": analysis.paths.quality_path,
            "discarded": dict(analysis.paths.discarded),
            "frontier": list(analysis.paths.frontier),
        },
    }


def _fmt(x: float) -> str:
    return f"{x:+.3f}"


def render_report(
    analysis: sr.Analysis,
    summary: Mapping[str, Mapping[str, float]],
    coverage: Mapping[str, Mapping[str, float]],
    header: Mapping[str, str],
    looks: int,
) -> str:
    lines = ["# S5-S6 retrieval selection: the single labelled read", ""]
    lines += [f"- {k}: `{v}`" for k, v in header.items()]
    lines += [f"- looks consumed: {looks}", ""]
    lines += ["## Systems on DEV (macro mean over the evaluable needs)", ""]
    lines += [
        "| system | latency class | Recall@10 | nDCG@10 | Judged@10 |",
        "|---|---|---|---|---|",
    ]
    for s in sr.SYSTEMS:
        m = summary[s]
        lines.append(
            f"| {s} | {analysis.classes[s]} | {m['Recall@10']:.3f} | {m['nDCG@10']:.3f} | "
            f"{m['Judged@10']:.3f} |"
        )
    lines += ["", "## Labelled comparisons (candidate against reference)", ""]
    lines += [
        "| comparison | dRecall [95 % CI] | d without top need | dnDCG [95 % CI] | guardrail "
        "| label (frozen qrels) | favourable / unfavourable label | coverage state | final label |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for pair, state in analysis.states.items():
        c = analysis.comparisons[pair]
        p = c.primary
        lines.append(
            f"| {pair[0]} vs {pair[1]} | {_fmt(p.recall.mean)} "
            f"[{_fmt(p.recall.low2)}; {_fmt(p.recall.high2)}] | {_fmt(p.drop_one)} | "
            f"{_fmt(p.ndcg.mean)} [{_fmt(p.ndcg.low2)}; {_fmt(p.ndcg.high2)}] | {p.guardrail} | "
            f"{p.label} | {c.favourable.label} / {c.unfavourable.label} | {state} | "
            f"{analysis.labels[pair]} |"
        )
    lines += ["", "## Coverage sensitivity envelope (dRecall and dnDCG, per scenario)", ""]
    lines += [
        "| comparison | dRecall unfav. | dRecall primary | dRecall fav. | dnDCG unfav. "
        "| dnDCG primary | dnDCG fav. |",
        "|---|---|---|---|---|---|---|",
    ]
    for pair, c in analysis.comparisons.items():
        lines.append(
            f"| {pair[0]} vs {pair[1]} | {_fmt(c.unfavourable.recall.mean)} | "
            f"{_fmt(c.primary.recall.mean)} | {_fmt(c.favourable.recall.mean)} | "
            f"{_fmt(c.unfavourable.ndcg.mean)} | {_fmt(c.primary.ndcg.mean)} | "
            f"{_fmt(c.favourable.ndcg.mean)} |"
        )
    lines += ["", "## Coverage figures (descriptive)", ""]
    lines += [
        "| comparison | Judged@10 candidate | Judged@10 reference | ratio | chunks only in "
        "candidate top-10 | of which unjudged | reference-only unjudged |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, v in coverage.items():
        lines.append(
            f"| {name.replace('|', ' vs ')} | {v['judged_at_10_candidate']:.3f} | "
            f"{v['judged_at_10_reference']:.3f} | {v['judged_ratio']:.3f} | "
            f"{v['mean_chunks_only_in_candidate_top10']:.2f} | "
            f"{v['mean_unjudged_among_them']:.2f} | {v['mean_unjudged_among_reference_only']:.2f} |"
        )
    lines += ["", "## H2b actions against E5 (after the coverage state)", ""]
    for system, action in analysis.actions.items():
        lines.append(f"- {system}: {action}")
    lines += [
        f"- N20 over RRF (only used if both displace): {analysis.n20_over_rrf} "
        f"(coverage state {analysis.tie_break_state})",
        "",
        "## H7 dominance (W dominates V on cost; result of the quality rule)",
        "",
    ]
    for pair, outcome in analysis.dominance.items():
        lines.append(
            f"- {pair[0]} over {pair[1]}: {outcome} (coverage {analysis.dominance_states[pair]})"
        )
    paths = analysis.paths
    lines += [
        "",
        "## Paths",
        "",
        f"- FAST PATH: {paths.fast_path}",
        f"- QUALITY PATH: {paths.quality_path} (by construction, TEST v2)",
        f"- discarded: {dict(paths.discarded) or 'none'}",
        f"- frontier (not discarded, no path): {', '.join(paths.frontier) or 'none'}",
        "",
        "selected/explored on DEV, not independently confirmed. Latencies and memory: "
        "measured on M1. Latency classes: product assumption \u2014 measured on M1.",
        "",
    ]
    return "\n".join(lines)
