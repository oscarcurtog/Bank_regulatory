"""Paired comparison of two dense baselines on the same frozen benchmark and chunk layer.

    PYTHONPATH=. .venv/bin/python scripts/compare_embedding_runs.py \\
        multilingual-e5-large bge-m3 --expect-version 0.10.0

Both runs must come from scripts/embed_baseline.py on the same DEV version, the same
chunk layer (same content fingerprint) and the same queries. Every number here is read
from their summaries and runs; the metrics themselves come from the official evaluator.
The comparison is paired: both models answer the same 81 queries, so each query (and
each need) is compared with itself. Intervals are a bootstrap over needs, because the
three queries of a need are not independent.

Writes comparison.json and comparison.md, versioned, in data/results/. Spanish queries
are named by id only.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ingest import benchmark as bm
from ingest import chunks as ch
from ingest import download as dl
from ingest import evaluation as ev

METRICS = [
    "nDCG@10",
    "Recall@10",
    "MRR@10",
    "Recall@5",
    "Recall@20",
    "Recall@T=1000",
    "Recall@T=2000",
]
SMALL_GROUP = 5
NDCG_STEP = 0.05  # a change of nDCG@10 under this is "the same" in the failure transitions
SEED = 20261013
TRANSITION_QUERIES = ["N026-es-1", "N024-es-1", "N015-en-2", "N023-en-2"]
TRANSITION_NEEDS = ["N035", "N034", "N022"]
FOCUS_NEEDS = ["N022", "N024", "N026", "N034", "N035"]
E5_TRUNCATION_NEEDS = ["N009", "N022", "N024", "N031"]
CORE_GROUPS = [
    ("type", "multi_provision"),
    ("driver", "cross_act"),
    ("driver", "cross_reference"),
    ("driver", "lexical_twin"),
]


def load_run(path: Path) -> dict[str, list[str]]:
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return {r["query_id"]: r["chunk_ids"] for r in rows}


def bootstrap_mean(
    values_by_need: Mapping[str, Sequence[float]], reps: int = 10_000
) -> dict[str, float]:
    """Mean over queries, with a 95% interval from resampling needs."""
    needs = sorted(values_by_need)
    flat = [v for n in needs for v in values_by_need[n]]
    rng = random.Random(SEED)
    means = []
    for _ in range(reps):
        sample = [v for n in rng.choices(needs, k=len(needs)) for v in values_by_need[n]]
        means.append(statistics.fmean(sample))
    means.sort()
    return {
        "mean": statistics.fmean(flat),
        "ci95_low": means[int(0.025 * reps)],
        "ci95_high": means[int(0.975 * reps) - 1],
    }


def wins(deltas: Sequence[float], eps: float = 1e-9) -> dict[str, int]:
    return {
        "b_wins": sum(d > eps for d in deltas),
        "ties": sum(abs(d) <= eps for d in deltas),
        "a_wins": sum(d < -eps for d in deltas),
    }


def side_by_side(
    a: Mapping[str, float], b: Mapping[str, float]
) -> dict[str, dict[str, float | None]]:
    out = {}
    for m in METRICS:
        delta = b[m] - a[m]
        out[m] = {"a": a[m], "b": b[m], "delta": delta, "relative": delta / a[m] if a[m] else None}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("a", help="model name of the baseline (config name)")
    parser.add_argument("b", help="model name of the challenger (config name)")
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--results", type=Path, default=Path("data/results"))
    args = parser.parse_args(argv)
    base = args.results / f"dev-{args.expect_version}"
    sa = json.loads(
        (base / f"{args.a}-layer-{args.max_tokens}" / "summary.json").read_text(encoding="utf-8")
    )
    sb = json.loads(
        (base / f"{args.b}-layer-{args.max_tokens}" / "summary.json").read_text(encoding="utf-8")
    )
    for key in ("benchmark_version", "parser_version", "queries_sha256"):
        if sa[key] != sb[key]:
            raise SystemExit(f"the two runs differ in {key}: {sa[key]} against {sb[key]}")
    if sa["layer"] != sb["layer"]:
        raise SystemExit("the two runs were made on different chunk layers")
    if sa["benchmark_version"] != args.expect_version:
        raise SystemExit(f"benchmark {sa['benchmark_version']}, expected {args.expect_version}")
    runs = {
        name: load_run(
            Path("data/runs")
            / f"{name}-layer-{args.max_tokens}-dev-{args.expect_version}"
            / "run.jsonl"
        )
        for name in (args.a, args.b)
    }
    dev = bm.load_benchmark(bm.DEV_DIR)
    docs = [ch.load_chunks(act, dl.load_manifest()) for act in bm.ACTS]
    layer = ev.layer_from_documents(docs)
    targets = {t.need.need_id: t for t in ev.targets(dev, layer)}
    qa, qb = sa["per_query"], sb["per_query"]
    if set(qa) != set(qb):
        raise SystemExit("the two runs scored different queries")
    queries = {q.query_id: q for q in dev.queries if q.query_id in qa}
    by_need: dict[str, list[str]] = defaultdict(list)
    for qid in sorted(qa):
        by_need[queries[qid].need_id].append(qid)

    def rank(model: str, qid: str, chunk: str) -> int:
        return runs[model][qid].index(chunk) + 1

    # ---- global, by style and by group (official numbers, read from the summaries)
    ma, mb = sa["metrics"], sb["metrics"]
    tables: dict[str, Any] = {
        "official_by_language": {
            lang: side_by_side(ma["official_by_language"][lang], mb["official_by_language"][lang])
            for lang in ma["official_by_language"]
        },
        "macro_by_need": side_by_side(ma["macro_by_need"], mb["macro_by_need"]),
        "by_style": {s: side_by_side(ma["by_style"][s], mb["by_style"][s]) for s in ma["by_style"]},
    }
    for key in ("by_type", "by_difficulty", "by_answerability", "by_driver"):
        tables[key] = {
            g: {"needs": ma[key][g]["needs"], "descriptive_only": ma[key][g]["descriptive_only"]}
            | side_by_side(ma[key][g], mb[key][g])
            for g in ma[key]
        }

    # ---- paired, per query and per need
    paired: dict[str, Any] = {}
    for metric in ("nDCG@10", "Recall@10", "MRR@10"):
        deltas = {qid: qb[qid][metric] - qa[qid][metric] for qid in qa}
        per_need = {n: [deltas[q] for q in qs] for n, qs in by_need.items()}
        need_means = {n: statistics.fmean(v) for n, v in per_need.items()}
        positive = sorted((d for d in need_means.values() if d > 0), reverse=True)
        largest = sorted(need_means, key=lambda n: abs(need_means[n]), reverse=True)[:3]
        paired[metric] = {
            "queries": wins(list(deltas.values())),
            "queries_en": wins([d for q, d in deltas.items() if queries[q].lang == "en"]),
            "queries_es": wins([d for q, d in deltas.items() if queries[q].lang == "es"]),
            "needs": wins(list(need_means.values())),
            "all": bootstrap_mean(per_need),
            "en": bootstrap_mean(
                {n: [deltas[q] for q in qs if queries[q].lang == "en"] for n, qs in by_need.items()}
            ),
            "es": bootstrap_mean(
                {n: [deltas[q] for q in qs if queries[q].lang == "es"] for n, qs in by_need.items()}
            ),
            "need_deltas": dict(sorted(need_means.items(), key=lambda kv: kv[1])),
            "top3_share_of_gains": (sum(positive[:3]) / sum(positive)) if positive else None,
            "three_largest_needs": largest,
            "all_without_three_largest": bootstrap_mean(
                {n: v for n, v in per_need.items() if n not in largest}
            ),
        }

    # ---- the core hypothesis: anchor found, dependent evidence missed
    anchor_rows: dict[str, dict[str, Any]] = {}
    for model in (args.a, args.b):
        rows: list[dict[str, Any]] = []
        for need_id, t in sorted(targets.items()):
            anchors = [c for c, g in t.chunk_grades.items() if g == 3]
            dependents = [c for c, g in t.chunk_grades.items() if g == 2]
            if not anchors or not dependents:
                continue
            for q in t.queries:
                ranks_dep = [rank(model, q.query_id, c) for c in dependents]
                anchor_hit = min(rank(model, q.query_id, c) for c in anchors) <= 10
                dep_recall = sum(r <= 10 for r in ranks_dep) / len(ranks_dep)
                rows.append(
                    {
                        "need": need_id,
                        "query": q.query_id,
                        "anchor_in_top10": anchor_hit,
                        "dependent_recall10": dep_recall,
                        "dependent_median_rank": statistics.median(ranks_dep),
                    }
                )
        anchor_rows[model] = {
            "queries": len(rows),
            "needs": len({r["need"] for r in rows}),
            "anchor_in_top10": statistics.fmean(r["anchor_in_top10"] for r in rows),
            "dependent_recall10": statistics.fmean(r["dependent_recall10"] for r in rows),
            "dependent_median_rank": statistics.median(r["dependent_median_rank"] for r in rows),
            "anchor_found_dependents_missed": statistics.fmean(
                r["anchor_in_top10"] and r["dependent_recall10"] < 1 for r in rows
            ),
        }
    need_info = {n["need"]: n for n in sa["per_need"]}

    def members(kind: str, name: str) -> list[str]:
        if kind == "type":
            return [n for n, i in need_info.items() if i["type"] == name]
        return [n for n, i in need_info.items() if name in i["drivers"]]

    core = {
        "groups": {
            f"{kind} {name}": tables["by_type" if kind == "type" else "by_driver"][name]
            | {
                f"paired {m}": bootstrap_mean(
                    {n: [qb[q][m] - qa[q][m] for q in by_need[n]] for n in members(kind, name)}
                )
                for m in ("nDCG@10", "Recall@10")
            }
            for kind, name in CORE_GROUPS
        },
        "anchor_dependent": anchor_rows,
        "focus_needs": {
            n: {
                m: {
                    "a": statistics.fmean(qa[q][m] for q in by_need[n]),
                    "b": statistics.fmean(qb[q][m] for q in by_need[n]),
                }
                for m in ("nDCG@10", "Recall@10")
            }
            | {
                "relevant_ranks": {
                    q: {"a": qa[q]["relevant_ranks"], "b": qb[q]["relevant_ranks"]}
                    for q in by_need[n]
                }
            }
            for n in FOCUS_NEEDS
        },
    }

    # ---- failure transitions
    def failing(q: Mapping[str, Any]) -> bool:
        return bool(q["Recall@10"] < 1 or q["first_relevant_rank"] > 10)

    def transition(qid: str) -> dict[str, Any]:
        a, b = qa[qid], qb[qid]
        recall_a, recall_b, ndcg_a, ndcg_b = (
            a["Recall@10"],
            b["Recall@10"],
            a["nDCG@10"],
            b["nDCG@10"],
        )
        if failing(a) and not failing(b):
            label = "A fixed by B"
        elif not failing(a) and failing(b):
            label = "D worse (a new failure)"
        elif not failing(a):
            label = "not failing under either"
        elif recall_b < recall_a or (recall_b == recall_a and ndcg_b < ndcg_a - NDCG_STEP):
            label = "D worse"
        elif recall_b > recall_a or (recall_b == recall_a and ndcg_b > ndcg_a + NDCG_STEP):
            label = "B improved but still failing"
        else:
            label = "C same failure"
        t = targets[queries[qid].need_id]
        relevant = {c for c, g in t.chunk_grades.items() if g >= bm.RELEVANT_GRADE}
        top_a, top_b = set(runs[args.a][qid][:10]), set(runs[args.b][qid][:10])
        distractors = {
            m: [(c, t.chunk_grades.get(c)) for c in runs[m][qid][:10] if c not in relevant][:3]
            for m in runs
        }
        return {
            "query": qid,
            "need": queries[qid].need_id,
            "lang": queries[qid].lang,
            "style": queries[qid].style,
            "text": queries[qid].text if queries[qid].lang == "en" else None,
            "label": label,
            "first_relevant_rank": {"a": a["first_relevant_rank"], "b": b["first_relevant_rank"]},
            "Recall@10": {"a": a["Recall@10"], "b": b["Recall@10"]},
            "nDCG@10": {"a": a["nDCG@10"], "b": b["nDCG@10"]},
            "relevant_gained_in_top10": sorted((relevant & top_b) - top_a),
            "relevant_lost_from_top10": sorted((relevant & top_a) - top_b),
            "top_distractors": distractors,
        }

    transition_ids = TRANSITION_QUERIES + [q for n in TRANSITION_NEEDS for q in by_need[n]]
    transitions = [transition(q) for q in transition_ids]
    all_transitions: defaultdict[str, int] = defaultdict(int)
    for qid in qa:
        all_transitions[transition(qid)["label"]] += 1

    # ---- language effect, per model, paired per need
    def per_style(
        summary_q: Mapping[str, Mapping[str, Any]], need: str, lang: str, style: str, metric: str
    ) -> float:
        [qid] = [q for q in by_need[need] if (queries[q].lang, queries[q].style) == (lang, style)]
        return float(summary_q[qid][metric])

    language: dict[str, Any] = {}
    for model, sq in ((args.a, qa), (args.b, qb)):
        language[model] = {
            metric: {
                "en_formal_minus_en_natural": bootstrap_mean(
                    {
                        n: [
                            per_style(sq, n, "en", "formal", metric)
                            - per_style(sq, n, "en", "natural", metric)
                        ]
                        for n in by_need
                    }
                ),
                "es_natural_minus_en_natural": bootstrap_mean(
                    {
                        n: [
                            per_style(sq, n, "es", "natural", metric)
                            - per_style(sq, n, "en", "natural", metric)
                        ]
                        for n in by_need
                    }
                ),
            }
            for metric in ("nDCG@10", "Recall@10", "Recall@T=1000")
        }

    def gap(sq: Mapping[str, Mapping[str, Any]], n: str, metric: str) -> float:
        es = per_style(sq, n, "es", "natural", metric)
        return es - per_style(sq, n, "en", "natural", metric)

    language["gap_change_b_minus_a"] = {
        metric: bootstrap_mean({n: [gap(qb, n, metric) - gap(qa, n, metric)] for n in by_need})
        for metric in ("nDCG@10", "Recall@10", "Recall@T=1000")
    }

    # ---- truncation as a control
    truncated_a = {c["chunk"] for c in sa["exposure"]["chunks"]}
    truncation: dict[str, Any] = {
        "a_truncated_chunks": sa["exposure"]["truncated_chunks"],
        "b_truncated_chunks": sb["exposure"]["truncated_chunks"],
        "needs": {},
    }
    for n in E5_TRUNCATION_NEEDS:
        t = targets[n]
        relevant = [c for c, g in t.chunk_grades.items() if g >= bm.RELEVANT_GRADE]
        rows = []
        for qid in by_need[n]:
            for c in relevant:
                rows.append(
                    {
                        "query": qid,
                        "chunk": c,
                        "truncated_in_a": c in truncated_a,
                        "rank_a": rank(args.a, qid, c),
                        "rank_b": rank(args.b, qid, c),
                    }
                )
        truncation["needs"][n] = rows
    pairs = [
        (c in truncated_a, rank(args.a, q, c), rank(args.b, q, c))
        for n, t in targets.items()
        for q in by_need[n]
        for c, g in t.chunk_grades.items()
        if g >= bm.RELEVANT_GRADE
    ]
    for label, cut in (("relevant_truncated_in_a", True), ("relevant_not_truncated", False)):
        group = [(ra, rb) for tr, ra, rb in pairs if tr is cut]
        truncation[label] = {
            "pairs": len(group),
            "better_in_b": sum(rb < ra for ra, rb in group),
            "same": sum(rb == ra for ra, rb in group),
            "worse_in_b": sum(rb > ra for ra, rb in group),
            "median_rank_a": statistics.median(ra for ra, _ in group),
            "median_rank_b": statistics.median(rb for _, rb in group),
        }
    top10_truncated = {m: sum(len(set(runs[m][q][:10]) & truncated_a) for q in qa) for m in runs}

    def bucket(chunk: str) -> str:
        lo = min(layer.chunks[chunk].token_count // 100, 4) * 100
        return f"{lo}-{lo + 99}" if lo < 400 else f"400-{args.max_tokens}"

    corpus_buckets = Counter(bucket(c) for c in layer.chunks)
    top10_buckets = {m: Counter(bucket(c) for q in qa for c in runs[m][q][:10]) for m in runs}
    truncation["length_of_top10_chunks"] = {
        "unit": "cl100k tokens of the official layer, the same for both models",
        "buckets": {
            k: {"corpus": corpus_buckets[k] / len(layer.chunks)}
            | {m: top10_buckets[m][k] / (10 * len(qa)) for m in runs}
            for k in sorted(corpus_buckets, key=lambda k: int(k.split("-")[0]))
        },
        "median": {
            m: statistics.median(layer.chunks[c].token_count for q in qa for c in runs[m][q][:10])
            for m in runs
        },
    }
    truncation["appearances_of_a_truncated_chunks_in_top10"] = top10_truncated

    performance = {m: s["performance"] for m, s in ((args.a, sa), (args.b, sb))}
    comparison = {
        "a": args.a,
        "b": args.b,
        "benchmark_version": sa["benchmark_version"],
        "layer": sa["layer"],
        "identity": {args.a: sa["identity"], args.b: sb["identity"]},
        "exposure": {
            args.a: {
                k: sa["exposure"][k]
                for k in ("truncated_chunks", "truncated_share", "dropped_tokens")
            },
            args.b: {
                k: sb["exposure"][k]
                for k in ("truncated_chunks", "truncated_share", "dropped_tokens")
            },
        },
        "tokenization": {args.a: sa["tokenization"], args.b: sb["tokenization"]},
        "tables": tables,
        "paired": paired,
        "core": core,
        "transitions": transitions,
        "transition_counts_all_queries": dict(sorted(all_transitions.items())),
        "language": language,
        "truncation": truncation,
        "performance": performance,
        "sanity": {args.a: sa["sanity"], args.b: sb["sanity"]},
    }
    out = base / "comparisons" / f"{args.a}-vs-{args.b}-layer-{args.max_tokens}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(
        json.dumps(comparison, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / "comparison.md").write_text(render(comparison), encoding="utf-8")
    print(f"written {out}")
    return 0


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------


def f3(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:+.1f}%"


def ci(s: Mapping[str, float]) -> str:
    return f"[{s['ci95_low']:+.3f}, {s['ci95_high']:+.3f}]"


def secs(timings: Mapping[str, float], key: str, digits: int) -> str:
    """A timing, or n/a when the step did not run (vectors read from the cache)."""
    return f"{timings[key]:.{digits}f}" if key in timings else "n/a"


def table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + [
        "| " + " | ".join(str(c) for c in r) + " |" for r in rows
    ]


def render(c: Mapping[str, Any]) -> str:
    a, b = c["a"], c["b"]
    t = c["tables"]
    lines = [
        f"# Paired comparison: {a} against {b}, DEV {c['benchmark_version']}, "
        f"layer {c['layer']['max_tokens']}",
        "",
        "Same benchmark, chunk layer, queries, qrels, projection and evaluator; only the model "
        "changes. Deltas are B minus A. Intervals: bootstrap over needs, 10,000 resamples, "
        "seed 20261013.",
        "",
    ]
    lines += ["## Global", ""]
    rows = []
    for scope, sbs in [
        ("EN (official)", t["official_by_language"]["en"]),
        ("ES (official)", t["official_by_language"]["es"]),
        ("macro by need (descriptive)", t["macro_by_need"]),
    ]:
        for m, v in sbs.items():
            rows.append(
                [scope, m, f3(v["a"]), f3(v["b"]), f"{v['delta']:+.3f}", pct(v["relative"])]
            )
    lines += table(["Scope", "Metric", a, b, "Delta", "Relative"], rows)
    lines += ["", "## By query style", ""]
    rows = [
        [s, m, f3(v["a"]), f3(v["b"]), f"{v['delta']:+.3f}"]
        for s, sbs in t["by_style"].items()
        for m, v in sbs.items()
    ]
    lines += table(["Style", "Metric", a, b, "Delta"], rows)
    lines += ["", "## By type, difficulty, answerability and driver (macro by need)", ""]
    rows = []
    for key in ("by_type", "by_difficulty", "by_answerability", "by_driver"):
        for g, v in t[key].items():
            mark = " (descriptive)" if v["descriptive_only"] else ""
            rows.append(
                [
                    key.replace("by_", ""),
                    f"{g}{mark}",
                    v["needs"],
                    f3(v["nDCG@10"]["a"]),
                    f3(v["nDCG@10"]["b"]),
                    f3(v["Recall@10"]["a"]),
                    f3(v["Recall@10"]["b"]),
                ]
            )
    lines += table(
        [
            "Group",
            "Value",
            "Needs",
            f"nDCG@10 {a}",
            f"nDCG@10 {b}",
            f"Recall@10 {a}",
            f"Recall@10 {b}",
        ],
        rows,
    )
    lines += [
        "",
        "## Paired, per query",
        "",
        "Mean of the per-query deltas. 'all' mixes languages and is descriptive; 'en' and 'es' "
        "follow D-020. Wins, ties and losses count queries (ties within 1e-9).",
        "",
    ]
    rows = []
    for m, p in c["paired"].items():
        for scope, counts in (("all", "queries"), ("en", "queries_en"), ("es", "queries_es")):
            s = p[scope]
            rows.append([m, scope, f"{s['mean']:+.3f}", ci(s), *p[counts].values()])
        s = p["all_without_three_largest"]
        rows.append(
            [m, f"all, without {', '.join(p['three_largest_needs'])}", f"{s['mean']:+.3f}", ci(s)]
            + ["", "", ""]
        )
    lines += table(
        ["Metric", "Queries", "Mean delta", "95% interval", f"{b} wins", "Ties", f"{a} wins"], rows
    )
    lines += ["", "Per need (mean over its three queries), B minus A:", ""]
    rows = []
    for m in ("nDCG@10", "Recall@10"):
        p = c["paired"][m]
        need_wins = p["needs"]
        rows.append(
            [
                m,
                f"{need_wins['b_wins']} / {need_wins['ties']} / {need_wins['a_wins']}",
                ", ".join(f"{n} {d:+.2f}" for n, d in list(p["need_deltas"].items())[:4]),
                ", ".join(f"{n} {d:+.2f}" for n, d in list(p["need_deltas"].items())[-4:]),
                f3(p["top3_share_of_gains"]),
            ]
        )
    lines += table(
        [
            "Metric",
            f"Needs: {b} better / equal / {a} better",
            "Largest losses",
            "Largest gains",
            "Share of all gains from the top 3 needs",
        ],
        rows,
    )
    lines += [
        "",
        "## Anchor found, dependent evidence missed",
        "",
        "Needs whose relevant chunks include a grade-3 anchor and at least one grade-2 "
        "dependent chunk.",
        "",
    ]
    rows = [
        [
            m,
            v["needs"],
            v["queries"],
            f3(v["anchor_in_top10"]),
            f3(v["dependent_recall10"]),
            v["dependent_median_rank"],
            f3(v["anchor_found_dependents_missed"]),
        ]
        for m, v in c["core"]["anchor_dependent"].items()
    ]
    lines += table(
        [
            "Model",
            "Needs",
            "Queries",
            "Anchor in top 10",
            "Dependent Recall@10",
            "Dependent median rank",
            "Anchor found, a dependent missed",
        ],
        rows,
    )
    lines += ["", "The groups of the core hypothesis (macro by need):", ""]
    rows = []
    for g, v in c["core"]["groups"].items():
        mark = " (descriptive)" if v["descriptive_only"] else ""
        rows += [
            [
                f"{g}{mark}",
                v["needs"],
                m,
                f3(v[m]["a"]),
                f3(v[m]["b"]),
                f"{v[m]['delta']:+.3f}",
                ci(v[f"paired {m}"]),
            ]
            for m in ("nDCG@10", "Recall@10")
        ]
    lines += table(["Group", "Needs", "Metric", a, b, "Delta", "95% interval"], rows)
    lines += ["", "The needs under review (mean over their three queries):", ""]
    rows = [
        [n, m, f3(v[m]["a"]), f3(v[m]["b"]), f"{v[m]['b'] - v[m]['a']:+.3f}"]
        for n, v in c["core"]["focus_needs"].items()
        for m in ("nDCG@10", "Recall@10")
    ]
    lines += table(["Need", "Metric", a, b, "Delta"], rows)
    lines += [
        "",
        "## Failure transitions",
        "",
        "A query fails when some relevant chunk is outside its top 10 (Recall@10 under 1; no "
        "need has more than 6 relevant chunks). A fixed: fails under A, not under B. D worse "
        "(a new failure): fails under B only. When it fails under both: D worse if Recall@10 is "
        "lower, or equal with nDCG@10 down by more than 0.05; B improved but still failing if "
        "Recall@10 is higher, or equal with nDCG@10 up by more than 0.05; otherwise C same "
        "failure. The rule is symmetric between the models.",
        "",
    ]
    rows = []
    for x in c["transitions"]:
        rows.append(
            [
                x["query"],
                x["label"],
                f"{x['first_relevant_rank']['a']} / {x['first_relevant_rank']['b']}",
                f"{f3(x['Recall@10']['a'])} / {f3(x['Recall@10']['b'])}",
                f"{f3(x['nDCG@10']['a'])} / {f3(x['nDCG@10']['b'])}",
                ", ".join(x["relevant_gained_in_top10"]) or "-",
                ", ".join(x["relevant_lost_from_top10"]) or "-",
            ]
        )
    lines += table(
        [
            "Query",
            "Transition",
            "First relevant",
            "Recall@10",
            "nDCG@10",
            "Gained in top 10",
            "Lost from top 10",
        ],
        rows,
    )
    lines += [
        "",
        "All 81 queries: "
        + ", ".join(f"{k} {v}" for k, v in c["transition_counts_all_queries"].items())
        + ".",
        "",
    ]
    lines += ["## Language effect, per model", "", "Paired per need.", ""]
    rows = []
    for m in (a, b):
        for metric, d in c["language"][m].items():
            for name, s in d.items():
                rows.append([m, metric, name.replace("_", " "), f"{s['mean']:+.3f}", ci(s)])
    for metric, s in c["language"]["gap_change_b_minus_a"].items():
        rows.append(
            [
                f"{b} minus {a}",
                metric,
                "change of (ES natural - EN natural)",
                f"{s['mean']:+.3f}",
                ci(s),
            ]
        )
    lines += table(["Model", "Metric", "Difference", "Mean", "95% interval"], rows)
    tr = c["truncation"]
    lines += [
        "",
        "## Truncation as a control",
        "",
        f"Truncated chunks: {a} {tr['a_truncated_chunks']}, {b} {tr['b_truncated_chunks']}. "
        f"Appearances of {a}'s truncated chunks in a top 10: "
        f"{tr['appearances_of_a_truncated_chunks_in_top10']}.",
        "",
        "Every (query, relevant chunk) pair of the 81 queries, by whether A truncated the chunk:",
        "",
    ]
    lines += table(
        [
            "Pairs",
            "Count",
            f"Better rank in {b}",
            "Same",
            f"Worse in {b}",
            f"Median rank {a}",
            f"Median rank {b}",
        ],
        [
            [
                k.replace("_", " "),
                v["pairs"],
                v["better_in_b"],
                v["same"],
                v["worse_in_b"],
                v["median_rank_a"],
                v["median_rank_b"],
            ]
            for k, v in tr.items()
            if k.startswith("relevant_")
        ],
    )
    lengths = tr["length_of_top10_chunks"]
    lines += [
        "",
        f"Length of the chunks that fill the top 10 of the 81 queries ({lengths['unit']}). "
        f"Median: {a} {lengths['median'][a]}, {b} {lengths['median'][b]}.",
        "",
    ]
    lines += table(
        ["Tokens", "Share of the corpus", f"Share of {a}'s top 10", f"Share of {b}'s top 10"],
        [[k, f3(v["corpus"]), f3(v[a]), f3(v[b])] for k, v in lengths["buckets"].items()],
    )
    lines += ["", "The needs where A truncated a relevant chunk:", ""]
    rows = [
        [
            n,
            r["query"],
            r["chunk"] + (" (truncated in A)" if r["truncated_in_a"] else ""),
            r["rank_a"],
            r["rank_b"],
        ]
        for n, rs in tr["needs"].items()
        for r in rs
    ]
    lines += table(["Need", "Query", "Relevant chunk", f"Rank {a}", f"Rank {b}"], rows)
    lines += [
        "",
        "## Performance",
        "",
        "Same machine, measured back to back in one session, one run per model: the variance "
        "between runs is not measured. Times exclude system sleep. Queries: the 90 queries in "
        "one batched pass, not a per-query latency. Peak RAM is the whole process.",
        "",
    ]
    rows = []
    for m, p in c["performance"].items():
        ts = p["timings_seconds"]
        rows.append(
            [
                m,
                secs(ts, "model load", 1),
                secs(ts, "document embeddings", 1),
                secs(ts, "query embeddings", 1),
                secs(ts, "exact retrieval", 3),
                p["dimension"],
                f"{p['embedding_bytes']['documents'] / 1e6:.2f}",
                f"{p['peak_rss_bytes'] / 1e9:.2f}",
                f"{p['model_cache_bytes'] / 1e9:.2f}",
            ]
        )
    lines += table(
        [
            "Model",
            "Load (s)",
            "Documents (s)",
            "Queries (s)",
            "Retrieval (s)",
            "Dimension",
            "Vectors (MB)",
            "Peak RAM (GB)",
            "Model cache (GB)",
        ],
        rows,
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
