"""B of retrieval iteration 3: evidence sets from the necessity scores
(config/retrieval/necessity-selection.md).

    PYTHONPATH=. .venv/bin/python scripts/necessity_selection.py --expect-version 0.10.0

Runs only if A's gate opens it (retrieval-iteration-3/a-necessity.json). In each of the 9
configurations, every query's ranking is E5's first n chunks, then at most k supports whose best
A3 probability under a kept anchor reaches the threshold, then E5's order. Each configuration is
written as a run, measured by ingest/run_report.py (the full-evidence coverage at 10 included)
and paired with E5. The selected one is checked against the success rules, and N022, N024, N034
and N035 are followed query by query: E5, R1's pool, the selector. Writes
data/results/dev-<version>/retrieval-iteration-3/b-selection.json and a summary per
configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ingest import benchmark as bm
from ingest import embeddings as em
from ingest import necessity as nc
from ingest import references as xr
from ingest import retrieval as rt
from ingest import run_report as rr

SCORES = "necessity-qwen3-reranker-0.6b"
XREF = Path("config/retrieval/xref-expansion.json")
ANCHORS = 8
OUTSIDE = 10
GRID = [(n, k, t) for n, k in ((6, 4), (7, 3), (8, 2)) for t in (0.50, 0.70, 0.85)]
FOLLOWED = ("N022", "N024", "N034", "N035")
E5_REFERENCE = {"Recall@10": 0.763, "nDCG@10": 0.528, "multi_provision": 0.481, "dependent": 0.300}


def name_of(n: int, k: int, t: float) -> str:
    return f"sets-a3-n{n}-k{k}-t{round(t * 100)}"


def columns_of(diag: dict[str, Any]) -> dict[str, float]:
    return {
        "nDCG@10": diag["global"]["nDCG@10"],
        "Recall@10": diag["global"]["Recall@10"],
        "Recall@20": diag["global"]["Recall@20"],
        "MRR@10": diag["global"]["MRR@10"],
        "Recall@T=1000": diag["global"]["Recall@T=1000"],
        "Recall@T=2000": diag["global"]["Recall@T=2000"],
        "Full-evidence@10": diag["full_evidence"]["Full-evidence@10"],
        "multi_provision": diag["groups"]["multi_provision"]["Recall@10"],
        "cross_reference": diag["groups"]["cross_reference"]["Recall@10"],
        "cross_act": diag["groups"]["cross_act"]["Recall@10"],
        "dependent": diag["dependent_evidence"]["Recall@10"],
    }


def success(c: dict[str, float]) -> dict[str, bool]:
    r = {k: round(v, 3) for k, v in c.items()}
    return {
        "useful": r["Recall@10"] >= 0.85
        and r["multi_provision"] >= 0.60
        and r["dependent"] >= 0.45,
        "strong": r["Recall@10"] >= 0.90 and r["nDCG@10"] >= 0.65,
    }


def material(c: dict[str, float]) -> dict[str, bool]:
    """Iteration 1's four tests of an interesting configuration, against E5, for the record."""
    d = {k: round(c[k], 3) - E5_REFERENCE[k] for k in E5_REFERENCE}
    return {
        "A: Recall@10 +0.05": d["Recall@10"] >= 0.05 - 1e-9,
        "B: nDCG@10 +0.05, Recall@10 down at most 0.02": d["nDCG@10"] >= 0.05 - 1e-9
        and d["Recall@10"] >= -0.02 - 1e-9,
        "C: multi_provision Recall@10 +0.10": d["multi_provision"] >= 0.10 - 1e-9,
        "D: dependent evidence Recall@10 +0.10": d["dependent"] >= 0.10 - 1e-9,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--benchmark", type=Path, default=bm.DEV_DIR)
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--results", type=Path, default=Path("data/results"))
    args = parser.parse_args(argv)
    frozen = rt.load_frozen_inputs(args.benchmark, args.expect_version, args.max_tokens)
    version = frozen.dev.manifest.benchmark_version
    out_dir = args.results / f"dev-{version}" / "retrieval-iteration-3"
    a = json.loads((out_dir / "a-necessity.json").read_text(encoding="utf-8"))
    if not a["decision"]["run_b"]:
        raise SystemExit("A's gate does not open B: A3 does not discriminate better than A1/A2")
    scores_path = rt.run_dir(SCORES, args.max_tokens, version) / "scores.jsonl"
    scores_bytes = scores_path.read_bytes()
    probability: dict[str, dict[tuple[str, str], float]] = {}
    for line in scores_bytes.decode("utf-8").splitlines():
        row = json.loads(line)
        if row["variant"] == "A3":
            key = (row["anchor"], row["candidate"])
            probability.setdefault(row["query_id"], {})[key] = nc.p_yes(row["logit_diff"])
    a3_rows = sum(len(v) for v in probability.values())
    if a3_rows != a["population"]["triples"]:
        raise SystemExit(f"{a3_rows} A3 scores, A recorded {a['population']['triples']} triples")
    e5_ranked, e5_summary = rt.load_system_run(
        "multilingual-e5-large", args.max_tokens, version, args.results
    )
    it2 = args.results / f"dev-{version}" / "retrieval-iteration-2"
    r1 = json.loads((it2 / "r1-candidates.json").read_text(encoding="utf-8"))
    r1_name = f"acdr-e5-bridge{r1['decision']['selected_k']}"
    r1_dir = rt.run_dir(r1_name, args.max_tokens, version)
    r1_lists, _ = rt.read_run(r1_dir / rt.RUN_FILE, r1["configurations"][r1_name]["run_sha256"])
    provenance = {
        row["query_id"]: row
        for row in map(json.loads, (r1_dir / "provenance.jsonl").read_text("utf-8").splitlines())
    }
    acts = xr.ExpansionConfig.model_validate_json(XREF.read_text(encoding="utf-8")).acts
    structure = nc.structure_of(frozen, acts)
    population = {
        r.query_id: nc.population(r, provenance[r.query_id], structure, ANCHORS, OUTSIDE)
        for r in e5_ranked
    }
    kinds = {(q, t.anchor, t.candidate): t.kinds for q, v in population.items() for t in v}
    if set(kinds) != {(q, a, c) for q, v in probability.items() for a, c in v}:
        raise SystemExit("the rebuilt population is not the one A scored")
    targets = frozen.targets
    queries = [q for t in targets for q in t.queries]
    e5_diag = rr.diagnostics(frozen.dev, frozen.layer, e5_ranked)
    bodies = {cid: c.body for cid, c in frozen.chunks.items()}
    grades = {q.query_id: dict(t.chunk_grades) for t in targets for q in t.queries}

    configurations: dict[str, dict[str, Any]] = {}
    chosen_supports: dict[str, dict[str, list[nc.Support]]] = {}
    for n, k, t in GRID:
        name = name_of(n, k, t)
        ranked, supports = [], {}
        for r in e5_ranked:
            r_set, kept = nc.evidence_set(r, probability.get(r.query_id, {}), n, k, t)
            ranked.append(r_set)
            supports[r.query_id] = kept
        chosen_supports[name] = supports
        sha = rt.write_run(
            rt.run_dir(name, args.max_tokens, version) / rt.RUN_FILE, ranked, frozen.evaluable
        )
        evaluated = rr.evaluate_ranked(frozen.dev, frozen.layer, ranked, bodies)
        diag = rr.diagnostics(frozen.dev, frozen.layer, ranked)
        paired = {
            m: rr.paired_bootstrap(
                {
                    q.query_id: diag["per_query"][m][q.query_id]
                    - e5_diag["per_query"][m][q.query_id]
                    for q in queries
                },
                queries,
            )
            for m in ("Recall@10", "nDCG@10", "Full-evidence@10")
        }
        columns = columns_of(diag)
        picked = [(q.query_id, s) for q in queries for s in supports[q.query_id]]
        support_stats = {
            "per_query_mean": statistics.fmean(len(supports[q.query_id]) for q in queries),
            "selected": len(picked),
            "relevant": sum(grades[qid].get(s.candidate, 0) >= 2 for qid, s in picked),
            "by_grade": dict(
                sorted(Counter(grades[qid].get(s.candidate, 0) for qid, s in picked).items())
            ),
            "by_link_kind": dict(
                Counter(k for qid, s in picked for k in kinds[(qid, s.anchor, s.candidate)])
            ),
        }
        summary = {
            "system": {
                "kind": "evidence_set",
                "scores": SCORES,
                "variant": "A3",
                "head": n,
                "supports": k,
                "threshold": t,
            },
            "identity": {
                "base_run_sha256": e5_summary["identity"]["run_sha256"],
                "scores_sha256": hashlib.sha256(scores_bytes).hexdigest(),
                "run_sha256": sha,
            },
            "benchmark_version": version,
            "parser_version": frozen.dev.manifest.parser_version,
            "layer": em.layer_source(list(frozen.docs)).model_dump(mode="json"),
            "queries_sha256": em.query_source(frozen.dev).queries_sha256,
            "metrics": evaluated["metrics"],
            "diagnostics": {key: v for key, v in diag.items() if key != "per_query"},
            "paired_against_e5": paired,
            "per_need": evaluated["per_need"],
            "per_query": evaluated["per_query"],
            "failures": evaluated["failures"],
        }
        out = args.results / f"dev-{version}" / f"{name}-layer-{args.max_tokens}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        configurations[name] = {
            "head": n,
            "supports": k,
            "threshold": t,
            "run_sha256": sha,
            "columns": columns,
            "per_query_recall10": diag["per_query"]["Recall@10"],
            "per_query_full10": diag["per_query"]["Full-evidence@10"],
            "by_language_recall10": diag["by_language"]["Recall@10"],
            "paired_against_e5": paired,
            "supports_selected": support_stats,
            "success": success(columns),
            "material": material(columns),
        }
        print(
            f"{name}: R@10 {columns['Recall@10']:.3f} nDCG@10 {columns['nDCG@10']:.3f} "
            f"full@10 {columns['Full-evidence@10']:.3f} multi {columns['multi_provision']:.3f} "
            f"dependent {columns['dependent']:.3f}; supports {support_stats['selected']} "
            f"({support_stats['relevant']} relevant); paired R@10 "
            f"{paired['Recall@10']['mean']:+.3f} [{paired['Recall@10']['ci95_low']:+.3f}, "
            f"{paired['Recall@10']['ci95_high']:+.3f}]",
            flush=True,
        )

    def order_key(name: str) -> tuple[float, float, int]:
        c = configurations[name]["columns"]
        return (round(c["Recall@10"], 3), round(c["nDCG@10"], 3), -list(configurations).index(name))

    chosen = max(configurations, key=order_key)
    decision = {
        "rule": "selected by global Recall@10, then nDCG@10, then grid order; useful success: "
        "Recall@10 >= 0.85, multi_provision >= 0.60 and dependent >= 0.45; strong success: "
        "Recall@10 >= 0.90 and nDCG@10 >= 0.65",
        "selected": chosen,
        "success": configurations[chosen]["success"],
        "material_tests_passed": [m for m, ok in configurations[chosen]["material"].items() if ok],
        "next": "stop: success"
        if any(configurations[chosen]["success"].values())
        else "research and C",
    }
    e5_columns = columns_of(e5_diag)
    chosen_ranked, _ = rt.read_run(
        rt.run_dir(chosen, args.max_tokens, version) / rt.RUN_FILE,
        configurations[chosen]["run_sha256"],
    )
    rank_in = {
        "e5": {r.query_id: {c: i for i, c in enumerate(r.chunk_ids, 1)} for r in e5_ranked},
        "r1": {r.query_id: {c: i for i, c in enumerate(r.chunk_ids, 1)} for r in r1_lists},
        "selector": {
            r.query_id: {c: i for i, c in enumerate(r.chunk_ids, 1)} for r in chosen_ranked
        },
    }
    followed: dict[str, list[dict[str, Any]]] = {}
    for target in targets:
        if target.need.need_id not in FOLLOWED:
            continue
        for q in target.queries:
            qid = q.query_id
            best: dict[str, tuple[float, str]] = {}
            for (anchor, c), p in probability.get(qid, {}).items():
                if c not in best or p > best[c][0]:
                    best[c] = (p, anchor)
            relevant = sorted(rt.relevant_chunks(target), key=lambda c: rank_in["e5"][qid][c])
            followed.setdefault(target.need.need_id, []).append(
                {
                    "query_id": qid,
                    "lang": q.lang,
                    "relevant": [
                        {
                            "chunk": c,
                            "grade": target.chunk_grades[c],
                            "e5_rank": rank_in["e5"][qid][c],
                            "r1_position": rank_in["r1"][qid][c],
                            "in_population": c in best,
                            "best_a3_probability": round(best[c][0], 4) if c in best else None,
                            "best_anchor": best[c][1] if c in best else None,
                            "selector_rank": rank_in["selector"][qid][c],
                        }
                        for c in relevant
                    ],
                    "supports": [
                        {
                            "candidate": s.candidate,
                            "anchor": s.anchor,
                            "probability": round(s.probability, 4),
                            "grade": target.chunk_grades.get(s.candidate, 0),
                            "links": list(kinds[(qid, s.anchor, s.candidate)]),
                        }
                        for s in chosen_supports[chosen][qid]
                    ],
                    "recall10": {
                        "e5": e5_diag["per_query"]["Recall@10"][qid],
                        "selector": configurations[chosen]["per_query_recall10"][qid],
                    },
                    "full_evidence10": {
                        "e5": e5_diag["per_query"]["Full-evidence@10"][qid],
                        "selector": configurations[chosen]["per_query_full10"][qid],
                    },
                }
            )
    payload = {
        "experiment": "B",
        "benchmark_version": version,
        "e5_run_sha256": e5_summary["identity"]["run_sha256"],
        "scores_sha256": hashlib.sha256(scores_bytes).hexdigest(),
        "e5_columns": e5_columns,
        "configurations": {
            name: {key: v for key, v in c.items() if not key.startswith("per_query")}
            for name, c in configurations.items()
        },
        "decision": decision,
        "followed_needs": followed,
    }
    (out_dir / "b-selection.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"E5: {e5_columns}")
    print(f"decision: {decision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
