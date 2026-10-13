"""E2 of retrieval iteration 2: dense-seeded Personalized PageRank over the legal structure
(config/retrieval/dependency-retrieval.md).

    PYTHONPATH=. .venv/bin/python scripts/structural_ppr.py --expect-version 0.10.0

E5's cached vectors (identity checked, re-ranking to E5's recorded run) give the seeds; the graph
is the law's own (references, defined terms, definitions by reference, ingest/dependencies.py).
Each configuration of the grid ranks every query, is written as a run and measured by
ingest/run_report.py; the selected one is paired with E5 and judged by the iteration's rule.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from ingest import benchmark as bm
from ingest import dense as dn
from ingest import dependencies as dp
from ingest import embeddings as em
from ingest import references as xr
from ingest import retrieval as rt
from ingest import run_report as rr

E5 = Path("config/embeddings/multilingual-e5-large.json")
XREF = Path("config/retrieval/xref-expansion.json")
GRID = [(10, 0.3), (10, 0.5), (20, 0.3), (20, 0.5)]
E5_REFERENCE = {"Recall@10": 0.763, "nDCG@10": 0.528, "multi_provision": 0.481, "dependent": 0.300}


def material(c: dict[str, float]) -> dict[str, bool]:
    """Iteration 1's four tests of an interesting configuration, against E5."""
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
    dense = dn.load_dense(E5, frozen, args.max_tokens)
    base = dense.rank()
    recorded, e5_summary = rt.load_system_run(
        "multilingual-e5-large", args.max_tokens, version, args.results
    )
    if base != recorded:
        raise SystemExit("the cached E5 vectors do not reproduce E5's recorded run")

    start = time.perf_counter()
    corpus = bm.load_corpus()
    texts = {
        (act, n.path): n.own_text for act, d in corpus.items() for a in d.articles for n in a.nodes
    }
    acts = xr.ExpansionConfig.model_validate_json(XREF.read_text(encoding="utf-8")).acts
    graph = xr.build_graph(frozen.layer, texts, list(frozen.chunks), acts)
    index = dp.build_index(frozen.layer, texts, graph, dp.glossary(texts))
    doc_ids = list(dense.doc_ids)
    transition = dp.structure_matrix(index, doc_ids)
    structure_seconds = time.perf_counter() - start
    edges = int((transition > 0).sum())
    print(f"graph: {len(doc_ids)} chunks, {edges} edges, built in {structure_seconds:.2f} s")

    scores = dense.query_vectors @ dense.doc_vectors.T
    dense_scores = {
        qid: dict(zip(doc_ids, (float(x) for x in scores[i]), strict=True))
        for i, qid in enumerate(dense.query_ids)
    }
    queries = [q for t in frozen.targets for q in t.queries]
    e5_diag = rr.diagnostics(frozen.dev, frozen.layer, base)
    bodies = {cid: c.body for cid, c in frozen.chunks.items()}
    configurations: dict[str, dict[str, Any]] = {}
    for n_seeds, follow in GRID:
        name = f"ppr-e5-n{n_seeds}-f{int(follow * 10)}"
        start = time.perf_counter()
        ranked = [
            dp.ppr_ranking(r, dense_scores[r.query_id], transition, doc_ids, n_seeds, follow)
            for r in base
        ]
        seconds = time.perf_counter() - start
        sha = rt.write_run(
            rt.run_dir(name, args.max_tokens, version) / rt.RUN_FILE, ranked, frozen.evaluable
        )
        evaluated = rr.evaluate_ranked(frozen.dev, frozen.layer, ranked, bodies)
        diag = rr.diagnostics(frozen.dev, frozen.layer, ranked)
        cand = rr.candidate_recall(frozen.dev, frozen.layer, ranked)
        paired = {
            m: rr.paired_bootstrap(
                {
                    q.query_id: diag["per_query"][m][q.query_id]
                    - e5_diag["per_query"][m][q.query_id]
                    for q in queries
                },
                queries,
            )
            for m in ("Recall@10", "nDCG@10")
        }
        columns = {
            "Recall@10": diag["global"]["Recall@10"],
            "nDCG@10": diag["global"]["nDCG@10"],
            "multi_provision": diag["groups"]["multi_provision"]["Recall@10"],
            "dependent": diag["dependent_evidence"]["Recall@10"],
        }
        summary = {
            "system": {
                "kind": "dense_seeded_ppr",
                "dense": "multilingual-e5-large",
                "seeds": n_seeds,
                "follow": follow,
                "graph": "reference, definition, definition_reference",
            },
            "identity": {
                "base_run_sha256": e5_summary["identity"]["run_sha256"],
                "run_sha256": sha,
            },
            "benchmark_version": version,
            "parser_version": frozen.dev.manifest.parser_version,
            "layer": em.layer_source(list(frozen.docs)).model_dump(mode="json"),
            "queries_sha256": em.query_source(frozen.dev).queries_sha256,
            "metrics": evaluated["metrics"],
            "diagnostics": {k: v for k, v in diag.items() if k != "per_query"},
            "candidate_recall": {
                at: {k: v for k, v in b.items() if k != "per_query"} for at, b in cand.items()
            },
            "paired_against_e5": paired,
            "per_need": evaluated["per_need"],
            "per_query": evaluated["per_query"],
            "failures": evaluated["failures"],
            "performance": {
                "graph_seconds": structure_seconds,
                "ranking_seconds_90_queries": seconds,
                "graph_edges": edges,
            },
        }
        out = args.results / f"dev-{version}" / f"{name}-layer-{args.max_tokens}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        configurations[name] = {
            "seeds": n_seeds,
            "follow": follow,
            "columns": columns,
            "recall20": diag["global"]["Recall@20"],
            "candidate_recall50": cand["@50"]["global"],
            "cross_reference": diag["groups"]["cross_reference"]["Recall@10"],
            "cross_act": diag["groups"]["cross_act"]["Recall@10"],
            "by_language_recall10": diag["by_language"]["Recall@10"],
            "paired_against_e5": paired,
            "material": material(columns),
        }
        c = columns
        print(
            f"{name}: R@10 {c['Recall@10']:.3f} nDCG@10 {c['nDCG@10']:.3f} R@20 "
            f"{diag['global']['Recall@20']:.3f} R@50 {cand['@50']['global']:.3f}; multi_provision "
            f"{c['multi_provision']:.3f}, dependent {c['dependent']:.3f}; paired R@10 "
            f"{paired['Recall@10']['mean']:+.3f} [{paired['Recall@10']['ci95_low']:+.3f}, "
            f"{paired['Recall@10']['ci95_high']:+.3f}]"
        )

    def key(name: str) -> tuple[float, float, int]:
        c = configurations[name]["columns"]
        return (round(c["Recall@10"], 3), round(c["nDCG@10"], 3), -list(configurations).index(name))

    chosen = max(configurations, key=key)
    passed = [t for t, ok in configurations[chosen]["material"].items() if ok]
    decision = {
        "rule": "selected by global Recall@10, then nDCG@10, then grid order; kept if one of "
        "iteration 1's four tests of an interesting configuration passes against E5",
        "selected": chosen,
        "passed": passed,
        "kept": bool(passed),
    }
    payload = {
        "experiment": "E2",
        "benchmark_version": version,
        "e5_run_sha256": e5_summary["identity"]["run_sha256"],
        "graph_edges": edges,
        "configurations": configurations,
        "decision": decision,
    }
    out = args.results / f"dev-{version}" / "retrieval-iteration-2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e2-ppr.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"decision: {decision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
