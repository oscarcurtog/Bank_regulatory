"""Experiment 3 of the retrieval iteration: one-hop explicit cross-reference expansion.

    PYTHONPATH=. .venv/bin/python scripts/xref_expansion.py --base multilingual-e5-large \\
        --expect-version 0.10.0

The contract is config/retrieval/xref-expansion.json and the protocol
config/retrieval/README.md (experiment 3). The base retriever's run is read as its summary
recorded it (SHA-256). For each anchor K of the contract, every query's base ranking is
expanded (ingest/references.py), written to data/runs/ with the trace of every added chunk,
and measured by ingest/run_report.py, the code that measures every other retriever. Then the
expansion precision, and rule 3 of the protocol on the K that the selection rule picks.

Writes one summary per K in data/results/dev-<version>/<name>-<base>-k<K>-layer-<n>/ and
data/results/dev-<version>/retrieval-iteration/experiment-3-xref.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from ingest import benchmark as bm
from ingest import embeddings as em
from ingest import references as xr
from ingest import retrieval as rt
from ingest import run_report as rr

CONFIG = Path("config/retrieval/xref-expansion.json")
SHORT = {"multilingual-e5-large": "e5", "bge-m3": "bge"}
KEEP_RECALL = 0.05
KEEP_MULTI = 0.10
RERANK_GATE = 0.95


def precision(
    traces: dict[str, list[xr.Placement]], relevant: dict[str, frozenset[str]]
) -> dict[str, Any]:
    added = [(q, p) for q, ps in traces.items() if q in relevant for p in ps]
    top = [(q, p) for q, p in added if p.position <= 10]
    per_query = [len(traces[q]) for q in relevant]
    return {
        "added_chunks": len(added),
        "relevant_added": sum(p.chunk_id in relevant[q] for q, p in added),
        "precision": (
            sum(p.chunk_id in relevant[q] for q, p in added) / len(added) if added else None
        ),
        "added_in_top10": len(top),
        "relevant_added_in_top10": sum(p.chunk_id in relevant[q] for q, p in top),
        "precision_in_top10": (
            sum(p.chunk_id in relevant[q] for q, p in top) / len(top) if top else None
        ),
        "added_per_query": {
            "median": statistics.median(per_query),
            "max": max(per_query),
            "queries_with_none": sum(n == 0 for n in per_query),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--benchmark", type=Path, default=bm.DEV_DIR)
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--results", type=Path, default=Path("data/results"))
    args = parser.parse_args(argv)

    config = xr.ExpansionConfig.model_validate_json(CONFIG.read_text(encoding="utf-8"))
    frozen = rt.load_frozen_inputs(args.benchmark, args.expect_version, args.max_tokens)
    dev, layer = frozen.dev, frozen.layer
    version = dev.manifest.benchmark_version
    base_ranked, base_summary = rt.load_system_run(
        args.base, args.max_tokens, version, args.results
    )
    base_by_query = {r.query_id: r for r in base_ranked}
    corpus = bm.load_corpus()
    texts = {
        (act, n.path): n.own_text for act, d in corpus.items() for a in d.articles for n in a.nodes
    }
    start = time.perf_counter()
    graph = xr.build_graph(layer, texts, list(frozen.chunks), config.acts)
    graph_seconds = time.perf_counter() - start
    graph_stats = xr.summarize_graph(graph)
    print(
        f"graph: {graph_stats['references_followed']} references followed, built in "
        f"{graph_seconds:.2f} s"
    )

    base_diag = rr.diagnostics(dev, layer, base_ranked)
    relevant = {q.query_id: rt.relevant_chunks(t) for t in frozen.targets for q in t.queries}
    bodies = {cid: c.body for cid, c in frozen.chunks.items()}
    results: dict[int, dict[str, Any]] = {}
    for k in config.anchor_k:
        name = f"{config.name}-{SHORT.get(args.base, args.base)}-k{k}"
        start = time.perf_counter()
        expanded, traces = [], {}
        for q in dev.queries:
            ranked, trace = xr.expand(base_by_query[q.query_id], graph, k)
            expanded.append(ranked)
            traces[q.query_id] = trace
        seconds = time.perf_counter() - start
        out_run = rt.run_dir(name, args.max_tokens, version)
        run_sha = rt.write_run(out_run / rt.RUN_FILE, expanded, frozen.evaluable)
        with (out_run / "trace.jsonl").open("w", encoding="utf-8") as f:
            for qid, trace in traces.items():
                for p in trace:
                    f.write(json.dumps({"query_id": qid} | p.__dict__) + "\n")
        lead = {
            r.query_id: set(base_by_query[r.query_id].chunk_ids[:k])
            | {p.chunk_id for p in traces[r.query_id]}
            for r in expanded
        }
        evaluated = rr.evaluate_ranked(dev, layer, expanded, bodies)
        diag = rr.diagnostics(dev, layer, expanded)
        prec = precision(traces, relevant)
        summary = {
            "system": {"kind": "xref_expansion", "base": args.base, "anchor_k": k}
            | config.model_dump(mode="json"),
            "config_hash": config.config_hash(),
            "identity": {
                "base_run_sha256": base_summary["identity"]["run_sha256"],
                "run_sha256": run_sha,
            },
            "benchmark_version": version,
            "parser_version": dev.manifest.parser_version,
            "layer": em.layer_source(list(frozen.docs)).model_dump(mode="json"),
            "queries_sha256": em.query_source(dev).queries_sha256,
            "graph": graph_stats,
            "expansion": prec,
            "metrics": evaluated["metrics"],
            "diagnostics": diag,
            "per_need": evaluated["per_need"],
            "per_query": evaluated["per_query"],
            "failures": evaluated["failures"],
            "sanity": {
                "every_ranking_is_a_permutation_of_the_layer": all(
                    sorted(r.chunk_ids) == sorted(frozen.chunks) for r in expanded
                ),
                # the first anchor stays first; anchors and their targets come before every
                # other chunk; the rest keeps the base order (an anchor that an earlier anchor
                # cites moves up, by design)
                "first_anchor_first": all(
                    r.chunk_ids[0] == base_by_query[r.query_id].chunk_ids[0] for r in expanded
                ),
                "anchors_and_targets_lead": all(
                    set(r.chunk_ids[: len(lead[r.query_id])]) == lead[r.query_id] for r in expanded
                ),
                "the_rest_keeps_the_base_order": all(
                    list(r.chunk_ids[len(lead[r.query_id]) :])
                    == [c for c in base_by_query[r.query_id].chunk_ids if c not in lead[r.query_id]]
                    for r in expanded
                ),
            },
            "performance": {
                "graph_build_seconds": graph_seconds,
                "expansion_seconds_90_queries": seconds,
                "environment": em.environment_description(),
            },
        }
        out = args.results / f"dev-{version}" / f"{name}-layer-{args.max_tokens}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        results[k] = {"name": name, "diagnostics": diag, "expansion": prec}
        g = diag["global"]
        print(
            f"{name}: global R@10 {g['Recall@10']:.3f} nDCG@10 {g['nDCG@10']:.3f} R@20 "
            f"{g['Recall@20']:.3f} R@50 {g['Recall@50']:.3f}; multi_provision "
            f"{diag['groups']['multi_provision']['Recall@10']:.3f}; dependent "
            f"{diag['dependent_evidence']['Recall@10']:.3f}; precision {prec['precision']}, top10 "
            f"{prec['precision_in_top10']}"
        )

    def key(k: int) -> tuple[float, float, int]:
        g = results[k]["diagnostics"]["global"]
        return (round(g["Recall@10"], 3), round(g["nDCG@10"], 3), -k)

    chosen = max(results, key=key)
    c = results[chosen]["diagnostics"]
    base_g, c_g = base_diag["global"], c["global"]
    d_recall = round(c_g["Recall@10"], 3) - round(base_g["Recall@10"], 3)
    d_multi = round(c["groups"]["multi_provision"]["Recall@10"], 3) - round(
        base_diag["groups"]["multi_provision"]["Recall@10"], 3
    )
    kept = d_recall >= KEEP_RECALL - 1e-9 or d_multi >= KEEP_MULTI - 1e-9
    if kept:
        nxt = (
            "experiment 4: reranker"
            if c_g["Recall@50"] >= RERANK_GATE
            else "end: the expansion is the candidate"
        )
    else:
        nxt = "experiment 5, if its own conditions hold"
    decision = {
        "rule": "kept if, against its base, global Recall@10 +0.05 or multi_provision Recall@10 "
        "+0.10; kept with Recall@50 >= 0.95 goes to 4; not kept goes to 5",
        "selected_k": chosen,
        "base_global_recall10": round(base_g["Recall@10"], 3),
        "global_recall10": round(c_g["Recall@10"], 3),
        "delta_recall10": round(d_recall, 3),
        "base_multi_provision_recall10": round(
            base_diag["groups"]["multi_provision"]["Recall@10"], 3
        ),
        "multi_provision_recall10": round(c["groups"]["multi_provision"]["Recall@10"], 3),
        "delta_multi_provision": round(d_multi, 3),
        "recall50": round(c_g["Recall@50"], 3),
        "kept": kept,
        "next": nxt,
    }
    payload = {
        "experiment": 3,
        "base": args.base,
        "base_run_sha256": base_summary["identity"]["run_sha256"],
        "config_hash": config.config_hash(),
        "graph": graph_stats,
        "base_diagnostics": {k: v for k, v in base_diag.items() if k != "per_query"},
        "configurations": {
            r["name"]: {
                "diagnostics": {k: v for k, v in r["diagnostics"].items() if k != "per_query"},
                "expansion": r["expansion"],
            }
            for r in results.values()
        },
        "decision": decision,
    }
    out = args.results / f"dev-{version}" / "retrieval-iteration"
    out.mkdir(parents=True, exist_ok=True)
    (out / "experiment-3-xref.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"decision: {decision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
