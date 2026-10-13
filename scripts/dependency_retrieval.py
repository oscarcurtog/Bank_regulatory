"""R1 of retrieval iteration 2: anchor-conditioned dependency retrieval, measured as candidate
recall before any reranking (config/retrieval/dependency-retrieval.md).

    PYTHONPATH=. .venv/bin/python scripts/dependency_retrieval.py --expect-version 0.10.0

E5's cached vectors (identity checked) give the base ranking, which must equal E5's recorded
run. The anchors' relations (ingest/dependencies.py), the bridges and their E5 hits build each
query's candidate list for every bridge k of the grid; their Candidate Recall@20/50/100 decide the
gate. Candidate lists and provenance go to data/runs/ (not versioned); the measures and the
decision to data/results/dev-<version>/retrieval-iteration-2/r1-candidates.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ingest import benchmark as bm
from ingest import dense as dn
from ingest import dependencies as dp
from ingest import references as xr
from ingest import retrieval as rt
from ingest import run_report as rr

E5 = Path("config/embeddings/multilingual-e5-large.json")
XREF = Path("config/retrieval/xref-expansion.json")
ANCHORS = 10
KEEP = 20
BRIDGES = 3
GRID = (3, 5)
GATE = 0.95


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
    recorded, summary = rt.load_system_run(
        "multilingual-e5-large", args.max_tokens, version, args.results
    )
    if base != recorded:
        raise SystemExit("the cached E5 vectors do not reproduce E5's recorded run")
    timings: dict[str, float] = {}

    start = time.perf_counter()
    corpus = bm.load_corpus()
    texts = {
        (act, n.path): n.own_text for act, d in corpus.items() for a in d.articles for n in a.nodes
    }
    acts = xr.ExpansionConfig.model_validate_json(XREF.read_text(encoding="utf-8")).acts
    graph = xr.build_graph(frozen.layer, texts, list(frozen.chunks), acts)
    definitions = dp.glossary(texts)
    index = dp.build_index(frozen.layer, texts, graph, definitions)
    timings["structure (graph, glossary, relations)"] = time.perf_counter() - start

    per_query: dict[str, tuple[list[dp.Relation], list[tuple[str, int]]]] = {
        r.query_id: dp.anchor_relations(r, index, ANCHORS) for r in base
    }
    bridge_texts = sorted({dp.bridge_text(t) for _, terms in per_query.values() for t, _ in terms})
    start = time.perf_counter()
    encoded = dense.encode_queries(bridge_texts)
    timings["bridge candidates encoded"] = time.perf_counter() - start
    bridge_vectors = dict(zip(bridge_texts, encoded, strict=True))
    qvec = dict(zip(dense.query_ids, dense.query_vectors, strict=True))
    doc_ids = list(dense.doc_ids)
    bridges = {
        qid: dp.choose_bridges(qvec[qid], terms, bridge_vectors, BRIDGES)
        for qid, (_, terms) in per_query.items()
    }
    bridge_rank = {
        b: rt.rank_by_scores(b, doc_ids, dense.doc_vectors @ bridge_vectors[b]).chunk_ids
        for b in sorted({b for bs in bridges.values() for b in bs})
    }

    results: dict[str, Any] = {}
    runs: dict[int, list[rt.Ranked]] = {}
    for k in GRID:
        start = time.perf_counter()
        lists: list[rt.Ranked] = []
        records: list[dict[str, Any]] = []
        for r in base:
            relations, _ = per_query[r.query_id]
            probes = [qvec[r.query_id], *(bridge_vectors[b] for b in bridges[r.query_id])]
            scores = np.max(np.vstack(probes) @ dense.doc_vectors.T, axis=0)
            score_of = {c: float(v) for c, v in zip(doc_ids, scores, strict=True)}
            hits = {b: bridge_rank[b][:k] for b in bridges[r.query_id]}
            qc = dp.candidate_list(r, relations, hits, score_of.__getitem__, KEEP)
            lists.append(qc.ranked)
            records.append(
                {
                    "query_id": r.query_id,
                    "bridges": qc.bridges,
                    "pool_size": qc.pool_size,
                    "candidates": {
                        c: [rel.__dict__ for rel in rels] for c, rels in qc.provenance.items()
                    },
                }
            )
        timings[f"candidate lists, k={k}"] = time.perf_counter() - start
        name = f"acdr-e5-bridge{k}"
        out_run = rt.run_dir(name, args.max_tokens, version)
        sha = rt.write_run(out_run / rt.RUN_FILE, lists, frozen.evaluable)
        with (out_run / "provenance.jsonl").open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        recall = rr.candidate_recall(frozen.dev, frozen.layer, lists)
        runs[k] = lists
        evaluated = [rec for rec in records if rec["query_id"] in frozen.evaluable]
        relevant = {q.query_id: rt.relevant_chunks(t) for t in frozen.targets for q in t.queries}
        entered: Counter[str] = Counter()
        for rec, lst in zip(records, lists, strict=True):
            if rec["query_id"] not in relevant:
                continue
            first50 = set(lst.chunk_ids[KEEP:50])
            for c in first50 & relevant[rec["query_id"]]:
                for kind in {rel["kind"] for rel in rec["candidates"].get(c, [])}:
                    entered[kind] += 1
        results[name] = {
            "run_sha256": sha,
            "candidate_recall": {
                at: {key: v for key, v in block.items() if key != "per_query"}
                for at, block in recall.items()
            },
            "pool_size": {
                "median": statistics.median(rec["pool_size"] for rec in evaluated),
                "max": max(rec["pool_size"] for rec in evaluated),
            },
            "relevant_beyond_e5_top20_in_first_50_by_relation": dict(sorted(entered.items())),
        }
        g = recall
        print(
            f"{name}: candidate recall global @20 {g['@20']['global']:.3f} @50 "
            f"{g['@50']['global']:.3f} @100 {g['@100']['global']:.3f}; @50 multi_provision "
            f"{g['@50']['groups']['multi_provision']:.3f}, cross_reference "
            f"{g['@50']['groups']['cross_reference']:.3f}, cross_act "
            f"{g['@50']['groups']['cross_act']:.3f}, dependent "
            f"{g['@50']['dependent_evidence']:.3f}; pool median "
            f"{results[name]['pool_size']['median']}"
        )

    e5 = rr.candidate_recall(frozen.dev, frozen.layer, base)
    best = max(
        GRID,
        key=lambda k: (
            round(results[f"acdr-e5-bridge{k}"]["candidate_recall"]["@50"]["global"], 3),
            -k,
        ),
    )
    chosen = results[f"acdr-e5-bridge{best}"]["candidate_recall"]["@50"]["global"]
    decision = {
        "rule": "the k with the higher global Candidate Recall@50 (then the smaller k); >= 0.95 "
        "goes to reranking, under 0.95 to research",
        "selected_k": best,
        "candidate_recall50": round(chosen, 3),
        "e5_recall50": round(e5["@50"]["global"], 3),
        "next": "reranking" if chosen >= GATE else "research",
    }
    payload = {
        "experiment": "R1",
        "benchmark_version": version,
        "e5_run_sha256": summary["identity"]["run_sha256"],
        "structure": {
            "definitions": len(definitions),
            "definitions_by_act": dict(Counter(d.act for d in definitions)),
            "graph": xr.summarize_graph(graph),
            "relations_per_chunk_median": statistics.median(
                len(v) for v in index.relations.values()
            ),
        },
        "bridges": {qid: bs for qid, bs in bridges.items()},
        "e5_candidate_recall": {
            at: {key: v for key, v in block.items() if key != "per_query"}
            for at, block in e5.items()
        },
        "configurations": results,
        "decision": decision,
        "timings_seconds": timings,
    }
    out = args.results / f"dev-{version}" / "retrieval-iteration-2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "r1-candidates.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"E5 alone: @20 {e5['@20']['global']:.3f} @50 {e5['@50']['global']:.3f} @100 "
        f"{e5['@100']['global']:.3f}; @50 multi_provision "
        f"{e5['@50']['groups']['multi_provision']:.3f}, dependent "
        f"{e5['@50']['dependent_evidence']:.3f}"
    )
    print(f"decision: {decision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
