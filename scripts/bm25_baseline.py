"""Experiment 1 of the retrieval iteration: BM25 alone on the frozen DEV benchmark.

    PYTHONPATH=. .venv/bin/python scripts/bm25_baseline.py --expect-version 0.10.0

The contract is config/retrieval/bm25.json and the protocol config/retrieval/README.md:
one configuration, no tuning. Steps:
1. the frozen inputs: the DEV benchmark at the expected version and the official chunk layer;
2. the index: the analyzer on every chunk.text, bm25s's Lucene BM25, and what the analyzer
   keeps of the queries in each language;
3. every query ranks every chunk; the run goes to data/runs/ (not versioned);
4. the official metrics and breakdowns (ingest/run_report.py, the code that measures the
   dense models) and the iteration's diagnostics;
5. sanity checks: the same ranking twice, and the scores against an independent numpy
   computation of the same formula.

The summary is versioned in data/results/dev-<version>/bm25-layer-<n>/summary.json. No
label is read for anything but scoring.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ingest import benchmark as bm
from ingest import embeddings as em
from ingest import lexical as lx
from ingest import retrieval as rt
from ingest import run_report as rr


def pct(values: list[int], p: float) -> int:
    s = sorted(values)
    return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=lx.CONFIG_DIR / "bm25.json")
    parser.add_argument("--benchmark", type=Path, default=bm.DEV_DIR)
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--results", type=Path, default=Path("data/results"))
    args = parser.parse_args(argv)
    timings: dict[str, float] = {}

    print("1. frozen inputs")
    config = lx.load_config(args.config)
    frozen = rt.load_frozen_inputs(args.benchmark, args.expect_version, args.max_tokens)
    dev, layer = frozen.dev, frozen.layer
    documents = [(cid, c.text) for cid, c in frozen.chunks.items()]
    print(
        f"  DEV {dev.manifest.benchmark_version}, parser {dev.manifest.parser_version}, "
        f"{len(documents)} chunks at {args.max_tokens}, contract {config.config_hash()[:12]}"
    )

    print("2. index")
    versions = lx.check_versions(config)
    start = time.perf_counter()
    index = lx.build_index(config, documents)
    timings["index build"] = time.perf_counter() - start
    lengths = list(index.doc_lengths)
    vocabulary = {t for t in index.retriever.vocab_dict if t}
    query_terms: dict[str, dict[str, Any]] = {}
    for lang in ("en", "es"):
        shares, empty = [], []
        for q in dev.queries:
            if q.lang != lang:
                continue
            terms = index.analyzer(q.text)
            known = [t for t in terms if t in vocabulary]
            shares.append(len(known) / len(terms) if terms else 0.0)
            if not known:
                empty.append(q.query_id)
        query_terms[lang] = {
            "queries": len(shares),
            "mean_share_of_terms_in_vocabulary": statistics.fmean(shares),
            "queries_with_no_known_term": empty,
        }
    analysis = {
        "vocabulary_size": len(vocabulary),
        "terms_per_chunk": {
            "min": min(lengths),
            "median": statistics.median(lengths),
            "p95": pct(lengths, 95),
            "max": max(lengths),
        },
        "query_terms": query_terms,
    }
    print(
        f"  {len(vocabulary)} terms; per chunk median {statistics.median(lengths)}; query terms "
        f"known: en {query_terms['en']['mean_share_of_terms_in_vocabulary']:.2f}, es "
        f"{query_terms['es']['mean_share_of_terms_in_vocabulary']:.2f}"
    )

    print("3. ranking")
    queries = [(q.query_id, q.text) for q in dev.queries]
    start = time.perf_counter()
    ranked = index.rank(queries)
    timings["retrieval (90 queries)"] = time.perf_counter() - start
    out_run = rt.run_dir(config.name, args.max_tokens, dev.manifest.benchmark_version)
    run_sha = rt.write_run(out_run / rt.RUN_FILE, ranked, frozen.evaluable)

    print("4. metrics")
    bodies = {cid: c.body for cid, c in frozen.chunks.items()}
    evaluated = rr.evaluate_ranked(dev, layer, ranked, bodies)
    diagnostics = rr.diagnostics(dev, layer, ranked)

    print("5. sanity checks")
    again = index.rank(queries)
    texts = [text for _, text in documents]
    ids = [cid for cid, _ in documents]
    same_top10 = 0
    worst = 0.0
    for r in ranked:
        reference = lx.reference_scores(index, texts, dict(queries)[r.query_id])
        ordered = rt.rank_by_scores(r.query_id, ids, reference)
        same_top10 += ordered.chunk_ids[:10] == r.chunk_ids[:10]
        mine = index.scores(dict(queries)[r.query_id])
        worst = max(worst, float(np.abs(mine - reference).max()))
    zero = {r.query_id: sum(s == 0 for s in r.scores) for r in ranked}
    sanity = {
        "same_ranking_twice": again == ranked,
        "top10_equals_numpy_reference": same_top10,
        "max_score_diff_vs_numpy_reference": worst,
        "ranked_every_chunk": all(len(r.chunk_ids) == len(ids) for r in ranked),
        "chunks_with_zero_score_median": statistics.median(zero.values()),
    }
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scores_matrix = index.retriever.scores
    performance = {
        "documents": len(ids),
        "queries": len(queries),
        "timings_seconds": timings,
        "index_nonzero_entries": int(len(scores_matrix["data"])),
        "index_bytes_in_memory": int(
            scores_matrix["data"].nbytes
            + scores_matrix["indices"].nbytes
            + scores_matrix["indptr"].nbytes
        ),
        "peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
        "environment": em.environment_description(),
    }
    summary = {
        "system": {"kind": "bm25"} | config.model_dump(mode="json"),
        "config_hash": config.config_hash(),
        "identity": {"versions": versions, "run_sha256": run_sha},
        "benchmark_version": dev.manifest.benchmark_version,
        "parser_version": dev.manifest.parser_version,
        "layer": em.layer_source(list(frozen.docs)).model_dump(mode="json"),
        "queries_sha256": em.query_source(dev).queries_sha256,
        "analysis": analysis,
        "metrics": evaluated["metrics"],
        "diagnostics": diagnostics,
        "per_need": evaluated["per_need"],
        "per_query": evaluated["per_query"],
        "failures": evaluated["failures"],
        "sanity": sanity,
        "performance": performance,
    }
    out = (
        args.results
        / f"dev-{dev.manifest.benchmark_version}"
        / f"{config.name}-layer-{args.max_tokens}"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"  written {out / 'summary.json'}")
    for name, row in evaluated["metrics"]["by_style"].items():
        print(f"  {name}: " + ", ".join(f"{m} {row[m]:.3f}" for m in rr.METRICS))
    g = diagnostics["global"]
    print("  global: " + ", ".join(f"{m} {g[m]:.3f}" for m in g))
    print(f"  sanity: {sanity}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
