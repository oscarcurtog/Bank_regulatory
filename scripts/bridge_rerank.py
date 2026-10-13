"""E3 of retrieval iteration 2: bridge-conditioned cross-encoder reranking
(config/retrieval/dependency-retrieval.md).

    PYTHONPATH=. .venv/bin/python scripts/bridge_rerank.py --expect-version 0.10.0

The candidate list is R1's selected list (its run and provenance as R1 recorded them), unless
E2's selected ranking has a higher global Recall@50. Its top 50 are scored by the reranker of
config/retrieval/bge-reranker-v2-m3.json: with the original query only, or, for a candidate
outside E5's top 20, with the maximum over the original query and the bridges that produced it.
Each configuration is written as a run, measured by ingest/run_report.py and paired with E5.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

from ingest import benchmark as bm
from ingest import dependencies as dp
from ingest import embeddings as em
from ingest import reranker as rk
from ingest import retrieval as rt
from ingest import run_report as rr

RERANKER = Path("config/retrieval/bge-reranker-v2-m3.json")
DEPTH = 50
KEEP = 20
E5_REFERENCE = {"Recall@10": 0.763, "nDCG@10": 0.528, "multi_provision": 0.481, "dependent": 0.300}


def material(c: dict[str, float]) -> dict[str, bool]:
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
    out_dir = args.results / f"dev-{version}" / "retrieval-iteration-2"
    r1 = json.loads((out_dir / "r1-candidates.json").read_text(encoding="utf-8"))
    e2 = json.loads((out_dir / "e2-ppr.json").read_text(encoding="utf-8"))
    r1_name = f"acdr-e5-bridge{r1['decision']['selected_k']}"
    r1_r50 = r1["configurations"][r1_name]["candidate_recall"]["@50"]["global"]
    e2_r50 = e2["configurations"][e2["decision"]["selected"]]["candidate_recall50"]
    if e2_r50 > r1_r50:
        raise SystemExit(
            "E2's ranking has the higher Recall@50: its candidate list is not wired here"
        )
    run_path = rt.run_dir(r1_name, args.max_tokens, version)
    lists, _ = rt.read_run(run_path / rt.RUN_FILE, r1["configurations"][r1_name]["run_sha256"])
    provenance = {
        row["query_id"]: row
        for row in (
            json.loads(line)
            for line in (run_path / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    e5_ranked, e5_summary = rt.load_system_run(
        "multilingual-e5-large", args.max_tokens, version, args.results
    )
    query_text = {q.query_id: q.text for q in frozen.dev.queries}
    chunk_text = {cid: c.text for cid, c in frozen.chunks.items()}

    config = rk.load_config(RERANKER)
    start = time.perf_counter()
    reranker = rk.Reranker(config)
    load_seconds = time.perf_counter() - start
    wanted: dict[tuple[str, str], rk.Pair] = {}
    bridges_of: dict[str, dict[str, list[str]]] = {}
    for r in lists:
        qid, row = r.query_id, provenance[r.query_id]
        for c in r.chunk_ids[:DEPTH]:
            wanted.setdefault((query_text[qid], c), reranker.pair(query_text[qid], chunk_text[c]))
        query_bridges = set(row["bridges"])
        bridges_of[qid] = {}
        for c in r.chunk_ids[KEEP:DEPTH]:
            found = set()
            for rel in row["candidates"].get(c, []):
                if rel["kind"] == "bridge":
                    found.add(rel["detail"])
                elif rel["kind"] == "definition" and dp.bridge_text(rel["detail"]) in query_bridges:
                    found.add(dp.bridge_text(rel["detail"]))
            bridges_of[qid][c] = sorted(found)
            for b in found:
                wanted.setdefault((b, c), reranker.pair(b, chunk_text[c]))
    keys = sorted(wanted)
    start = time.perf_counter()
    values = reranker.score([wanted[k] for k in keys])
    score_seconds = time.perf_counter() - start
    score = dict(zip(keys, (float(v) for v in values), strict=True))
    truncated = sum(p.truncated for p in wanted.values())
    print(
        f"reranker: {len(keys)} pairs scored in {score_seconds:.0f} s (load {load_seconds:.1f} s), "
        f"{truncated} truncated, longest {max(len(p.ids) for p in wanted.values())} tokens"
    )

    queries = [q for t in frozen.targets for q in t.queries]
    e5_diag = rr.diagnostics(frozen.dev, frozen.layer, e5_ranked)
    bodies = {cid: c.body for cid, c in frozen.chunks.items()}
    configurations: dict[str, dict[str, Any]] = {}
    for rule in ("orig", "bridgemax"):
        name = f"rerank-{r1_name}-{rule}"
        ranked = []
        for r in lists:
            qid = r.query_id
            position = {c: i for i, c in enumerate(r.chunk_ids)}

            def value(
                c: str, qid: str = qid, rule: str = rule, position: dict[str, int] = position
            ) -> float:
                s = score[(query_text[qid], c)]
                if rule == "bridgemax" and position[c] >= KEEP:
                    s = max([s, *(score[(b, c)] for b in bridges_of[qid].get(c, []))])
                return s

            head = sorted(r.chunk_ids[:DEPTH], key=lambda c: (-value(c), position[c]))
            order = head + list(r.chunk_ids[DEPTH:])
            n = len(order)
            ranked.append(rt.Ranked(qid, tuple(order), tuple(float(n - i) for i in range(n))))
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
                "kind": "bridge_conditioned_rerank",
                "candidates": r1_name,
                "depth": DEPTH,
                "rule": rule,
                "reranker": config.model_dump(mode="json"),
            },
            "identity": {
                "candidates_run_sha256": r1["configurations"][r1_name]["run_sha256"],
                "reranker_config_hash": config.config_hash(),
                "reranker_tokenizer_sha256": reranker.tokenizer_sha256,
                "run_sha256": sha,
            },
            "benchmark_version": version,
            "parser_version": frozen.dev.manifest.parser_version,
            "layer": em.layer_source(list(frozen.docs)).model_dump(mode="json"),
            "queries_sha256": em.query_source(frozen.dev).queries_sha256,
            "metrics": evaluated["metrics"],
            "diagnostics": {k: v for k, v in diag.items() if k != "per_query"},
            "paired_against_e5": paired,
            "per_need": evaluated["per_need"],
            "per_query": evaluated["per_query"],
            "failures": evaluated["failures"],
            "performance": {
                "pairs_scored": len(keys),
                "pairs_truncated": truncated,
                "model_load_seconds": load_seconds,
                "scoring_seconds": score_seconds,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "environment": em.environment_description(),
            },
        }
        out = args.results / f"dev-{version}" / f"{name}-layer-{args.max_tokens}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        configurations[name] = {
            "rule": rule,
            "columns": columns,
            "recall20": diag["global"]["Recall@20"],
            "mrr10": diag["global"]["MRR@10"],
            "recall_t": {k: diag["global"][k] for k in ("Recall@T=1000", "Recall@T=2000")},
            "cross_reference": diag["groups"]["cross_reference"]["Recall@10"],
            "cross_act": diag["groups"]["cross_act"]["Recall@10"],
            "by_language": {m: diag["by_language"][m] for m in ("Recall@10", "nDCG@10")},
            "paired_against_e5": paired,
            "material": material(columns),
        }
        print(
            f"{name}: R@10 {columns['Recall@10']:.3f} nDCG@10 {columns['nDCG@10']:.3f} MRR@10 "
            f"{diag['global']['MRR@10']:.3f} R@20 {diag['global']['Recall@20']:.3f}; "
            f"multi_provision {columns['multi_provision']:.3f}, dependent "
            f"{columns['dependent']:.3f}; EN/ES "
            f"R@10 {diag['by_language']['Recall@10']['en']:.3f}/"
            f"{diag['by_language']['Recall@10']['es']:.3f}; paired nDCG@10 "
            f"{paired['nDCG@10']['mean']:+.3f} [{paired['nDCG@10']['ci95_low']:+.3f}, "
            f"{paired['nDCG@10']['ci95_high']:+.3f}]"
        )

    def key(name: str) -> tuple[float, float, int]:
        c = configurations[name]["columns"]
        return (round(c["Recall@10"], 3), round(c["nDCG@10"], 3), -list(configurations).index(name))

    chosen = max(configurations, key=key)
    cols: dict[str, float] = configurations[chosen]["columns"]
    passed = [t for t, ok in configurations[chosen]["material"].items() if ok]
    stop_a = cols["Recall@10"] >= 0.90 and cols["nDCG@10"] >= 0.65
    stop_b = (
        cols["Recall@10"] >= 0.85
        and round(cols["multi_provision"], 3) - E5_REFERENCE["multi_provision"] >= 0.10 - 1e-9
        and round(cols["dependent"], 3) - E5_REFERENCE["dependent"] >= 0.10 - 1e-9
    )
    decision = {
        "selected": chosen,
        "passed": passed,
        "kept": bool(passed),
        "stop_A_strong": stop_a,
        "stop_B_useful": stop_b,
    }
    payload = {
        "experiment": "E3",
        "benchmark_version": version,
        "candidates": r1_name,
        "e5_run_sha256": e5_summary["identity"]["run_sha256"],
        "reranker": config.model_dump(mode="json"),
        "pairs_scored": len(keys),
        "configurations": configurations,
        "decision": decision,
    }
    (out_dir / "e3-rerank.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"decision: {decision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
