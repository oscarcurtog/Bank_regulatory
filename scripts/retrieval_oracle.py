"""Experiment 1, second half: the oracle union of BM25 with each dense model, and the
decision it takes (config/retrieval/README.md, rule 1).

    PYTHONPATH=. .venv/bin/python scripts/retrieval_oracle.py --expect-version 0.10.0

For every scored query, the relevant chunks found in the union of two top-10 lists. It is a
diagnostic of complementarity, not a system: the union holds up to 20 chunks. The same is
read per language, per group and for the dependent evidence. Every run is the one its
summary recorded (SHA-256).

Writes data/results/dev-<version>/retrieval-iteration/experiment-1-bm25.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ingest import benchmark as bm
from ingest import retrieval as rt
from ingest import run_report as rr

DENSE = ("multilingual-e5-large", "bge-m3")
LEXICAL = "bm25"
GAIN_TO_HYBRID = 0.05


def union_block(
    frozen: rt.FrozenInputs,
    first: dict[str, rt.Ranked],
    second: dict[str, rt.Ranked],
) -> dict[str, Any]:
    targets = frozen.targets
    queries = [q for t in targets for q in t.queries]
    relevant = {q.query_id: rt.relevant_chunks(t) for t in targets for q in t.queries}
    union = {
        q.query_id: rt.oracle_union_recall(
            first[q.query_id].chunk_ids, second[q.query_id].chunk_ids, relevant[q.query_id]
        )
        for q in queries
    }
    alone = {
        q.query_id: rt.recall_of(first[q.query_id].chunk_ids[:10], relevant[q.query_id])
        for q in queries
    }
    needs = {t.need.need_id: t.need for t in targets}
    groups = {}
    for name, (kind, label) in rr.GROUPS.items():
        ids = {
            n
            for n, need in needs.items()
            if (need.type == label if kind == "type" else label in need.drivers)
        }
        mine = [q for q in queries if q.need_id in ids]
        groups[name] = {
            "needs": len(ids),
            "first": rt.macro_by_need(alone, mine),
            "union": rt.macro_by_need(union, mine),
        }
    dep = rt.dependent_evidence(targets)
    dep_queries = [q for q in queries if q.need_id in dep]
    dep_union = {
        q.query_id: rt.recall_of(
            set(first[q.query_id].chunk_ids[:10]) | set(second[q.query_id].chunk_ids[:10]),
            dep[q.need_id].dependents,
        )
        for q in dep_queries
    }
    dep_first = {
        q.query_id: rt.recall_of(first[q.query_id].chunk_ids[:10], dep[q.need_id].dependents)
        for q in dep_queries
    }
    gained = sorted(q for q in union if union[q] > alone[q])
    return {
        "global": {
            "first": rt.macro_by_need(alone, queries),
            "union": rt.macro_by_need(union, queries),
        },
        "by_language": {
            "first": rt.per_language(alone, queries),
            "union": rt.per_language(union, queries),
        },
        "groups": groups,
        "dependent_evidence": {
            "needs": len(dep),
            "first": rt.macro_by_need(dep_first, dep_queries),
            "union": rt.macro_by_need(dep_union, dep_queries),
        },
        "queries_gaining": {
            "count": len(gained),
            "en": sum("-en-" in q for q in gained),
            "es": sum("-es-" in q for q in gained),
            "ids": gained,
        },
        "per_query": {"first": alone, "union": union},
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
    runs, shas, diagnostics = {}, {}, {}
    for system in (*DENSE, LEXICAL):
        ranked, summary = rt.load_system_run(system, args.max_tokens, version, args.results)
        runs[system] = {r.query_id: r for r in ranked}
        shas[system] = summary["identity"]["run_sha256"]
        diagnostics[system] = rr.diagnostics(frozen.dev, frozen.layer, ranked)
    pairs = {
        f"{a} + {b}": union_block(frozen, runs[a], runs[b])
        for a, b in (
            ("multilingual-e5-large", LEXICAL),
            ("bge-m3", LEXICAL),
            ("multilingual-e5-large", "bge-m3"),
        )
    }
    e5 = pairs["multilingual-e5-large + bm25"]["global"]
    gain = round(e5["union"], 3) - round(e5["first"], 3)
    decision = {
        "rule": "oracle(E5, BM25) global Recall@10 at least E5 + 0.05 goes to experiment 2 "
        "(hybrid); otherwise to experiment 3 (cross-reference expansion)",
        "e5_global_recall10": round(e5["first"], 3),
        "oracle_global_recall10": round(e5["union"], 3),
        "gain": round(gain, 3),
        "next": "experiment 2: hybrid" if gain >= GAIN_TO_HYBRID - 1e-9 else "experiment 3",
    }
    out = args.results / f"dev-{version}" / "retrieval-iteration"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": 1,
        "benchmark_version": version,
        "run_sha256": shas,
        "systems": {
            s: {k: v for k, v in d.items() if k != "per_query"} for s, d in diagnostics.items()
        },
        "oracle_unions": {
            pair: {k: v for k, v in block.items() if k != "per_query"}
            for pair, block in pairs.items()
        },
        "decision": decision,
    }
    (out / "experiment-1-bm25.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for pair, block in pairs.items():
        g, lang, dep = block["global"], block["by_language"], block["dependent_evidence"]
        print(
            f"{pair}: global R@10 {g['first']:.3f} -> union {g['union']:.3f}; en "
            f"{lang['first']['en']:.3f} -> {lang['union']['en']:.3f}; es "
            f"{lang['first']['es']:.3f} -> {lang['union']['es']:.3f}; dependent "
            f"{dep['first']:.3f} -> {dep['union']:.3f}; queries gaining "
            f"{block['queries_gaining']['count']}"
        )
        for name, v in block["groups"].items():
            print(f"    {name} ({v['needs']} needs): {v['first']:.3f} -> {v['union']:.3f}")
    print(f"decision: {decision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
