"""The leaderboard of the retrieval iteration: every system that ran, measured by one code path.

    PYTHONPATH=. .venv/bin/python scripts/retrieval_leaderboard.py --expect-version 0.10.0

Each system's run is read as its summary recorded it (SHA-256) and measured by
ingest/run_report.diagnostics: the official metrics as global (macro by need) and per-language
values, Recall@50 and Recall@100, the group recalls and the dependent evidence. Before anything
is written, the global values are checked against the summary's own macro block: one
evaluator, one number. Then decision 5 of the protocol for the third dense model, and a paired
bootstrap of the best system against E5.

Writes data/results/dev-<version>/retrieval-iteration/leaderboard.json and leaderboard.md.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

from ingest import benchmark as bm
from ingest import retrieval as rt
from ingest import run_report as rr

SYSTEMS = [
    ("multilingual-e5-large", "dense, E5 (reference)"),
    ("bge-m3", "dense, BGE-M3"),
    ("bm25", "BM25"),
    ("xref1-e5-k5", "E5 + one-hop references, K = 5"),
    ("xref1-e5-k10", "E5 + one-hop references, K = 10"),
    ("qwen3-embedding-0.6b", "dense, Qwen3-Embedding-0.6B"),
    ("acdr-e5-bridge3", "R1: E5 top 20 + dependency pool (candidate list)"),
    ("ppr-e5-n10-f3", "E2: dense-seeded PageRank, 10 seeds, follow 0.3"),
    ("rerank-acdr-e5-bridge3-orig", "E3: cross-encoder on R1's top 50, original query"),
    ("rerank-acdr-e5-bridge3-bridgemax", "E3: cross-encoder, bridge maximum"),
    ("quota2-acdr-e5-bridge3", "E4: 2 top-10 places for dependencies"),
    (
        "c-functional-qwen3-4b",
        "C: legal-function set selection, expanded population (Qwen3-4B, 4-bit)",
    ),
    ("r4a-facts-rrf60", "R4-A: facts rewrite fused with E5 (RRF, k = 60)"),
    ("r4b-catalog-rrf60", "R4-B: facts and closed-catalog concepts fused with E5 (RRF, k = 60)"),
]
REFERENCE = "multilingual-e5-large"
THIRD_MODEL = "qwen3-embedding-0.6b"
COLUMNS = [
    ("nDCG@10", "global", "nDCG@10"),
    ("Recall@10", "global", "Recall@10"),
    ("MRR@10", "global", "MRR@10"),
    ("Recall@20", "global", "Recall@20"),
    ("Recall@T=1000", "global", "Recall@T=1000"),
    ("Recall@T=2000", "global", "Recall@T=2000"),
    ("Recall@50", "global", "Recall@50"),
    ("multi_provision R@10", "groups", "multi_provision"),
    ("cross_reference R@10", "groups", "cross_reference"),
    ("cross_act R@10", "groups", "cross_act"),
    ("dependent R@10", "dependent_evidence", "Recall@10"),
    ("Full-evidence@10", "full_evidence", "Full-evidence@10"),
]
SEED = 20261013


def value(diag: dict[str, Any], where: str, key: str) -> float:
    if where == "groups":
        return float(diag["groups"][key]["Recall@10"])
    return float(diag[where][key])


def bootstrap(
    deltas: dict[str, float], queries: list[bm.Query], reps: int = 10_000
) -> dict[str, float]:
    """Mean per-query delta, with a 95% interval from resampling needs (a need's queries
    are not independent)."""
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--benchmark", type=Path, default=bm.DEV_DIR)
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--results", type=Path, default=Path("data/results"))
    args = parser.parse_args(argv)
    frozen = rt.load_frozen_inputs(args.benchmark, args.expect_version, args.max_tokens)
    version = frozen.dev.manifest.benchmark_version
    queries = [q for t in frozen.targets for q in t.queries]

    rows: dict[str, dict[str, Any]] = {}
    per_query: dict[str, dict[str, dict[str, float]]] = {}
    for system, label in SYSTEMS:
        ranked, summary = rt.load_system_run(system, args.max_tokens, version, args.results)
        diag = rr.diagnostics(frozen.dev, frozen.layer, ranked)
        macro = summary["metrics"]["macro_by_need"]
        for m in rr.METRICS:
            if abs(diag["global"][m] - macro[m]) > 1e-12:
                raise SystemExit(
                    f"{system}: {m} {diag['global'][m]} here, {macro[m]} in its summary"
                )
        rows[system] = {
            "label": label,
            "run_sha256": summary["identity"]["run_sha256"],
            "columns": {name: value(diag, where, key) for name, where, key in COLUMNS},
            "Recall@100": diag["global"]["Recall@100"],
            "by_language": {
                m: diag["by_language"][m] for m in ("nDCG@10", "Recall@10", "Recall@T=1000")
            },
            "dependent_evidence": {
                k: v for k, v in diag["dependent_evidence"].items() if k != "per_query"
            },
            "groups": diag["groups"],
        }
        per_query[system] = diag["per_query"]

    ref = rows[REFERENCE]["columns"]
    third = rows[THIRD_MODEL]["columns"]
    gain = round(third["Recall@10"], 3) - round(ref["Recall@10"], 3)
    decision_5 = {
        "rule": "the third dense model must reach E5 + 0.05 global Recall@10; otherwise dense "
        "model swapping stops",
        "e5_global_recall10": round(ref["Recall@10"], 3),
        "third_model_global_recall10": round(third["Recall@10"], 3),
        "gain": round(gain, 3),
        "outcome": "keep testing dense models"
        if gain >= 0.05 - 1e-9
        else "stop testing dense embedding models: model swapping has diminishing returns",
    }
    reranker_gate = {s: round(r["columns"]["Recall@50"], 3) for s, r in rows.items()}

    def key(system: str) -> tuple[float, float]:
        c = rows[system]["columns"]
        return (round(c["Recall@10"], 3), round(c["nDCG@10"], 3))

    best = max(rows, key=key)
    paired: dict[str, dict[str, Any]] = {}
    interesting: dict[str, dict[str, bool]] = {}
    for system in rows:
        if system == REFERENCE:
            continue
        paired[system] = {}
        for m in ("Recall@10", "nDCG@10"):
            deltas = {
                q.query_id: per_query[system][m][q.query_id] - per_query[REFERENCE][m][q.query_id]
                for q in queries
            }
            paired[system][m] = bootstrap(deltas, queries)
        c = rows[system]["columns"]

        def d(name: str, c: dict[str, float] = c) -> float:
            return float(round(c[name], 3) - round(ref[name], 3))

        interesting[system] = {
            "A: Recall@10 +0.05": d("Recall@10") >= 0.05 - 1e-9,
            "B: nDCG@10 +0.05, Recall@10 down at most 0.02": d("nDCG@10") >= 0.05 - 1e-9
            and d("Recall@10") >= -0.02 - 1e-9,
            "C: multi_provision Recall@10 +0.10": d("multi_provision R@10") >= 0.10 - 1e-9,
            "D: dependent evidence Recall@10 +0.10": d("dependent R@10") >= 0.10 - 1e-9,
        }
    payload = {
        "benchmark_version": version,
        "systems": rows,
        "best_by_global_recall10_then_ndcg10": best,
        "paired_against_e5": paired,
        "interesting_against_e5": interesting,
        "decision_5": decision_5,
        "reranker_gate_recall50": reranker_gate,
        "reranker_gate_met_by": [s for s, v in reranker_gate.items() if v >= 0.95],
    }
    out = args.results / f"dev-{version}" / "retrieval-iteration"
    out.mkdir(parents=True, exist_ok=True)
    (out / "leaderboard.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    names = [name for name, _, _ in COLUMNS]
    lines = [
        f"# Retrieval leaderboard, DEV {version}",
        "",
        "Every system ranks all 1,354 chunks for the 81 scored queries; one code path measures "
        "them all (`ingest/run_report.py`). Global values are macro means by need over the 27 "
        "evaluable needs, mixing English and Spanish (descriptive under D-020). `cross_act` has "
        "2 needs and is a diagnostic only. Dependent evidence: the grade-2 chunks of the 11 "
        "needs that also have a grade-3 chunk.",
        "",
        "| System | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for r in rows.values():
        c = r["columns"]
        lines.append(f"| {r['label']} | " + " | ".join(f"{c[n]:.3f}" for n in names) + " |")
    lines += ["", "Per language (official, D-020), nDCG@10 and Recall@10:", ""]
    lines += [
        "| System | EN nDCG@10 | ES nDCG@10 | EN Recall@10 | ES Recall@10 |",
        "|---|---|---|---|---|",
    ]
    for r in rows.values():
        b = r["by_language"]
        lines.append(
            f"| {r['label']} | {b['nDCG@10']['en']:.3f} | {b['nDCG@10']['es']:.3f} | "
            f"{b['Recall@10']['en']:.3f} | {b['Recall@10']['es']:.3f} |"
        )
    lines += [
        "",
        f"Best by global Recall@10, then nDCG@10: {rows[best]['label']}. This is an ordering, "
        "not a decision: each iteration's protocol fixes its own rules for keeping a system.",
        "",
        "Each system against E5, paired by query (81 queries), with a 95% bootstrap interval over "
        "needs, and the protocol's four tests of an interesting configuration (A to D):",
        "",
        "| System | Recall@10 delta | nDCG@10 delta | Queries better / equal / worse (nDCG@10) "
        "| Interesting |",
        "|---|---|---|---|---|",
    ]
    for system, p in paired.items():
        r10, nd = p["Recall@10"], p["nDCG@10"]
        passed = [t.split(":")[0] for t, ok in interesting[system].items() if ok] or ["none"]
        lines.append(
            f"| {rows[system]['label']} | {r10['mean']:+.3f} [{r10['ci95_low']:+.3f}, "
            f"{r10['ci95_high']:+.3f}] | {nd['mean']:+.3f} [{nd['ci95_low']:+.3f}, "
            f"{nd['ci95_high']:+.3f}] | {nd['better']} / {nd['equal']} / {nd['worse']} | "
            f"{', '.join(passed)} |"
        )
    lines += [
        "",
        f"Decision 5: {decision_5['outcome']} (third model {decision_5['gain']:+.3f} global "
        "Recall@10 against E5; the rule asks for +0.05).",
        "",
        "Reranker gate (Recall@50 >= 0.95): "
        + ", ".join(f"{rows[s]['label']} {v:.3f}" for s, v in reranker_gate.items())
        + ("; met by none." if not payload["reranker_gate_met_by"] else "."),
        "",
    ]
    (out / "leaderboard.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
