"""E4 of retrieval iteration 2: a fixed quota of the top 10 for the best dependency candidates
(config/retrieval/dependency-retrieval.md).

    PYTHONPATH=. .venv/bin/python scripts/dependency_quota.py --expect-version 0.10.0

E5's recorded run and R1's selected candidate list (both checked by SHA-256): E5's top 10 - s,
then the first s chunks of R1's list after its position 20, then E5's ranks 11 - s to 20, then the
rest of R1's list. Each s of the grid is written as a run, measured by ingest/run_report.py and
paired with E5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ingest import benchmark as bm
from ingest import embeddings as em
from ingest import retrieval as rt
from ingest import run_report as rr

GRID = (2, 3)
KEEP = 20
E5_REFERENCE = {"Recall@10": 0.763, "nDCG@10": 0.528, "multi_provision": 0.481, "dependent": 0.300}


def quota(e5: rt.Ranked, candidates: rt.Ranked, s: int) -> rt.Ranked:
    head = list(e5.chunk_ids[: 10 - s])
    slots = [c for c in candidates.chunk_ids[KEEP:] if c not in head][:s]
    order = head + slots
    placed = set(order)
    order += [c for c in e5.chunk_ids[10 - s : KEEP] if c not in placed]
    placed = set(order)
    order += [c for c in candidates.chunk_ids if c not in placed]
    n = len(order)
    return rt.Ranked(e5.query_id, tuple(order), tuple(float(n - i) for i in range(n)))


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
    r1_name = f"acdr-e5-bridge{r1['decision']['selected_k']}"
    lists, _ = rt.read_run(
        rt.run_dir(r1_name, args.max_tokens, version) / rt.RUN_FILE,
        r1["configurations"][r1_name]["run_sha256"],
    )
    by_query = {r.query_id: r for r in lists}
    e5_ranked, e5_summary = rt.load_system_run(
        "multilingual-e5-large", args.max_tokens, version, args.results
    )
    queries = [q for t in frozen.targets for q in t.queries]
    e5_diag = rr.diagnostics(frozen.dev, frozen.layer, e5_ranked)
    bodies = {cid: c.body for cid, c in frozen.chunks.items()}
    configurations: dict[str, dict[str, Any]] = {}
    for s in GRID:
        name = f"quota{s}-{r1_name}"
        ranked = [quota(r, by_query[r.query_id], s) for r in e5_ranked]
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
        d = {k: round(columns[k], 3) - E5_REFERENCE[k] for k in E5_REFERENCE}
        material = {
            "A: Recall@10 +0.05": d["Recall@10"] >= 0.05 - 1e-9,
            "B: nDCG@10 +0.05, Recall@10 down at most 0.02": d["nDCG@10"] >= 0.05 - 1e-9
            and d["Recall@10"] >= -0.02 - 1e-9,
            "C: multi_provision Recall@10 +0.10": d["multi_provision"] >= 0.10 - 1e-9,
            "D: dependent evidence Recall@10 +0.10": d["dependent"] >= 0.10 - 1e-9,
        }
        summary = {
            "system": {"kind": "dependency_quota", "candidates": r1_name, "slots": s},
            "identity": {
                "base_run_sha256": e5_summary["identity"]["run_sha256"],
                "candidates_run_sha256": r1["configurations"][r1_name]["run_sha256"],
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
        }
        out = args.results / f"dev-{version}" / f"{name}-layer-{args.max_tokens}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        configurations[name] = {
            "slots": s,
            "columns": columns,
            "recall20": diag["global"]["Recall@20"],
            "cross_reference": diag["groups"]["cross_reference"]["Recall@10"],
            "cross_act": diag["groups"]["cross_act"]["Recall@10"],
            "by_language_recall10": diag["by_language"]["Recall@10"],
            "paired_against_e5": paired,
            "material": material,
        }
        print(
            f"{name}: R@10 {columns['Recall@10']:.3f} nDCG@10 {columns['nDCG@10']:.3f}; "
            f"multi_provision {columns['multi_provision']:.3f}, dependent "
            f"{columns['dependent']:.3f}; paired R@10 {paired['Recall@10']['mean']:+.3f} "
            f"[{paired['Recall@10']['ci95_low']:+.3f}, {paired['Recall@10']['ci95_high']:+.3f}]"
        )

    def key(name: str) -> tuple[float, float, int]:
        c = configurations[name]["columns"]
        return (round(c["Recall@10"], 3), round(c["nDCG@10"], 3), -list(configurations).index(name))

    chosen = max(configurations, key=key)
    cols: dict[str, float] = configurations[chosen]["columns"]
    passed = [t for t, ok in configurations[chosen]["material"].items() if ok]
    decision = {
        "selected": chosen,
        "passed": passed,
        "kept": bool(passed),
        "stop_A_strong": cols["Recall@10"] >= 0.90 and cols["nDCG@10"] >= 0.65,
        "stop_B_useful": cols["Recall@10"] >= 0.85
        and round(cols["multi_provision"], 3) - E5_REFERENCE["multi_provision"] >= 0.10 - 1e-9
        and round(cols["dependent"], 3) - E5_REFERENCE["dependent"] >= 0.10 - 1e-9,
    }
    payload = {
        "experiment": "E4",
        "benchmark_version": version,
        "candidates": r1_name,
        "configurations": configurations,
        "decision": decision,
    }
    (out_dir / "e4-quota.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"decision: {decision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
