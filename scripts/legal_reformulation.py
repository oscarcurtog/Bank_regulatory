"""Round 4: closed-catalog legal reformulation for first-hop failures
(config/retrieval/necessity-selection.md, round 4).

    PYTHONPATH=. caffeinate -i .venv/bin/python scripts/legal_reformulation.py \\
        --expect-version 0.10.0

The catalog (every article title and every defined term of the corpus) is built once. For every
query the MLX worker (scripts/mlx_generate.py) rewrites the question twice: R4-A, its legally
material facts; R4-B, the facts and at most 3 catalog concepts, each with the words of the question
that motivate it. E5 ranks each rewrite (its cached document vectors, the rewrite encoded as E5
encodes a query), and that ranking is fused with E5's ranking of the original question by
Reciprocal Rank Fusion, k = 60, equal weights. A query whose rewrite holds neither facts nor
concepts keeps E5's ranking, and is counted. Each variant is written as a run, measured as every
system, paired with E5 and judged by the round's gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ingest import benchmark as bm
from ingest import dense as dn
from ingest import dependencies as dp
from ingest import embeddings as em
from ingest import reformulation as rf
from ingest import retrieval as rt
from ingest import run_report as rr

E5 = Path("config/embeddings/multilingual-e5-large.json")
MLX_PYTHON = Path(".cache/mlx-venv/bin/python")
MODEL = Path("config/retrieval/qwen3-4b-instruct-mlx.json")
NAME = "r4-reformulation"
MAX_NEW_TOKENS = 256
VARIANTS = {"r4a": "r4a-facts-rrf60", "r4b": "r4b-catalog-rrf60"}
FOLLOWED = ("N022", "N024", "N034", "N035")


def run_worker(prompts: Path, outputs: Path) -> None:
    if not MLX_PYTHON.exists():
        raise SystemExit(f"{MLX_PYTHON} is missing: the MLX environment of D-040 is not set up")
    command = [
        str(MLX_PYTHON),
        "scripts/mlx_generate.py",
        "--config",
        str(MODEL),
        "--prompts",
        str(prompts),
        "--out",
        str(outputs),
        "--max-new-tokens",
        str(MAX_NEW_TOKENS),
    ]
    subprocess.run(command, check=True, env=os.environ | {"HF_HOME": str(em.hf_home())})


def columns_of(d: dict[str, Any]) -> dict[str, float]:
    return {
        "nDCG@10": d["global"]["nDCG@10"],
        "Recall@10": d["global"]["Recall@10"],
        "Recall@20": d["global"]["Recall@20"],
        "MRR@10": d["global"]["MRR@10"],
        "Recall@T=1000": d["global"]["Recall@T=1000"],
        "Recall@T=2000": d["global"]["Recall@T=2000"],
        "Full-evidence@10": d["full_evidence"]["Full-evidence@10"],
        "multi_provision": d["groups"]["multi_provision"]["Recall@10"],
        "cross_reference": d["groups"]["cross_reference"]["Recall@10"],
        "cross_act": d["groups"]["cross_act"]["Recall@10"],
        "dependent": d["dependent_evidence"]["Recall@10"],
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

    # 1. the catalog and the prompts: no label is read here
    corpus = bm.load_corpus()
    texts = {
        (act, n.path): n.own_text for act, d in corpus.items() for a in d.articles for n in a.nodes
    }
    entries = rf.catalog(corpus, dp.glossary(texts))
    catalog_sha = hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()
    question = {q.query_id: q.text for q in frozen.dev.queries}
    rows = [{"id": f"r4a#{qid}", "messages": rf.facts_prompt(q)} for qid, q in question.items()]
    rows += [
        {"id": f"r4b#{qid}", "messages": rf.concepts_prompt(q, entries)}
        for qid, q in question.items()
    ]
    run_path = rt.run_dir(NAME, args.max_tokens, version)
    run_path.mkdir(parents=True, exist_ok=True)
    prompts_path, outputs_path = run_path / "prompts.jsonl", run_path / "outputs.jsonl"
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    prompts_path.write_bytes(data)
    prompts_sha = hashlib.sha256(data).hexdigest()
    run_worker(prompts_path, outputs_path)
    outputs = {
        r["id"]: r for r in map(json.loads, outputs_path.read_text(encoding="utf-8").splitlines())
    }
    if missing := [r["id"] for r in rows if r["id"] not in outputs]:
        raise SystemExit(f"{len(missing)} prompts have no output")

    # 2. the rewrites, ranked by E5 and fused with E5's ranking of the original question
    dense = dn.load_dense(E5, frozen, args.max_tokens)
    base = dense.rank()
    e5_ranked, e5_summary = rt.load_system_run(
        "multilingual-e5-large", args.max_tokens, version, args.results
    )
    if base != e5_ranked:
        raise SystemExit("the cached E5 vectors do not reproduce E5's recorded run")
    e5_of = {r.query_id: r for r in e5_ranked}
    doc_ids = list(dense.doc_ids)
    reformulated: dict[str, dict[str, rf.Reformulation]] = {}
    for variant in VARIANTS:
        reformulated[variant] = {
            qid: rf.parse(
                outputs[f"{variant}#{qid}"]["output"],
                q,
                entries if variant == "r4b" else None,
            )
            for qid, q in question.items()
        }
    to_encode = sorted(
        {rf.retrieval_text(r) for v in reformulated.values() for r in v.values()} - {""}
    )
    vectors = dict(zip(to_encode, dense.encode_queries(to_encode), strict=True))
    targets = frozen.targets
    queries = [q for t in targets for q in t.queries]
    e5_diag = rr.diagnostics(frozen.dev, frozen.layer, e5_ranked)
    e5_cols = columns_of(e5_diag)
    bodies = {cid: c.body for cid, c in frozen.chunks.items()}
    results: dict[str, Any] = {}
    rank_of: dict[str, dict[str, dict[str, int]]] = {
        "e5": {r.query_id: {c: i for i, c in enumerate(r.chunk_ids, 1)} for r in e5_ranked}
    }
    for variant, system in VARIANTS.items():
        ranked, kept_e5 = [], []
        for qid in question:
            text = rf.retrieval_text(reformulated[variant][qid])
            if not text:
                kept_e5.append(qid)
                ranked.append(e5_of[qid])
                continue
            rewrite = rt.rank_by_scores(qid, doc_ids, dense.doc_vectors @ vectors[text])
            ranked.append(
                rt.rrf(qid, [e5_of[qid].chunk_ids, rewrite.chunk_ids], [1.0, 1.0], rf.RRF_K)
            )
        sha = rt.write_run(
            rt.run_dir(system, args.max_tokens, version) / rt.RUN_FILE, ranked, frozen.evaluable
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
        cols = columns_of(diag)
        r, e = (
            {k: round(v, 3) for k, v in cols.items()},
            {k: round(v, 3) for k, v in e5_cols.items()},
        )
        gate = {
            "Recall@10 +0.03": r["Recall@10"] - e["Recall@10"] >= 0.03 - 1e-9,
            "nDCG@10 >= 0.518": r["nDCG@10"] >= 0.518 - 1e-9,
            "Full-evidence@10 down at most 0.01": r["Full-evidence@10"]
            >= e["Full-evidence@10"] - 0.01 - 1e-9,
        }
        success = {
            "useful": r["Recall@10"] >= 0.85
            and r["multi_provision"] >= 0.60
            and r["dependent"] >= 0.45
            and r["Full-evidence@10"] >= 0.75,
            "strong": r["Recall@10"] >= 0.90 and r["nDCG@10"] >= 0.65,
        }
        rank_of[variant] = {
            x.query_id: {c: i for i, c in enumerate(x.chunk_ids, 1)} for x in ranked
        }
        reforms = list(reformulated[variant].values())
        stats: dict[str, Any] = {
            "queries_keeping_e5_ranking": len(kept_e5),
            "facts_per_query_median": statistics.median(len(x.facts) for x in reforms),
        }
        if variant == "r4b":
            concepts = [c for x in reforms for c in x.concepts]
            stats |= {
                "concepts_per_query": dict(
                    sorted(Counter(len(x.concepts) for x in reforms).items())
                ),
                "concepts_kept": len(concepts),
                "concepts_rejected": sum(len(x.rejected) for x in reforms),
                "fragments_found_in_the_question": sum(c.fragment_in_question for c in concepts),
                "most_chosen": Counter(c.entry for c in concepts).most_common(15),
            }
        summary = {
            "system": {
                "kind": "closed_catalog_reformulation"
                if variant == "r4b"
                else "facts_reformulation",
                "fusion": f"RRF k={rf.RRF_K}, equal weights, with E5 on the original question",
                "model": json.loads(MODEL.read_text(encoding="utf-8")),
            },
            "identity": {
                "base_run_sha256": e5_summary["identity"]["run_sha256"],
                "catalog_sha256": catalog_sha,
                "prompts_sha256": prompts_sha,
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
        sys_dir = args.results / f"dev-{version}" / f"{system}-layer-{args.max_tokens}"
        sys_dir.mkdir(parents=True, exist_ok=True)
        (sys_dir / "summary.json").write_text(
            json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        results[system] = {
            "variant": variant,
            "run_sha256": sha,
            "columns": cols,
            "by_language_recall10": diag["by_language"]["Recall@10"],
            "paired_against_e5": paired,
            "gate": gate,
            "kept": all(gate.values()),
            "success": success,
            "rewrites": stats,
        }
        print(
            f"{system}: R@10 {cols['Recall@10']:.3f} nDCG@10 {cols['nDCG@10']:.3f} full "
            f"{cols['Full-evidence@10']:.3f}; paired R@10 {paired['Recall@10']['mean']:+.3f} "
            f"[{paired['Recall@10']['ci95_low']:+.3f}, {paired['Recall@10']['ci95_high']:+.3f}]; "
            f"kept {all(gate.values())}",
            flush=True,
        )
    followed: dict[str, list[dict[str, Any]]] = {}
    for target in targets:
        if target.need.need_id not in FOLLOWED:
            continue
        for q in target.queries:
            qid = q.query_id
            followed.setdefault(target.need.need_id, []).append(
                {
                    "query_id": qid,
                    "lang": q.lang,
                    "relevant_ranks": {
                        c: {k: rank_of[k][qid][c] for k in ("e5", "r4a", "r4b")}
                        for c in sorted(rt.relevant_chunks(target))
                    },
                    "r4b_concepts": [c.entry for c in reformulated["r4b"][qid].concepts],
                }
            )
    payload = {
        "experiment": "R4",
        "benchmark_version": version,
        "e5_run_sha256": e5_summary["identity"]["run_sha256"],
        "catalog": {"entries": len(entries), "sha256": catalog_sha},
        "prompts": {
            "facts": rf.FACTS_TASK,
            "concepts": rf.CONCEPTS_TASK,
            "sha256": prompts_sha,
        },
        "e5_columns": e5_cols,
        "systems": results,
        "followed_needs": followed,
        "performance": {
            "generations": len(outputs),
            "seconds_total": sum(o["seconds"] for o in outputs.values()),
            "seconds_median": statistics.median(o["seconds"] for o in outputs.values()),
            "prompt_tokens_median": statistics.median(o["prompt_tokens"] for o in outputs.values()),
            "reused_prefix_tokens_total": sum(o["reused_prefix_tokens"] for o in outputs.values()),
            "versions": next(iter(outputs.values()))["versions"],
        },
    }
    out_dir = args.results / f"dev-{version}" / "retrieval-iteration-3"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "r4-reformulation.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
