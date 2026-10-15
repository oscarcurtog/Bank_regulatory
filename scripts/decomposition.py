"""D of retrieval iteration 4: requirements from a local model, ranked by E5, merged by a fixed rule
(config/retrieval/decomposition.md).

    PYTHONPATH=. caffeinate -i .venv/bin/python scripts/decomposition.py --expect-version 0.10.0 \\
        --model qwen3-4b [--check]

With --check the run uses the 6 fixture queries (data/benchmark/fixtures, not DEV) and only says
whether the answers are readable, how long they took and how much memory the model needed; for
Qwen3-8B that check is the pilot that decides, by the protocol's limits, whether the 8B runs at
all. Without it, for every DEV query the MLX worker (scripts/mlx_generate.py) writes the
requirements from the question and E5's first 3 results; E5 ranks each requirement over the whole
corpus and ingest/decomposition.py merges them with E5's ranking. The run is written, measured
as every system, paired with E5 and judged by the protocol's gates; the labels are read only
after every ranking exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ingest import benchmark as bm
from ingest import decomposition as dc
from ingest import dense as dn
from ingest import embeddings as em
from ingest import retrieval as rt
from ingest import run_report as rr

E5 = Path("config/embeddings/multilingual-e5-large.json")
MLX_PYTHON = Path(".cache/mlx-venv/bin/python")
MODELS = {
    "qwen3-4b": Path("config/retrieval/qwen3-4b-instruct-mlx.json"),
    "qwen3-8b": Path("config/retrieval/qwen3-8b-mlx.json"),
}
FIXTURES = Path("data/benchmark/fixtures/queries.jsonl")
MAX_NEW_TOKENS = 256
PILOT = {"peak_memory_gib": 10.0, "projected_hours": 3.0}  # the 8B runs only within both
FOLLOWED = ("N022", "N024", "N034", "N035")
GIB = 1024**3
# letters of the Spanish queries that English does not use: a requirement must be in English
SPANISH_LETTERS = {
    chr(c)
    for c in (
        0xF1,
        0xE1,
        0xE9,
        0xED,
        0xF3,
        0xFA,
        0xFC,
        0xBF,
        0xA1,
        0xD1,
        0xC1,
        0xC9,
        0xCD,
        0xD3,
        0xDA,
    )
}


def run_worker(config: Path, prompts: Path, outputs: Path) -> None:
    if not MLX_PYTHON.exists():
        raise SystemExit(f"{MLX_PYTHON} is missing: the MLX environment of D-040 is not set up")
    command = [str(MLX_PYTHON), "scripts/mlx_generate.py", "--config", str(config)]
    command += ["--prompts", str(prompts), "--out", str(outputs)]
    command += ["--max-new-tokens", str(MAX_NEW_TOKENS)]
    subprocess.run(command, check=True, env=os.environ | {"HF_HOME": str(em.hf_home())})


def generate(
    config: Path, run_path: Path, rows: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], str]:
    run_path.mkdir(parents=True, exist_ok=True)
    prompts_path, outputs_path = run_path / "prompts.jsonl", run_path / "outputs.jsonl"
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    prompts_path.write_bytes(data)
    run_worker(config, prompts_path, outputs_path)
    outputs = {
        r["id"]: r for r in map(json.loads, outputs_path.read_text(encoding="utf-8").splitlines())
    }
    if missing := [r["id"] for r in rows if r["id"] not in outputs]:
        raise SystemExit(f"{len(missing)} prompts have no output")
    return outputs, hashlib.sha256(data).hexdigest()


def swap_used() -> str:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True)
    return out.stdout.strip()


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


def requirement_quality(
    parsed: dict[str, dc.Requirements], excerpts: dict[str, list[str]]
) -> dict[str, Any]:
    """What the answers look like, without a label."""
    items = [(qid, x) for qid, p in parsed.items() for x in p.items]
    invented = [
        (qid, x)
        for qid, x in items
        if dc.articles_named(x) - set().union(*(dc.articles_named(t) for t in excerpts[qid]))
    ]
    return {
        "requirements_per_answer": dict(
            sorted(Counter(len(p.items) for p in parsed.values()).items())
        ),
        "answers_unreadable_kept_e5": sorted(q for q, p in parsed.items() if not p.readable),
        "items_dropped": sum(p.dropped for p in parsed.values()),
        "words_per_requirement_median": statistics.median(len(x.split()) for _, x in items)
        if items
        else 0,
        "requirements_not_a_question": sum(not x.endswith("?") for _, x in items),
        "requirements_writing_an_article_number": sum(bool(dc.articles_named(x)) for _, x in items),
        "requirements_writing_an_article_no_excerpt_names": len(invented),
        "requirements_writing_a_year": sum(
            bool(re.search(r"\b(?:19|20)\d\d\b", x)) for _, x in items
        ),
        "requirements_with_spanish_letters": sum(bool(set(x) & SPANISH_LETTERS) for _, x in items),
    }


def check(args: argparse.Namespace, frozen: rt.FrozenInputs, dense: dn.DenseSet) -> int:
    """The fixture queries: readable answers, time and memory; for the 8B, the pilot's limits."""
    queries = [json.loads(line) for line in FIXTURES.read_text(encoding="utf-8").splitlines()]
    vectors = dense.encode_queries([q["text"] for q in queries])
    doc_ids = list(dense.doc_ids)
    rows: list[dict[str, Any]] = []
    excerpts: dict[str, list[str]] = {}
    for q, v in zip(queries, vectors, strict=True):
        top = rt.rank_by_scores(q["query_id"], doc_ids, dense.doc_vectors @ v).chunk_ids
        excerpts[q["query_id"]] = [frozen.chunks[c].text for c in top[: dc.EXCERPTS]]
        rows.append(
            {
                "id": f"check#{q['query_id']}",
                "messages": dc.requirements_prompt(q["text"], excerpts[q["query_id"]]),
            }
        )
    version = frozen.dev.manifest.benchmark_version
    run_path = rt.run_dir(f"d-check-{args.model}", args.max_tokens, version)
    before = swap_used()
    outputs, prompts_sha = generate(MODELS[args.model], run_path, rows)
    after = swap_used()
    parsed = {r["id"].split("#")[1]: dc.parse(outputs[r["id"]]["output"]) for r in rows}
    seconds = [o["seconds"] for o in outputs.values()]
    peak = max(o["peak_memory_bytes"] for o in outputs.values()) / GIB
    projected = max(seconds) * len(frozen.dev.queries) / 3600
    record: dict[str, Any] = {
        "experiment": "D, check on the fixture queries (not DEV)",
        "model": json.loads(MODELS[args.model].read_text(encoding="utf-8")),
        "prompts_sha256": prompts_sha,
        "readable": {q: p.readable for q, p in parsed.items()},
        "quality": requirement_quality(parsed, excerpts),
        "seconds": seconds,
        "prompt_tokens": [o["prompt_tokens"] for o in outputs.values()],
        "new_tokens": [o["new_tokens"] for o in outputs.values()],
        "peak_memory_gib": round(peak, 2),
        "projected_hours_for_dev": round(projected, 2),
        "projection": "the slowest fixture prompt times the DEV queries",
        "swap_before": before,
        "swap_after": after,
        "versions": next(iter(outputs.values()))["versions"],
    }
    if args.model == "qwen3-8b":
        gate = {
            f"peak memory <= {PILOT['peak_memory_gib']} GiB": peak <= PILOT["peak_memory_gib"],
            f"projected <= {PILOT['projected_hours']} h": projected <= PILOT["projected_hours"],
        }
        record |= {"pilot_limits": PILOT, "gate": gate, "passes": all(gate.values())}
    out = args.results / f"dev-{version}" / "retrieval-iteration-4"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"d-check-{args.model}.json").write_text(
        json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for qid, p in parsed.items():
        print(f"{qid}: {len(p.items)} requirements, readable {p.readable}")
        for x in p.items:
            print(f"   - {x}")
    print(
        f"seconds {min(seconds):.1f} to {max(seconds):.1f}; peak {peak:.2f} GiB; projected "
        f"{projected:.2f} h for DEV; swap {before} -> {after}"
        + (f"; pilot passes: {record['passes']}" if "passes" in record else "")
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--benchmark", type=Path, default=bm.DEV_DIR)
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--results", type=Path, default=Path("data/results"))
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--check", action="store_true", help="the fixture queries, not DEV")
    args = parser.parse_args(argv)
    frozen = rt.load_frozen_inputs(args.benchmark, args.expect_version, args.max_tokens)
    version = frozen.dev.manifest.benchmark_version
    dense = dn.load_dense(E5, frozen, args.max_tokens)
    if args.check:
        return check(args, frozen, dense)
    out_dir = args.results / f"dev-{version}" / "retrieval-iteration-4"
    if args.model == "qwen3-8b":
        pilot = out_dir / "d-check-qwen3-8b.json"
        if not pilot.exists() or not json.loads(pilot.read_text(encoding="utf-8"))["passes"]:
            raise SystemExit("the 8B runs only after its pilot (--check) passes the limits")

    # 1. the prompts and the requirements: no label is read here
    e5_ranked, e5_summary = rt.load_system_run(
        "multilingual-e5-large", args.max_tokens, version, args.results
    )
    if dense.rank() != e5_ranked:
        raise SystemExit("the cached E5 vectors do not reproduce E5's recorded run")
    e5_of = {r.query_id: r for r in e5_ranked}
    question = {q.query_id: q.text for q in frozen.dev.queries}
    excerpts = {
        qid: [frozen.chunks[c].text for c in e5_of[qid].chunk_ids[: dc.EXCERPTS]]
        for qid in question
    }
    rows = [
        {"id": f"d#{qid}", "messages": dc.requirements_prompt(q, excerpts[qid])}
        for qid, q in question.items()
    ]
    system = f"d-{args.model}-k{dc.K}"
    outputs, prompts_sha = generate(
        MODELS[args.model], rt.run_dir(system, args.max_tokens, version), rows
    )
    parsed = {qid: dc.parse(outputs[f"d#{qid}"]["output"]) for qid in question}

    # 2. E5 ranks every requirement over the whole corpus; the rule builds each ranking
    texts = sorted({x for p in parsed.values() if p.readable for x in p.items})
    vectors = dict(zip(texts, dense.encode_queries(texts), strict=True))
    doc_ids = list(dense.doc_ids)
    hits: dict[str, list[tuple[str, ...]]] = {}
    for qid, p in parsed.items():
        hits[qid] = [
            rt.rank_by_scores(qid, doc_ids, dense.doc_vectors @ vectors[x]).chunk_ids[: dc.POOL]
            for x in (p.items if p.readable else ())
        ]
    assembled = {qid: dc.assemble(e5_of[qid].chunk_ids, hits[qid]) for qid in question}
    ranked = []
    for qid in question:
        order = assembled[qid].ranking
        n = len(order)
        ranked.append(rt.Ranked(qid, order, tuple(float(n - i) for i in range(n))))
    sha = rt.write_run(
        rt.run_dir(system, args.max_tokens, version) / rt.RUN_FILE, ranked, frozen.evaluable
    )

    # 3. the labels enter here, once every ranking exists
    targets = frozen.targets
    queries = [q for t in targets for q in t.queries]
    rel = {q.query_id: rt.relevant_chunks(t) for t in targets for q in t.queries}
    bodies = {cid: c.body for cid, c in frozen.chunks.items()}
    e5_diag = rr.diagnostics(frozen.dev, frozen.layer, e5_ranked)
    e5_cols = columns_of(e5_diag)
    evaluated = rr.evaluate_ranked(frozen.dev, frozen.layer, ranked, bodies)
    diag = rr.diagnostics(frozen.dev, frozen.layer, ranked)
    cols = columns_of(diag)
    paired = {
        m: rr.paired_bootstrap(
            {
                q.query_id: diag["per_query"][m][q.query_id] - e5_diag["per_query"][m][q.query_id]
                for q in queries
            },
            queries,
        )
        for m in ("Recall@10", "nDCG@10", "Full-evidence@10")
    }
    r, e = ({k: round(v, 3) for k, v in c.items()} for c in (cols, e5_cols))
    signal = {
        "Recall@10 +0.03": r["Recall@10"] - e["Recall@10"] >= 0.03 - 1e-9,
        "Full-evidence@10 +0.03": r["Full-evidence@10"] - e["Full-evidence@10"] >= 0.03 - 1e-9,
        "nDCG@10 >= 0.518": r["nDCG@10"] >= 0.518 - 1e-9,
    }
    success = {
        "useful": r["Recall@10"] >= 0.85
        and r["multi_provision"] >= 0.60
        and r["dependent"] >= 0.45
        and r["Full-evidence@10"] >= 0.75,
        "strong": r["Recall@10"] >= 0.90 and r["nDCG@10"] >= 0.65,
    }
    e5_top = {qid: set(e5_of[qid].chunk_ids[: dc.DEPTH]) for qid in question}
    d_top = {qid: set(assembled[qid].ranking[: dc.DEPTH]) for qid in question}
    gained = {
        q.query_id: sorted((d_top[q.query_id] - e5_top[q.query_id]) & rel[q.query_id])
        for q in queries
    }
    lost = {
        q.query_id: sorted((e5_top[q.query_id] - d_top[q.query_id]) & rel[q.query_id])
        for q in queries
    }

    def pool_recall(pool: dict[str, set[str]]) -> float:
        return rt.macro_by_need(
            {
                q.query_id: len(pool[q.query_id] & rel[q.query_id]) / len(rel[q.query_id])
                for q in queries
            },
            queries,
        )

    pool_k = {qid: {c for h in hits[qid] for c in h[: dc.K]} for qid in question}
    pool_wide = {qid: {c for h in hits[qid] for c in h[: dc.POOL]} for qid in question}
    rank_of = {
        name: {x.query_id: {c: i for i, c in enumerate(x.chunk_ids, 1)} for x in run}
        for name, run in (("e5", e5_ranked), ("d", ranked))
    }
    followed = {
        n: [
            {
                "query_id": q.query_id,
                "requirements": list(parsed[q.query_id].items),
                "relevant_ranks": {
                    c: {k: rank_of[k][q.query_id][c] for k in ("e5", "d")}
                    for c in sorted(rel[q.query_id])
                },
            }
            for q in queries
            if q.need_id == n
        ]
        for n in FOLLOWED
    }
    model = json.loads(MODELS[args.model].read_text(encoding="utf-8"))
    summary = {
        "system": {
            "kind": "decomposition_requirements_merge",
            "rule": "E5's first 6; places 7 to 10 to corroborated E5 ranks 7 to 10, then each "
            "requirement's best chunk outside E5's top 10 in turns, then the rest of E5's 7 to "
            f"10; the 10 in E5's order (K = {dc.K})",
            "model": model,
        },
        "identity": {
            "base_run_sha256": e5_summary["identity"]["run_sha256"],
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
    payload = {
        "experiment": f"D, {args.model}",
        "benchmark_version": version,
        "e5_run_sha256": e5_summary["identity"]["run_sha256"],
        "model": model,
        "prompt": {
            "system": dc.SYSTEM,
            "necessary_evidence": dc.NECESSARY,
            "task": dc.TASK,
            "excerpts": dc.EXCERPTS,
            "sha256": prompts_sha,
        },
        "rule": {"keep": dc.KEEP, "depth": dc.DEPTH, "k": dc.K},
        "system_name": system,
        "run_sha256": sha,
        "columns": cols,
        "e5_columns": e5_cols,
        "by_language_recall10": diag["by_language"]["Recall@10"],
        "paired_against_e5": paired,
        "minimum_signal": signal,
        "signal_found": all(signal.values()),
        "success": success,
        "requirements": requirement_quality(parsed, excerpts),
        "assembly": {
            "queries_changed": sum(bool(a.added) for a in assembled.values()),
            "chunks_added": sum(len(a.added) for a in assembled.values()),
            "e5_ranks_7_to_10_corroborated": sum(len(a.corroborated) for a in assembled.values()),
            "e5_ranks_7_to_10_displaced": sum(len(a.displaced) for a in assembled.values()),
        },
        "evidence": {
            "relevant_pairs_gained": sum(len(v) for v in gained.values()),
            "relevant_pairs_lost": sum(len(v) for v in lost.values()),
            "queries_gaining": sorted(q for q, v in gained.items() if v),
            "queries_losing": sorted(q for q, v in lost.items() if v),
            "recall_requirements_top5_pool": pool_recall(pool_k),
            f"recall_requirements_top{dc.POOL}_pool": pool_recall(pool_wide),
            "recall_e5_top10_plus_top5_pool": pool_recall(
                {qid: e5_top[qid] | pool_k[qid] for qid in question}
            ),
        },
        "followed_needs": followed,
        "performance": {
            "generations": len(outputs),
            "seconds_total": sum(o["seconds"] for o in outputs.values()),
            "seconds_median": statistics.median(o["seconds"] for o in outputs.values()),
            "prompt_tokens_median": statistics.median(o["prompt_tokens"] for o in outputs.values()),
            "new_tokens_median": statistics.median(o["new_tokens"] for o in outputs.values()),
            "peak_memory_gib": round(
                max(o.get("peak_memory_bytes", 0) for o in outputs.values()) / GIB, 2
            ),
            "versions": next(iter(outputs.values()))["versions"],
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"d-{args.model}.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"{system}: R@10 {cols['Recall@10']:.3f} nDCG@10 {cols['nDCG@10']:.3f} full "
        f"{cols['Full-evidence@10']:.3f} multi {cols['multi_provision']:.3f} dependent "
        f"{cols['dependent']:.3f}; paired R@10 {paired['Recall@10']['mean']:+.3f} "
        f"[{paired['Recall@10']['ci95_low']:+.3f}, {paired['Recall@10']['ci95_high']:+.3f}]; "
        f"gained {payload['evidence']['relevant_pairs_gained']}, lost "
        f"{payload['evidence']['relevant_pairs_lost']}; signal {all(signal.values())}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
