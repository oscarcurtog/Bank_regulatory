"""E of retrieval iteration 4: D-4B's requirements, each one's first 50 E5 results reranked by
Qwen3-Reranker-0.6B zero-shot, then D's rule unchanged (config/retrieval/requirement-reranking.md).

    PYTHONPATH=. caffeinate -i .venv/bin/python scripts/requirement_reranking.py \\
        --expect-version 0.10.0 [--check]

With --check, the 4B's requirements for the fixture queries (not DEV) time the reranker: the first
2 requirements, 50 chunks each; the seconds per input, times the DEV inputs, give the projection,
and past the protocol's limit E does not run. Without it: D-4B's recorded answers (their prompts
checked against D-4B's record), D's parser, and E5's first 50 chunks for each requirement. D's
recorded ranking is rebuilt first, exactly, from E5's order. Every (requirement, chunk) pair is
scored once and appended to data/runs/e-qwen3-reranker-0.6b-layer-<n>-dev-<version>/scores.jsonl
as it comes (a rerun scores only what is missing and refuses a stored score whose input changed);
each requirement's 50 are reordered by the scores; D's rule builds the ten from the reranked lists.
The run is written, measured, paired with E5 and with D-4B, and judged by the protocol; the labels
are read only after every ranking exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from ingest import benchmark as bm
from ingest import decomposition as dc
from ingest import dense as dn
from ingest import embeddings as em
from ingest import necessity as nc
from ingest import requirement_rerank as rq
from ingest import retrieval as rt
from ingest import run_report as rr

E5 = Path("config/embeddings/multilingual-e5-large.json")
RERANKER = Path("config/retrieval/qwen3-reranker-0.6b.json")
D_SYSTEM = "d-qwen3-4b-k5"
SYSTEM = "e-d4b-qwen3-reranker-k5"
NAME = "e-qwen3-reranker-0.6b"
LIMIT_HOURS = 4.0
CHECK_REQUIREMENTS = 2


def input_sha(fields: tuple[str, str, str]) -> str:
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode("utf-8")).hexdigest()


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


def requirement_hits(
    dense: dn.DenseSet, answers: dict[str, dc.Requirements]
) -> dict[str, list[tuple[str, ...]]]:
    """E5's first 50 chunks for each readable requirement, in the model's order."""
    texts = sorted({x for p in answers.values() if p.readable for x in p.items})
    vectors = dict(zip(texts, dense.encode_queries(texts), strict=True))
    doc_ids = list(dense.doc_ids)
    return {
        qid: [
            rt.rank_by_scores(qid, doc_ids, dense.doc_vectors @ vectors[x]).chunk_ids[: rq.DEPTH]
            for x in (p.items if p.readable else ())
        ]
        for qid, p in answers.items()
    }


def answers_of(outputs_path: Path) -> dict[str, dc.Requirements]:
    rows = map(json.loads, outputs_path.read_text(encoding="utf-8").splitlines())
    return {r["id"].split("#")[1]: dc.parse(r["output"]) for r in rows}


def jobs_of(
    answers: dict[str, dc.Requirements],
    hits: dict[str, list[tuple[str, ...]]],
    frozen: rt.FrozenInputs,
) -> dict[tuple[str, str], tuple[str, str, str]]:
    """Every (requirement, chunk) pair to score, once, with the reranker's three fields."""
    jobs: dict[tuple[str, str], tuple[str, str, str]] = {}
    for qid, p in answers.items():
        for x, ranking in zip(p.items if p.readable else (), hits[qid], strict=True):
            for c in ranking:
                jobs.setdefault((x, c), (rq.INSTRUCTION, x, frozen.chunks[c].text))
    return jobs


def score_all(
    reranker: nc.NecessityReranker,
    jobs: dict[tuple[str, str], tuple[str, str, str]],
    scores_path: Path | None,
) -> tuple[dict[tuple[str, str], dict[str, Any]], float, int]:
    """Each pair's score, the stored ones reused; one shared opening per requirement."""
    done: dict[tuple[str, str], dict[str, Any]] = {}
    if scores_path is not None and scores_path.exists():
        for line in scores_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (row["requirement"], row["chunk"])
            if key not in jobs or row["input_sha256"] != input_sha(jobs[key]):
                raise SystemExit(f"{scores_path}: a stored score has another input: {key}")
            done[key] = row
    pending = [k for k in jobs if k not in done]
    groups: dict[str, list[tuple[str, str]]] = {}
    for k in pending:
        groups.setdefault(k[0], []).append(k)
    start = time.perf_counter()
    scored = 0
    for g, part in enumerate(groups.values(), start=1):
        prompts = [reranker.prompt(*jobs[k]) for k in part]
        values = reranker.score_shared(prompts)
        rows: list[dict[str, Any]] = [
            {
                "requirement": k[0],
                "chunk": k[1],
                "input_sha256": input_sha(jobs[k]),
                "tokens": len(p.ids),
                "truncated": p.truncated,
                "logit_diff": float(v),
            }
            for k, p, v in zip(part, prompts, values, strict=True)
        ]
        if scores_path is not None:
            with scores_path.open("a", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        done |= {(str(r["requirement"]), str(r["chunk"])): r for r in rows}
        scored += len(part)
        if g % 20 == 0 or g == len(groups):
            elapsed = time.perf_counter() - start
            print(
                f"scored {scored}/{len(pending)} (requirement {g}/{len(groups)}) in "
                f"{elapsed:.0f} s, about {elapsed / scored * (len(pending) - scored) / 60:.0f} "
                "min left",
                flush=True,
            )
    return done, time.perf_counter() - start, len(pending)


def reranked(
    answers: dict[str, dc.Requirements],
    hits: dict[str, list[tuple[str, ...]]],
    done: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, list[tuple[str, ...]]]:
    out = {}
    for qid, p in answers.items():
        out[qid] = [
            tuple(rq.reorder(ranking, {c: done[(x, c)]["logit_diff"] for c in ranking}))
            for x, ranking in zip(p.items if p.readable else (), hits[qid], strict=True)
        ]
    return out


def check(
    args: argparse.Namespace, frozen: rt.FrozenInputs, dense: dn.DenseSet, dev_inputs: int
) -> int:
    version = frozen.dev.manifest.benchmark_version
    fixture_run = rt.run_dir("d-check-qwen3-4b", args.max_tokens, version) / "outputs.jsonl"
    answers = answers_of(fixture_run)
    first = [(qid, x) for qid, p in answers.items() for x in p.items][:CHECK_REQUIREMENTS]
    texts = [x for _, x in first]
    doc_ids = list(dense.doc_ids)
    rankings = [
        rt.rank_by_scores(qid, doc_ids, dense.doc_vectors @ v).chunk_ids[: rq.DEPTH]
        for (qid, _), v in zip(first, dense.encode_queries(texts), strict=True)
    ]
    jobs = {
        (x, c): (rq.INSTRUCTION, x, frozen.chunks[c].text)
        for x, ranking in zip(texts, rankings, strict=True)
        for c in ranking
    }
    start = time.perf_counter()
    reranker = nc.NecessityReranker(nc.load_config(RERANKER))
    load_seconds = time.perf_counter() - start
    done, seconds, scored = score_all(reranker, jobs, None)
    per_input = seconds / scored
    projected = per_input * dev_inputs / 3600
    tokens = [r["tokens"] for r in done.values()]
    record = {
        "experiment": "E, check on the fixture queries' requirements (not DEV)",
        "reranker": nc.load_config(RERANKER).model_dump(mode="json"),
        "instruction": rq.INSTRUCTION,
        "requirements": texts,
        "inputs": scored,
        "seconds": seconds,
        "seconds_per_input": per_input,
        "tokens": {"median": statistics.median(tokens), "max": max(tokens)},
        "truncated": sum(r["truncated"] for r in done.values()),
        "model_load_seconds": load_seconds,
        "dev_inputs": dev_inputs,
        "projected_hours_for_dev": round(projected, 2),
        "limit_hours": LIMIT_HOURS,
        "passes": projected <= LIMIT_HOURS,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "environment": em.environment_description(),
    }
    out = args.results / f"dev-{version}" / "retrieval-iteration-4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e-check.json").write_text(
        json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"{scored} inputs in {seconds:.1f} s ({per_input:.3f} s each, {max(tokens)} tokens at "
        f"most, {record['truncated']} truncated); {dev_inputs} DEV inputs: {projected:.2f} h "
        f"projected; passes {record['passes']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--benchmark", type=Path, default=bm.DEV_DIR)
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--results", type=Path, default=Path("data/results"))
    parser.add_argument("--check", action="store_true", help="fixture requirements, not DEV")
    args = parser.parse_args(argv)
    frozen = rt.load_frozen_inputs(args.benchmark, args.expect_version, args.max_tokens)
    version = frozen.dev.manifest.benchmark_version
    out_dir = args.results / f"dev-{version}" / "retrieval-iteration-4"

    # 1. D-4B as it ran: its answers, its requirements' first 50, its ranking rebuilt exactly
    dense = dn.load_dense(E5, frozen, args.max_tokens)
    e5_ranked, e5_summary = rt.load_system_run(
        "multilingual-e5-large", args.max_tokens, version, args.results
    )
    if dense.rank() != e5_ranked:
        raise SystemExit("the cached E5 vectors do not reproduce E5's recorded run")
    e5_of = {r.query_id: r for r in e5_ranked}
    d_record = json.loads((out_dir / "d-qwen3-4b.json").read_text(encoding="utf-8"))
    d_run = rt.run_dir(D_SYSTEM, args.max_tokens, version)
    prompts_sha = hashlib.sha256((d_run / "prompts.jsonl").read_bytes()).hexdigest()
    if prompts_sha != d_record["prompt"]["sha256"]:
        raise SystemExit("D-4B's prompts are not the ones its record names")
    answers = answers_of(d_run / "outputs.jsonl")
    if set(answers) != {q.query_id for q in frozen.dev.queries}:
        raise SystemExit("D-4B's answers do not cover the DEV queries")
    hits = requirement_hits(dense, answers)
    d_ranked, d_summary = rt.load_system_run(D_SYSTEM, args.max_tokens, version, args.results)
    for r in d_ranked:
        if dc.assemble(e5_of[r.query_id].chunk_ids, hits[r.query_id]).ranking != r.chunk_ids:
            raise SystemExit(f"{r.query_id}: D-4B's ranking is not rebuilt from its requirements")
    jobs = jobs_of(answers, hits, frozen)
    if args.check:
        return check(args, frozen, dense, len(jobs))
    check_record = json.loads((out_dir / "e-check.json").read_text(encoding="utf-8"))
    if not check_record["passes"]:
        raise SystemExit("E runs only after its check projects at most the protocol's limit")

    # 2. the scores, then each requirement's 50 reordered and D's rule on them
    config = nc.load_config(RERANKER)
    run_path = rt.run_dir(NAME, args.max_tokens, version)
    run_path.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    reranker = nc.NecessityReranker(config)
    load_seconds = time.perf_counter() - start
    done, scoring_seconds, scored = score_all(reranker, jobs, run_path / "scores.jsonl")
    hits_rr = reranked(answers, hits, done)
    question = [q.query_id for q in frozen.dev.queries]
    assembled = {qid: dc.assemble(e5_of[qid].chunk_ids, hits_rr[qid]) for qid in question}
    ranked = []
    for qid in question:
        order = assembled[qid].ranking
        n = len(order)
        ranked.append(rt.Ranked(qid, order, tuple(float(n - i) for i in range(n))))
    sha = rt.write_run(
        rt.run_dir(SYSTEM, args.max_tokens, version) / rt.RUN_FILE, ranked, frozen.evaluable
    )

    # 3. the labels enter here, once every ranking exists
    targets = frozen.targets
    queries = [q for t in targets for q in t.queries]
    rel = {q.query_id: rt.relevant_chunks(t) for t in targets for q in t.queries}
    grade = {q.query_id: dict(t.chunk_grades) for t in targets for q in t.queries}
    bodies = {cid: c.body for cid, c in frozen.chunks.items()}
    diag = {
        name: rr.diagnostics(frozen.dev, frozen.layer, run)
        for name, run in (("e5", e5_ranked), ("d", d_ranked), ("e", ranked))
    }
    cols = {name: columns_of(d) for name, d in diag.items()}
    evaluated = rr.evaluate_ranked(frozen.dev, frozen.layer, ranked, bodies)

    def paired(a: str, b: str) -> dict[str, Any]:
        return {
            m: rr.paired_bootstrap(
                {
                    q.query_id: diag[a]["per_query"][m][q.query_id]
                    - diag[b]["per_query"][m][q.query_id]
                    for q in queries
                },
                queries,
            )
            for m in ("Recall@10", "nDCG@10", "Full-evidence@10")
        }

    before, after = {}, {}
    per_query_r5 = {}
    for k in rq.CUTOFFS:
        before[f"Recall@{k}"], b_values = rq.requirement_recall(hits, rel, queries, k)
        after[f"Recall@{k}"], a_values = rq.requirement_recall(hits_rr, rel, queries, k)
        if k == 5:
            per_query_r5 = {q: a_values[q] - b_values[q] for q in a_values}
    before["MRR"] = rq.requirement_mrr(hits, rel, queries)
    after["MRR"] = rq.requirement_mrr(hits_rr, rel, queries)
    r5_paired = rr.paired_bootstrap(per_query_r5, queries)
    r5_gain = round(after["Recall@5"], 3) - round(before["Recall@5"], 3)
    reranking_signal = r5_gain >= rq.SIGNAL_GAIN - 1e-9 and r5_paired["ci95_low"] > 0

    # E5's missing pairs: where each one stands, by its best rank over the requirements
    moves: dict[str, int] = {}
    blocked = {"d": 0, "e": 0}
    d_of = {x.query_id: x for x in d_ranked}
    for q in queries:
        qid = q.query_id
        e5_top = set(e5_of[qid].chunk_ids[: dc.DEPTH])
        b_best, a_best = rq.best_rank(hits[qid]), rq.best_rank(hits_rr[qid])
        d_top = set(d_of[qid].chunk_ids[: dc.DEPTH])
        e_top = set(assembled[qid].ranking[: dc.DEPTH])
        for c in rel[qid] - e5_top:
            b, a = b_best.get(c, 99), a_best.get(c, 99)
            key = (
                "outside the 50"
                if b == 99
                else "1-5 to 1-5"
                if b <= 5 and a <= 5
                else "1-5 to 6-50"
                if b <= 5
                else "6-50 to 1-5"
                if a <= 5
                else "6-50 to 6-50"
            )
            moves[key] = moves.get(key, 0) + 1
            blocked["d"] += b <= 5 and c not in d_top
            blocked["e"] += a <= 5 and c not in e_top
    e_top10 = {qid: set(assembled[qid].ranking[: dc.DEPTH]) for qid in question}
    e5_top10 = {qid: set(e5_of[qid].chunk_ids[: dc.DEPTH]) for qid in question}
    gained = sum(
        len((e_top10[q.query_id] - e5_top10[q.query_id]) & rel[q.query_id]) for q in queries
    )
    lost = sum(len((e5_top10[q.query_id] - e_top10[q.query_id]) & rel[q.query_id]) for q in queries)
    gained_by_grade = {
        g: sum(
            grade[q.query_id][c] == g
            for q in queries
            for c in (e_top10[q.query_id] - e5_top10[q.query_id]) & rel[q.query_id]
        )
        for g in (3, 2)
    }

    er = {k: round(v, 3) for k, v in cols["e"].items()}
    e = {k: round(v, 3) for k, v in cols["e5"].items()}
    dd = {k: round(v, 3) for k, v in cols["d"].items()}
    signal = {
        "Recall@10 +0.03": er["Recall@10"] - e["Recall@10"] >= 0.03 - 1e-9,
        "Full-evidence@10 +0.03": er["Full-evidence@10"] - e["Full-evidence@10"] >= 0.03 - 1e-9,
        "nDCG@10 >= 0.518": er["nDCG@10"] >= 0.518 - 1e-9,
    }
    success = {
        "useful": er["Recall@10"] >= 0.85
        and er["multi_provision"] >= 0.60
        and er["dependent"] >= 0.45
        and er["Full-evidence@10"] >= 0.75,
        "strong": er["Recall@10"] >= 0.90 and er["nDCG@10"] >= 0.65,
    }
    final_gain = er["Recall@10"] - dd["Recall@10"]
    if not reranking_signal:
        conclusion = "NO RERANKING SIGNAL"
    elif final_gain >= 0.03 - 1e-9:
        conclusion = "RERANKER WORKS"
    elif blocked["e"] > blocked["d"]:
        conclusion = "RERANKER WORKS / ASSEMBLER BOTTLENECK"
    else:
        conclusion = "RERANKER WORKS / NO FINAL GAIN, NOT THE PLACES"
    tokens = [row["tokens"] for row in done.values()]
    summary = {
        "system": {
            "kind": "decomposition_requirements_reranked_merge",
            "rule": "D's rule on each requirement's first 50 E5 chunks reordered by the reranker",
            "decomposer": d_record["model"],
            "reranker": config.model_dump(mode="json"),
            "instruction": rq.INSTRUCTION,
        },
        "identity": {
            "base_run_sha256": e5_summary["identity"]["run_sha256"],
            "decomposition_run_sha256": d_summary["identity"]["run_sha256"],
            "prompts_sha256": prompts_sha,
            "run_sha256": sha,
        },
        "benchmark_version": version,
        "parser_version": frozen.dev.manifest.parser_version,
        "layer": em.layer_source(list(frozen.docs)).model_dump(mode="json"),
        "queries_sha256": em.query_source(frozen.dev).queries_sha256,
        "metrics": evaluated["metrics"],
        "diagnostics": {k: v for k, v in diag["e"].items() if k != "per_query"},
        "paired_against_e5": paired("e", "e5"),
        "per_need": evaluated["per_need"],
        "per_query": evaluated["per_query"],
        "failures": evaluated["failures"],
    }
    sys_dir = args.results / f"dev-{version}" / f"{SYSTEM}-layer-{args.max_tokens}"
    sys_dir.mkdir(parents=True, exist_ok=True)
    (sys_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    payload = {
        "experiment": "E, zero-shot requirement reranking on D-4B",
        "benchmark_version": version,
        "reranker": config.model_dump(mode="json"),
        "reranker_config_hash": config.config_hash(),
        "reranker_tokenizer_sha256": reranker.tokenizer_sha256,
        "instruction": rq.INSTRUCTION,
        "system_name": SYSTEM,
        "run_sha256": sha,
        "requirement_ranking": {
            "e5_order": before,
            "reranked": after,
            "recall5_paired": r5_paired,
            "recall5_gain": round(r5_gain, 3),
            "signal_rule": f"Recall@5 +{rq.SIGNAL_GAIN} and its paired interval above zero",
            "reranking_signal": reranking_signal,
        },
        "migrations_of_e5_missing_pairs": dict(sorted(moves.items())),
        "blocked_by_the_places": blocked,
        "columns": cols,
        "by_language_recall10": diag["e"]["by_language"]["Recall@10"],
        "paired_against_e5": paired("e", "e5"),
        "paired_against_d4b": paired("e", "d"),
        "evidence": {
            "relevant_pairs_gained_against_e5": gained,
            "relevant_pairs_lost_against_e5": lost,
            "gained_by_grade": gained_by_grade,
        },
        "minimum_signal": signal,
        "signal_found": all(signal.values()),
        "success": success,
        "conclusion": conclusion,
        "performance": {
            "inputs": len(jobs),
            "inputs_scored_this_run": scored,
            "tokens": {
                "total": sum(tokens),
                "median": statistics.median(tokens),
                "max": max(tokens),
            },
            "truncated": sum(row["truncated"] for row in done.values()),
            "model_load_seconds": load_seconds,
            "scoring_seconds_this_run": scoring_seconds,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "environment": em.environment_description(),
        },
    }
    (out_dir / "e-requirement-reranking.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"requirement Recall@5 {before['Recall@5']:.3f} -> {after['Recall@5']:.3f} "
        f"({r5_paired['mean']:+.3f} [{r5_paired['ci95_low']:+.3f}, "
        f"{r5_paired['ci95_high']:+.3f}]); "
        f"moves {dict(sorted(moves.items()))}; blocked {blocked}\n"
        f"{SYSTEM}: R@10 {cols['e']['Recall@10']:.3f} nDCG@10 {cols['e']['nDCG@10']:.3f} full "
        f"{cols['e']['Full-evidence@10']:.3f} dependent {cols['e']['dependent']:.3f}; "
        f"{conclusion}; REQUIREMENT RERANKING SIGNAL: {'YES' if reranking_signal else 'NO'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
