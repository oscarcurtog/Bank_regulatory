"""A of retrieval iteration 3: anchor-conditioned necessity scoring of R1's pool
(config/retrieval/necessity-selection.md).

    PYTHONPATH=. caffeinate -i .venv/bin/python scripts/necessity_scoring.py --expect-version 0.10.0

E5's recorded run gives the anchors; R1's selected run (checked by SHA-256) and its provenance
the pool; R1's structure, rebuilt as R1 built it and checked against that provenance, the
relations and their citation contexts. Every (query, anchor, candidate) of the population is
scored under A3 and A2, and every (query, candidate) under A1, by the reranker of
config/retrieval/qwen3-reranker-0.6b.json. Each score is appended to
data/runs/necessity-qwen3-reranker-0.6b-layer-<n>-dev-<version>/scores.jsonl (not versioned) as
it is computed, with the SHA-256 of its input; a rerun scores only what is missing and refuses a
stored score whose input changed. Once every score exists, the labels measure each variant's
discrimination and the protocol's gates decide what follows:
data/results/dev-<version>/retrieval-iteration-3/a-necessity.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ingest import benchmark as bm
from ingest import dependencies as dp
from ingest import embeddings as em
from ingest import necessity as nc
from ingest import references as xr
from ingest import retrieval as rt
from ingest import run_report as rr

RERANKER = Path("config/retrieval/qwen3-reranker-0.6b.json")
XREF = Path("config/retrieval/xref-expansion.json")
NAME = "necessity-qwen3-reranker-0.6b"
ANCHORS = 8
OUTSIDE = 10
SUPPORTS = 4
CLEAR_GAIN = 0.05


def structure_of(frozen: rt.FrozenInputs) -> nc.Structure:
    """R1's structure, built as scripts/dependency_retrieval.py builds it."""
    corpus = bm.load_corpus()
    texts = {
        (act, n.path): n.own_text for act, d in corpus.items() for a in d.articles for n in a.nodes
    }
    acts = xr.ExpansionConfig.model_validate_json(XREF.read_text(encoding="utf-8")).acts
    graph = xr.build_graph(frozen.layer, texts, list(frozen.chunks), acts)
    definitions = dp.glossary(texts)
    index = dp.build_index(frozen.layer, texts, graph, definitions)
    article = {cid: f"{nc.act_of(cid)}:{c.article_id}" for cid, c in frozen.chunks.items()}
    return nc.Structure(texts, frozen.layer, graph, index, tuple(definitions), article)


def check_structure(structure: nc.Structure, provenance: dict[str, dict[str, Any]]) -> int:
    """Every relation R1 recorded is one the rebuilt structure holds."""
    held = {
        (anchor, kind, detail, node, target)
        for anchor, rels in structure.index.relations.items()
        for kind, detail, node, target in rels
    }
    checked = 0
    for record in provenance.values():
        for candidate, rels in record["candidates"].items():
            for r in rels:
                if r["kind"] == "bridge":
                    continue
                key = (r["anchor"], r["kind"], r["detail"], r["target_node"], candidate)
                if key not in held:
                    raise SystemExit(f"R1 recorded {key}; the rebuilt structure does not hold it")
                checked += 1
    return checked


def input_sha(fields: tuple[str, str, str]) -> str:
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--benchmark", type=Path, default=bm.DEV_DIR)
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--results", type=Path, default=Path("data/results"))
    args = parser.parse_args(argv)
    frozen = rt.load_frozen_inputs(args.benchmark, args.expect_version, args.max_tokens)
    version = frozen.dev.manifest.benchmark_version
    results = args.results / f"dev-{version}"
    r1 = json.loads((results / "retrieval-iteration-2" / "r1-candidates.json").read_text("utf-8"))
    r1_name = f"acdr-e5-bridge{r1['decision']['selected_k']}"
    r1_dir = rt.run_dir(r1_name, args.max_tokens, version)
    rt.read_run(r1_dir / rt.RUN_FILE, r1["configurations"][r1_name]["run_sha256"])
    provenance_bytes = (r1_dir / "provenance.jsonl").read_bytes()
    provenance = {
        row["query_id"]: row
        for row in (json.loads(line) for line in provenance_bytes.decode("utf-8").splitlines())
    }
    e5_ranked, e5_summary = rt.load_system_run(
        "multilingual-e5-large", args.max_tokens, version, args.results
    )

    # 1. the population and its inputs: no label is read here
    start = time.perf_counter()
    structure = structure_of(frozen)
    relations_checked = check_structure(structure, provenance)
    population = {
        r.query_id: nc.population(r, provenance[r.query_id], structure, ANCHORS, OUTSIDE)
        for r in e5_ranked
    }
    population_seconds = time.perf_counter() - start
    question = {q.query_id: q.text for q in frozen.dev.queries}
    text = {cid: c.text for cid, c in frozen.chunks.items()}
    jobs: dict[tuple[str, str, str, str], tuple[str, str, str]] = {}
    for variant in ("A3", "A2", "A1"):
        for qid, triples in population.items():
            for t in triples:
                anchor = "" if variant == "A1" else t.anchor
                key = (variant, qid, anchor, t.candidate)
                if key not in jobs:
                    jobs[key] = nc.model_input(variant, question[qid], t, text)
    triples_total = sum(len(v) for v in population.values())
    print(
        f"population: {triples_total} (query, anchor, candidate) triples over "
        f"{len(population)} queries, {relations_checked} R1 relations checked; "
        f"{len(jobs)} inputs ({Counter(k[0] for k in jobs)})",
        flush=True,
    )

    # 2. the scores, appended as they come; a rerun scores only what is missing
    config = nc.load_config(RERANKER)
    out_run = rt.run_dir(NAME, args.max_tokens, version)
    out_run.mkdir(parents=True, exist_ok=True)
    scores_path = out_run / "scores.jsonl"
    done: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    if scores_path.exists():
        for line in scores_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (row["variant"], row["query_id"], row["anchor"], row["candidate"])
            if key not in jobs or row["input_sha256"] != input_sha(jobs[key]):
                raise SystemExit(f"{scores_path}: a stored score has another input: {key}")
            done[key] = row
    pending = [k for k in jobs if k not in done]
    start = time.perf_counter()
    reranker = nc.NecessityReranker(config)
    load_seconds = time.perf_counter() - start
    print(f"reranker loaded in {load_seconds:.1f} s; {len(pending)} inputs to score", flush=True)
    prompts = {k: reranker.prompt(*jobs[k]) for k in pending}
    # one group per query and anchor (per query for A1): its prompts share their opening
    groups: dict[tuple[str, str, str], list[tuple[str, str, str, str]]] = {}
    for k in sorted(pending, key=lambda k: (("A3", "A2", "A1").index(k[0]), k)):
        groups.setdefault(k[:3], []).append(k)
    start = time.perf_counter()
    scored = 0
    for g, part in enumerate(groups.values(), start=1):
        values = reranker.score_shared([prompts[k] for k in part])
        with scores_path.open("a", encoding="utf-8") as f:
            for k, v in zip(part, values, strict=True):
                row = {
                    "variant": k[0],
                    "query_id": k[1],
                    "anchor": k[2],
                    "candidate": k[3],
                    "input_sha256": input_sha(jobs[k]),
                    "tokens": len(prompts[k].ids),
                    "truncated": prompts[k].truncated,
                    "logit_diff": float(v),
                }
                f.write(json.dumps(row) + "\n")
                done[k] = row
        scored += len(part)
        if g % 25 == 0 or g == len(groups):
            elapsed = time.perf_counter() - start
            print(
                f"scored {scored}/{len(pending)} ({part[-1][0]}, group {g}/{len(groups)}) in "
                f"{elapsed:.0f} s, about {elapsed / scored * (len(pending) - scored) / 60:.0f} "
                "min left",
                flush=True,
            )
    scoring_seconds = time.perf_counter() - start

    # 3. discrimination: the labels enter here, once every score exists
    targets = frozen.targets
    queries = [q for t in targets for q in t.queries]
    grades = {q.query_id: dict(t.chunk_grades) for t in targets for q in t.queries}
    top10 = {r.query_id: set(r.chunk_ids[:OUTSIDE]) for r in e5_ranked}
    dep = rt.dependent_evidence(targets)
    missing = {
        q.query_id: rt.relevant_chunks(t) - top10[q.query_id] for t in targets for q in t.queries
    }
    dependents = {
        q.query_id: dep[q.need_id].dependents - top10[q.query_id]
        for q in queries
        if q.need_id in dep
    }
    measured = {q.query_id: population[q.query_id] for q in queries}
    variants: dict[str, dict[str, Any]] = {}
    for variant in nc.VARIANTS:
        score = {
            (qid, t.anchor, t.candidate): done[
                (variant, qid, "" if variant == "A1" else t.anchor, t.candidate)
            ]["logit_diff"]
            for qid, triples in measured.items()
            for t in triples
        }
        variants[variant] = nc.discrimination(
            measured, score, grades, missing, dependents, queries, k=SUPPORTS
        )
    ap: dict[str, dict[str, float]] = {
        v: variants[v]["per_query"]["average_precision"] for v in nc.VARIANTS
    }
    with_positive = [q for q in queries if q.query_id in ap["A3"]]
    paired = {
        f"{a} - {b}": rr.paired_bootstrap(
            {q.query_id: ap[a][q.query_id] - ap[b][q.query_id] for q in with_positive},
            with_positive,
        )
        for a, b in (("A3", "A1"), ("A3", "A2"), ("A2", "A1"))
    }
    mean_ap = {v: variants[v]["average_precision"] for v in nc.VARIANTS}
    clear = mean_ap["A3"] - mean_ap["A1"] >= CLEAR_GAIN - 1e-9 and paired["A3 - A1"]["ci95_low"] > 0
    run_b = mean_ap["A3"] > mean_ap["A1"] and mean_ap["A3"] > mean_ap["A2"]
    decision = {
        "rule": "B runs if A3's average precision is above A1's and A2's; A3 improves clearly "
        "on A1 if its average precision is higher by at least 0.05 with the 95% paired "
        "interval above zero; if not, or if B does not run or reaches no success, research "
        "and C follow",
        "average_precision": {v: round(x, 3) for v, x in mean_ap.items()},
        "a3_clearly_better_than_a1": clear,
        "run_b": run_b,
        "research": not clear or not run_b,
    }
    tokens = [row["tokens"] for row in done.values()]
    payload = {
        "experiment": "A",
        "benchmark_version": version,
        "e5_run_sha256": e5_summary["identity"]["run_sha256"],
        "candidates": r1_name,
        "candidates_run_sha256": r1["configurations"][r1_name]["run_sha256"],
        "provenance_sha256": hashlib.sha256(provenance_bytes).hexdigest(),
        "reranker": config.model_dump(mode="json"),
        "reranker_config_hash": config.config_hash(),
        "reranker_tokenizer_sha256": reranker.tokenizer_sha256,
        "instructions": dict(nc.INSTRUCTIONS),
        "population": {
            "anchors": ANCHORS,
            "outside_e5_top": OUTSIDE,
            "triples": triples_total,
            "query_candidate_pairs": len({(k[1], k[3]) for k in jobs}),
            "anchor_groups": len({(k[1], k[2]) for k in jobs if k[0] == "A3"}),
            "triples_per_query_median": statistics.median(len(v) for v in population.values()),
            "triples_by_link_kind": dict(
                Counter(kind for v in population.values() for t in v for kind in t.kinds)
            ),
            "triples_by_relation_head": dict(
                Counter(
                    label.split(":")[0]
                    for v in population.values()
                    for t in v
                    for label in t.relation.split("; ")
                )
            ),
            "r1_relations_checked_against_the_rebuilt_structure": relations_checked,
        },
        "discrimination": {
            v: {k: x for k, x in d.items() if k != "per_query"} for v, d in variants.items()
        },
        "paired_average_precision": paired,
        "decision": decision,
        "performance": {
            "inputs": len(jobs),
            "tokens": {
                "total": sum(tokens),
                "median": statistics.median(tokens),
                "max": max(tokens),
            },
            "truncated": sum(row["truncated"] for row in done.values()),
            "population_seconds": population_seconds,
            "model_load_seconds": load_seconds,
            "scoring_seconds_this_run": scoring_seconds,
            "inputs_scored_this_run": len(pending),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "environment": em.environment_description(),
        },
    }
    out = results / "retrieval-iteration-3"
    out.mkdir(parents=True, exist_ok=True)
    (out / "a-necessity.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for v in nc.VARIANTS:
        d = variants[v]
        print(
            f"{v}: AP {d['average_precision']:.3f} AUROC {d['auroc']:.3f} P@1 "
            f"{d['precision_at_1']:.3f} P@3 {d['precision_at_3']:.3f} support@4 "
            f"{d['support_recall@4']:.3f} dependent@4 {d['dependent_support_recall@4']:.3f}"
        )
    for name, p in paired.items():
        print(f"AP {name}: {p['mean']:+.3f} [{p['ci95_low']:+.3f}, {p['ci95_high']:+.3f}]")
    print(f"decision: {decision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
