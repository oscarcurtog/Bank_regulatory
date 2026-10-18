"""T3: the dense E5 baseline moved into PostgreSQL with pgvector, and shown to be the same.

    docker compose up -d --wait postgres
    PYTHONPATH=. .venv/bin/python scripts/pgvector_baseline.py --expect-version 0.10.0

An infrastructure migration, not an experiment: nothing here may change a number, and the
script reports where it does. In order:

1. Frozen inputs: the embedding contract, the chunk layer DEV was sealed against, and
   the benchmark's queries. The benchmark loader also reads the labels; nothing reads
   them before step 8.
2. The cached E5 vectors of the documents and of the queries, verified under their full
   identity (D-028). A stale or missing set stops the script: T3 moves these vectors, it
   never computes new ones.
3. The index: schema, chunk layer (rebuilt from its rows and compared with the files),
   embedding set (matrix read back and hashed against meta.json), and no approximate
   index on the vectors.
4. A small check on 3 queries before the whole run.
5. Reproduction: all 90 queries at full depth (every chunk) against the stored E5 run.
   Rankings, scores and ties are compared; no label is read.
6. Provenance: the first 10 chunks of each evaluated query resolved to their content
   and source.
7. Latency: one warm-up pass, then --repetitions timed passes of top-10 searches, the
   server's own execution times, and the plan; next to the in-memory search of the
   baseline on the same vectors.
8. Retrieval metrics of the pgvector run through the same evaluator, against the stored
   summary. The labels are read here, last, and only to measure.

Writes data/results/dev-<version>/t3-pgvector/report.json and report.md, and the run to
data/runs/pgvector-multilingual-e5-large-layer-<max_tokens>-dev-<version>/run.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import psycopg

from ingest import benchmark as bm
from ingest import chunks as ch
from ingest import download as dl
from ingest import embeddings as em
from ingest import evaluation as ev
from ingest import retrieval as rt
from ingest import run_report as rr
from ingest import vector_index as vi

BASELINE = "multilingual-e5-large"
SYSTEM = f"pgvector-{BASELINE}"
TOP_K = 10
# What the report records about the server, one query each.
SERVER_FACTS = {
    "server": "SELECT version()",
    "pgvector": "SELECT extversion FROM pg_extension WHERE extname = 'vector'",
    "shared_buffers": "SELECT current_setting('shared_buffers')",
    "collation": "SELECT datcollate FROM pg_database WHERE datname = current_database()",
    "chunk_embedding_bytes": "SELECT pg_total_relation_size('chunk_embedding')::text",
}


def one(conn: vi.Connection, sql: str) -> str:
    row = conn.execute(sql).fetchone()
    if row is None:
        raise SystemExit(f"no answer from the server to: {sql}")
    return str(row[0])


def nearest_rank(values: Sequence[float], p: float) -> float:
    """The p-th percentile by the nearest-rank method: a value that was observed."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def latency_summary(seconds: Sequence[float]) -> dict[str, Any]:
    ms = [s * 1000 for s in seconds]
    out: dict[str, Any] = {"n": len(ms), "method": "nearest rank", "unit": "ms"}
    out |= {"p50": nearest_rank(ms, 50), "max": max(ms), "mean": statistics.fmean(ms)}
    if len(ms) >= 100:
        out["p95"] = nearest_rank(ms, 95)
    if len(ms) >= 1000:
        out["p99"] = nearest_rank(ms, 99)
    return out


def verified_set(
    directory: Path, identity: em.ArtifactIdentity, inputs: Sequence[em.ModelInput]
) -> tuple[em.EmbeddingSetMeta, np.ndarray]:
    try:
        meta, vectors = em.load_embedding_set(directory, identity)
    except em.EmbeddingError as e:
        raise SystemExit(f"{e}\nNot recomputed: T3 moves the cached vectors, it makes none.") from e
    if meta.item_ids != [i.item_id for i in inputs] or meta.token_records != [
        i.record for i in inputs
    ]:
        raise SystemExit(f"{directory}: other items or token records than the inputs")
    return meta, vectors


def ulps(delta: float, at: float) -> float:
    return abs(delta) / float(np.spacing(np.float32(abs(at))))


def compare(
    stored: Sequence[rt.Ranked], moved: Sequence[rt.Ranked], evaluated: frozenset[str]
) -> dict[str, Any]:
    """Rankings and scores of two runs of the same queries, position by position and
    chunk by chunk. No label is involved."""
    by_id = {r.query_id: r for r in moved}
    if [r.query_id for r in stored] != [r.query_id for r in moved]:
        raise SystemExit("the two runs do not hold the same queries in the same order")
    full_same, top_same, exact = 0, 0, 0
    first_diff: dict[str, int] = {}
    deltas: list[float] = []
    ulp_values: list[float] = []
    for s in stored:
        m = by_id[s.query_id]
        if m.chunk_ids == s.chunk_ids:
            full_same += 1
        else:
            first_diff[s.query_id] = next(
                i for i, (a, b) in enumerate(zip(s.chunk_ids, m.chunk_ids, strict=True)) if a != b
            )
        if s.query_id in evaluated and m.chunk_ids[:TOP_K] == s.chunk_ids[:TOP_K]:
            top_same += 1
        moved_score = dict(zip(m.chunk_ids, m.scores, strict=True))
        for chunk_id, score in zip(s.chunk_ids, s.scores, strict=True):
            delta = moved_score[chunk_id] - score
            deltas.append(abs(delta))
            ulp_values.append(ulps(delta, score))
            exact += delta == 0.0
    inversions = explain_inversions(stored, moved)
    return {
        "inversions": inversions,
        "queries": len(stored),
        "evaluated_queries": len(evaluated),
        "identical_full_rankings": full_same,
        "identical_top10_of_evaluated": top_same,
        "first_differing_position": first_diff,
        "scores_compared": len(deltas),
        "scores_bit_identical": exact,
        "abs_score_difference": {
            "max": max(deltas),
            "p99": nearest_rank(deltas, 99),
            "mean": statistics.fmean(deltas),
        },
        "float32_ulps": {"max": max(ulp_values), "p99": nearest_rank(ulp_values, 99)},
    }


def explain_inversions(stored: Sequence[rt.Ranked], moved: Sequence[rt.Ranked]) -> dict[str, Any]:
    """Every pair of chunks the two runs order differently, and whether the two scores'
    own floating-point differences explain it: an inverted pair whose stored gap exceeds
    the sum of its two score differences would be a real disagreement."""
    by_id = {r.query_id: r for r in moved}
    total, from_ties, unexplained, largest_gap, displacement = 0, 0, 0, 0.0, 0
    first: list[int] = []
    for s in stored:
        m = by_id[s.query_id]
        stored_score = dict(zip(s.chunk_ids, s.scores, strict=True))
        moved_score = dict(zip(m.chunk_ids, m.scores, strict=True))
        moved_pos = {c: i for i, c in enumerate(m.chunk_ids)}
        shift = max(abs(i - moved_pos[c]) for i, c in enumerate(s.chunk_ids))
        displacement = max(displacement, shift)
        diffs = [i for i, (a, b) in enumerate(zip(s.chunk_ids, m.chunk_ids, strict=True)) if a != b]
        if diffs:
            first.append(diffs[0] + 1)
        window = 2 * shift + 1  # an inverted pair is at most this far apart
        for i, a in enumerate(s.chunk_ids):
            for b in s.chunk_ids[i + 1 : i + 1 + window]:
                if moved_pos[a] < moved_pos[b]:
                    continue
                total += 1
                gap = stored_score[a] - stored_score[b]
                from_ties += gap == 0.0
                largest_gap = max(largest_gap, gap)
                bound = abs(moved_score[a] - stored_score[a]) + abs(
                    moved_score[b] - stored_score[b]
                )
                unexplained += gap > bound
    return {
        "inverted_pairs": total,
        "from_stored_exact_ties": from_ties,
        "not_explained_by_float_differences": unexplained,
        "largest_stored_gap_of_an_inverted_pair": largest_gap,
        "max_displacement_positions": displacement,
        "first_differing_position_1based": {
            "min": min(first) if first else None,
            "median": statistics.median(first) if first else None,
            "max": max(first) if first else None,
        },
    }


def fragile_pairs(
    stored: Sequence[rt.Ranked],
    moved: Sequence[rt.Ranked],
    evaluated: frozenset[str],
    threshold: float,
) -> list[dict[str, Any]]:
    """Neighbours in the top 11 of an evaluated query whose stored gap is under
    `threshold` (twice the tolerance): positions a floating-point difference could swap.
    Reported with the gap they kept in the moved run."""
    by_id = {r.query_id: r for r in moved}
    out = []
    for s in stored:
        if s.query_id not in evaluated:
            continue
        moved_score = dict(zip(by_id[s.query_id].chunk_ids, by_id[s.query_id].scores, strict=True))
        for i in range(TOP_K):
            gap = s.scores[i] - s.scores[i + 1]
            if gap < threshold:
                a, b = s.chunk_ids[i], s.chunk_ids[i + 1]
                out.append(
                    {
                        "query_id": s.query_id,
                        "positions": [i + 1, i + 2],
                        "stored_gap": gap,
                        "moved_gap": moved_score[a] - moved_score[b],
                    }
                )
    return out


def ties_and_margins(
    stored: Sequence[rt.Ranked],
    evaluated: frozenset[str],
    doc_ids: Sequence[str],
    doc_vectors: np.ndarray,
    texts: dict[str, str],
) -> dict[str, Any]:
    """Where a floating-point difference could reorder a ranking: exact ties, and the
    smallest gaps between neighbours, at full depth and around the top 10."""
    row = {c: i for i, c in enumerate(doc_ids)}
    ties, tie_same_vector, tie_same_text = 0, 0, 0
    gaps_all, gaps_top = [], []
    for r in stored:
        for i in range(len(r.scores) - 1):
            gap = r.scores[i] - r.scores[i + 1]
            gaps_all.append(gap)
            if r.query_id in evaluated and i < TOP_K:  # pairs (1,2) ... (10,11)
                gaps_top.append(gap)
            if gap == 0.0:
                ties += 1
                a, b = r.chunk_ids[i], r.chunk_ids[i + 1]
                tie_same_vector += bool(np.array_equal(doc_vectors[row[a]], doc_vectors[row[b]]))
                tie_same_text += texts[a] == texts[b]
    groups: dict[bytes, list[str]] = {}
    for i, c in enumerate(doc_ids):
        groups.setdefault(doc_vectors[i].tobytes(), []).append(c)
    duplicates = [sorted(g) for g in groups.values() if len(g) > 1]
    positive = [g for g in gaps_all if g > 0]
    return {
        "adjacent_exact_ties": ties,
        "ties_with_bit_identical_vectors": tie_same_vector,
        "ties_with_identical_text": tie_same_text,
        "duplicate_vector_groups": len(duplicates),
        "chunks_in_duplicate_groups": sum(len(g) for g in duplicates),
        "duplicate_examples": duplicates[:5],
        "smallest_positive_gap_full_depth": min(positive) if positive else None,
        "smallest_gap_top11_evaluated": min(gaps_top),
        "gaps_top11_evaluated": len(gaps_top),
    }


def time_searches(
    conn: vi.Connection, set_id: int, queries: np.ndarray, passes: int
) -> tuple[list[float], float]:
    first = time.perf_counter()
    for q in queries:  # warm-up: prepares the statement and fills the buffers
        vi.search(conn, set_id, q, TOP_K)
    warm_up = time.perf_counter() - first
    samples = []
    for _ in range(passes):
        for q in queries:
            start = time.perf_counter()
            vi.search(conn, set_id, q, TOP_K)
            samples.append(time.perf_counter() - start)
    return samples, warm_up


def server_times(conn: vi.Connection, set_id: int, queries: np.ndarray) -> dict[str, Any]:
    execution, planning, hit, read = [], [], 0, 0
    for q in queries:
        row = conn.execute(
            "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + vi.SEARCH_SQL,
            {"query": vi.vector_literal(q), "set_id": set_id, "k": TOP_K},
        ).fetchone()
        assert row is not None
        plan = row[0][0]
        execution.append(plan["Execution Time"] / 1000)
        planning.append(plan["Planning Time"] / 1000)
        hit += plan["Plan"].get("Shared Hit Blocks", 0)
        read += plan["Plan"].get("Shared Read Blocks", 0)
    return {
        "execution": latency_summary(execution),
        "planning": latency_summary(planning),
        "shared_blocks_hit": hit,
        "shared_blocks_read": read,
    }


def in_memory_times(doc_vectors: np.ndarray, queries: np.ndarray, passes: int) -> list[float]:
    samples = []
    for _ in range(passes):
        for q in queries:
            start = time.perf_counter()
            scores = doc_vectors @ q
            top = np.argpartition(-scores, TOP_K)[:TOP_K]
            top[np.argsort(-scores[top], kind="stable")]
            samples.append(time.perf_counter() - start)
    return samples


def plan_nodes(plan: dict[str, Any]) -> list[str]:
    method = f" ({plan['Sort Method']})" if "Sort Method" in plan else ""
    nodes = [f"{plan['Node Type']}{method}"]
    for child in plan.get("Plans", []):
        nodes += plan_nodes(child)
    return nodes


def machine() -> dict[str, str]:
    def sysctl(key: str) -> str:
        try:
            return subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    return {
        "platform": platform.platform(),
        "cpu": sysctl("machdep.cpu.brand_string"),
        "cpu_count": sysctl("hw.ncpu"),
        "memory_bytes": sysctl("hw.memsize"),
        "python": platform.python_version(),
        "psycopg": psycopg.__version__,
        "libpq": str(psycopg.pq.version()),
    }


def render(report: dict[str, Any]) -> str:
    """The report for a reader: what was compared, and every number that says so."""
    c, m, lat = report["comparison"], report["ties_and_margins"], report["latency"]
    diff, ulp = c["abs_score_difference"], c["float32_ulps"]
    client, server = lat["client_round_trip"], lat["server"]["execution"]
    memory = lat["in_memory_baseline_numpy"]
    idx = report["index"]
    tol = report["score_tolerance"]
    metric_max = max(abs(v) for v in report["metrics"]["official_differences"].values())
    lines = [
        f"# T3: the E5 baseline in PostgreSQL with pgvector, DEV {report['benchmark_version']}",
        "",
        "An infrastructure migration, not an experiment: the stored E5 vectors and the stored "
        "E5 run, moved into the database and compared. Produced by "
        "`scripts/pgvector_baseline.py`.",
        "",
        "## Same data, same embeddings, same similarity",
        "",
        f"- {idx['vectors']} chunks and their vectors ({idx['dimension']} dimensions) loaded; the "
        "layer rebuilt from its rows equals the chunk files field by field, and the matrix "
        "read back from PostgreSQL has the SHA-256 that meta.json records "
        f"(`{report['inputs']['documents_sha256'][:16]}`).",
        f"- Search: {idx['search']}. Operator: {idx['operator']}. Approximate indexes: "
        f"{idx['approximate_indexes'] or 'none'}.",
        f"- {idx['server']['server'].split(' on ')[0]}, pgvector {idx['server']['pgvector']}, "
        f'database collation {idx["server"]["collation"]} (chunk ids are COLLATE "C").',
        "",
        "## Same ranking",
        "",
        f"- Full rankings identical: {c['identical_full_rankings']} of {c['queries']} queries "
        "(every chunk, every position).",
        f"- Top 10 identical: {c['identical_top10_of_evaluated']} of {c['evaluated_queries']} "
        "evaluated queries.",
        f"- Scores: {c['scores_bit_identical']} of {c['scores_compared']} bit-identical; largest "
        f"absolute difference {diff['max']:.3e} ({ulp['max']:.1f} float32 ulps), p99 "
        f"{diff['p99']:.3e}.",
        f"- Ties in the stored run: {m['adjacent_exact_ties']} adjacent pairs with equal scores, "
        f"{m['ties_with_bit_identical_vectors']} of them between bit-identical vectors "
        f"({m['duplicate_vector_groups']} groups of duplicate vectors in the layer). Smallest "
        f"gap between neighbours in the top 11 of an evaluated query: "
        f"{m['smallest_gap_top11_evaluated']:.3e}.",
    ]
    inv = c["inversions"]
    fragile = m["fragile_top11_pairs_under_twice_max_difference"]
    lines += [
        f"- Where the full rankings differ: {inv['inverted_pairs']} inverted neighbour pairs, "
        f"{inv['from_stored_exact_ties']} of them exact ties in the stored run; "
        f"{inv['not_explained_by_float_differences']} not explained by the two scores' own "
        "floating-point differences. No chunk moves more than "
        f"{inv['max_displacement_positions']} positions; the first difference is at position "
        f"{inv['first_differing_position_1based']['min']} at the earliest (median "
        f"{inv['first_differing_position_1based']['median']}).",
        "- Fragile top-11 neighbours (stored gap under twice the largest difference): "
        + (
            "; ".join(
                f"{f['query_id']} {f['positions'][0]}/{f['positions'][1]}: gap "
                f"{f['stored_gap']:.2e} stored, {f['moved_gap']:.2e} moved"
                for f in fragile
            )
            or "none"
        )
        + ". Their order held in this run, but the margins alone do not guarantee it.",
    ]
    if tol is not None:
        lines.append(
            f"- Tolerance τ = {tol['tau']:.0e}: every score within τ: {tol['all_scores_within']}; "
            f"every top-11 gap above 2τ: {tol['top11_gap_above_two_tau']}."
        )
    lines += [
        "",
        "## Same metrics",
        "",
        f"- Official metrics: largest difference {metric_max:.1e}. Every diagnostic block "
        f"(global, per language, groups, dependent evidence, full evidence) identical: "
        f"{report['metrics']['all_diagnostic_blocks_identical']}. Labels were read only for "
        "this step, after the rankings had been compared.",
        "",
        "## Provenance",
        "",
        f"- {report['provenance']['resolved']} of {report['provenance']['expected']} top-10 "
        "results resolved to act, path, text and source document.",
        "",
        "## Latency",
        "",
        f"- {lat['what']}; {lat['queries']} queries × {lat['repetitions']} passes, {lat['cache']}.",
        f"- Client round trip: p50 {client['p50']:.2f} ms, p95 {client['p95']:.2f} ms, max "
        f"{client['max']:.2f} ms (n = {client['n']}, {client['method']}).",
        f"- Server execution: p50 {server['p50']:.2f} ms, max {server['max']:.2f} ms "
        f"(n = {server['n']}). Plan: {' > '.join(lat['plan'])}.",
        f"- In-memory exact search of the baseline (numpy), same vectors: p50 "
        f"{memory['p50'] * 1:.3f} ms, p95 {memory['p95']:.3f} ms.",
        f"- Machine: {report['machine']['cpu']}, {report['machine']['platform']}; PostgreSQL "
        "in Docker Desktop on the same machine.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=em.CONFIG_DIR / f"{BASELINE}.json")
    parser.add_argument("--benchmark", type=Path, default=bm.DEV_DIR)
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--results", type=Path, default=Path("data/results"))
    parser.add_argument("--repetitions", type=int, default=5, help="timed passes of all queries")
    parser.add_argument(
        "--score-tolerance",
        type=float,
        default=None,
        help="the tolerance proposed from the observed differences (D-052); reported against",
    )
    args = parser.parse_args(argv)

    print("1. frozen inputs (no label is read before step 8)")
    config = em.load_config(args.config)
    dev = bm.load_benchmark(args.benchmark)
    version = dev.manifest.benchmark_version
    if version != args.expect_version:
        raise SystemExit(f"benchmark {version}, expected {args.expect_version}")
    docs = [ch.load_chunks(act, dl.load_manifest()) for act in bm.ACTS]
    if {d.max_tokens for d in docs} != {args.max_tokens}:
        raise SystemExit(f"the chunk layer was built at {sorted({d.max_tokens for d in docs})}")
    ev.check_layer(dev.manifest, ev.layer_from_documents(docs))
    doc_ids = [c.chunk_id for d in docs for c in d.chunks]
    texts = {c.chunk_id: c.text for d in docs for c in d.chunks}
    print(
        f"  DEV {version}, {len(doc_ids)} chunks at {args.max_tokens}, {len(dev.queries)} queries"
    )

    print("2. cached E5 vectors, verified (D-028)")
    loaded = em.load_tokenizer(config)
    doc_inputs = em.prepare_inputs(list(texts.items()), "document", loaded.tokenizer, config)
    query_inputs = em.prepare_inputs(
        [(q.query_id, q.text) for q in dev.queries], "query", loaded.tokenizer, config
    )
    runtime = em.runtime_identity(config)
    doc_meta, doc_vectors = verified_set(
        em.embedding_dir(config, f"layer-{args.max_tokens}"),
        em.artifact_identity(config, loaded, runtime, em.layer_source(docs)),
        doc_inputs,
    )
    query_meta, query_vectors = verified_set(
        em.embedding_dir(config, f"queries-{version}"),
        em.artifact_identity(config, loaded, runtime, em.query_source(dev)),
        query_inputs,
    )
    print(f"  documents {doc_meta.vectors_sha256[:12]}, queries {query_meta.vectors_sha256[:12]}")

    run_path = rt.run_dir(BASELINE, args.max_tokens, version) / rt.RUN_FILE
    summary_path = (
        args.results / f"dev-{version}" / f"{BASELINE}-layer-{args.max_tokens}" / "summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stored, evaluated = rt.read_run(run_path, summary["identity"]["run_sha256"])

    with vi.connect() as conn:
        print("3. index")
        start = time.perf_counter()
        vi.create_schema(conn)
        layer_id = vi.load_layer(conn, docs)
        set_id = vi.load_embedding_set(conn, layer_id, doc_meta, doc_vectors)
        load_seconds = time.perf_counter() - start
        approximate = vi.vector_indexes(conn)
        if approximate:
            raise SystemExit(f"approximate indexes on the vectors: {approximate}; T3 is exact")
        server = {key: one(conn, sql) for key, sql in SERVER_FACTS.items()}
        print(
            f"  layer {layer_id}, set {set_id}: layer rebuilt equal to the files, matrix read back "
            f"with the hash of meta.json, no approximate index ({load_seconds:.1f} s)"
        )

        print("4. small check, 3 queries")
        for qid, q in list(zip(query_meta.item_ids, query_vectors, strict=True))[:3]:
            got = vi.search(conn, set_id, q, 5)
            want = next(r for r in stored if r.query_id == qid)
            same = [c for c, _ in got] == list(want.chunk_ids[:5])
            print(f"  {qid}: top 5 {'identical' if same else 'DIFFERENT'} {[c for c, _ in got]}")

        print("5. reproduction: 90 queries at full depth, against the stored run")
        moved = []
        for qid, q in zip(query_meta.item_ids, query_vectors, strict=True):
            rows = vi.search(conn, set_id, q, len(doc_ids))
            moved.append(rt.Ranked(qid, tuple(c for c, _ in rows), tuple(s for _, s in rows)))
        comparison = compare(stored, moved, evaluated)
        margins = ties_and_margins(stored, evaluated, doc_ids, doc_vectors, texts)
        diff = comparison["abs_score_difference"]
        print(
            f"  full rankings identical {comparison['identical_full_rankings']}/90, top 10 "
            f"identical {comparison['identical_top10_of_evaluated']}/{len(evaluated)}; scores "
            f"bit-identical {comparison['scores_bit_identical']}/{comparison['scores_compared']}, "
            f"max |difference| {diff['max']:.3e} ({comparison['float32_ulps']['max']:.1f} ulps)"
        )
        print(
            f"  exact ties {margins['adjacent_exact_ties']} (identical vectors "
            f"{margins['ties_with_bit_identical_vectors']}); smallest gap in the top 11 "
            f"{margins['smallest_gap_top11_evaluated']:.3e}"
        )

        print("6. provenance of the top 10 of the evaluated queries")
        resolved = 0
        for r in moved:
            if r.query_id in evaluated:
                found = vi.resolve(conn, layer_id, r.chunk_ids[:TOP_K])
                resolved += sum(
                    bool(p.act and p.source_celex and p.root_path and p.text and p.source_paths)
                    for p in found
                )
        print(f"  {resolved}/{len(evaluated) * TOP_K} resolved to act, path, text and source")

        print("7. latency")
        plan = vi.search_plan(conn, set_id, query_vectors[0], TOP_K)
        samples, warm_up = time_searches(conn, set_id, query_vectors, args.repetitions)
        on_server = server_times(conn, set_id, query_vectors)
    in_memory = in_memory_times(doc_vectors, query_vectors, args.repetitions)
    client = latency_summary(samples)
    print(
        f"  top-10 search, client: p50 {client['p50']:.2f} ms, p95 {client['p95']:.2f} ms "
        f"(n={client['n']}); server execution p50 {on_server['execution']['p50']:.2f} ms; "
        f"plan {' > '.join(plan_nodes(plan))}"
    )

    print("8. retrieval metrics through the evaluator (labels read here, last)")
    moved_sha = rt.write_run(
        rt.run_dir(SYSTEM, args.max_tokens, version) / rt.RUN_FILE, moved, evaluated
    )
    frozen = rt.load_frozen_inputs(args.benchmark, version, args.max_tokens)
    diag_moved = rr.diagnostics(frozen.dev, frozen.layer, moved)
    diag_stored = rr.diagnostics(frozen.dev, frozen.layer, stored)
    macro = summary["metrics"]["macro_by_need"]
    metric_diffs = {m: diag_moved["global"][m] - macro[m] for m in rr.METRICS}
    blocks = ("global", "by_language", "groups", "dependent_evidence", "full_evidence")
    same_diagnostics = all(diag_moved[b] == diag_stored[b] for b in blocks)
    print(
        f"  official metrics: largest difference {max(abs(v) for v in metric_diffs.values()):.1e}; "
        f"every diagnostic block identical: {same_diagnostics}"
    )

    tolerance: dict[str, Any] | None = None
    if args.score_tolerance is not None:
        tau = args.score_tolerance
        tolerance = {
            "tau": tau,
            "all_scores_within": diff["max"] <= tau,
            "top11_gap_above_two_tau": margins["smallest_gap_top11_evaluated"] > 2 * tau,
            "fragile_top11_pairs_under_two_tau": fragile_pairs(stored, moved, evaluated, 2 * tau),
        }
    margins["fragile_top11_pairs_under_twice_max_difference"] = fragile_pairs(
        stored, moved, evaluated, 2 * diff["max"]
    )
    report = {
        "what": "T3: the E5 baseline moved into PostgreSQL with pgvector; a migration, "
        "not an experiment",
        "benchmark_version": version,
        "contract": {
            "name": config.name,
            "config_hash": config.config_hash(),
            "revision": config.revision,
        },
        "index": {
            "layer_id": layer_id,
            "set_id": set_id,
            "vectors": len(doc_ids),
            "dimension": vi.DIMENSION,
            "search": "exact: no HNSW or IVFFlat index; ORDER BY embedding <#> query, chunk_id",
            "operator": "<#> (negative inner product); score = -(embedding <#> query)",
            "approximate_indexes": approximate,
            "load_and_verify_seconds": load_seconds,
            "server": server,
        },
        "inputs": {
            "documents_sha256": doc_meta.vectors_sha256,
            "queries_sha256": query_meta.vectors_sha256,
            "stored_run_sha256": summary["identity"]["run_sha256"],
            "moved_run_sha256": moved_sha,
        },
        "comparison": comparison,
        "ties_and_margins": margins,
        "score_tolerance": tolerance,
        "metrics": {
            "official_differences": metric_diffs,
            "all_diagnostic_blocks_identical": same_diagnostics,
        },
        "provenance": {"resolved": resolved, "expected": len(evaluated) * TOP_K},
        "latency": {
            "what": "one top-10 search, the query vector already computed",
            "queries": len(query_vectors),
            "repetitions": args.repetitions,
            "cache": "warm: one untimed pass first (statement prepared, table in shared buffers)",
            "warm_up_pass_seconds": warm_up,
            "client_round_trip": client,
            "server": on_server,
            "plan": plan_nodes(plan),
            "in_memory_baseline_numpy": latency_summary(in_memory),
        },
        "machine": machine(),
    }
    out = args.results / f"dev-{version}" / "t3-pgvector"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render(report), encoding="utf-8")
    print(f"wrote {out / 'report.json'} and report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
