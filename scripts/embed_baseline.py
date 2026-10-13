"""Dense retrieval baseline: one embedding model, exact search, the frozen DEV benchmark.

    PYTHONPATH=. .venv/bin/python scripts/embed_baseline.py \\
        --config config/embeddings/multilingual-e5-large.json --expect-version 0.10.0

Steps, each one checked before the next:
1. the frozen inputs: the DEV benchmark at the expected version, the official chunk layer
   (500 cl100k tokens) and the embedding contract;
2. the tokenizer compatibility check and the truncation exposure report, before any
   vector exists;
3. document and query embeddings, read from data/embeddings/ when they are not stale and
   computed otherwise (needs the optional 'embed' extra);
4. exact search: every query against every chunk;
5. the official metrics (ingest/evaluation.py), then the breakdowns, the failure analysis
   and the sanity checks.

It writes the embeddings and the run outside git (data/embeddings/, data/runs/) and a
summary and a report that ARE versioned (data/results/). No label of the benchmark is
read for anything but scoring, and none is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import resource
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ingest import benchmark as bm
from ingest import chunks as ch
from ingest import download as dl
from ingest import embeddings as em
from ingest import evaluation as ev
from ingest import hierarchy as h

METRICS = [
    "nDCG@10",
    "Recall@10",
    "MRR@10",
    "Recall@5",
    "Recall@20",
    "Recall@T=1000",
    "Recall@T=2000",
]
STYLES = [("en", "formal"), ("en", "natural"), ("es", "natural")]
SMALL_GROUP = 5
LOW_NDCG = 0.5
SEED = 20261013


def pct(values: Sequence[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))]


def mean_of(rows: Sequence[Mapping[str, float]], metric: str) -> float:
    return statistics.fmean(r[metric] for r in rows)


# ---------------------------------------------------------------------------
# exposure: which judged text a truncation hides
# ---------------------------------------------------------------------------


def node_spans(
    chunk: ch.Chunk, act: str, nodes: Mapping[bm.NodeKey, h.Node]
) -> list[tuple[str, int, int]]:
    """Where the own text of each source node sits in chunk.text."""
    pos = len(chunk.context) + len(ch.SECTION_SEPARATOR)
    spans = []
    for path in chunk.source_paths:
        rendered = ch.render_node(nodes[(act, path)])
        if chunk.text[pos : pos + len(rendered)] != rendered:
            raise SystemExit(f"{chunk.chunk_id}: cannot place {path} in the chunk text")
        spans.append((path, pos, pos + len(rendered)))
        pos += len(rendered) + len(ch.LINE_SEPARATOR)
    return spans


def exposure(
    docs: Sequence[ch.ChunkedDocument],
    records: Mapping[str, em.TokenRecord],
    nodes: Mapping[bm.NodeKey, h.Node],
    dev: bm.Benchmark,
) -> dict[str, Any]:
    judged: dict[bm.NodeKey, list[tuple[str, int]]] = defaultdict(list)
    for r in dev.qrels:
        if not r.ambiguous:
            judged[r.node].append((r.need_id, r.grade))
    truncated: list[tuple[str, str, em.TokenRecord]] = []
    cut_nodes: list[dict[str, Any]] = []
    for doc in docs:
        for c in doc.chunks:
            rec = records[c.chunk_id]
            if not rec.truncated:
                continue
            truncated.append((doc.act, c.chunk_id, rec))
            for path, start, end in node_spans(c, doc.act, nodes):
                if end <= rec.kept_text_chars:
                    continue
                state = "partly cut" if start < rec.kept_text_chars else "cut"
                for need_id, grade in judged.get((doc.act, path), []):
                    cut_nodes.append(
                        {
                            "need": need_id,
                            "node": f"{doc.act}:{path}",
                            "grade": grade,
                            "state": state,
                            "chunk": c.chunk_id,
                        }
                    )
    dropped = [rec.dropped_tokens for _, _, rec in truncated]
    relevant_needs: dict[str, list[str]] = defaultdict(list)
    truncated_ids = {cid for _, cid, _ in truncated}
    owner = {(d.act, p): c.chunk_id for d in docs for c in d.chunks for p in c.source_paths}
    for need in dev.needs:
        rel = {
            r.node
            for r in dev.qrels_of(need.need_id)
            if r.grade >= bm.RELEVANT_GRADE and not r.ambiguous
        }
        hit = sorted({owner[n] for n in rel} & truncated_ids)
        if hit:
            relevant_needs[need.need_id] = hit
    total = sum(len(d.chunks) for d in docs)
    return {
        "total_chunks": total,
        "truncated_chunks": len(truncated),
        "truncated_share": len(truncated) / total,
        "dropped_tokens": {
            "min": min(dropped, default=0),
            "median": statistics.median(dropped) if dropped else 0,
            "p95": pct(dropped, 95) if dropped else 0,
            "max": max(dropped, default=0),
        },
        "by_act": dict(sorted(Counter(act for act, _, _ in truncated).items())),
        "chunks": [
            {"chunk": cid, "original": rec.original_model_tokens, "dropped": rec.dropped_tokens}
            for _, cid, rec in sorted(truncated, key=lambda t: -t[2].dropped_tokens)
        ],
        "needs_with_a_truncated_relevant_chunk": dict(sorted(relevant_needs.items())),
        "judged_nodes_in_truncated_text": {
            "relevant": [n for n in cut_nodes if n["grade"] >= bm.RELEVANT_GRADE],
            "grade_1": [n for n in cut_nodes if n["grade"] == 1],
            "grade_0": [n for n in cut_nodes if n["grade"] == 0],
        },
    }


# ---------------------------------------------------------------------------
# breakdowns
# ---------------------------------------------------------------------------


def group_means(
    per_need: Mapping[str, Mapping[str, float]], keys: Mapping[str, Sequence[str]]
) -> dict[str, dict[str, Any]]:
    """Macro means over needs for each group a need belongs to (a need may be in several)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for need_id, labels in keys.items():
        for label in labels:
            groups[label].append(need_id)
    out = {}
    for label, ids in sorted(groups.items()):
        rows = [per_need[i] for i in ids]
        out[label] = {"needs": len(ids), "descriptive_only": len(ids) < SMALL_GROUP} | {
            m: mean_of(rows, m) for m in METRICS
        }
    return out


def paired(diffs: Sequence[float]) -> dict[str, float]:
    rng = random.Random(SEED)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(10_000))
    return {
        "mean": statistics.fmean(diffs),
        "ci95_low": means[249],
        "ci95_high": means[9749],
        "es_better": sum(d > 0 for d in diffs),
        "en_better": sum(d < 0 for d in diffs),
        "equal": sum(d == 0 for d in diffs),
    }


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def model_cache_bytes(config: em.EmbeddingConfig) -> int:
    """Size on disk of the model's files at its pinned revision in the Hugging Face cache.
    The snapshot holds links to blobs that may live elsewhere in the cache; each blob
    counts once, and files of any other revision are not the model."""
    folder = em.hf_home() / "hub" / f"models--{config.model_id.replace('/', '--')}"
    snapshots = [folder / "snapshots" / config.revision]
    if config.tokenizer_commit != config.revision:
        snapshots.append(folder / "snapshots" / config.tokenizer_commit)
    blobs = {f.resolve() for s in snapshots for f in s.rglob("*") if f.is_file()}
    return sum(b.stat().st_size for b in blobs)


def timed(label: str, timings: dict[str, float], fn: Callable[[], Any]) -> Any:
    start = time.perf_counter()
    value = fn()
    timings[label] = time.perf_counter() - start
    print(f"  {label}: {timings[label]:.1f} s", flush=True)
    return value


def embeddings_for(
    directory: Path,
    identity: em.ArtifactIdentity,
    inputs: Sequence[em.ModelInput],
    encoder_factory: Callable[[], Any],
    timings: dict[str, float],
    label: str,
    recompute: bool,
) -> tuple[np.ndarray, dict[str, str], bool]:
    try:
        if recompute:
            raise em.EmbeddingsNotFoundError("--recompute")
        meta, vectors = em.load_embedding_set(directory, identity)
        if meta.item_ids != [i.item_id for i in inputs] or meta.token_records != [
            i.record for i in inputs
        ]:
            raise em.StaleEmbeddingsError(f"{directory}: other items or token records")
        print(f"  {label}: read from {directory} (not stale)")
        return vectors, meta.environment, False
    except em.EmbeddingsNotFoundError as e:
        print(f"  {label}: {'recomputing' if recompute else 'none cached'} ({e}), computing")
    except em.StaleEmbeddingsError as e:
        print(f"  {label}: {e}; computing again")
    encoder = encoder_factory()
    vectors = timed(label, timings, lambda: encoder.encode(inputs))
    em.write_embedding_set(directory, identity, inputs, vectors, encoder.environment())
    return vectors, encoder.environment(), True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=em.CONFIG_DIR / "multilingual-e5-large.json")
    parser.add_argument("--benchmark", type=Path, default=bm.DEV_DIR)
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--results", type=Path, default=Path("data/results"))
    parser.add_argument(
        "--recompute", action="store_true", help="embed again even if a cached set is not stale"
    )
    args = parser.parse_args(argv)
    timings: dict[str, float] = {}

    print("1. frozen inputs")
    config = em.load_config(args.config)
    dev = bm.load_benchmark(args.benchmark)
    if dev.manifest.benchmark_version != args.expect_version:
        raise SystemExit(
            f"benchmark {dev.manifest.benchmark_version}, expected {args.expect_version}"
        )
    manifest = dl.load_manifest()
    docs = [ch.load_chunks(act, manifest) for act in bm.ACTS]
    if {d.max_tokens for d in docs} != {args.max_tokens}:
        raise SystemExit(f"the chunk layer was built at {sorted({d.max_tokens for d in docs})}")
    layer = ev.layer_from_documents(docs)
    ev.check_layer(dev.manifest, layer)
    corpus = bm.load_corpus()
    nodes = {(act, n.path): n for act, d in corpus.items() for a in d.articles for n in a.nodes}
    chunks = [(c.chunk_id, c.text) for d in docs for c in d.chunks]
    chunk_by_id = {c.chunk_id: c for d in docs for c in d.chunks}
    query_text = {q.query_id: q.text for q in dev.queries}
    print(
        f"  DEV {dev.manifest.benchmark_version}, parser {dev.manifest.parser_version}, "
        f"{len(chunks)} chunks at {args.max_tokens}, contract {config.config_hash()[:12]}"
    )

    print("2. tokenizer compatibility and truncation exposure")
    loaded = em.load_tokenizer(config)
    tokenizer = loaded.tokenizer
    doc_inputs = em.prepare_inputs(chunks, "document", tokenizer, config)
    query_inputs = em.prepare_inputs(
        [(q.query_id, q.text) for q in dev.queries], "query", tokenizer, config
    )
    records = {i.item_id: i.record for i in doc_inputs}
    counts = [r.original_model_tokens for r in records.values()]
    query_counts = [i.record.original_model_tokens for i in query_inputs]
    tokenization: dict[str, Any] = {
        "limit": config.max_input_tokens,
        "documents": {
            "min": min(counts),
            "median": statistics.median(counts),
            "p95": pct(counts, 95),
            "p99": pct(counts, 99),
            "max": max(counts),
            "over_limit": sum(c > config.max_input_tokens for c in counts),
        },
        "queries": {
            "min": min(query_counts),
            "median": statistics.median(query_counts),
            "max": max(query_counts),
            "over_limit": sum(c > config.max_input_tokens for c in query_counts),
        },
    }
    exposed = exposure(docs, records, nodes, dev)
    d, q = tokenization["documents"], tokenization["queries"]
    print(
        f"  documents: model tokens min {d['min']}, median {d['median']}, p95 {d['p95']}, "
        f"p99 {d['p99']}, max {d['max']}; over the {config.max_input_tokens} limit "
        f"{d['over_limit']}, truncated {exposed['truncated_chunks']} of {exposed['total_chunks']}"
    )
    print(
        f"  queries: model tokens min {q['min']}, median {q['median']}, max {q['max']}; "
        f"over the limit {q['over_limit']}"
    )

    print("3. embeddings")
    from ingest.encoder import Encoder

    encoder_cache: list[Encoder] = []

    def encoder() -> Encoder:
        if not encoder_cache:
            encoder_cache.append(timed("model load", timings, lambda: Encoder(config)))
        return encoder_cache[0]

    layer_src, query_src = em.layer_source(docs), em.query_source(dev)
    runtime = em.runtime_identity(config)
    doc_identity = em.artifact_identity(config, loaded, runtime, layer_src)
    query_identity = em.artifact_identity(config, loaded, runtime, query_src)
    doc_dir = em.embedding_dir(config, f"layer-{args.max_tokens}")
    query_dir = em.embedding_dir(config, f"queries-{dev.manifest.benchmark_version}")
    doc_vectors, environment, fresh_docs = embeddings_for(
        doc_dir,
        doc_identity,
        doc_inputs,
        encoder,
        timings,
        "document embeddings",
        args.recompute,
    )
    query_vectors, _, _ = embeddings_for(
        query_dir,
        query_identity,
        query_inputs,
        encoder,
        timings,
        "query embeddings",
        args.recompute,
    )

    print("4. exact search")
    doc_ids = [i.item_id for i in doc_inputs]
    query_ids = [i.item_id for i in query_inputs]
    ranked = timed(
        "exact retrieval",
        timings,
        lambda: em.rank_exact(query_ids, query_vectors, doc_ids, doc_vectors),
    )
    # Every query is ranked and kept in the run file; only the queries of evaluable needs
    # (complete or partial) go to the evaluator. A 'none' need is kept for abstention.
    evaluable = {q.query_id for t in ev.targets(dev, layer) for q in t.queries}
    run = em.to_run([r for r in ranked if r.query_id in evaluable])
    run_dir = (
        Path("data/runs")
        / f"{config.name}-layer-{args.max_tokens}-dev-{dev.manifest.benchmark_version}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run.jsonl").open("w", encoding="utf-8") as f:
        for r in ranked:
            f.write(
                json.dumps(
                    {
                        "query_id": r.query_id,
                        "evaluated": r.query_id in evaluable,
                        "chunk_ids": r.chunk_ids,
                        "scores": r.scores,
                    }
                )
                + "\n"
            )

    print("5. metrics")
    scores, missing = ev.score_queries(dev, layer, run)
    if missing:
        raise SystemExit(f"queries absent from the run: {missing}")
    report = ev.aggregate(scores, [s.query.query_id for s in scores])
    by_query = {s.query.query_id: s for s in scores}
    targets = {t.need.need_id: t for t in ev.targets(dev, layer)}
    rank_of = {r.query_id: {c: i + 1 for i, c in enumerate(r.chunk_ids)} for r in ranked}
    score_of = {r.query_id: dict(zip(r.chunk_ids, r.scores, strict=True)) for r in ranked}

    def first_relevant(query_id: str, need_id: str) -> int:
        rel = [c for c, g in targets[need_id].chunk_grades.items() if g >= bm.RELEVANT_GRADE]
        return min(rank_of[query_id][c] for c in rel)

    per_need: dict[str, dict[str, float]] = {}
    per_need_style: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for need_id, t in targets.items():
        rows = [by_query[q.query_id].values for q in t.queries]
        per_need[need_id] = {m: mean_of(rows, m) for m in METRICS}
        for q in t.queries:
            per_need_style[need_id][f"{q.lang} {q.style}"] = dict(by_query[q.query_id].values) | {
                "first_relevant_rank": first_relevant(q.query_id, need_id),
                "query_id": q.query_id,
            }
    styles = {}
    for lang, style in STYLES:
        rows = [per_need_style[n][f"{lang} {style}"] for n in per_need]
        styles[f"{lang} {style}"] = {"needs": len(rows)} | {m: mean_of(rows, m) for m in METRICS}
    official = {
        lang: {"needs": s.needs, "queries": s.queries} | {m: s.means[m] for m in METRICS}
        for lang, s in report.by_language.items()
    }
    macro = {"needs": len(per_need), "queries": len(scores)} | {
        m: mean_of(list(per_need.values()), m) for m in METRICS
    }

    needs = {n.need_id: n for n in dev.needs}
    by_type = group_means(per_need, {n: [needs[n].type] for n in per_need})
    by_difficulty = group_means(per_need, {n: [needs[n].difficulty] for n in per_need})
    by_answerability = group_means(per_need, {n: [needs[n].answerability] for n in per_need})
    by_driver = group_means(per_need, {n: list(needs[n].drivers) or ["(none)"] for n in per_need})
    language = {
        metric: {
            "es_natural_minus_en_natural": paired(
                [
                    per_need_style[n]["es natural"][metric]
                    - per_need_style[n]["en natural"][metric]
                    for n in per_need
                ]
            ),
            "es_natural_minus_en_formal": paired(
                [
                    per_need_style[n]["es natural"][metric] - per_need_style[n]["en formal"][metric]
                    for n in per_need
                ]
            ),
        }
        for metric in ("nDCG@10", "Recall@10")
    }

    rel_chunks = {
        n: sum(g >= bm.RELEVANT_GRADE for g in t.chunk_grades.values()) for n, t in targets.items()
    }
    truncated_ids = {c["chunk"] for c in exposed["chunks"]}
    need_table = []
    for need_id in sorted(per_need):
        n = needs[need_id]
        firsts = {k: v["first_relevant_rank"] for k, v in per_need_style[need_id].items()}
        ceiling = min(1.0, 5 / rel_chunks[need_id])
        flags = []
        if any(per_need_style[need_id][k]["Recall@10"] < 1 for k in per_need_style[need_id]):
            flags.append("Recall@10<1")
        if any(r > 10 for r in firsts.values()):
            flags.append("first relevant >10")
        if per_need[need_id]["nDCG@10"] < LOW_NDCG:
            flags.append("low nDCG@10")
        es, en = (
            per_need_style[need_id]["es natural"]["nDCG@10"],
            per_need_style[need_id]["en natural"]["nDCG@10"],
        )
        if es < en - 0.2:
            flags.append("ES much worse")
        rel = {c for c, g in targets[need_id].chunk_grades.items() if g >= bm.RELEVANT_GRADE}
        if rel & truncated_ids:
            flags.append("truncation-exposed")
        need_table.append(
            {
                "need": need_id,
                "type": n.type,
                "difficulty": n.difficulty,
                "answerability": n.answerability,
                "drivers": list(n.drivers),
                "relevant_chunks": rel_chunks[need_id],
                "recall5_ceiling": ceiling,
                "recall5_over_ceiling": per_need[need_id]["Recall@5"] / ceiling,
            }
            | {m: per_need[need_id][m] for m in METRICS}
            | {"first_relevant_rank": firsts, "flags": flags}
        )

    # failures: the queries with the lowest nDCG@10, then the latest first relevant chunk
    ordered = sorted(
        scores,
        key=lambda s: (
            s.values["nDCG@10"],
            -first_relevant(s.query.query_id, s.query.need_id),
            s.query.query_id,
        ),
    )
    failures = []
    for s in ordered[:8]:
        q, t = s.query, targets[s.query.need_id]
        top = []
        for c in ranked[query_ids.index(q.query_id)].chunk_ids[:10]:
            grade = t.chunk_grades.get(c)
            text = chunk_by_id[c].body.replace("\n", " ")
            top.append(
                {
                    "rank": rank_of[q.query_id][c],
                    "chunk": c,
                    "score": score_of[q.query_id][c],
                    "grade": grade,
                    "relevant": grade is not None and grade >= bm.RELEVANT_GRADE,
                    "truncated": c in truncated_ids,
                    "preview": text[:160],
                }
            )
        rel_ranked = sorted(
            (c for c, g in t.chunk_grades.items() if g >= bm.RELEVANT_GRADE),
            key=lambda c: rank_of[q.query_id][c],
        )
        failures.append(
            {
                "query_id": q.query_id,
                "need": q.need_id,
                "lang": q.lang,
                "style": q.style,
                "text": q.text if q.lang == "en" else None,
                "values": dict(s.values),
                "first_relevant_rank": first_relevant(q.query_id, q.need_id),
                "top10": top,
                "relevant_chunks": [
                    {
                        "chunk": c,
                        "rank": rank_of[q.query_id][c],
                        "score": score_of[q.query_id][c],
                        "grade": t.chunk_grades[c],
                        "truncated": c in truncated_ids,
                    }
                    for c in rel_ranked
                ],
                "truncation_exposed": bool(set(rel_ranked) & truncated_ids)
                or any(x["truncated"] for x in top),
            }
        )

    print("6. sanity checks")
    sanity: dict[str, Any] = {}
    again = em.rank_exact(query_ids, query_vectors, doc_ids, doc_vectors)
    sanity["same_ranking_twice"] = again == ranked
    norms = np.linalg.norm(doc_vectors.astype(np.float64), axis=1)
    qnorms = np.linalg.norm(query_vectors.astype(np.float64), axis=1)
    sanity["dimension"] = int(doc_vectors.shape[1])
    sanity["max_norm_error"] = float(max(np.abs(norms - 1).max(), np.abs(qnorms - 1).max()))
    exact = np.zeros(len(query_ids), dtype=bool)
    worst = 0.0
    for i, r in enumerate(ranked):
        cos = (doc_vectors.astype(np.float64) @ query_vectors[i].astype(np.float64)) / (
            norms * qnorms[i]
        )
        order = sorted(range(len(doc_ids)), key=lambda j: (-cos[j], doc_ids[j]))
        exact[i] = [doc_ids[j] for j in order[:10]] == list(r.chunk_ids[:10])
        worst = max(
            worst,
            float(
                np.abs(
                    np.array([cos[doc_ids.index(c)] for c in r.chunk_ids[:10]])
                    - np.array(r.scores[:10])
                ).max()
            ),
        )
    sanity["top10_equals_float64_cosine_bruteforce"] = int(exact.sum())
    sanity["max_score_diff_vs_float64_cosine"] = worst
    sanity["ranked_every_chunk"] = all(len(r.chunk_ids) == len(doc_ids) for r in ranked)
    sanity["search"] = "exact: one dot product per query and chunk, numpy matrix product, no index"
    decoder: Any = tokenizer  # tokenizers.Tokenizer, untyped: decode is not in TokenizerLike
    prefix_ok = {
        "document": all(
            decoder.decode(list(i.input_ids), skip_special_tokens=True).startswith(
                config.document_prefix.strip()
            )
            for i in doc_inputs
        ),
        "query": all(
            decoder.decode(list(i.input_ids), skip_special_tokens=True).startswith(
                config.query_prefix.strip()
            )
            for i in query_inputs
        ),
    }
    sanity["prefixes"] = prefix_ok
    embedded_is_text = all(
        list(i.input_ids)
        == list(
            tokenizer.encode(em.model_input(config, "document", chunk_by_id[i.item_id].text)).ids
        )
        for i in doc_inputs
        if not i.record.truncated
    ) and all(
        list(i.input_ids)
        == list(tokenizer.encode(em.model_input(config, "query", query_text[i.item_id])).ids)
        for i in query_inputs
    )
    sanity["embedded_string_is_prefix_plus_text_only"] = embedded_is_text
    enc = encoder()
    twice = timed("query embeddings, second pass", timings, lambda: enc.encode(query_inputs))
    sanity["same_query_vectors_twice"] = bool(np.array_equal(twice, query_vectors))
    sanity["documents_embedded_in_this_run"] = fresh_docs

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    performance = {
        "documents": len(doc_ids),
        "queries": len(query_ids),
        "dimension": int(doc_vectors.shape[1]),
        "timings_seconds": timings,
        "embedding_bytes": {
            "documents": (doc_dir / em.VECTORS_FILE).stat().st_size,
            "documents_meta": (doc_dir / em.META_FILE).stat().st_size,
            "queries": (query_dir / em.VECTORS_FILE).stat().st_size,
        },
        "peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
        "model_cache_bytes": model_cache_bytes(config),
        "environment": environment,
    }
    summary = {
        "model": config.model_dump(mode="json"),
        "config_hash": config.config_hash(),
        "identity": {
            "tokenizer_sha256": loaded.sha256,
            "runtime": runtime.model_dump(mode="json"),
            "document_vectors_sha256": em.matrix_sha256(doc_vectors),
            "query_vectors_sha256": em.matrix_sha256(query_vectors),
            "run_sha256": hashlib.sha256((run_dir / "run.jsonl").read_bytes()).hexdigest(),
        },
        "benchmark_version": dev.manifest.benchmark_version,
        "parser_version": dev.manifest.parser_version,
        "layer": layer_src.model_dump(mode="json"),
        "queries_sha256": query_src.queries_sha256,
        "tokenization": tokenization,
        "exposure": exposed,
        "metrics": {
            "official_by_language": official,
            "by_style": styles,
            "macro_by_need": macro,
            "by_type": by_type,
            "by_difficulty": by_difficulty,
            "by_answerability": by_answerability,
            "by_driver": by_driver,
            "language_effect": language,
        },
        "per_need": need_table,
        "per_query": {
            s.query.query_id: dict(s.values)
            | {
                "first_relevant_rank": first_relevant(s.query.query_id, s.query.need_id),
                "relevant_ranks": {
                    c: rank_of[s.query.query_id][c]
                    for c, g in sorted(targets[s.query.need_id].chunk_grades.items())
                    if g >= bm.RELEVANT_GRADE
                },
            }
            for s in scores
        },
        "failures": failures,
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
    for lang_style, row in styles.items():
        print(f"  {lang_style}: " + ", ".join(f"{m} {row[m]:.3f}" for m in METRICS))
    print("  macro by need: " + ", ".join(f"{m} {macro[m]:.3f}" for m in METRICS))
    print(f"  sanity: {sanity}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
