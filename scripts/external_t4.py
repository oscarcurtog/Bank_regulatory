"""T4 (D-053): the frozen retrievers on JuriFindIT's anti-money-laundering questions, once, then
evaluated. Protocol fixed before any download: config/external/jurifindit-aml.md.

    PYTHONPATH=. .venv/bin/python scripts/external_t4.py <step>

Steps, in order; each refuses to run before the ones it needs:
  download   the three pinned parquet files, checked by SHA-256
  adapt      corpus, queries and qrels in separate files (ingest/external.py)
  audit      lengths, truncation exposure, duplicates and sources, without qrels
  bm25       BM25 with the standard Italian analysis (config/retrieval/bm25-italian.json)
  e5-embed   the frozen E5 contract on the corpus and the questions
  e5-run     E5's exact search
  e-gate     E's decomposer prompts, counted against the frozen 16,384-token limit
  e-estimate E's runtime on three civil-law questions, never an evaluated one
  e-run      E exactly as frozen, within six hours
  freeze     the SHA-256 of every run, written before any qrel is read
  evaluate   the metrics, the paired differences and the strata: the qrels enter here
  analyse    the failure analysis, after the metrics

No step tunes anything: there is one run per system. Every retrieval step reads only the
corpus and the questions; the qrels path is known to evaluate and analyse alone.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import resource
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ingest import decomposition as dc
from ingest import embeddings as em
from ingest import external as ex
from ingest import ir_metrics as im
from ingest import lexical as lx
from ingest import requirement_rerank as rq
from ingest import retrieval as rt

E5 = Path("config/embeddings/multilingual-e5-large.json")
BM25 = Path("config/retrieval/bm25-italian.json")
RERANKER = Path("config/retrieval/qwen3-reranker-0.6b.json")
DECOMPOSER = Path("config/retrieval/qwen3-4b-instruct-mlx.json")
MLX_PYTHON = Path(".cache/mlx-venv/bin/python")
MAX_NEW_TOKENS = 256  # D's, scripts/decomposition.py
DEPTH = 1000
E_LIMIT_HOURS = 6.0
SEED = 20261013  # ingest/run_report.py's
REPS = 10_000
RECALL = (5, 10, 20, 60, 100)
NDCG = (5, 10, 100)
RR = (5, 10, 100)
AP = (5, 100)
OFFICIAL = (
    "Recall@5",
    "Recall@20",
    "Recall@60",
    "Recall@100",
    "nDCG@5",
    "nDCG@100",
    "MRR@5",
    "MRR@100",
    "mAP@5",
    "mAP@100",
)
OURS = ("Recall@10", "nDCG@10", "MRR@10")
PRIMARY = ("Recall@10", "nDCG@10")
STRATA = ((1, 4, "1-4"), (5, 15, "5-15"), (16, 10**9, "16+"))
SYSTEMS = ("bm25-italian", "multilingual-e5-large", "e")
EVAL_SETS = {"validation-27": ("aml-validation",), "pooled-108": ("aml-validation", "aml-train")}
AML_SETS = ("aml-validation", "aml-train")

RAW = ex.DATA_DIR / ex.RAW
ADAPTED = ex.DATA_DIR / ex.ADAPTED
WORK = ex.DATA_DIR / "work"
RESULTS = Path(f"data/results/external/jurifindit-aml@{ex.REVISION[:8]}")


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def peak_rss() -> int:
    """This process's peak resident memory, in bytes (macOS reports bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"{path} is missing: run the step that writes it first")
    return json.loads(path.read_text(encoding="utf-8"))


def perf(step: str, record: Mapping[str, Any]) -> None:
    write_json(
        WORK / "perf" / f"{step}.json",
        {"step": step, "finished": now(), **record, "environment": em.environment_description()},
    )


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aml_queries() -> list[tuple[str, str]]:
    return [q for name in AML_SETS for q in ex.load_queries(ADAPTED, name)]


# ---------------------------------------------------------------------------
# E5 vectors of the external corpus and questions, apart from Fundamento's
# ---------------------------------------------------------------------------


def vector_identity(
    config: em.EmbeddingConfig,
    loaded: em.LoadedTokenizer,
    manifest: ex.AdaptManifest,
    source: str,
) -> dict[str, Any]:
    """What a vector set was computed from: the contract, the tokenizer, the runtime, and the
    adapted file (never Fundamento's chunk layer)."""
    return {
        "contract": config.model_dump(mode="json"),
        "config_hash": config.config_hash(),
        "tokenizer_sha256": loaded.sha256,
        "runtime": em.runtime_identity(config).model_dump(mode="json"),
        "data": {
            "dataset": ex.DATASET,
            "revision": ex.REVISION,
            "adapter_version": manifest.adapter_version,
            "file": source,
            "sha256": manifest.outputs[source].sha256,
        },
    }


def vector_dir(name: str) -> Path:
    return WORK / "embeddings" / "multilingual-e5-large" / name


def load_vectors(
    config: em.EmbeddingConfig, loaded: em.LoadedTokenizer, name: str, source: str
) -> tuple[list[str], np.ndarray, list[dict[str, Any]]]:
    """A stored set, refused unless its identity is the current one and its matrix is the one
    its meta records."""
    manifest = ex.load_manifest(ADAPTED)
    meta = read_json(vector_dir(name) / "meta.json")
    if meta["identity"] != vector_identity(config, loaded, manifest, source):
        raise SystemExit(f"{vector_dir(name)}: computed with another identity; not reused")
    vectors = np.load(vector_dir(name) / "vectors.npy")
    if em.matrix_sha256(vectors) != meta["vectors_sha256"]:
        raise SystemExit(f"{vector_dir(name)}: the matrix is not the one its meta records")
    em.check_vectors(vectors, len(meta["item_ids"]), config)
    return list(meta["item_ids"]), vectors, list(meta["token_records"])


def encode_set(
    config: em.EmbeddingConfig,
    loaded: em.LoadedTokenizer,
    encoder: Any,
    items: Sequence[tuple[str, str]],
    kind: em.Kind,
    name: str,
    source: str,
) -> dict[str, Any]:
    manifest = ex.load_manifest(ADAPTED)
    start = time.perf_counter()
    inputs = em.prepare_inputs(items, kind, loaded.tokenizer, config)
    vectors = encoder.encode(inputs)
    seconds = time.perf_counter() - start
    out = vector_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", vectors)
    records = [i.record.model_dump(mode="json") for i in inputs]
    write_json(
        out / "meta.json",
        {
            "identity": vector_identity(config, loaded, manifest, source),
            "item_ids": [i.item_id for i in inputs],
            "token_records": records,
            "vectors_sha256": em.matrix_sha256(vectors),
            "created_at": now(),
            "seconds": seconds,
            "environment": encoder.environment(),
        },
    )
    return {
        "items": len(inputs),
        "seconds": seconds,
        "truncated": sum(r["truncated"] for r in records),
        "vectors_sha256": em.matrix_sha256(vectors),
    }


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------


def run_path(system: str) -> Path:
    return WORK / "runs" / system / rt.RUN_FILE


def write_system_run(system: str, ranked: Sequence[rt.Ranked]) -> str:
    cut = [rt.Ranked(r.query_id, r.chunk_ids[:DEPTH], r.scores[:DEPTH]) for r in ranked]
    if any(len(r.chunk_ids) != DEPTH for r in cut):
        raise SystemExit(f"{system}: a ranking shorter than the depth of {DEPTH}")
    return rt.write_run(run_path(system), cut, [r.query_id for r in cut])


def read_system_run(system: str) -> dict[str, tuple[str, ...]]:
    frozen = read_json(RESULTS / "runs.json")
    ranked, _ = rt.read_run(run_path(system), frozen["runs"][system]["sha256"])
    return {r.query_id: r.chunk_ids for r in ranked}


# ---------------------------------------------------------------------------
# steps before any system runs
# ---------------------------------------------------------------------------


def step_download(args: argparse.Namespace) -> int:
    fetched = ex.download(RAW)
    print(f"verified {len(ex.FILES)} pinned files; fetched {fetched or 'none'}")
    return 0


def step_adapt(args: argparse.Namespace) -> int:
    manifest = ex.adapt(RAW, ADAPTED)
    print(json.dumps(manifest.counts))
    return 0


def quantiles(values: Sequence[int]) -> dict[str, float]:
    ordered = sorted(values)

    def at(q: float) -> float:
        return float(ordered[min(len(ordered) - 1, int(q * len(ordered)))])

    return {
        "min": float(ordered[0]),
        "median": float(statistics.median(ordered)),
        "p90": at(0.90),
        "p99": at(0.99),
        "max": float(ordered[-1]),
        "mean": float(statistics.fmean(ordered)),
    }


def step_audit(args: argparse.Namespace) -> int:
    """What the adapted data look like, read without a single qrel."""
    corpus = ex.load_corpus(ADAPTED)
    documents = ex.load_documents(ADAPTED)
    sets = {name: ex.load_queries(ADAPTED, name) for name in ex.QUERY_SETS}
    config = em.load_config(E5)
    loaded = em.load_tokenizer(config)
    tok = loaded.tokenizer

    def tokens(kind: em.Kind, text: str) -> int:
        return len(tok.encode(em.model_input(config, kind, text)).ids)

    doc_tokens = [tokens("document", t) for _, t in corpus]
    over = [n for n in doc_tokens if n > config.max_input_tokens]
    query_tokens = {name: [tokens("query", t) for _, t in qs] for name, qs in sets.items()}
    analyzer = lx.Analyzer(lx.load_config(BM25).analyzer)
    terms = [len(analyzer(t)) for _, t in corpus]
    texts = Counter(t for _, t in corpus)
    question_texts = Counter(t for name in AML_SETS for _, t in sets[name])
    sources = Counter(
        ("eu" if f.split("§")[2] == "eu" else "it") for f, _ in documents.values() if "§" in f
    )
    record: dict[str, Any] = {
        "corpus": {
            "articles": len(corpus),
            "e5_tokens": quantiles(doc_tokens),
            "e5_truncated": len(over),
            "e5_truncated_share": len(over) / len(corpus),
            "e5_dropped_tokens": sum(n - config.max_input_tokens for n in over),
            "bm25_terms": quantiles(terms),
            "bm25_empty": [d for (d, _), n in zip(corpus, terms, strict=True) if n == 0],
            "chars": quantiles([len(t) for _, t in corpus]),
            "duplicate_texts": {
                "groups": sum(1 for n in texts.values() if n > 1),
                "articles": sum(n for n in texts.values() if n > 1),
            },
            "articles_by_jurisdiction": dict(sources),
            "documents": len({f for f, _ in documents.values()}),
        },
        "queries": {
            name: {
                "questions": len(qs),
                "e5_tokens": quantiles(query_tokens[name]) if qs else None,
                "over_e5_limit": sum(n > config.max_input_tokens for n in query_tokens[name]),
                "chars": quantiles([len(t) for _, t in qs]) if qs else None,
            }
            for name, qs in sets.items()
        },
        "duplicate_questions_aml": sum(n - 1 for n in question_texts.values() if n > 1),
        "e5_limit": config.max_input_tokens,
        "read_qrels": False,
    }
    write_json(WORK / "audit.json", record)
    print(json.dumps({k: record["corpus"][k] for k in ("articles", "e5_truncated")}))
    over_limit = {n: v["over_e5_limit"] for n, v in record["queries"].items()}
    print("queries over the E5 limit:", over_limit)
    print("BM25 documents with no term:", len(record["corpus"]["bm25_empty"]))
    return 0


# ---------------------------------------------------------------------------
# BM25 and E5
# ---------------------------------------------------------------------------


def step_bm25(args: argparse.Namespace) -> int:
    """BM25 once. A document with no index term scores 0 for every query (the protocol's
    rule): it is left out of the index and ranked with the other zeros, by id."""
    config = lx.load_config(BM25)
    corpus = ex.load_corpus(ADAPTED)
    queries = aml_queries()
    analyzer = lx.Analyzer(config.analyzer)
    indexed = [(d, t) for d, t in corpus if analyzer(t)]
    position = {d: i for i, (d, _) in enumerate(corpus)}
    doc_ids = [d for d, _ in corpus]
    start = time.perf_counter()
    index = lx.build_index(config, indexed)
    build_seconds = time.perf_counter() - start
    ranked = []
    start = time.perf_counter()
    for qid, text in queries:
        scores = np.zeros(len(corpus), dtype=np.float64)
        partial = index.scores(text)
        for d, s in zip(index.doc_ids, partial, strict=True):
            scores[position[d]] = s
        ranked.append(rt.rank_by_scores(qid, doc_ids, scores))
    query_seconds = time.perf_counter() - start
    sha = write_system_run("bm25-italian", ranked)
    perf(
        "bm25",
        {
            "config_hash": config.config_hash(),
            "indexed_documents": len(indexed),
            "unindexed_documents": len(corpus) - len(indexed),
            "index_seconds": build_seconds,
            "query_seconds": query_seconds,
            "seconds_per_query": query_seconds / len(queries),
            "peak_rss_bytes": peak_rss(),
            "run_sha256": sha,
        },
    )
    print(f"BM25: {len(queries)} queries, run {sha[:12]}")
    return 0


def step_e5_embed(args: argparse.Namespace) -> int:
    from ingest.encoder import Encoder

    config = em.load_config(E5)
    loaded = em.load_tokenizer(config)
    start = time.perf_counter()
    encoder = Encoder(config)
    load_seconds = time.perf_counter() - start
    done: dict[str, Any] = {"model_load_seconds": load_seconds}
    done["corpus"] = encode_set(
        config, loaded, encoder, ex.load_corpus(ADAPTED), "document", "corpus", "corpus.jsonl"
    )
    for name in ex.QUERY_SETS:
        done[name] = encode_set(
            config,
            loaded,
            encoder,
            ex.load_queries(ADAPTED, name),
            "query",
            name,
            f"queries/{name}.jsonl",
        )
    perf("e5-embed", {**done, "peak_rss_bytes": peak_rss(), "encoder": encoder.environment()})
    print(json.dumps({k: v for k, v in done.items() if k != "model_load_seconds"}))
    return 0


def e5_rankings(names: Sequence[str]) -> tuple[dict[str, rt.Ranked], list[str], np.ndarray]:
    """E5's full ranking of every question of the named sets, from the stored vectors."""
    config = em.load_config(E5)
    loaded = em.load_tokenizer(config)
    doc_ids, doc_vectors, _ = load_vectors(config, loaded, "corpus", "corpus.jsonl")
    out: dict[str, rt.Ranked] = {}
    for name in names:
        query_ids, query_vectors, _ = load_vectors(config, loaded, name, f"queries/{name}.jsonl")
        for r in em.rank_exact(query_ids, query_vectors, doc_ids, doc_vectors):
            out[r.query_id] = rt.Ranked(r.query_id, r.chunk_ids, r.scores)
    return out, doc_ids, doc_vectors


def step_e5_run(args: argparse.Namespace) -> int:
    from ingest.encoder import Encoder

    start = time.perf_counter()
    ranked, doc_ids, doc_vectors = e5_rankings(AML_SETS)
    search_seconds = time.perf_counter() - start
    order = [qid for qid, _ in aml_queries()]
    sha = write_system_run("multilingual-e5-large", [ranked[q] for q in order])
    # one warm question at a time, encoding included: the latency a user would see
    config = em.load_config(E5)
    loaded = em.load_tokenizer(config)
    encoder = Encoder(config)
    texts = aml_queries()
    encoder.encode(em.prepare_inputs([texts[0]], "query", loaded.tokenizer, config))  # warm-up
    latencies = []
    for qid, text in texts:
        t0 = time.perf_counter()
        vector = encoder.encode(em.prepare_inputs([(qid, text)], "query", loaded.tokenizer, config))
        scores = doc_vectors @ vector[0]
        rt.rank_by_scores(qid, doc_ids, scores)
        latencies.append(time.perf_counter() - t0)
    latencies.sort()
    perf(
        "e5-run",
        {
            "rank_all_seconds_from_stored_vectors": search_seconds,
            "warm_single_question_seconds": {
                "p50": latencies[len(latencies) // 2],
                "p95": latencies[int(0.95 * len(latencies)) - 1],
                "max": latencies[-1],
                "n": len(latencies),
            },
            "documents": len(doc_ids),
            "peak_rss_bytes": peak_rss(),
            "run_sha256": sha,
        },
    )
    print(f"E5: {len(order)} queries, run {sha[:12]}")
    return 0


# ---------------------------------------------------------------------------
# E, exactly as frozen
# ---------------------------------------------------------------------------


def text_of() -> dict[str, str]:
    return dict(ex.load_corpus(ADAPTED))


def decomposer_rows(
    queries: Sequence[tuple[str, str]], e5: Mapping[str, rt.Ranked], texts: Mapping[str, str]
) -> list[dict[str, Any]]:
    """D's prompt for each question: the question and E5's first 3 results, whole, as D's
    script builds them (scripts/decomposition.py)."""
    return [
        {
            "id": f"d#{qid}",
            "messages": dc.requirements_prompt(
                text, [texts[c] for c in e5[qid].chunk_ids[: dc.EXCERPTS]]
            ),
        }
        for qid, text in queries
    ]


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def count_tokens(prompts: Path, out: Path) -> dict[str, Any]:
    command = [str(MLX_PYTHON), "scripts/mlx_count_tokens.py", "--config", str(DECOMPOSER)]
    command += ["--prompts", str(prompts), "--out", str(out)]
    subprocess.run(command, check=True, env=os.environ | {"HF_HOME": str(em.hf_home())})
    counts: dict[str, Any] = read_json(out)
    return counts


def run_decomposer(prompts: Path, outputs: Path) -> float:
    """The MLX worker of D-040, as D runs it (scripts/decomposition.py, run_worker)."""
    if not MLX_PYTHON.exists():
        raise SystemExit(f"{MLX_PYTHON} is missing: the MLX environment of D-040 is not set up")
    command = [str(MLX_PYTHON), "scripts/mlx_generate.py", "--config", str(DECOMPOSER)]
    command += ["--prompts", str(prompts), "--out", str(outputs)]
    command += ["--max-new-tokens", str(MAX_NEW_TOKENS)]
    start = time.perf_counter()
    subprocess.run(command, check=True, env=os.environ | {"HF_HOME": str(em.hf_home())})
    return time.perf_counter() - start


def parsed_answers(outputs: Path) -> dict[str, dc.Requirements]:
    rows = [json.loads(line) for line in outputs.read_text(encoding="utf-8").splitlines()]
    return {r["id"].split("#", 1)[1]: dc.parse(r["output"]) for r in rows}


def requirement_hits(
    answers: Mapping[str, dc.Requirements], doc_ids: Sequence[str], doc_vectors: np.ndarray
) -> dict[str, list[tuple[str, ...]]]:
    """E5's first 50 articles for each readable requirement, in the model's order: the frozen
    script's requirement_hits, on this corpus."""
    from ingest.encoder import Encoder

    config = em.load_config(E5)
    loaded = em.load_tokenizer(config)
    texts = sorted({x for p in answers.values() if p.readable for x in p.items})
    encoder = Encoder(config)
    inputs = em.prepare_inputs(
        [(str(i), t) for i, t in enumerate(texts)], "query", loaded.tokenizer, config
    )
    vectors = dict(zip(texts, encoder.encode(inputs), strict=True))
    return {
        qid: [
            rt.rank_by_scores(qid, doc_ids, doc_vectors @ vectors[x]).chunk_ids[: rq.DEPTH]
            for x in (p.items if p.readable else ())
        ]
        for qid, p in answers.items()
    }


def jobs_of(
    answers: Mapping[str, dc.Requirements],
    hits: Mapping[str, list[tuple[str, ...]]],
    texts: Mapping[str, str],
) -> dict[tuple[str, str], tuple[str, str, str]]:
    jobs: dict[tuple[str, str], tuple[str, str, str]] = {}
    for qid, p in answers.items():
        for x, ranking in zip(p.items if p.readable else (), hits[qid], strict=True):
            for c in ranking:
                jobs.setdefault((x, c), (rq.INSTRUCTION, x, texts[c]))
    return jobs


def input_sha(fields: tuple[str, str, str]) -> str:
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode("utf-8")).hexdigest()


def score_jobs(
    jobs: Mapping[tuple[str, str], tuple[str, str, str]],
    scores_path: Path,
    deadline: float | None,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """The frozen script's score_all: one shared opening per requirement, every score appended
    as it comes, a stored one reused only if its input is the same. With a deadline (a
    perf_counter value), it stops before a group that starts after it."""
    from ingest import necessity as nc

    done: dict[tuple[str, str], dict[str, Any]] = {}
    if scores_path.exists():
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
    reranker = nc.NecessityReranker(nc.load_config(RERANKER))
    load_seconds = time.perf_counter() - start
    # every prompt built before any score, only to record their lengths first: a long pair
    # is what can exhaust the device's memory. The same prompts, scored in the same order.
    built = {g: [reranker.prompt(*jobs[k]) for k in part] for g, part in groups.items()}
    lengths = [len(p.ids) for prompts in built.values() for p in prompts]
    if lengths:
        write_json(
            scores_path.with_name(scores_path.stem + "-inputs.json"),
            {
                "pending_pairs": len(lengths),
                "tokens": quantiles(lengths),
                "over": {str(n): sum(x > n for x in lengths) for n in (1024, 2048, 4096, 8192)},
                "truncated": sum(p.truncated for prompts in built.values() for p in prompts),
                "groups": len(built),
                "groups_with_a_pair_over_4096": sum(
                    any(len(p.ids) > 4096 for p in prompts) for prompts in built.values()
                ),
            },
        )
    scored, stopped, mps_peak = 0, False, 0
    torch = reranker._torch
    for g, (group, part) in enumerate(groups.items(), start=1):
        if deadline is not None and time.perf_counter() > deadline:
            stopped = True
            break
        prompts = built.pop(group)
        values = reranker.score_shared(prompts)
        if torch.backends.mps.is_available():
            mps_peak = max(mps_peak, int(torch.mps.driver_allocated_memory()))
        rows = [
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
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        with scores_path.open("a", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        done |= {(str(r["requirement"]), str(r["chunk"])): r for r in rows}
        scored += len(part)
        if g % 20 == 0 or g == len(groups):
            elapsed = time.perf_counter() - start
            left = elapsed / scored * (len(pending) - scored) / 60
            print(
                f"scored {scored}/{len(pending)} (group {g}/{len(groups)}) in {elapsed:.0f} s, "
                f"about {left:.0f} min left",
                flush=True,
            )
    seconds = time.perf_counter() - start
    tokens = [r["tokens"] for r in done.values()]
    return done, {
        "pairs": len(jobs),
        "scored_now": scored,
        "seconds": seconds,
        "model_load_seconds": load_seconds,
        "seconds_per_pair": seconds / scored if scored else None,
        "tokens": quantiles(tokens) if tokens else None,
        "truncated": sum(r["truncated"] for r in done.values()),
        "stopped_at_deadline": stopped,
        "mps_driver_peak_bytes": mps_peak,
        "peak_rss_bytes": peak_rss(),
    }


def reranked_hits(
    answers: Mapping[str, dc.Requirements],
    hits: Mapping[str, list[tuple[str, ...]]],
    done: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, list[tuple[str, ...]]]:
    return {
        qid: [
            tuple(rq.reorder(ranking, {c: done[(x, c)]["logit_diff"] for c in ranking}))
            for x, ranking in zip(p.items if p.readable else (), hits[qid], strict=True)
        ]
        for qid, p in answers.items()
    }


def step_e_gate(args: argparse.Namespace) -> int:
    """E's decomposer prompts for the 108 questions, counted with the frozen limit before a
    single token is generated. The gate covers all 108: one over the limit and E is not
    applicable under its frozen contract, for both evaluation sets."""
    e5, _, _ = e5_rankings(AML_SETS)
    rows = decomposer_rows(aml_queries(), e5, text_of())
    sha = write_rows(WORK / "e" / "prompts.jsonl", rows)
    counts = count_tokens(WORK / "e" / "prompts.jsonl", WORK / "e" / "prompt-tokens.json")
    limit = counts["max_prompt_tokens"]
    over = sorted(rid for rid, n in counts["counts"].items() if n > limit)
    record = {
        "prompts": len(rows),
        "prompts_sha256": sha,
        "limit": limit,
        "tokens": quantiles(list(counts["counts"].values())),
        "over_limit": over,
        "passes": not over,
        "read_qrels": False,
    }
    write_json(WORK / "e" / "gate.json", record)
    print(
        f"gate: {len(rows)} prompts, {record['tokens']['max']:.0f} tokens at most, "
        f"{len(over)} over {limit}; passes {record['passes']}"
    )
    return 0


def step_e_estimate(args: argparse.Namespace) -> int:
    """E on the three civil-law questions, end to end, to project its time on the 108: no
    evaluated question and no qrel enters."""
    if not read_json(WORK / "e" / "gate.json")["passes"]:
        raise SystemExit("E's gate did not pass: E is not applicable, nothing to estimate")
    out = WORK / "e-estimate"
    e5, doc_ids, doc_vectors = e5_rankings(("estimate",))
    texts = text_of()
    queries = ex.load_queries(ADAPTED, "estimate")
    rows = decomposer_rows(queries, e5, texts)
    write_rows(out / "prompts.jsonl", rows)
    counts = count_tokens(out / "prompts.jsonl", out / "prompt-tokens.json")
    if max(counts["counts"].values()) > counts["max_prompt_tokens"]:
        raise SystemExit("an estimate prompt is over the limit: choose nothing else, report it")
    decompose_seconds = run_decomposer(out / "prompts.jsonl", out / "outputs.jsonl")
    answers = parsed_answers(out / "outputs.jsonl")
    start = time.perf_counter()
    hits = requirement_hits(answers, doc_ids, doc_vectors)
    hits_seconds = time.perf_counter() - start
    jobs = jobs_of(answers, hits, texts)
    _, scoring = score_jobs(jobs, out / "scores.jsonl", None)
    per_question = (decompose_seconds + hits_seconds + scoring["seconds"]) / len(queries)
    projected = per_question * len(aml_queries()) / 3600
    record = {
        "questions": [q for q, _ in queries],
        "area": ex.ESTIMATE_AREA,
        "decompose_seconds": decompose_seconds,
        "requirement_search_seconds": hits_seconds,
        "scoring": scoring,
        "requirements": sum(len(p.items) for p in answers.values() if p.readable),
        "seconds_per_question": per_question,
        "projected_hours_for_108": projected,
        "limit_hours": E_LIMIT_HOURS,
        "passes": projected <= E_LIMIT_HOURS,
        "read_qrels": False,
    }
    write_json(out / "estimate.json", record)
    print(
        f"estimate: {per_question:.0f} s per question, {projected:.2f} h projected for 108; "
        f"passes {record['passes']}"
    )
    return 0


def step_e_run(args: argparse.Namespace) -> int:
    """E once, exactly as frozen, on the 108 questions. Stops, and E is reported as not
    completed, if its time passes six hours."""
    if not read_json(WORK / "e" / "gate.json")["passes"]:
        raise SystemExit("E's gate did not pass: E is not applicable under its frozen contract")
    if not read_json(WORK / "e-estimate" / "estimate.json")["passes"]:
        raise SystemExit("E's estimate projects more than six hours: E does not run")
    started = time.perf_counter()
    deadline = started + E_LIMIT_HOURS * 3600
    out = WORK / "e"
    e5, doc_ids, doc_vectors = e5_rankings(AML_SETS)
    texts = text_of()
    queries = aml_queries()
    rows = decomposer_rows(queries, e5, texts)
    if write_rows(out / "prompts.jsonl", rows) != read_json(out / "gate.json")["prompts_sha256"]:
        raise SystemExit("the prompts are not the ones the gate counted")
    decompose_seconds = run_decomposer(out / "prompts.jsonl", out / "outputs.jsonl")
    answers = parsed_answers(out / "outputs.jsonl")
    if set(answers) != {q for q, _ in queries}:
        raise SystemExit("the decomposer's answers do not cover the 108 questions")
    t0 = time.perf_counter()
    hits = requirement_hits(answers, doc_ids, doc_vectors)
    hits_seconds = time.perf_counter() - t0
    write_json(out / "hits.json", {q: [list(h) for h in v] for q, v in hits.items()})
    jobs = jobs_of(answers, hits, texts)
    done, scoring = score_jobs(jobs, out / "scores.jsonl", deadline)
    if scoring["stopped_at_deadline"] or len(done) < len(jobs):
        write_json(
            out / "e-record.json",
            {
                "completed": False,
                "reason": "six hours passed before every pair was scored",
                "decompose_seconds": decompose_seconds,
                "scoring": scoring,
            },
        )
        raise SystemExit("E stopped at its six-hour limit: not completed, reported as such")
    hits_rr = reranked_hits(answers, hits, done)
    assembled = {q: dc.assemble(e5[q].chunk_ids, hits_rr[q]) for q, _ in queries}
    ranked = []
    for qid, _ in queries:
        order = assembled[qid].ranking
        n = len(order)
        ranked.append(rt.Ranked(qid, order, tuple(float(n - i) for i in range(n))))
    sha = write_system_run("e", ranked)
    total = time.perf_counter() - started
    record = {
        "completed": True,
        "questions": len(queries),
        "readable_answers": sum(p.readable for p in answers.values()),
        "unreadable_kept_e5": sorted(q for q, p in answers.items() if not p.readable),
        "requirements": sum(len(p.items) for p in answers.values() if p.readable),
        "pairs": len(jobs),
        "added_from_outside_e5_top10": sum(len(a.added) for a in assembled.values()),
        "corroborated": sum(len(a.corroborated) for a in assembled.values()),
        "displaced": sum(len(a.displaced) for a in assembled.values()),
        "decompose_seconds": decompose_seconds,
        "requirement_search_seconds": hits_seconds,
        "scoring": scoring,
        "total_seconds": total,
        "run_sha256": sha,
        "read_qrels": False,
    }
    write_json(out / "e-record.json", record)
    write_json(
        out / "assembly.json",
        {
            q: {
                "added": list(a.added),
                "corroborated": list(a.corroborated),
                "displaced": list(a.displaced),
            }
            for q, a in assembled.items()
        },
    )
    print(f"E: {len(queries)} queries in {total / 3600:.2f} h, run {sha[:12]}")
    return 0


# ---------------------------------------------------------------------------
# the freeze: every run's hash before the qrels are read
# ---------------------------------------------------------------------------


def step_freeze(args: argparse.Namespace) -> int:
    manifest = ex.load_manifest(ADAPTED)
    runs = {}
    for system in SYSTEMS:
        path = run_path(system)
        if path.exists():
            runs[system] = {"sha256": file_sha(path), "depth": DEPTH}
    e_record = WORK / "e" / "e-record.json"
    gate = read_json(WORK / "e" / "gate.json")
    frozen = {
        "frozen_at": now(),
        "dataset": {
            "id": ex.DATASET,
            "revision": ex.REVISION,
            "license": ex.LICENSE,
            "inputs": {k: v.model_dump() for k, v in manifest.inputs.items()},
            "adapter_version": manifest.adapter_version,
            "adapted_outputs": {k: v.sha256 for k, v in manifest.outputs.items()},
        },
        "configs": {
            "bm25-italian": lx.load_config(BM25).config_hash(),
            "multilingual-e5-large": em.load_config(E5).config_hash(),
            "qwen3-reranker-0.6b": file_sha(RERANKER),
            "qwen3-4b-instruct-mlx": file_sha(DECOMPOSER),
        },
        "e_gate": {k: gate[k] for k in ("prompts", "limit", "over_limit", "passes")},
        "e_completed": read_json(e_record)["completed"] if e_record.exists() else False,
        "runs": runs,
        "qrels_read_before": False,
    }
    write_json(RESULTS / "runs.json", frozen)
    print(json.dumps(runs))
    return 0


# ---------------------------------------------------------------------------
# evaluation: the qrels enter here
# ---------------------------------------------------------------------------


def paired(diffs: Sequence[float]) -> dict[str, Any]:
    """Mean difference with a 95% percentile bootstrap over queries (run_report's seed and
    resamples), and how many queries got better, stayed equal or got worse."""
    rng = random.Random(SEED)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(REPS))
    return {
        "mean": statistics.fmean(diffs),
        "median": statistics.median(diffs),
        "ci95_low": means[int(0.025 * REPS)],
        "ci95_high": means[int(0.975 * REPS) - 1],
        "better": sum(d > 1e-9 for d in diffs),
        "equal": sum(abs(d) <= 1e-9 for d in diffs),
        "worse": sum(d < -1e-9 for d in diffs),
        "min": min(diffs),
        "max": max(diffs),
        "n": len(diffs),
    }


def verdict(p: Mapping[str, Any]) -> str:
    """The protocol's reading of a primary difference, fixed in advance."""
    if p["ci95_low"] > 0:
        return "holds"
    if p["ci95_high"] < 0:
        return "reverses"
    return "inconclusive"


def stratum_of(n: int) -> str:
    return next(label for lo, hi, label in STRATA if lo <= n <= hi)


def step_evaluate(args: argparse.Namespace) -> int:
    frozen = read_json(RESULTS / "runs.json")
    systems = [s for s in SYSTEMS if s in frozen["runs"]]
    runs = {s: read_system_run(s) for s in systems}
    loaded_qrels = {name: ex.load_qrels(ADAPTED, name) for name in AML_SETS}
    # a question with no relevant article cannot be scored: left out, as trec_eval does,
    # and counted (the protocol's rule)
    empty = sorted(q for name in AML_SETS for q, docs in loaded_qrels[name].items() if not docs)
    qrels = {name: {q: docs for q, docs in loaded_qrels[name].items() if docs} for name in AML_SETS}
    corpus_ids = {d for d, _ in ex.load_corpus(ADAPTED)}
    relevant = {q: docs for name in AML_SETS for q, docs in qrels[name].items()}
    outside = {q: sorted(d for d in docs if d not in corpus_ids) for q, docs in relevant.items()}
    per_query = {
        s: im.binary_scores(relevant, {q: list(runs[s][q]) for q in relevant}, RECALL, NDCG, RR, AP)
        for s in systems
    }
    if "e" in systems:  # E keeps E5's first 6: every @5 metric must be equal, by construction
        for q in relevant:
            for m in ("Recall@5", "nDCG@5", "MRR@5", "mAP@5"):
                if abs(per_query["e"][q][m] - per_query["multilingual-e5-large"][q][m]) > 1e-12:
                    raise SystemExit(f"{q}: E and E5 differ on {m}; E did not keep E5's top 6")
    metrics = [*OFFICIAL, *OURS]
    report: dict[str, Any] = {
        "frozen_runs": frozen["runs"],
        "relevant_per_query": {
            name: quantiles([len(qrels[name][q]) for q in qrels[name]]) for name in AML_SETS
        },
        "relevant_outside_corpus": {q: v for q, v in outside.items() if v},
        "questions_without_relevant": empty,
        "sets": {},
    }
    comparisons = [("multilingual-e5-large", "bm25-italian")]
    if "e" in systems:
        comparisons.append(("e", "multilingual-e5-large"))
    for label, names in EVAL_SETS.items():
        ids = [q for name in names for q in qrels[name]]
        block: dict[str, Any] = {
            "questions": len(ids),
            "means": {
                s: {m: statistics.fmean(per_query[s][q][m] for q in ids) for m in metrics}
                for s in systems
            },
            "paired": {},
            "strata": {},
        }
        for a, b in comparisons:
            block["paired"][f"{a} - {b}"] = {
                m: paired([per_query[a][q][m] - per_query[b][q][m] for q in ids])
                for m in (*PRIMARY, "MRR@10", "Recall@20")
            }
            for m in PRIMARY:
                block["paired"][f"{a} - {b}"][m]["verdict"] = verdict(
                    block["paired"][f"{a} - {b}"][m]
                )
        for _, _, label_s in STRATA:
            members = [q for q in ids if stratum_of(len(relevant[q])) == label_s]
            if not members:
                continue
            block["strata"][label_s] = {
                "questions": len(members),
                "means": {
                    s: {m: statistics.fmean(per_query[s][q][m] for q in members) for m in OURS}
                    for s in systems
                },
                "mean_differences": {
                    f"{a} - {b}": {
                        m: statistics.fmean(
                            per_query[a][q][m] - per_query[b][q][m] for q in members
                        )
                        for m in PRIMARY
                    }
                    for a, b in comparisons
                },
            }
        report["sets"][label] = block
    report["per_query"] = {
        q: {
            "set": next(n for n in AML_SETS if q in qrels[n]),
            "relevant": len(relevant[q]),
            **{s: {m: per_query[s][q][m] for m in metrics} for s in systems},
        }
        for q in relevant
    }
    report["backend"] = im.backend().__dict__
    report["evaluated_at"] = now()
    write_json(RESULTS / "report.json", report)
    (RESULTS / "report.md").write_text(render(report, systems), encoding="utf-8")
    print(json.dumps({k: v["means"] for k, v in report["sets"].items()}, indent=1)[:4000])
    return 0


NAMES = {"bm25-italian": "BM25 (Italian)", "multilingual-e5-large": "E5", "e": "E"}


def render(report: Mapping[str, Any], systems: Sequence[str]) -> str:
    """The report as Markdown: numbers and question ids only, never a text of the dataset."""
    lines = [
        "# T4: the frozen retrieval on JuriFindIT's anti-money-laundering questions",
        "",
        f"Protocol: `config/external/jurifindit-aml.md`. Dataset `{ex.DATASET}` at "
        f"`{ex.REVISION[:12]}`. One run per system, frozen before any qrel was read "
        "(`runs.json`). **validation-27 is the external result; pooled-108 is a secondary "
        "analysis, not an official validation score.** JuriFindIT's numbers are not comparable "
        "with Fundamento's: different corpus, questions, labels and units.",
        "",
    ]
    for label, block in report["sets"].items():
        lines += [f"## {label} ({block['questions']} questions)", ""]
        lines += ["Official metrics (JuriFindIT):", ""]
        head = "| System | " + " | ".join(OFFICIAL) + " |"
        lines += [head, "|---" * (len(OFFICIAL) + 1) + "|"]
        for s in systems:
            row = " | ".join(f"{block['means'][s][m]:.3f}" for m in OFFICIAL)
            lines.append(f"| {NAMES[s]} | {row} |")
        lines += ["", "Our compatible metrics:", ""]
        lines += ["| System | " + " | ".join(OURS) + " |", "|---" * (len(OURS) + 1) + "|"]
        for s in systems:
            row = " | ".join(f"{block['means'][s][m]:.3f}" for m in OURS)
            lines.append(f"| {NAMES[s]} | {row} |")
        lines += [
            "",
            "Paired differences (mean, 95% bootstrap interval over questions, better / equal / "
            "worse):",
            "",
            "| Comparison | Metric | Mean | 95% interval | Better / equal / worse |",
            "|---|---|---|---|---|",
        ]
        for comparison, by_metric in block["paired"].items():
            a, b = comparison.split(" - ")
            for m, p in by_metric.items():
                primary = f" (primary: {p['verdict']})" if m in PRIMARY else ""
                lines.append(
                    f"| {NAMES[a]} − {NAMES[b]} | {m}{primary} | {p['mean']:+.3f} | "
                    f"[{p['ci95_low']:+.3f}, {p['ci95_high']:+.3f}] | "
                    f"{p['better']} / {p['equal']} / {p['worse']} |"
                )
        lines += ["", "By number of relevant articles (descriptive):", ""]
        lines += [
            "| Relevant | Questions | "
            + " | ".join(f"{NAMES[s]} R@10 / nDCG@10" for s in systems)
            + " |"
        ]
        lines += ["|---" * (len(systems) + 2) + "|"]
        for stratum, sb in block["strata"].items():
            cells = " | ".join(
                f"{sb['means'][s]['Recall@10']:.3f} / {sb['means'][s]['nDCG@10']:.3f}"
                for s in systems
            )
            lines.append(f"| {stratum} | {sb['questions']} | {cells} |")
        lines.append("")
    outside = report["relevant_outside_corpus"]
    lines += [
        "## Notes",
        "",
        f"- Relevant ids outside the 23,458 articles: {sum(len(v) for v in outside.values())} "
        f"in {len(outside)} questions (kept in the qrels; they count in the denominators).",
        f"- Questions without any relevant article: {len(report['questions_without_relevant'])}.",
        "- E keeps E5's first 6 results: every @5 metric of E equals E5's, checked per question.",
        f"- Metrics computed by ir-measures {report['backend']['ir_measures']} with "
        f"pytrec-eval-terrier {report['backend']['pytrec_eval_terrier']}.",
        "",
    ]
    return "\n".join(lines)


def step_analyse(args: argparse.Namespace) -> int:
    """The per-question material of the failure analysis, private (texts of the dataset are
    not redistributed): written under data/external/, never into the repository."""
    report = read_json(RESULTS / "report.json")
    frozen = read_json(RESULTS / "runs.json")
    systems = [s for s in SYSTEMS if s in frozen["runs"]]
    runs = {s: read_system_run(s) for s in systems}
    queries = dict(aml_queries())
    texts = text_of()
    documents = ex.load_documents(ADAPTED)
    relevant = {q: docs for n in AML_SETS for q, docs in ex.load_qrels(ADAPTED, n).items()}
    answers = parsed_answers(WORK / "e" / "outputs.jsonl") if "e" in systems else {}
    assembly = read_json(WORK / "e" / "assembly.json") if "e" in systems else {}
    hits = read_json(WORK / "e" / "hits.json") if "e" in systems else {}
    config = em.load_config(E5)
    loaded = em.load_tokenizer(config)
    _, _, records = load_vectors(config, loaded, "corpus", "corpus.jsonl")
    truncated = {r["item_id"] for r in records if r["truncated"]}
    out = {}
    for q, text in queries.items():
        rel = relevant[q]
        ranks = {s: sorted(i + 1 for i, d in enumerate(runs[s][q]) if d in rel) for s in systems}
        out[q] = {
            "question": text,
            "chars": len(text),
            "relevant": len(rel),
            "relevant_sources": Counter(
                documents[d][0].split("§")[2] if d in documents else "outside" for d in rel
            ),
            "relevant_files": Counter(documents[d][0] for d in rel if d in documents),
            "relevant_truncated_by_e5": sum(d in truncated for d in rel),
            "relevant_chars_median": statistics.median(len(texts[d]) for d in rel if d in texts)
            if any(d in texts for d in rel)
            else None,
            "ranks": {s: v[:20] for s, v in ranks.items()},
            "first_rank": {s: (v[0] if v else None) for s, v in ranks.items()},
            "metrics": {s: report["per_query"][q][s] for s in systems},
            "requirements": list(answers[q].items) if q in answers else [],
            "e_added": [
                {"doc": d, "relevant": d in rel} for d in assembly.get(q, {}).get("added", [])
            ],
            "e_displaced": [
                {"doc": d, "relevant": d in rel} for d in assembly.get(q, {}).get("displaced", [])
            ],
            "requirement_hits_relevant_top5_vs_top50": [
                {"top5_e5": sum(d in rel for d in h[:5]), "top50_e5": sum(d in rel for d in h)}
                for h in hits.get(q, [])
            ],
        }
    write_json(WORK / "analysis.json", out)
    print(f"analysis material for {len(out)} questions in {WORK / 'analysis.json'}")
    return 0


STEPS = {
    "download": step_download,
    "adapt": step_adapt,
    "audit": step_audit,
    "bm25": step_bm25,
    "e5-embed": step_e5_embed,
    "e5-run": step_e5_run,
    "e-gate": step_e_gate,
    "e-estimate": step_e_estimate,
    "e-run": step_e_run,
    "freeze": step_freeze,
    "evaluate": step_evaluate,
    "analyse": step_analyse,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=list(STEPS))
    args = parser.parse_args(argv)
    return STEPS[args.step](args)


if __name__ == "__main__":
    sys.exit(main())
